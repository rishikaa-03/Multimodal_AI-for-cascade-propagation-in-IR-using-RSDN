"""
Integration tests -- verify DATA FLOWING BETWEEN modules stays correct,
not just each module in isolation (that's test_core.py / test_ai_modules.py).
Each test here exercises a real handoff: Module 2 -> 3 -> 4/5/7/8.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
from multiplex_graph import MultiplexGraph
from cascade_simulator import CascadeSimulator
from cross_zone_detector import CrossZoneFrontierDetector
from cascade_predictor import CascadePredictor, add_graph_features
from cause_classifier import CauseClassifier
from recommendation_engine import RecommendationEngine
from network_risk_summary import compute_network_risk

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data"))


@pytest.fixture(scope="module")
def mg():
    return MultiplexGraph.build_from_csv(DATA_DIR)


# ---------------------------------------------------------------------
# 2 -> 3: graph feeds the simulator correctly
# ---------------------------------------------------------------------

def test_graph_to_simulator_every_train_in_routes_is_simulable(mg):
    """Every (train, station) pair that actually appears in routes.csv
    must be simulate-able without raising, proving Module 3 correctly
    reads the SCHED_DEP edges Module 2 built."""
    sim = CascadeSimulator(mg)
    routes = pd.read_csv(f"{DATA_DIR}/routes.csv")
    sample = routes.sample(min(30, len(routes)), random_state=1)
    for _, row in sample.iterrows():
        result = sim.simulate_event(
            train_number=int(row.train_number), station_id=row.station_id, delay_minutes=60.0,
        )
        assert result is not None  # no exception, well-formed result object


# ---------------------------------------------------------------------
# 3 -> 5: simulator's traversed edges line up with frontier detector's
# own zone-crossing bookkeeping
# ---------------------------------------------------------------------

def test_simulator_frontier_checks_only_fire_at_real_frontier_stations(mg):
    sim = CascadeSimulator(mg)
    detector = CrossZoneFrontierDetector(mg)
    events = pd.read_csv(f"{DATA_DIR}/delay_events.csv").sample(20, random_state=2)
    for _, row in events.iterrows():
        result = sim.simulate_event(
            train_number=int(row.train_number), station_id=row.station_id,
            delay_minutes=float(row.delay_minutes), frontier_detector=detector,
        )
        for check in result.frontier_checks:
            assert detector.is_frontier(check["station_id"]), (
                f"simulator fired a frontier check at {check['station_id']}, "
                f"which the detector itself doesn't consider a frontier"
            )


# ---------------------------------------------------------------------
# 3 -> 4: simulator's graph-grounded output is what add_graph_features/
# CascadePredictor actually train on -- confirm the merge doesn't silently
# drop or misalign rows
# ---------------------------------------------------------------------

def test_simulator_output_merges_cleanly_into_predictor_pipeline(mg):
    sim = CascadeSimulator(mg)
    events = pd.read_csv(f"{DATA_DIR}/delay_events.csv")
    gsd = sim.simulate_from_delay_events(f"{DATA_DIR}/delay_events.csv")

    merged = events.merge(gsd, on="event_id", suffixes=("", "_sim"))
    assert len(merged) == len(events), "merge on event_id dropped or duplicated rows"

    merged = add_graph_features(merged, mg)
    predictor = CascadePredictor()
    report = predictor.train(merged)
    # end-to-end: this only succeeds if every feature column survived the
    # merge with correct dtypes and no NaNs the pipeline chokes on
    assert report["cascade_depth"]["r2"] is not None


# ---------------------------------------------------------------------
# 3 -> 7: recommendation engine's counterfactual re-simulation must use
# the SAME simulator/graph as the baseline run, or the comparison is invalid
# ---------------------------------------------------------------------

def test_recommendation_counterfactual_uses_same_graph_as_baseline(mg):
    sim = CascadeSimulator(mg)
    events = pd.read_csv(f"{DATA_DIR}/delay_events.csv")
    # find one event that actually cascades
    for _, row in events.sample(50, random_state=3).iterrows():
        baseline = sim.simulate_event(
            train_number=int(row.train_number), station_id=row.station_id,
            delay_minutes=float(row.delay_minutes),
        )
        if baseline.traversed_edges:
            break
    else:
        pytest.skip("no cascading event found in this sample")

    engine = RecommendationEngine(sim)
    recs = engine.generate(
        train_number=int(row.train_number), station_id=row.station_id,
        delay_minutes=float(row.delay_minutes), baseline=baseline,
    )
    # re-running the identical baseline call must reproduce identical
    # delay_spread_minutes -- proves no hidden state leaked between the
    # engine's internal re-simulations and this baseline
    repeat = sim.simulate_event(
        train_number=int(row.train_number), station_id=row.station_id,
        delay_minutes=float(row.delay_minutes),
    )
    assert repeat.delay_spread_minutes == baseline.delay_spread_minutes


# ---------------------------------------------------------------------
# 3 -> 8: network_risk_summary's per-edge tallies must be internally
# consistent with re-simulating those same events directly
# ---------------------------------------------------------------------

def test_network_risk_summary_matches_direct_resimulation(mg):
    sim = CascadeSimulator(mg)
    events = pd.read_csv(f"{DATA_DIR}/delay_events.csv")
    station_df, edge_df = compute_network_risk(mg, f"{DATA_DIR}/delay_events.csv")

    # re-simulate the same events independently and check that every
    # traversed edge is reflected as at least "cascade" risk in edge_df
    seen_edges = set()
    for _, row in events.sample(30, random_state=4).iterrows():
        result = sim.simulate_event(
            train_number=int(row.train_number), station_id=row.station_id,
            delay_minutes=float(row.delay_minutes),
        )
        for hop in result.traversed_edges:
            seen_edges.add((hop["src"], hop["dst"], hop["edge_type"]))

    edge_lookup = {(r.src, r.dst, r.edge_type): r.risk_level for r in edge_df.itertuples()}
    for key in seen_edges:
        assert edge_lookup.get(key) in ("cascade", "cross_zone"), (
            f"edge {key} was actually traversed but network_risk_summary marked it normal"
        )


# ---------------------------------------------------------------------
# 4a -> dashboard: cause classifier's output must be a valid label the
# dashboard can display directly, not raw indices or unexpected types
# ---------------------------------------------------------------------

def test_cause_classifier_output_is_dashboard_ready(mg):
    events = pd.read_csv(f"{DATA_DIR}/delay_events.csv")
    clf = CauseClassifier()
    clf.train(events["reported_text"], events["cause_category"])
    pred = clf.predict(pd.Series([events["reported_text"].iloc[0]]))[0]
    assert isinstance(pred, str)
    assert pred in set(events["cause_category"].unique())

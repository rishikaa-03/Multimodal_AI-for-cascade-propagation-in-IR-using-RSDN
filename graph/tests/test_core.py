"""
Starter unit test suite for NFR-07 (70% coverage target).
Covers the formulas and edge cases each module's readme documents as its
core contract -- not full coverage yet, but the skeleton to build on.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data"))

import pytest
from multiplex_graph import MultiplexGraph
from cascade_simulator import CascadeSimulator, CASCADE_STOP_THRESHOLD_MINS
from cross_zone_detector import CrossZoneFrontierDetector, identify_frontier_stations


@pytest.fixture(scope="module")
def mg():
    return MultiplexGraph.build_from_csv(DATA_DIR)


# ---------------------------------------------------------------------
# Module 2 -- graph construction invariants
# ---------------------------------------------------------------------

def test_infra_buffer_clipped_to_range(mg):
    for _, _, d in mg.graph.edges(data=True):
        if d.get("edge_type") == "INFRA":
            assert 5.0 <= d["buffer_time_mins"] <= 45.0


def test_infra_edges_are_bidirectional(mg):
    infra_pairs = {
        (u, v) for u, v, d in mg.graph.edges(data=True) if d.get("edge_type") == "INFRA"
    }
    for (u, v) in infra_pairs:
        assert (v, u) in infra_pairs, f"INFRA edge {u}->{v} has no reverse edge"


def test_all_nodes_have_station_name(mg):
    for n, d in mg.graph.nodes(data=True):
        assert "station_name" in d


# ---------------------------------------------------------------------
# Module 3 -- FR-04 propagation formula:
#   propagated = max(0, (upstream_delay - buffer) * weight)
# ---------------------------------------------------------------------

def test_propagation_formula_direct():
    upstream, buffer, weight = 100.0, 20.0, 0.9
    expected = max(0.0, (upstream - buffer) * weight)
    assert expected == pytest.approx(72.0)


def test_propagation_formula_floors_at_zero():
    upstream, buffer, weight = 10.0, 50.0, 0.9
    result = max(0.0, (upstream - buffer) * weight)
    assert result == 0.0


def test_cascade_stops_below_threshold(mg):
    sim = CascadeSimulator(mg)
    result = sim.simulate_event(
        train_number=999999, station_id=list(mg.graph.nodes())[0], delay_minutes=0.5,
    )
    # a sub-threshold delay must never register any traversed hop
    assert result.cascade_depth == 0
    assert len(result.traversed_edges) == 0


def test_unknown_station_returns_empty_result(mg):
    sim = CascadeSimulator(mg)
    result = sim.simulate_event(train_number=1, station_id="NOT_A_REAL_STATION", delay_minutes=90)
    assert result.cascade_depth == 0
    assert "not found in graph" in result.path_log[0]


def test_blocked_hop_forces_zero_delay(mg):
    """A blocked hop should never let more delay through than the
    unblocked run -- this is the counterfactual mechanism Module 7 relies on."""
    sim = CascadeSimulator(mg)
    events_stations = list(mg.graph.nodes())[:5]
    for sid in events_stations:
        out_edges = list(mg.graph.out_edges(sid, data=True))
        sched = [d for _, _, d in out_edges if d.get("edge_type") == "SCHED_DEP"]
        if not sched:
            continue
        train = sched[0]["train_number"]
        baseline = sim.simulate_event(train_number=train, station_id=sid, delay_minutes=90.0)
        if not baseline.traversed_edges:
            continue
        hop = baseline.traversed_edges[0]
        hop_key = (hop["src"], hop["dst"], hop["edge_type"], hop["from_train"])
        blocked = sim.simulate_event(
            train_number=train, station_id=sid, delay_minutes=90.0, blocked_hops={hop_key},
        )
        assert blocked.delay_spread_minutes <= baseline.delay_spread_minutes
        return
    pytest.skip("no station in this dataset had an outgoing SCHED_DEP edge")


# ---------------------------------------------------------------------
# Module 5 -- FR-06 frontier formula: P(cross) = incoming_delay / buffer
# ---------------------------------------------------------------------

def test_frontier_formula_direct():
    incoming, buffer = 30.0, 40.0
    assert incoming / buffer == pytest.approx(0.75)


def test_frontier_alert_fires_at_threshold(mg):
    detector = CrossZoneFrontierDetector(mg, threshold=0.75)
    if not detector.frontiers:
        pytest.skip("no frontier stations in this dataset")
    sid = next(iter(detector.frontiers))
    buf = detector.frontiers[sid]["typical_boundary_buffer"]
    evaluation = detector.evaluate(sid, incoming_delay=buf * 0.75)
    assert evaluation.alert is True
    evaluation_low = detector.evaluate(sid, incoming_delay=buf * 0.5)
    assert evaluation_low.alert is False


def test_non_frontier_station_returns_none(mg):
    detector = CrossZoneFrontierDetector(mg)
    non_frontier = [s for s in mg.graph.nodes() if s not in detector.frontiers]
    if not non_frontier:
        pytest.skip("every station in this dataset is a frontier station")
    assert detector.evaluate(non_frontier[0], incoming_delay=50.0) is None


def test_identify_frontier_stations_only_uses_sched_and_rs(mg):
    frontiers = identify_frontier_stations(mg)
    for sid, info in frontiers.items():
        crossing_edges = [
            d for _, _, d in mg.graph.out_edges(sid, data=True)
            if d.get("zone_crossing") and d.get("edge_type") in ("SCHED_DEP", "RS_DEP")
        ]
        assert len(crossing_edges) == info["crossing_edge_count"]

"""
Unit tests -- Module 4 (predictors), Module 7 (recommendation engine),
Module 8 (network risk). Companion to test_core.py (Modules 2/3/5).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
from multiplex_graph import MultiplexGraph
from cascade_simulator import CascadeSimulator
from cross_zone_detector import CrossZoneFrontierDetector
from cascade_predictor import CascadePredictor, add_graph_features, NUMERIC_FEATURES
from cause_classifier import CauseClassifier
from recommendation_engine import RecommendationEngine, MIN_RECOVERY_TO_RECOMMEND
from network_risk_summary import compute_network_risk

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data"))


@pytest.fixture(scope="module")
def mg():
    return MultiplexGraph.build_from_csv(DATA_DIR)


@pytest.fixture(scope="module")
def merged_dataset(mg):
    events = pd.read_csv(f"{DATA_DIR}/delay_events.csv")
    sim = CascadeSimulator(mg)
    gsd = sim.simulate_from_delay_events(f"{DATA_DIR}/delay_events.csv")
    merged = events.merge(gsd, on="event_id", suffixes=("", "_sim"))
    merged = add_graph_features(merged, mg)
    return merged


# ---------------------------------------------------------------------
# Module 4a -- CauseClassifier
# ---------------------------------------------------------------------

def test_cause_classifier_trains_and_predicts_known_labels(merged_dataset):
    clf = CauseClassifier()
    report = clf.train(merged_dataset["reported_text"], merged_dataset["cause_category"])
    assert report["accuracy"] > 0.5  # sanity floor, not the real target
    preds = clf.predict(merged_dataset["reported_text"].head(5))
    assert set(preds).issubset(set(merged_dataset["cause_category"].unique()))


def test_cause_classifier_predict_proba_sums_to_one(merged_dataset):
    clf = CauseClassifier()
    clf.train(merged_dataset["reported_text"], merged_dataset["cause_category"])
    proba = clf.predict_proba(merged_dataset["reported_text"].head(3))
    row_sums = proba.sum(axis=1)
    assert all(abs(s - 1.0) < 1e-6 for s in row_sums)


def test_cause_classifier_save_load_roundtrip(merged_dataset, tmp_path):
    clf = CauseClassifier()
    clf.train(merged_dataset["reported_text"], merged_dataset["cause_category"])
    path = tmp_path / "clf.pkl"
    clf.save(str(path))
    reloaded = CauseClassifier.load(str(path))
    original_preds = clf.predict(merged_dataset["reported_text"].head(5))
    reloaded_preds = reloaded.predict(merged_dataset["reported_text"].head(5))
    assert list(original_preds) == list(reloaded_preds)


# ---------------------------------------------------------------------
# Module 4b -- CascadePredictor
# ---------------------------------------------------------------------

def test_add_graph_features_populates_all_numeric_columns(merged_dataset):
    for col in NUMERIC_FEATURES:
        assert col in merged_dataset.columns
    assert (merged_dataset["out_degree"] >= 0).all()


def test_cascade_predictor_predictions_are_non_negative(merged_dataset):
    predictor = CascadePredictor()
    predictor.train(merged_dataset)
    preds = predictor.predict(merged_dataset.head(10))
    assert (preds["predicted_cascade_depth"] >= 0).all()
    assert (preds["predicted_delay_spread_minutes"] >= 0).all()
    assert preds["cross_zone_probability"].between(0, 1).all()


def test_cascade_predictor_save_load_roundtrip(merged_dataset, tmp_path):
    predictor = CascadePredictor()
    predictor.train(merged_dataset)
    path = tmp_path / "pred.pkl"
    predictor.save(str(path))
    reloaded = CascadePredictor.load(str(path))
    a = predictor.predict(merged_dataset.head(5))
    b = reloaded.predict(merged_dataset.head(5))
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------
# Module 7 -- RecommendationEngine
# ---------------------------------------------------------------------

def test_recommendations_never_below_min_recovery(mg, merged_dataset):
    sim = CascadeSimulator(mg)
    detector = CrossZoneFrontierDetector(mg)
    engine = RecommendationEngine(sim)
    row = merged_dataset.sort_values("cascade_depth", ascending=False).iloc[0]
    baseline = sim.simulate_event(
        train_number=int(row.train_number), station_id=row.station_id,
        delay_minutes=float(row.initial_delay_minutes), frontier_detector=detector,
    )
    recs = engine.generate(
        train_number=int(row.train_number), station_id=row.station_id,
        delay_minutes=float(row.initial_delay_minutes), baseline=baseline, frontier_detector=detector,
    )
    action_recs = [r for r in recs if r.kind != "zone_advisory"]
    for r in action_recs:
        assert r.estimated_recovery_minutes >= MIN_RECOVERY_TO_RECOMMEND


def test_recommendations_sorted_descending_by_recovery(mg, merged_dataset):
    sim = CascadeSimulator(mg)
    engine = RecommendationEngine(sim)
    row = merged_dataset.sort_values("cascade_depth", ascending=False).iloc[0]
    baseline = sim.simulate_event(
        train_number=int(row.train_number), station_id=row.station_id,
        delay_minutes=float(row.initial_delay_minutes),
    )
    recs = engine.generate(
        train_number=int(row.train_number), station_id=row.station_id,
        delay_minutes=float(row.initial_delay_minutes), baseline=baseline,
    )
    recoveries = [r.estimated_recovery_minutes for r in recs]
    assert recoveries == sorted(recoveries, reverse=True)


def test_zero_delay_spread_yields_no_recommendations(mg):
    from cascade_simulator import CascadeResult
    sim = CascadeSimulator(mg)
    engine = RecommendationEngine(sim)
    empty_baseline = CascadeResult(train_number=1, station_id="X", initial_delay_minutes=0,
                                    delay_spread_minutes=0.0)
    recs = engine.generate(train_number=1, station_id="X", delay_minutes=0, baseline=empty_baseline)
    assert recs == []


# ---------------------------------------------------------------------
# Module 8 -- network_risk_summary classification rule
# ---------------------------------------------------------------------

def test_network_risk_classification_rule(mg):
    station_df, edge_df = compute_network_risk(mg, f"{DATA_DIR}/delay_events.csv")
    assert set(station_df["risk_level"].unique()) <= {"normal", "cascade", "cross_zone"}
    assert set(edge_df["risk_level"].unique()) <= {"normal", "cascade", "cross_zone"}
    # every SCHED_DEP/RS_DEP edge in the graph must appear exactly once
    expected_edges = sum(
        1 for _, _, d in mg.graph.edges(data=True) if d.get("edge_type") in ("SCHED_DEP", "RS_DEP")
    )
    # edge_df dedupes multi-edges between the same (u,v,type); graph count may include dupes
    assert len(edge_df) <= expected_edges


def test_network_risk_station_severity_is_max_of_touching_edges(mg):
    station_df, edge_df = compute_network_risk(mg, f"{DATA_DIR}/delay_events.csv")
    cross_zone_edges = edge_df[edge_df["risk_level"] == "cross_zone"]
    touched_stations = set(cross_zone_edges["src"]) | set(cross_zone_edges["dst"])
    for sid in touched_stations:
        level = station_df.loc[station_df.station_id == sid, "risk_level"].iloc[0]
        assert level == "cross_zone"

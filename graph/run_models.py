"""
Run this file to execute Module 4 end-to-end:
    cd RailwayCascadeAI/graph
    python3 run_models.py

Merges delay_events.csv with graph_simulation_dataset.csv (Module 3's
output) on event_id, trains the cause classifier and the cascade
predictor, prints evaluation metrics, saves both trained models, and
demonstrates the fused output on a few sample events.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiplex_graph import MultiplexGraph  # noqa: E402
from cause_classifier import CauseClassifier  # noqa: E402
from cascade_predictor import CascadePredictor, add_graph_features  # noqa: E402


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(here, "..", "data"))

    delay_events_path = os.path.join(data_dir, "delay_events.csv")
    graph_sim_path = os.path.join(here, "graph_simulation_dataset.csv")

    print("Loading and merging data...")
    delay_events = pd.read_csv(delay_events_path)
    graph_sim = pd.read_csv(graph_sim_path)
    merged = delay_events.merge(graph_sim, on="event_id", suffixes=("", "_sim"))
    print(f"  Merged dataset: {len(merged)} rows, {merged.shape[1]} columns")

    print("Attaching graph-derived features (degree, fan-out, criticality)...")
    mg = MultiplexGraph.load(os.path.join(here, "multiplex_graph.pkl"))
    merged = add_graph_features(merged, mg)

    # ---- Module 4a: cause classifier ----------------------------------------
    print("\n=== Training cause classifier (TF-IDF + Logistic Regression) ===")
    clf = CauseClassifier()
    clf_report = clf.train(merged["reported_text"], merged["cause_category"])
    print(f"  Train/test split: {clf_report['n_train']}/{clf_report['n_test']}")
    print(f"  Accuracy:  {clf_report['accuracy']:.3f}")
    print(f"  Macro F1:  {clf_report['macro_f1']:.3f}")
    print("\n  Full classification report:")
    print(clf_report["classification_report"])

    clf_path = os.path.join(here, "cause_classifier.pkl")
    clf.save(clf_path)
    print(f"  Saved to: {clf_path}")

    # ---- Module 4b: cascade predictor ---------------------------------------
    print("\n=== Training cascade predictor (Random Forest ensemble) ===")
    predictor = CascadePredictor()
    pred_report = predictor.train(merged)

    print(f"  cascade_depth          MAE={pred_report['cascade_depth']['mae']:.3f}  "
          f"R2={pred_report['cascade_depth']['r2']:.3f}")
    print(f"  delay_spread_minutes   MAE={pred_report['delay_spread_minutes']['mae']:.3f}  "
          f"R2={pred_report['delay_spread_minutes']['r2']:.3f}")
    print(f"  cross_zone_propagation Accuracy={pred_report['cross_zone_propagation']['accuracy']:.3f}  "
          f"F1={pred_report['cross_zone_propagation']['f1']:.3f}")

    pred_path = os.path.join(here, "cascade_predictor.pkl")
    predictor.save(pred_path)
    print(f"  Saved to: {pred_path}")

    # ---- Fused output demo on 5 real sample events --------------------------
    print("\n=== Fused prediction demo (5 real sample events) ===")
    sample = merged.sample(5, random_state=1).reset_index(drop=True)

    cause_preds = clf.predict(sample["reported_text"])
    cascade_preds = predictor.predict(sample)

    for i in range(len(sample)):
        print(f"\n  Event {sample.loc[i, 'event_id']} — train {sample.loc[i, 'train_number']} "
              f"at {sample.loc[i, 'station_id']}, {sample.loc[i, 'initial_delay_minutes']} min delay")
        print(f"    Actual cause: {sample.loc[i, 'cause_category']:<15} | Predicted cause: {cause_preds[i]}")
        print(f"    Actual cascade_depth: {sample.loc[i, 'cascade_depth']} | "
              f"Predicted: {cascade_preds.loc[i, 'predicted_cascade_depth']:.1f}")
        print(f"    Actual cross_zone: {bool(sample.loc[i, 'cross_zone_propagation'])} | "
              f"Predicted: {bool(cascade_preds.loc[i, 'predicted_cross_zone_propagation'])} "
              f"(prob={cascade_preds.loc[i, 'cross_zone_probability']:.2f})")

    print("\nModule 4 complete. Ready for Module 5 (Cross-Zone Cascade Frontier Detector).")


if __name__ == "__main__":
    main()
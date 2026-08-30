"""
Historical validation (Phase 5 deliverable requirement): "validating
predictions against historical cascade events."

"Historical events" = delay_events.csv run through the cascade simulator
(Module 3), the project's ground truth for what a cascade actually did --
same convention run_frontier_detector.py uses for precision/recall.

Trains on an 80% split, evaluates ONLY on the held-out 20% -- genuinely
unseen events, not the training set. Writes a per-event comparison CSV
and prints the aggregate metrics used in the test report.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.model_selection import train_test_split

from multiplex_graph import MultiplexGraph
from cascade_simulator import CascadeSimulator
from cross_zone_detector import CrossZoneFrontierDetector
from cascade_predictor import CascadePredictor, add_graph_features

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR = os.path.dirname(HERE)
DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data"))


def main():
    mg = MultiplexGraph.build_from_csv(DATA_DIR)
    sim = CascadeSimulator(mg)
    detector = CrossZoneFrontierDetector(mg)

    events = pd.read_csv(os.path.join(DATA_DIR, "delay_events.csv"))
    sim_df = sim.simulate_from_delay_events(
        os.path.join(DATA_DIR, "delay_events.csv"), frontier_detector=detector
    )
    merged = events.merge(sim_df, on="event_id", suffixes=("", "_sim"))
    merged["initial_delay_minutes"] = merged["delay_minutes"]
    merged = add_graph_features(merged, mg)

    # ---- 80/20 split: train only on 80%, evaluate only on the held-out 20% ----
    train_df, test_df = train_test_split(merged, test_size=0.2, random_state=42)

    predictor = CascadePredictor()
    predictor.train(train_df)
    preds = predictor.predict(test_df).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    report = pd.DataFrame({
        "event_id": test_df["event_id"],
        "train_number": test_df["train_number"],
        "station_id": test_df["station_id"],
        "actual_cascade_depth": test_df["cascade_depth"],
        "predicted_cascade_depth": preds["predicted_cascade_depth"].round(1),
        "depth_error": (preds["predicted_cascade_depth"] - test_df["cascade_depth"]).round(1),
        "actual_delay_spread": test_df["delay_spread_minutes"],
        "predicted_delay_spread": preds["predicted_delay_spread_minutes"].round(1),
        "spread_error": (preds["predicted_delay_spread_minutes"] - test_df["delay_spread_minutes"]).round(1),
        "actual_cross_zone": test_df["cross_zone_propagation"].astype(bool),
        "predicted_cross_zone": preds["predicted_cross_zone_propagation"].astype(bool),
    })
    report["cross_zone_correct"] = report["actual_cross_zone"] == report["predicted_cross_zone"]

    out_path = os.path.join(HERE, "historical_validation_report.csv")
    report.to_csv(out_path, index=False)

    depth_mae = report["depth_error"].abs().mean()
    exact_match = (report["depth_error"] == 0).mean()
    within_1 = (report["depth_error"].abs() <= 1).mean()
    spread_mae = report["spread_error"].abs().mean()
    cz_acc = report["cross_zone_correct"].mean()

    print(f"Held-out events: {len(report)} (trained on {len(train_df)})")
    print(f"\ncascade_depth        MAE={depth_mae:.2f}  exact_match={100*exact_match:.1f}%  "
          f"within_1_hop={100*within_1:.1f}%")
    print(f"delay_spread_minutes MAE={spread_mae:.1f} min")
    print(f"cross_zone_propagation accuracy={100*cz_acc:.1f}%")

    # ---- Frontier detector: all historical zone-crossing evaluations ----
    all_checks = [c for checks in sim_df["frontier_checks"] for c in checks]
    tp = sum(1 for c in all_checks if c["alert"] and c["actually_crossed"])
    fp = sum(1 for c in all_checks if c["alert"] and not c["actually_crossed"])
    fn = sum(1 for c in all_checks if not c["alert"] and c["actually_crossed"])
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    print(f"\nFrontier detector over {len(all_checks)} historical zone-crossing evaluations:")
    print(f"  precision={precision:.3f}  (SRS target >= 0.80)")
    print(f"  recall={recall:.3f}  (SRS target >= 0.75)")

    print(f"\nSaved per-event comparison to: {out_path}")


if __name__ == "__main__":
    main()

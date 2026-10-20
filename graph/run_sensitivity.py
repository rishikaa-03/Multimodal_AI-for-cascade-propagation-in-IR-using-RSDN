"""
Sensitivity of cascade results and detector performance to the hand-set
edge weights and buffers (addresses reviewer point 3).

    cd RailwayCascadeAI/graph
    python3 run_sensitivity.py

Writes sensitivity_results.csv.
"""

import itertools
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiplex_graph import MultiplexGraph  # noqa: E402
from cascade_simulator import CascadeSimulator  # noqa: E402
from cross_zone_detector import CrossZoneFrontierDetector  # noqa: E402

SCHED_WEIGHTS = [0.70, 0.80, 0.90, 0.95, 1.00]
RS_WEIGHTS = [0.70, 0.80, 0.90, 0.97, 1.00]
BUFFER_SCALES = [0.5, 0.75, 1.5, 2.0]
BASE_SCHED, BASE_RS = 0.95, 0.97


def configure(data_dir, w_sched=None, w_rs=None, buffer_scale=1.0, drop_rs=False):
    mg = MultiplexGraph.build_from_csv(data_dir)
    to_remove = []
    for u, v, k, d in mg.graph.edges(keys=True, data=True):
        et = d.get("edge_type")
        if et == "RS_DEP":
            if drop_rs:
                to_remove.append((u, v, k))
                continue
            if w_rs is not None:
                d["propagation_weight"] = w_rs
        elif et == "SCHED_DEP" and w_sched is not None:
            d["propagation_weight"] = w_sched
        if buffer_scale != 1.0:
            d["buffer_time_mins"] = d["buffer_time_mins"] * buffer_scale
    mg.graph.remove_edges_from(to_remove)
    return mg


def evaluate(mg, events_path, threshold=0.75):
    detector = CrossZoneFrontierDetector(mg, threshold=threshold)
    df = CascadeSimulator(mg).simulate_from_delay_events(events_path, frontier_detector=detector)
    checks = [c for cs in df["frontier_checks"] for c in cs]
    tp = sum(c["alert"] and c["actually_crossed"] for c in checks)
    fp = sum(c["alert"] and not c["actually_crossed"] for c in checks)
    fn = sum((not c["alert"]) and c["actually_crossed"] for c in checks)
    tn = sum((not c["alert"]) and (not c["actually_crossed"]) for c in checks)
    return {
        "mean_depth": df["cascade_depth"].mean(),
        "mean_spread_min": df["delay_spread_minutes"].mean(),
        "cross_zone_pct": 100 * df["cross_zone_propagation"].mean(),
        "reach_other_train_pct": 100 * (df["affected_trains_count"] > 0).mean(),
        "frontier_stations": len(detector.frontiers),
        "n_checks": len(checks),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(here, "..", "data"))
    events_path = os.path.join(data_dir, "delay_events.csv")

    rows = []

    for ws, wr in itertools.product(SCHED_WEIGHTS, RS_WEIGHTS):
        mg = configure(data_dir, w_sched=ws, w_rs=wr)
        rows.append({"experiment": "weight_grid", "w_sched": ws, "w_rs": wr, "buffer_scale": 1.0,
                     "rs_layer": "on", **evaluate(mg, events_path)})
        print(f"weights w_sched={ws:.2f} w_rs={wr:.2f} done")

    for scale in BUFFER_SCALES:
        mg = configure(data_dir, buffer_scale=scale)
        rows.append({"experiment": "buffer_scale", "w_sched": BASE_SCHED, "w_rs": BASE_RS,
                     "buffer_scale": scale, "rs_layer": "on", **evaluate(mg, events_path)})
        print(f"buffer scale {scale} done")

    mg = configure(data_dir, drop_rs=True)
    rows.append({"experiment": "no_rs_layer", "w_sched": BASE_SCHED, "w_rs": float("nan"),
                 "buffer_scale": 1.0, "rs_layer": "off", **evaluate(mg, events_path)})

    out = pd.DataFrame(rows)
    out_path = os.path.join(here, "sensitivity_results.csv")
    out.to_csv(out_path, index=False)

    pd.set_option("display.width", 200)
    grid = out[out["experiment"] == "weight_grid"]
    for metric in ["mean_depth", "mean_spread_min", "cross_zone_pct", "precision", "recall"]:
        print(f"\n{metric} (rows: w_sched, cols: w_rs)")
        print(grid.pivot(index="w_sched", columns="w_rs", values=metric).round(3).to_string())

    print("\nbuffer scale sweep (baseline weights) and RS-layer ablation")
    print(out[out["experiment"] != "weight_grid"].drop(columns=["experiment"]).round(3).to_string(index=False))

    for metric in ["mean_depth", "mean_spread_min", "cross_zone_pct", "precision", "recall"]:
        print(f"range over weight grid, {metric}: {grid[metric].min():.3f} to {grid[metric].max():.3f}")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
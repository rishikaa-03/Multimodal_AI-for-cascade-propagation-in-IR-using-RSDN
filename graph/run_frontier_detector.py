"""
Run this file to execute Module 5 end-to-end:
    cd RailwayCascadeAI/graph
    python3 run_frontier_detector.py

Identifies frontier stations, re-runs the cascade simulator over all
delay_events.csv with the frontier detector attached, prints a manual
trace, and computes precision/recall of the detector's alerts against
the simulator's own ground truth of whether a delay actually crossed.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiplex_graph import MultiplexGraph  # noqa: E402
from cascade_simulator import CascadeSimulator  # noqa: E402
from cross_zone_detector import CrossZoneFrontierDetector  # noqa: E402


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(here, "..", "data"))
    graph_path = os.path.join(here, "multiplex_graph.pkl")
    delay_events_path = os.path.join(data_dir, "delay_events.csv")

    print("Loading graph and identifying frontier stations...")
    mg = MultiplexGraph.load(graph_path)
    detector = CrossZoneFrontierDetector(mg)

    print(f"  Found {len(detector.frontiers)} frontier stations "
          f"(out of {mg.graph.number_of_nodes()} total)")

    print("\nTop 5 frontier stations by number of crossing dependency edges:")
    ranked = sorted(
        detector.frontiers.items(), key=lambda x: x[1]["crossing_edge_count"], reverse=True
    )
    for station_id, info in ranked[:5]:
        name = mg.graph.nodes[station_id]["station_name"]
        print(
            f"  {station_id} ({name}): {info['crossing_edge_count']} crossing edges, "
            f"typical_boundary_buffer={info['typical_boundary_buffer']:.1f} min, "
            f"adjacent zones={info['adjacent_zones']}"
        )

    # ---- Manual trace: force a large delay through a real frontier station ----
    sample_station = ranked[0][0]
    print(f"\nManual trace: simulating a 90-minute delay originating AT frontier station {sample_station}")
    sim = CascadeSimulator(mg)

    # find a real train that departs from this station per SCHED_DEP, to keep the trace realistic
    routes = pd.read_csv(os.path.join(data_dir, "routes.csv"))
    sample_train = routes[routes["station_id"] == sample_station]["train_number"].iloc[0]

    result = sim.simulate_event(
        train_number=int(sample_train), station_id=sample_station, delay_minutes=90.0,
        frontier_detector=detector,
    )
    for line in result.path_log:
        print(f"  {line}")
    print("  Frontier checks triggered during this trace:")
    for check in result.frontier_checks:
        print(f"    {check}")

    # ---- Full batch run + precision/recall validation -------------------------
    print(f"\nRunning full simulation with frontier detection over all delay events...")
    df = sim.simulate_from_delay_events(delay_events_path, frontier_detector=detector)

    all_checks = [c for checks in df["frontier_checks"] for c in checks]
    print(f"  Total zone-crossing evaluations across all events: {len(all_checks)}")

    tp = sum(1 for c in all_checks if c["alert"] and c["actually_crossed"])
    fp = sum(1 for c in all_checks if c["alert"] and not c["actually_crossed"])
    fn = sum(1 for c in all_checks if not c["alert"] and c["actually_crossed"])
    tn = sum(1 for c in all_checks if not c["alert"] and not c["actually_crossed"])

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")

    print(f"\n  Confusion matrix @ threshold {detector.threshold}:")
    print(f"    True positives (alerted, and it did cross):  {tp}")
    print(f"    False positives (alerted, but it didn't):    {fp}")
    print(f"    False negatives (missed, but it did cross):  {fn}")
    print(f"    True negatives (no alert, and it didn't):    {tn}")
    print(f"\n  Precision: {precision:.3f}  (SRS target: >= 0.80)")
    print(f"  Recall:    {recall:.3f}  (SRS target: >= 0.75)")

    output_path = os.path.join(here, "frontier_alerts_summary.csv")
    df.drop(columns=["frontier_checks"]).to_csv(output_path, index=False)
    print(f"\nSaved event-level results to: {output_path}")
    print("Module 5 complete. This is the last core prediction module — "
          "remaining work is the dashboard and integration/testing per your Gantt.")


if __name__ == "__main__":
    main()
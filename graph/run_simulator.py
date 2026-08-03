"""
Run this file to execute Module 3 end-to-end:
    cd RailwayCascadeAI/graph
    python3 run_simulator.py

Loads the Module 2 graph, runs the cascade simulator on delay_events.csv,
prints a manual trace for inspection, and saves the graph-grounded scenario
dataset to graph_simulation_dataset.csv for later comparison against the
formula-based simulation_dataset.csv from Module 1.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiplex_graph import MultiplexGraph  # noqa: E402
from cascade_simulator import CascadeSimulator  # noqa: E402


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(here, "..", "data"))
    graph_path = os.path.join(here, "multiplex_graph.pkl")
    delay_events_path = os.path.join(data_dir, "delay_events.csv")
    output_path = os.path.join(here, "graph_simulation_dataset.csv")

    print(f"Loading graph from: {graph_path}")
    mg = MultiplexGraph.load(graph_path)
    sim = CascadeSimulator(mg)

    # ---- Manual trace on one real event, so you can see the algorithm work ----
    events = pd.read_csv(delay_events_path)
    biggest = events.sort_values("delay_minutes", ascending=False).iloc[0]

    print(
        f"\nManual trace: event {biggest['event_id']} — train {biggest['train_number']} "
        f"delayed {biggest['delay_minutes']} min at {biggest['station_id']}"
    )
    result = sim.simulate_event(
        train_number=biggest["train_number"],
        station_id=biggest["station_id"],
        delay_minutes=float(biggest["delay_minutes"]),
    )
    for line in result.path_log:
        print(f"  {line}")
    print(f"  RESULT: {result.as_dict()}")

    # ---- Batch run over all 6,000 events ----
    print(f"\nRunning full simulation over {len(events)} delay events...")
    df = sim.simulate_from_delay_events(delay_events_path)
    df.to_csv(output_path, index=False)
    print(f"Saved graph-grounded scenario dataset to: {output_path}")

    # ---- Sanity check against the existing formula-based dataset ----
    print("\nComparison with the existing formula-based simulation_dataset.csv:")
    formula_df = pd.read_csv(os.path.join(data_dir, "simulation_dataset.csv"))

    print(f"  {'metric':<28}{'graph-based (new)':<22}{'formula-based (Module 1)'}")
    print(
        f"  {'mean cascade_depth':<28}"
        f"{df['cascade_depth'].mean():<22.2f}"
        f"{formula_df['cascade_depth'].mean():.2f}"
    )
    print(
        f"  {'mean affected_trains':<28}"
        f"{df['affected_trains_count'].mean():<22.2f}"
        f"{formula_df['affected_trains_count'].mean():.2f}"
    )
    print(
        f"  {'mean delay_spread_minutes':<28}"
        f"{df['delay_spread_minutes'].mean():<22.2f}"
        f"{formula_df['delay_spread_minutes'].mean():.2f}"
    )
    print(
        f"  {'% cross_zone_propagation':<28}"
        f"{100*df['cross_zone_propagation'].mean():<22.1f}"
        f"{100*formula_df['cross_zone_propagation'].mean():.1f}"
    )

    corr = df["initial_delay_minutes"].corr(df["cascade_depth"])
    print(f"\n  correlation(initial_delay, cascade_depth) = {corr:.3f}")
    print("  (Module 1's report cites 0.97 for the formula-based data — a positive,")
    print("   reasonably strong correlation here confirms the graph simulator behaves sensibly.)")

    print("\nModule 3 complete. Ready for Module 4 (AI prediction models).")


if __name__ == "__main__":
    main()
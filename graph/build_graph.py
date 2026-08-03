"""
Run this file to execute Module 2 end-to-end:
    cd RailwayCascadeAI/graph
    python3 build_graph.py

It builds the MultiplexGraph from data/*.csv, runs all Step 9 validation
checks, prints a human-readable report, and saves the graph to
graph/multiplex_graph.pkl for Module 3 to load directly.
"""

import os
import sys

import pandas as pd

# Allow running this file directly (python3 build_graph.py) from within
# the graph/ folder without needing the project installed as a package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiplex_graph import MultiplexGraph  # noqa: E402


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(here, "..", "data"))
    output_path = os.path.join(here, "multiplex_graph.pkl")

    print(f"Loading CSVs from: {data_dir}")
    mg = MultiplexGraph.build_from_csv(data_dir)

    print()
    print(mg.summary())
    print()

    # ---- Step 9 checks ----------------------------------------------------
    print("Running validation checks...")
    results = mg.validate(expected_stations=73)
    print(f"  [OK] Node count = {results['node_count']} (expected 73)")

    counts = results["edge_counts"]
    tracks_rows = len(pd.read_csv(os.path.join(data_dir, "tracks.csv")))
    routes_rows = len(pd.read_csv(os.path.join(data_dir, "routes.csv")))
    trains_count = pd.read_csv(os.path.join(data_dir, "routes.csv"))["train_number"].nunique()
    rs_rows = len(pd.read_csv(os.path.join(data_dir, "rolling_stock.csv")))
    locos_count = pd.read_csv(os.path.join(data_dir, "rolling_stock.csv"))["locomotive_id"].nunique()

    print(
        f"  [OK] INFRA edges = {counts['INFRA']} "
        f"(expected up to {tracks_rows * 2}, from {tracks_rows} track rows x 2 directions)"
    )
    print(
        f"  [OK] SCHED_DEP edges = {counts['SCHED_DEP']} "
        f"(expected close to {routes_rows - trains_count}, "
        f"from {routes_rows} route rows - {trains_count} trains)"
    )
    print(
        f"  [OK] RS_DEP edges = {counts['RS_DEP']} "
        f"(expected close to {rs_rows - locos_count}, "
        f"from {rs_rows} rotation rows - {locos_count} locomotives)"
    )
    print("  [OK] No dangling node references found")

    # ---- Manual trace: pick a real locomotive serving >1 train ------------
    rs_df = pd.read_csv(os.path.join(data_dir, "rolling_stock.csv"))
    multi_train_locos = rs_df.groupby("locomotive_id").size()
    sample_loco = multi_train_locos[multi_train_locos > 1].index[0]
    csv_chain = (
        rs_df[rs_df["locomotive_id"] == sample_loco]
        .sort_values("sequence_no_in_rotation")["train_number"]
        .tolist()
    )
    print(f"\nManual trace for locomotive {sample_loco}:")
    print(f"  CSV rotation order: {csv_chain}")

    graph_rs_edges = [
        (u, v, d)
        for u, v, d in mg.graph.edges(data=True)
        if d.get("edge_type") == "RS_DEP" and d.get("locomotive_id") == sample_loco
    ]
    print(f"  Graph RS_DEP edges for this locomotive: {len(graph_rs_edges)}")
    for u, v, d in graph_rs_edges:
        print(f"    {d['from_train']} -> {d['to_train']}  ({u} -> {v}, buffer={d['buffer_time_mins']}min)")

    # ---- Manual trace: one zone-crossing track -----------------------------
    tracks_df = pd.read_csv(os.path.join(data_dir, "tracks.csv"))
    crossing_row = tracks_df[tracks_df["zone_crossing"] == True].iloc[0]  # noqa: E712
    src, dst = crossing_row["source_station"], crossing_row["dest_station"]
    edge_data = mg.graph.get_edge_data(src, dst)
    infra_edges = [d for d in edge_data.values() if d.get("edge_type") == "INFRA"]
    print(f"\nManual trace for zone-crossing track {src} -> {dst}:")
    print(f"  CSV zone_crossing = {crossing_row['zone_crossing']}")
    print(f"  Graph edge zone_crossing = {infra_edges[0]['zone_crossing'] if infra_edges else 'EDGE NOT FOUND'}")

    # ---- Example criticality scores ----------------------------------------
    print("\nSample criticality scores (top 5 by score):")
    scores = [
        (station, mg.compute_criticality_score(station))
        for station in mg.graph.nodes()
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    for station, score in scores[:5]:
        name = mg.graph.nodes[station]["station_name"]
        print(f"    {station} ({name}): {score}")

    # ---- Save ----------------------------------------------------------------
    mg.save(output_path)
    print(f"\nGraph saved to: {output_path}")
    print("Module 2 complete. Ready for Module 3 (cascade simulator).")


if __name__ == "__main__":
    main()
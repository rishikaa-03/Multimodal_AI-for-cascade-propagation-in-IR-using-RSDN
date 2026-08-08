"""
Module 8 — Network-wide Risk Summary
========================================

Runs the simulator across EVERY delay event (not just one), and tallies,
per station and per SCHED_DEP/RS_DEP dependency edge, whether it was ever
part of a cascade, and whether that cascade ever crossed a zone boundary.

This is what powers the dashboard's "Full Network Map" — a system-wide
view (green / red / blue) that's separate from the "Live Simulation" tab,
which only ever shows ONE chosen event's path.

Classification rule (applied identically to stations and edges):
    - never appeared in any propagated cascade                -> "normal" (green)
    - appeared in a cascade, but never a zone-crossing one     -> "cascade" (red)
    - appeared in at least one cascade that crossed a zone     -> "cross_zone" (blue)

A station's classification is the MOST severe classification of any edge
touching it, since a station is "at risk" if anything connected to it is.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiplex_graph import MultiplexGraph  # noqa: E402
from cascade_simulator import CascadeSimulator  # noqa: E402
from cross_zone_detector import CrossZoneFrontierDetector  # noqa: E402


def compute_network_risk(mg: MultiplexGraph, delay_events_path: str):
    sim = CascadeSimulator(mg)
    detector = CrossZoneFrontierDetector(mg)
    events = pd.read_csv(delay_events_path)

    # edge_key -> {"times_traversed": int, "times_cross_zone": int}
    edge_stats: dict[tuple, dict] = {}
    # station_id -> {"cascade": bool, "cross_zone": bool}
    station_stats: dict[str, dict] = {sid: {"cascade": False, "cross_zone": False} for sid in mg.graph.nodes()}

    for _, row in events.iterrows():
        result = sim.simulate_event(
            train_number=row["train_number"],
            station_id=row["station_id"],
            delay_minutes=float(row["delay_minutes"]),
            frontier_detector=detector,
        )
        for edge in result.traversed_edges:
            key = (edge["src"], edge["dst"], edge["edge_type"])
            stats = edge_stats.setdefault(key, {"times_traversed": 0, "times_cross_zone": 0})
            stats["times_traversed"] += 1
            is_cross_zone = edge["zone_crossing"]
            if is_cross_zone:
                stats["times_cross_zone"] += 1

            for sid in (edge["src"], edge["dst"]):
                station_stats[sid]["cascade"] = True
                if is_cross_zone:
                    station_stats[sid]["cross_zone"] = True

    # ---- Build output tables ----
    # Include EVERY SCHED_DEP/RS_DEP edge in the graph, not just ones that
    # appeared in a cascade -- otherwise "never triggered" edges are
    # missing entirely instead of showing up green.
    edge_rows = []
    seen_keys = set()
    for u, v, data in mg.graph.edges(data=True):
        edge_type = data.get("edge_type")
        if edge_type not in ("SCHED_DEP", "RS_DEP"):
            continue
        key = (u, v, edge_type)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        stats = edge_stats.get(key, {"times_traversed": 0, "times_cross_zone": 0})
        if stats["times_cross_zone"] > 0:
            risk_level = "cross_zone"
        elif stats["times_traversed"] > 0:
            risk_level = "cascade"
        else:
            risk_level = "normal"
        edge_rows.append(
            {
                "src": u, "dst": v, "edge_type": edge_type,
                "times_traversed": stats["times_traversed"],
                "times_cross_zone": stats["times_cross_zone"],
                "risk_level": risk_level,
            }
        )
    edge_df = pd.DataFrame(edge_rows)

    station_rows = []
    for sid, stats in station_stats.items():
        if stats["cross_zone"]:
            risk_level = "cross_zone"
        elif stats["cascade"]:
            risk_level = "cascade"
        else:
            risk_level = "normal"
        station_rows.append({"station_id": sid, "risk_level": risk_level})
    station_df = pd.DataFrame(station_rows)

    return station_df, edge_df


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(here, "..", "data"))

    mg = MultiplexGraph.load(os.path.join(here, "multiplex_graph.pkl"))
    print("Running full simulation across all events to build the network risk map...")
    station_df, edge_df = compute_network_risk(mg, os.path.join(data_dir, "delay_events.csv"))

    print("\nStation risk breakdown:")
    print(station_df["risk_level"].value_counts())
    print("\nDependency edge risk breakdown:")
    print(edge_df["risk_level"].value_counts())

    station_df.to_csv(os.path.join(here, "station_risk_summary.csv"), index=False)
    edge_df.to_csv(os.path.join(here, "edge_risk_summary.csv"), index=False)
    print("\nSaved station_risk_summary.csv and edge_risk_summary.csv")


if __name__ == "__main__":
    main()
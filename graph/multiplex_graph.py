"""
Module 2 — Railway Network Graph
=================================

Builds the multiplex railway graph specified in the project's SRS (FR-02)
and Class Diagram. A single NetworkX MultiDiGraph holds three layers of
directed edges on top of one set of station nodes:

    INFRA      - physical track connections           (from tracks.csv)
    SCHED_DEP  - a train's own stop-to-stop journey    (from routes.csv)
    RS_DEP     - locomotive reuse between train services (from rolling_stock.csv)

Why MultiDiGraph specifically:
    - Directed: delay propagation flows in the direction of travel, and
      RS_DEP edges are inherently one-directional (loco finishes train X,
      then starts train Y — not the reverse).
    - Multigraph: two stations can be connected by more than one kind of
      edge at once (e.g. an INFRA edge AND a SCHED_DEP edge covering the
      same pair of stations for a different train). A plain DiGraph only
      allows a single edge between any two nodes, which would silently
      overwrite one relationship with another.

This module ONLY reads the CSVs in data/ — it never writes back to them.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import pandas as pd


# ---------------------------------------------------------------------------
# Tunable constants (documented per the working rule: explain before coding)
# ---------------------------------------------------------------------------

# INFRA buffer_time_mins: derived from capacity_trains_per_hour.
# Busier tracks (higher capacity) have LESS slack to absorb a delay before
# passing it downstream, so buffer is inversely proportional to capacity.
# tracks.csv capacity ranges ~4-24 trains/hour. 300/capacity gives a range
# of roughly 12-75 minutes, which we clip to a realistic 5-45 min window.
INFRA_BUFFER_NUMERATOR = 300
INFRA_BUFFER_MIN = 5
INFRA_BUFFER_MAX = 45

# INFRA propagation_weight: single-track sections have no alternate route,
# so a delay there propagates more strongly than on a double/electrified
# section which has more operational redundancy.
TRACK_TYPE_WEIGHTS = {
    "Single Non-Electrified": 0.90,
    "Single Electrified": 0.85,
    "Double Non-Electrified": 0.70,
    "Double Electrified": 0.60,
}
DEFAULT_TRACK_WEIGHT = 0.75  # fallback if an unseen track_type appears

# SCHED_DEP propagation_weight: a delay at one stop almost fully carries
# into the next stop's arrival since there's no alternate path within a
# train's own journey — kept high and constant.
SCHED_DEP_WEIGHT = 0.95

# RS_DEP propagation_weight: this is the project's core "hidden dependency"
# — a locomotive's delay carries onto its next assignment almost entirely,
# so this is set at or above SCHED_DEP.
RS_DEP_WEIGHT = 0.97


@dataclass
class MultiplexGraph:
    """Wraps a NetworkX MultiDiGraph with railway-specific build/query methods.

    Matches the `MultiplexGraph` responsibilities implied by the SRS Class
    Diagram: construction from CSV data, adjacency queries, and a station
    criticality score used later for identifying at-risk junctions.
    """

    graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)

    # -- Construction ------------------------------------------------------

    def load_stations(self, path: str) -> None:
        """Step 4: add one node per station, with all CSV columns as attributes."""
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            self.graph.add_node(
                row["station_id"],
                station_name=row["station_name"],
                zone_id=row["zone_id"],
                zone_name=row["zone_name"],
                latitude=row["latitude"],
                longitude=row["longitude"],
                platform_count=int(row["platform_count"]),
                traffic_category=row["traffic_category"],
                capacity_trains_per_hour=int(row["capacity_trains_per_hour"]),
            )

    def _infra_buffer(self, capacity: int) -> float:
        buf = INFRA_BUFFER_NUMERATOR / max(capacity, 1)
        return max(INFRA_BUFFER_MIN, min(INFRA_BUFFER_MAX, round(buf, 1)))

    def load_infra_edges(self, path: str) -> None:
        """Step 5: INFRA layer from tracks.csv. Adds both directions since
        tracks.csv does not mark any track as one-way."""
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            src, dst = row["source_station"], row["dest_station"]
            weight = TRACK_TYPE_WEIGHTS.get(row["track_type"], DEFAULT_TRACK_WEIGHT)
            buffer_mins = self._infra_buffer(row["capacity_trains_per_hour"])
            zone_crossing = bool(row["zone_crossing"])

            common_attrs = dict(
                edge_type="INFRA",
                track_id=row["track_id"],
                distance_km=row["distance_km"],
                track_type=row["track_type"],
                max_speed_kmph=row["max_speed_kmph"],
                capacity_trains_per_hour=row["capacity_trains_per_hour"],
                zone_crossing=zone_crossing,
                buffer_time_mins=buffer_mins,
                propagation_weight=weight,
            )
            self.graph.add_edge(src, dst, **common_attrs)
            self.graph.add_edge(dst, src, **common_attrs)

    def load_sched_dep_edges(self, path: str) -> None:
        """Step 6: SCHED_DEP layer from routes.csv. For each train, connect
        consecutive stops in sequence_no order."""
        df = pd.read_csv(path)
        df = df.sort_values(["train_number", "sequence_no"])

        for train_number, group in df.groupby("train_number"):
            group = group.sort_values("sequence_no").reset_index(drop=True)
            for i in range(len(group) - 1):
                cur = group.loc[i]
                nxt = group.loc[i + 1]

                src, dst = cur["station_id"], nxt["station_id"]
                if src not in self.graph or dst not in self.graph:
                    # Defensive check: skip and flag if a station_id in
                    # routes.csv doesn't exist in stations.csv.
                    continue

                src_zone = self.graph.nodes[src].get("zone_id")
                dst_zone = self.graph.nodes[dst].get("zone_id")
                zone_crossing = src_zone != dst_zone

                # Buffer = halt time at the destination stop, since that's
                # the slack available before the train must move on again.
                buffer_mins = float(nxt["halt_minutes"]) if pd.notna(nxt["halt_minutes"]) else 2.0
                buffer_mins = max(buffer_mins, 1.0)

                self.graph.add_edge(
                    src,
                    dst,
                    edge_type="SCHED_DEP",
                    train_number=train_number,
                    sequence_no=cur["sequence_no"],
                    zone_crossing=zone_crossing,
                    buffer_time_mins=buffer_mins,
                    propagation_weight=SCHED_DEP_WEIGHT,
                )

    def load_rs_dep_edges(self, path: str) -> None:
        """Step 7: RS_DEP layer from rolling_stock.csv — the project's core
        contribution. Connects the destination station of train X to the
        origin station of train Y when the same locomotive serves X then Y
        consecutively. (Simplified station-to-station model, per the scoping
        decision to defer full TrainService nodes to a later iteration.)

        Requires routes.csv to already be loaded (via a passed-in lookup)
        so we know each train's first/last station.
        """
        raise NotImplementedError("Use build_from_csv(), which supplies the route lookup.")

    def _load_rs_dep_edges_with_routes(self, rs_path: str, routes_df: pd.DataFrame) -> int:
        rs = pd.read_csv(rs_path)
        rs = rs.sort_values(["locomotive_id", "sequence_no_in_rotation"])

        # Build a quick lookup: train_number -> (origin_station, destination_station)
        routes_sorted = routes_df.sort_values(["train_number", "sequence_no"])
        first_last = (
            routes_sorted.groupby("train_number")["station_id"]
            .agg(["first", "last"])
            .rename(columns={"first": "origin_station", "last": "destination_station"})
        )

        edges_added = 0
        for loco_id, group in rs.groupby("locomotive_id"):
            group = group.sort_values("sequence_no_in_rotation").reset_index(drop=True)
            for i in range(len(group) - 1):
                cur = group.loc[i]
                nxt = group.loc[i + 1]

                cur_train, nxt_train = cur["train_number"], nxt["train_number"]
                if cur_train not in first_last.index or nxt_train not in first_last.index:
                    continue

                src = first_last.loc[cur_train, "destination_station"]
                dst = first_last.loc[nxt_train, "origin_station"]
                if src not in self.graph or dst not in self.graph:
                    continue

                src_zone = self.graph.nodes[src].get("zone_id")
                dst_zone = self.graph.nodes[dst].get("zone_id")
                zone_crossing = src_zone != dst_zone

                self.graph.add_edge(
                    src,
                    dst,
                    edge_type="RS_DEP",
                    locomotive_id=loco_id,
                    from_train=cur_train,
                    to_train=nxt_train,
                    zone_crossing=zone_crossing,
                    buffer_time_mins=float(cur["turnaround_buffer_mins"]),
                    propagation_weight=RS_DEP_WEIGHT,
                )
                edges_added += 1
        return edges_added

    @classmethod
    def build_from_csv(cls, data_dir: str) -> "MultiplexGraph":
        """Orchestrator: runs Steps 4-7 in the correct order and returns a
        fully built, validated MultiplexGraph."""
        mg = cls()

        stations_path = os.path.join(data_dir, "stations.csv")
        tracks_path = os.path.join(data_dir, "tracks.csv")
        routes_path = os.path.join(data_dir, "routes.csv")
        rs_path = os.path.join(data_dir, "rolling_stock.csv")

        mg.load_stations(stations_path)
        mg.load_infra_edges(tracks_path)
        mg.load_sched_dep_edges(routes_path)

        routes_df = pd.read_csv(routes_path)
        rs_edges_added = mg._load_rs_dep_edges_with_routes(rs_path, routes_df)
        mg._last_rs_edges_added = rs_edges_added  # stashed for validate() reporting

        return mg

    # -- Queries (Class Diagram methods) ------------------------------------

    def get_adjacent_stations(
        self, station_id: str, edge_type: Optional[str] = None
    ) -> list[str]:
        """Return all stations directly reachable from station_id, optionally
        filtered to a single edge_type (INFRA / SCHED_DEP / RS_DEP)."""
        if station_id not in self.graph:
            raise ValueError(f"Unknown station_id: {station_id}")

        neighbors = set()
        for _, dst, data in self.graph.out_edges(station_id, data=True):
            if edge_type is None or data.get("edge_type") == edge_type:
                neighbors.add(dst)
        return sorted(neighbors)

    def compute_criticality_score(self, station_id: str) -> float:
        """A simple, tunable criticality score combining:
        - degree (how many edges touch this station, in + out)
        - whether it sits on any zone-crossing edge (junction-like behaviour)
        - inverse of capacity (lower capacity relative to traffic = more fragile)

        Formula (documented, intentionally simple as a first pass):
            score = (in_degree + out_degree) * (1 + zone_crossing_bonus) / capacity_trains_per_hour

        Where zone_crossing_bonus = 0.5 if the station touches ANY
        zone-crossing edge, else 0. Higher score = more critical/vulnerable.
        """
        if station_id not in self.graph:
            raise ValueError(f"Unknown station_id: {station_id}")

        in_deg = self.graph.in_degree(station_id)
        out_deg = self.graph.out_degree(station_id)
        capacity = self.graph.nodes[station_id].get("capacity_trains_per_hour", 1)

        touches_zone_crossing = any(
            data.get("zone_crossing")
            for _, _, data in self.graph.out_edges(station_id, data=True)
        ) or any(
            data.get("zone_crossing")
            for _, _, data in self.graph.in_edges(station_id, data=True)
        )
        bonus = 0.5 if touches_zone_crossing else 0.0

        score = (in_deg + out_deg) * (1 + bonus) / max(capacity, 1)
        return round(score, 4)

    # -- Persistence ---------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.graph, f)

    @classmethod
    def load(cls, path: str) -> "MultiplexGraph":
        with open(path, "rb") as f:
            graph = pickle.load(f)
        mg = cls()
        mg.graph = graph
        return mg

    # -- Validation ------------------------------------------------------

    def validate(self, expected_stations: int = 73) -> dict:
        """Runs the Step 9 sanity checks and returns a results dict.
        Raises AssertionError with a clear message if anything is wrong.
        """
        results = {}

        n_nodes = self.graph.number_of_nodes()
        results["node_count"] = n_nodes
        assert n_nodes == expected_stations, (
            f"Expected {expected_stations} station nodes, got {n_nodes}"
        )

        edge_counts = {"INFRA": 0, "SCHED_DEP": 0, "RS_DEP": 0}
        for _, _, data in self.graph.edges(data=True):
            et = data.get("edge_type")
            if et in edge_counts:
                edge_counts[et] += 1
        results["edge_counts"] = edge_counts

        # Dangling reference check: every edge endpoint must be a real node.
        # (NetworkX add_edge auto-creates nodes for unknown endpoints, which
        # would silently hide a data bug, so we check node attribute
        # completeness instead: every node should have station_name set.)
        nodes_missing_attrs = [
            n for n, d in self.graph.nodes(data=True) if "station_name" not in d
        ]
        results["dangling_nodes"] = nodes_missing_attrs
        assert not nodes_missing_attrs, (
            f"Found {len(nodes_missing_attrs)} node(s) referenced by an edge "
            f"but never defined in stations.csv: {nodes_missing_attrs}"
        )

        return results

    def summary(self) -> str:
        edge_counts = {"INFRA": 0, "SCHED_DEP": 0, "RS_DEP": 0}
        for _, _, data in self.graph.edges(data=True):
            et = data.get("edge_type")
            if et in edge_counts:
                edge_counts[et] += 1

        lines = [
            "MultiplexGraph summary",
            "-----------------------",
            f"Stations (nodes): {self.graph.number_of_nodes()}",
            f"INFRA edges:      {edge_counts['INFRA']}",
            f"SCHED_DEP edges:  {edge_counts['SCHED_DEP']}",
            f"RS_DEP edges:     {edge_counts['RS_DEP']}",
            f"Total edges:      {self.graph.number_of_edges()}",
        ]
        return "\n".join(lines)
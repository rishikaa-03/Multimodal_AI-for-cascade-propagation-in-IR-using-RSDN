"""
generate_dataset.py
--------------------
Module 1 — Data Collection, Railway Dataset Design, and Synthetic Dataset
Generation.

Since real, granular, station-level Indian Railways operational data
(live NTES feeds, loco assignment logs, per-station delay causation text)
is not available for academic use at the volume needed for ML training,
this module generates a *topologically and operationally realistic*
synthetic dataset, seeded from real zone/station reference data
(see railway_reference_data.py).

Design principles followed:
  1. The station graph is built from real zones/stations, not randomly
     scattered points, so degree distribution and cross-zone junctions are
     realistic (a handful of high-degree junction hubs, many low-degree
     leaf stations) — this matters because the cascade simulator and GNN
     in later modules depend on realistic graph topology.
  2. Rolling stock rotations are generated as *chains*: a locomotive is
     assigned to a sequence of trains with realistic turnaround buffers,
     which is exactly the hidden dependency structure the project's core
     contribution (RSDG) needs to expose.
  3. All nine required CSVs are cross-referenced by shared keys
     (station_id, train_number, locomotive_id) so downstream modules
     (graph construction, simulation, ML) can join them directly.
  4. simulation_dataset.csv is the ML-ready, feature-engineered table
     (one row per simulated primary-delay scenario) with >= 5000 rows,
     built by actually running a lightweight cascade propagation over the
     generated graph (not just randomly sampled labels) so that feature-
     label relationships are learnable and not pure noise.

Output files (written to RailwayCascadeAI/data/):
  stations.csv, tracks.csv, routes.csv, trains.csv, locomotives.csv,
  weather.csv, delay_events.csv, rolling_stock.csv, simulation_dataset.csv

Usage:
    python generate_dataset.py --seed 42 --out ./  (defaults shown)

Author: RailwayCascadeAI Project (BE Major Project - Phase 2)
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Local import of the real-world-seeded reference data
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from railway_reference_data import ZONES, STATIONS, CROSS_ZONE_CORRIDORS  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

DELAY_CAUSE_CATEGORIES = [
    "Signal", "Infrastructure", "Rolling Stock", "Crew",
    "Weather", "Passenger", "Operational",
]

# Free-text templates used to generate delay_events "reported_text" field,
# consumed later by the NLP module (Module 7). Kept intentionally varied in
# phrasing so TF-IDF / BERT models have signal to learn from.
DELAY_TEXT_TEMPLATES: Dict[str, List[str]] = {
    "Signal": [
        "Signal failure near {station} causing hold-up.",
        "Automatic signalling fault reported at {station} yard.",
        "Points failure at {station} delaying departure.",
    ],
    "Infrastructure": [
        "Track maintenance block in effect near {station}.",
        "Overhead equipment (OHE) tripping reported near {station}.",
        "Speed restriction due to track renewal work near {station}.",
    ],
    "Rolling Stock": [
        "Locomotive failure reported, engine change required at {station}.",
        "Coach fault detected during pre-departure check at {station}.",
        "Brake power certification delay at {station}.",
    ],
    "Crew": [
        "Crew shortage reported at {station} link.",
        "Loco pilot booking-on delay at {station}.",
        "Guard unavailability causing hold at {station}.",
    ],
    "Weather": [
        "Heavy rain reducing visibility near {station}.",
        "Dense fog conditions near {station}, speed restricted.",
        "High wind speed advisory near {station} affecting running.",
    ],
    "Passenger": [
        "Passenger chain-pulling incident near {station}.",
        "Medical emergency on board, halted at {station}.",
        "Overcrowding-related boarding delay at {station}.",
    ],
    "Operational": [
        "Platform congestion at {station} delaying berthing.",
        "Crossing conflict with another service at {station}.",
        "Late running of connecting rake into {station}.",
    ],
}

TRAIN_TYPES = ["Rajdhani", "Shatabdi", "Express", "Superfast", "Passenger", "Freight", "MEMU"]
LOCO_CLASSES = ["WAP-7", "WAP-5", "WAG-9", "WDG-4", "WDP-4", "WAP-4"]
WEATHER_CONDITIONS = ["Clear", "Rain", "Heavy Rain", "Fog", "Storm"]

RNG_SEED_DEFAULT = 42


# ---------------------------------------------------------------------------
# Helper: haversine distance (used for track distance & propagation weight)
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two lat/lon points, in kilometres."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class RailwayDatasetGenerator:
    """
    Encapsulates the full synthetic dataset generation pipeline.

    Each `build_*` method returns a pandas DataFrame and (where relevant)
    stores intermediate state on `self` so downstream builders can
    reference already-generated entities (e.g. tracks need stations,
    rolling_stock needs trains + locomotives).
    """

    def __init__(self, seed: int = RNG_SEED_DEFAULT, n_days: int = 60):
        self.seed = seed
        self.n_days = n_days
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # populated by build_* methods
        self.stations_df: pd.DataFrame | None = None
        self.tracks_df: pd.DataFrame | None = None
        self.trains_df: pd.DataFrame | None = None
        self.routes_df: pd.DataFrame | None = None
        self.locomotives_df: pd.DataFrame | None = None
        self.rolling_stock_df: pd.DataFrame | None = None
        self.weather_df: pd.DataFrame | None = None
        self.delay_events_df: pd.DataFrame | None = None
        self.simulation_df: pd.DataFrame | None = None

        # adjacency built after tracks are generated: station_id -> list[station_id]
        self._adjacency: Dict[str, List[str]] = {}
        self._station_lookup: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # 1. STATIONS
    # ------------------------------------------------------------------
    def build_stations(self) -> pd.DataFrame:
        rows = []
        zone_name_lookup = {z.zone_id: z.zone_name for z in ZONES}
        for code, name, zone_id, lat, lon, platforms, category in STATIONS:
            rows.append({
                "station_id": code,
                "station_name": name,
                "zone_id": zone_id,
                "zone_name": zone_name_lookup[zone_id],
                "latitude": lat,
                "longitude": lon,
                "platform_count": platforms,
                "traffic_category": category,
                "capacity_trains_per_hour": {
                    "HIGH": self.rng.randint(18, 30),
                    "MEDIUM": self.rng.randint(8, 17),
                    "LOW": self.rng.randint(2, 7),
                }[category],
            })
        df = pd.DataFrame(rows)

        self.stations_df = df
        self._station_lookup = {r["station_id"]: r for r in rows}
        return df

    # ------------------------------------------------------------------
    # 2. TRACKS (graph edges - infrastructure layer)
    # ------------------------------------------------------------------
    def build_tracks(self) -> pd.DataFrame:
        """
        Builds the infrastructure edge list. Strategy:
          (a) Connect stations within each zone using a minimum-spanning-tree
              -like nearest-neighbour chain so every station is reachable
              (avoids disconnected components).
          (b) Add the explicit CROSS_ZONE_CORRIDORS as inter-zone edges.
          (c) Add a handful of extra "express corridor" long edges between
              major (HIGH traffic) stations across zones for realism
              (real IR has several parallel long-haul trunk routes).
        """
        assert self.stations_df is not None, "call build_stations() first"

        edges: List[Tuple[str, str]] = []
        seen_pairs = set()

        def add_edge(a, b):
            key = tuple(sorted((a, b)))
            if key not in seen_pairs and a != b:
                seen_pairs.add(key)
                edges.append((a, b))

        # (a) Within-zone nearest-neighbour chain
        for zone in ZONES:
            zone_stations = self.stations_df[self.stations_df.zone_id == zone.zone_id]
            codes = zone_stations["station_id"].tolist()
            if len(codes) < 2:
                continue
            # sort by longitude then latitude as a cheap way to create a
            # geographically sensible chain (approximates a line of route)
            coords = {r.station_id: (r.latitude, r.longitude) for r in zone_stations.itertuples()}
            ordered = sorted(codes, key=lambda c: (coords[c][1], coords[c][0]))
            for a, b in zip(ordered, ordered[1:]):
                add_edge(a, b)
            # connect the zone HQ to every other station in the zone (hub pattern)
            hq = next((z.headquarter for z in ZONES if z.zone_id == zone.zone_id), None)
            if hq in codes:
                for c in codes:
                    add_edge(hq, c)

        # (b) explicit cross-zone corridors
        valid_codes = set(self.stations_df["station_id"])
        for a, b in CROSS_ZONE_CORRIDORS:
            if a in valid_codes and b in valid_codes:
                add_edge(a, b)

        # (c) a few extra long-haul express trunk edges between HIGH stations
        high_stations = self.stations_df[self.stations_df.traffic_category == "HIGH"]["station_id"].tolist()
        self.rng.shuffle(high_stations)
        for a, b in zip(high_stations, high_stations[1:]):
            add_edge(a, b)

        rows = []
        for idx, (a, b) in enumerate(edges, start=1):
            sa, sb = self._station_lookup[a], self._station_lookup[b]
            dist = round(haversine_km(sa["latitude"], sa["longitude"], sb["latitude"], sb["longitude"]), 1)
            same_zone = sa["zone_id"] == sb["zone_id"]
            track_type = self.rng.choices(
                ["Double Electrified", "Single Electrified", "Double Non-Electrified", "Single Non-Electrified"],
                weights=[0.5, 0.25, 0.15, 0.10],
            )[0]
            max_speed = {"Double Electrified": 130, "Single Electrified": 110,
                         "Double Non-Electrified": 100, "Single Non-Electrified": 80}[track_type]
            capacity = self.rng.randint(4, 12) if "Single" in track_type else self.rng.randint(10, 24)
            rows.append({
                "track_id": f"TRK{idx:04d}",
                "source_station": a,
                "dest_station": b,
                "distance_km": max(dist, 3.0),
                "track_type": track_type,
                "max_speed_kmph": max_speed,
                "capacity_trains_per_hour": capacity,
                "zone_crossing": not same_zone,
            })
        df = pd.DataFrame(rows)
        self.tracks_df = df

        # build undirected adjacency for routing/simulation use
        adjacency: Dict[str, List[str]] = {c: [] for c in valid_codes}
        for r in rows:
            adjacency[r["source_station"]].append(r["dest_station"])
            adjacency[r["dest_station"]].append(r["source_station"])
        self._adjacency = adjacency
        return df

    # ------------------------------------------------------------------
    # 3. TRAINS + 4. ROUTES
    # ------------------------------------------------------------------
    def _shortest_path(self, src: str, dst: str, max_hops: int = 12) -> List[str]:
        """Simple BFS shortest path over the track adjacency graph."""
        if src == dst:
            return [src]
        visited = {src}
        queue = [[src]]
        while queue:
            path = queue.pop(0)
            node = path[-1]
            if len(path) > max_hops:
                continue
            for nxt in self._adjacency.get(node, []):
                if nxt == dst:
                    return path + [nxt]
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(path + [nxt])
        return [src, dst]  # fallback: direct (shouldn't normally happen)

    def build_trains_and_routes(self, n_trains: int = 320) -> Tuple[pd.DataFrame, pd.DataFrame]:
        assert self.tracks_df is not None, "call build_tracks() first"
        station_codes = self.stations_df["station_id"].tolist()
        high_med = self.stations_df[self.stations_df.traffic_category.isin(["HIGH", "MEDIUM"])]["station_id"].tolist()

        train_rows = []
        route_rows = []
        train_number_start = 12000

        for i in range(n_trains):
            train_number = str(train_number_start + i)
            origin, destination = self.rng.sample(high_med, 2)
            path = self._shortest_path(origin, destination)
            if len(path) < 2:
                continue

            train_type = self.rng.choices(
                TRAIN_TYPES, weights=[0.05, 0.05, 0.30, 0.20, 0.15, 0.10, 0.15]
            )[0]
            priority = {"Rajdhani": 1, "Shatabdi": 1, "Superfast": 2, "Express": 3,
                        "MEMU": 4, "Passenger": 5, "Freight": 6}[train_type]
            avg_speed = {"Rajdhani": 90, "Shatabdi": 95, "Superfast": 75, "Express": 60,
                         "MEMU": 45, "Passenger": 40, "Freight": 35}[train_type]

            origin_zone = self._station_lookup[origin]["zone_id"]

            train_rows.append({
                "train_number": train_number,
                "train_name": f"{self._station_lookup[origin]['station_name']} - "
                               f"{self._station_lookup[destination]['station_name']} {train_type}",
                "train_type": train_type,
                "origin_station": origin,
                "destination_station": destination,
                "zone_id": origin_zone,
                "priority": priority,
                "total_stops": len(path),
                "avg_speed_kmph": avg_speed,
            })

            # build route (sequence of stations with cumulative distance/time)
            cum_dist = 0.0
            base_time = datetime(2026, 1, 1, self.rng.randint(0, 23), self.rng.choice([0, 15, 30, 45]))
            for seq, station in enumerate(path, start=1):
                if seq > 1:
                    prev = path[seq - 2]
                    edge_dist = haversine_km(
                        self._station_lookup[prev]["latitude"], self._station_lookup[prev]["longitude"],
                        self._station_lookup[station]["latitude"], self._station_lookup[station]["longitude"],
                    )
                    cum_dist += edge_dist
                travel_minutes = (cum_dist / avg_speed) * 60 if cum_dist > 0 else 0
                arrival = base_time + timedelta(minutes=travel_minutes)
                halt = 2 if seq not in (1, len(path)) else 0
                departure = arrival + timedelta(minutes=halt)
                route_rows.append({
                    "route_id": f"{train_number}_{seq:02d}",
                    "train_number": train_number,
                    "sequence_no": seq,
                    "station_id": station,
                    "distance_from_origin_km": round(cum_dist, 1),
                    "scheduled_arrival": arrival.strftime("%H:%M"),
                    "scheduled_departure": departure.strftime("%H:%M"),
                    "halt_minutes": halt,
                })

        self.trains_df = pd.DataFrame(train_rows)
        self.routes_df = pd.DataFrame(route_rows)
        return self.trains_df, self.routes_df

    # ------------------------------------------------------------------
    # 5. LOCOMOTIVES + 6. ROLLING STOCK (dependency chains)
    # ------------------------------------------------------------------
    def build_locomotives_and_rolling_stock(self, n_locomotives: int = 140):
        """
        Builds locomotives and, crucially, the rolling-stock ROTATION chains:
        each locomotive is assigned a sequence of trains across a service day
        with a turnaround buffer between consecutive services. This directly
        seeds the Rolling Stock Dependency Graph (RSDG) built in Module 3.
        """
        assert self.trains_df is not None, "call build_trains_and_routes() first"

        loco_rows = []
        zone_ids = [z.zone_id for z in ZONES]
        for i in range(n_locomotives):
            loco_id = f"LOCO{i+1:04d}"
            zone = self.rng.choice(zone_ids)
            base_station = self.stations_df[self.stations_df.zone_id == zone]["station_id"].sample(
                random_state=self.rng.randint(0, 10_000)).iloc[0]
            loco_rows.append({
                "locomotive_id": loco_id,
                "loco_class": self.rng.choice(LOCO_CLASSES),
                "home_zone": zone,
                "base_station": base_station,
                "max_haul_capacity_tonnes": self.rng.choice([2000, 2500, 3000, 3500, 4500]),
                "commissioned_year": self.rng.randint(2005, 2024),
            })
        self.locomotives_df = pd.DataFrame(loco_rows)
        loco_ids = self.locomotives_df["locomotive_id"].tolist()

        # Build rotation chains: shuffle trains, assign in groups to locomotives
        rs_rows = []
        train_list = self.trains_df.to_dict("records")
        self.rng.shuffle(train_list)

        assignment_id = 1
        idx = 0
        n_trains = len(train_list)
        while idx < n_trains:
            loco_id = loco_ids[self.rng.randint(0, len(loco_ids) - 1)]
            chain_length = self.rng.choices([1, 2, 3, 4], weights=[0.35, 0.35, 0.20, 0.10])[0]
            chain = train_list[idx: idx + chain_length]
            idx += chain_length
            if not chain:
                break

            current_time = datetime(2026, 1, 1, self.rng.randint(4, 10), 0)
            for seq_no, train in enumerate(chain, start=1):
                scheduled_start = current_time
                # rough service duration based on stops * avg dwell + travel proxy
                service_minutes = train["total_stops"] * self.rng.randint(20, 45)
                scheduled_end = scheduled_start + timedelta(minutes=service_minutes)
                # turnaround buffer before the NEXT service in the chain
                turnaround_buffer = self.rng.choice([15, 20, 25, 30, 45, 60, 90])
                rs_rows.append({
                    "assignment_id": f"RS{assignment_id:05d}",
                    "locomotive_id": loco_id,
                    "train_number": train["train_number"],
                    "sequence_no_in_rotation": seq_no,
                    "scheduled_start": scheduled_start.strftime("%Y-%m-%d %H:%M"),
                    "scheduled_end": scheduled_end.strftime("%Y-%m-%d %H:%M"),
                    "turnaround_buffer_mins": turnaround_buffer,
                })
                assignment_id += 1
                current_time = scheduled_end + timedelta(minutes=turnaround_buffer)

        self.rolling_stock_df = pd.DataFrame(rs_rows)
        return self.locomotives_df, self.rolling_stock_df

    # ------------------------------------------------------------------
    # 7. WEATHER
    # ------------------------------------------------------------------
    def build_weather(self) -> pd.DataFrame:
        assert self.stations_df is not None
        rows = []
        start_date = datetime(2026, 1, 1)
        weather_id = 1
        for station in self.stations_df.itertuples():
            for d in range(self.n_days):
                date = start_date + timedelta(days=d)
                condition = self.rng.choices(
                    WEATHER_CONDITIONS, weights=[0.65, 0.15, 0.06, 0.10, 0.04]
                )[0]
                severity = {"Clear": 0, "Rain": 1, "Heavy Rain": 2, "Fog": 2, "Storm": 3}[condition]
                visibility = {
                    "Clear": self.rng.uniform(8, 10), "Rain": self.rng.uniform(4, 8),
                    "Heavy Rain": self.rng.uniform(1, 3), "Fog": self.rng.uniform(0.2, 2),
                    "Storm": self.rng.uniform(1, 4),
                }[condition]
                rows.append({
                    "weather_id": f"W{weather_id:06d}",
                    "station_id": station.station_id,
                    "date": date.strftime("%Y-%m-%d"),
                    "condition": condition,
                    "severity": severity,
                    "visibility_km": round(visibility, 1),
                    "temperature_celsius": round(self.rng.uniform(8, 42), 1),
                })
                weather_id += 1
        self.weather_df = pd.DataFrame(rows)
        return self.weather_df

    # ------------------------------------------------------------------
    # 8. DELAY EVENTS
    # ------------------------------------------------------------------
    def build_delay_events(self, n_events: int = 6000) -> pd.DataFrame:
        assert self.trains_df is not None and self.routes_df is not None
        rows = []
        train_numbers = self.trains_df["train_number"].tolist()
        route_by_train = self.routes_df.groupby("train_number")

        start_date = datetime(2026, 1, 1)
        for i in range(n_events):
            train_number = self.rng.choice(train_numbers)
            train_routes = route_by_train.get_group(train_number)
            station_row = train_routes.sample(random_state=self.rng.randint(0, 1_000_000)).iloc[0]
            station_id = station_row["station_id"]

            cause = self.rng.choices(
                DELAY_CAUSE_CATEGORIES,
                weights=[0.18, 0.15, 0.15, 0.12, 0.15, 0.10, 0.15],
            )[0]
            delay_minutes = max(1, int(self.rng.gauss(30, 25)))
            delay_minutes = min(delay_minutes, 240)
            date = start_date + timedelta(days=self.rng.randint(0, self.n_days - 1))
            timestamp = date + timedelta(hours=self.rng.randint(0, 23), minutes=self.rng.choice([0, 15, 30, 45]))

            text_template = self.rng.choice(DELAY_TEXT_TEMPLATES[cause])
            reported_text = text_template.format(station=self._station_lookup[station_id]["station_name"])

            rows.append({
                "event_id": f"EVT{i+1:06d}",
                "train_number": train_number,
                "station_id": station_id,
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
                "delay_minutes": delay_minutes,
                "cause_category": cause,
                "reported_text": reported_text,
            })
        self.delay_events_df = pd.DataFrame(rows)
        return self.delay_events_df

    # ------------------------------------------------------------------
    # 9. SIMULATION DATASET (ML-ready feature/label table)
    # ------------------------------------------------------------------
    def build_simulation_dataset(self, n_rows: int = 5200) -> pd.DataFrame:
        """
        Generates the ML training table by running a lightweight cascade
        propagation for each sampled scenario:

            propagated_delay(child) = max(0, (parent_delay - buffer) * decay)

        over BOTH the infrastructure graph (track-shared neighbours) and the
        rolling-stock chain (next service on the same locomotive), so labels
        (cascade_depth, affected_trains_count, delay_spread_minutes,
        cross_zone_flag) are causally related to the input features rather
        than being independently random — this matters for Module 6 (ML)
        to be able to learn a non-trivial mapping.
        """
        assert self.tracks_df is not None and self.rolling_stock_df is not None
        assert self.delay_events_df is not None and self.weather_df is not None

        # station degree lookup (graph topology feature)
        degree = {c: 0 for c in self.stations_df["station_id"]}
        for r in self.tracks_df.itertuples():
            degree[r.source_station] += 1
            degree[r.dest_station] += 1

        # next-service-in-rotation lookup, keyed by (locomotive_id, train_number)
        rs_sorted = self.rolling_stock_df.sort_values(["locomotive_id", "sequence_no_in_rotation"])
        next_service: Dict[Tuple[str, str], Tuple[str, int]] = {}
        prev_key = None
        for r in rs_sorted.itertuples():
            if prev_key is not None and prev_key[0] == r.locomotive_id:
                next_service[(prev_key[0], prev_key[1])] = (r.train_number, r.turnaround_buffer_mins)
            prev_key = (r.locomotive_id, r.train_number, r.turnaround_buffer_mins)

        weather_by_station_date = {
            (r.station_id, r.date): r for r in self.weather_df.itertuples()
        }

        events = self.delay_events_df.sample(n=min(n_rows, len(self.delay_events_df) * 3),
                                              replace=True,
                                              random_state=self.seed).reset_index(drop=True)
        if len(events) < n_rows:
            extra = self.delay_events_df.sample(n=n_rows - len(events), replace=True,
                                                 random_state=self.seed + 1)
            events = pd.concat([events, extra], ignore_index=True)
        events = events.iloc[:n_rows].reset_index(drop=True)

        # lookup: train_number -> its locomotive assignment(s)
        train_to_loco = self.rolling_stock_df.set_index("train_number")[
            ["locomotive_id", "sequence_no_in_rotation", "turnaround_buffer_mins"]
        ].to_dict("index")

        sim_rows = []
        DECAY = 0.75  # propagation decay factor per hop (shared with Module 3 simulator)

        for i, ev in events.iterrows():
            station_id = ev["station_id"]
            train_number = ev["train_number"]
            initial_delay = ev["delay_minutes"]
            cause = ev["cause_category"]

            station_deg = degree.get(station_id, 1)
            station_meta = self._station_lookup[station_id]
            platform_capacity = station_meta["platform_count"]
            zone_id = station_meta["zone_id"]

            date_str = ev["timestamp"][:10]
            w = weather_by_station_date.get((station_id, date_str))
            weather_condition = w.condition if w else "Clear"
            weather_severity = w.severity if w else 0

            track_congestion = round(self.rng.uniform(0.1, 1.0) *
                                      (1.3 if station_meta["traffic_category"] == "HIGH" else 1.0), 2)

            # A train has a genuine rolling-stock dependency only if its
            # locomotive goes on to serve ANOTHER train afterwards (i.e. it
            # is not the last leg of its rotation chain) -- that is exactly
            # the condition under which a delay can inherit forward.
            loco_info = train_to_loco.get(train_number)
            has_rs_dependency = bool(
                loco_info and (loco_info["locomotive_id"], train_number) in next_service
            )

            # --- run a small forward propagation (infra + rolling stock) ---
            affected = set()
            frontier = [(train_number, station_id, initial_delay, zone_id, 0)]
            cascade_depth = 0
            total_delay_spread = initial_delay
            cross_zone_hit = False
            visited_trains = {train_number}

            hop = 0
            while frontier and hop < 6:
                next_frontier = []
                for (t_num, s_id, delay, z_id, depth) in frontier:
                    if delay <= 5:
                        continue
                    # 1) infrastructure propagation to adjacent stations (proxy for
                    #    platform/track conflict affecting other trains at this station)
                    neighbours = self._adjacency.get(s_id, [])
                    for nb in neighbours[:3]:
                        buffer = self.rng.uniform(3, 12)
                        propagated = max(0.0, (delay - buffer) * DECAY)
                        if propagated > 5:
                            nb_zone = self._station_lookup[nb]["zone_id"]
                            if nb_zone != z_id:
                                cross_zone_hit = True
                            cascade_depth = max(cascade_depth, depth + 1)
                            total_delay_spread += propagated
                            affected.add(nb)
                            next_frontier.append((t_num, nb, propagated, nb_zone, depth + 1))

                    # 2) rolling-stock propagation to the next service of the same loco
                    if t_num in train_to_loco:
                        info = train_to_loco[t_num]
                        key = (info["locomotive_id"], t_num)
                        nxt = next_service.get((info["locomotive_id"], t_num))
                        if nxt:
                            nxt_train, buf = nxt
                            propagated = max(0.0, (delay - buf) * DECAY)
                            if propagated > 5 and nxt_train not in visited_trains:
                                visited_trains.add(nxt_train)
                                cascade_depth = max(cascade_depth, depth + 1)
                                total_delay_spread += propagated
                                affected.add(nxt_train)
                                next_frontier.append((nxt_train, s_id, propagated, z_id, depth + 1))
                frontier = next_frontier
                hop += 1

            affected_trains_count = len(visited_trains) - 1 + sum(
                1 for a in affected if a in self.trains_df["train_number"].values
            )
            affected_trains_count = max(affected_trains_count, len(visited_trains) - 1)

            sim_rows.append({
                "scenario_id": f"SIM{i+1:06d}",
                "train_number": train_number,
                "station_id": station_id,
                "zone_id": zone_id,
                "initial_delay_minutes": initial_delay,
                "station_degree": station_deg,
                "platform_capacity": platform_capacity,
                "track_congestion_index": track_congestion,
                "has_rolling_stock_dependency": int(has_rs_dependency),
                "weather_condition": weather_condition,
                "weather_severity": weather_severity,
                "delay_cause": cause,
                "traffic_category": station_meta["traffic_category"],
                # ---- labels (what the ML models in Module 6 will predict) ----
                "cascade_depth": cascade_depth,
                "affected_trains_count": affected_trains_count,
                "delay_spread_minutes": round(total_delay_spread, 1),
                "cross_zone_propagation": int(cross_zone_hit),
            })

        self.simulation_df = pd.DataFrame(sim_rows)
        return self.simulation_df

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def generate_all(self) -> Dict[str, pd.DataFrame]:
        self.build_stations()
        self.build_tracks()
        self.build_trains_and_routes()
        self.build_locomotives_and_rolling_stock()
        self.build_weather()
        self.build_delay_events()
        self.build_simulation_dataset()
        return {
            "stations": self.stations_df,
            "tracks": self.tracks_df,
            "routes": self.routes_df,
            "trains": self.trains_df,
            "locomotives": self.locomotives_df,
            "weather": self.weather_df,
            "delay_events": self.delay_events_df,
            "rolling_stock": self.rolling_stock_df,
            "simulation_dataset": self.simulation_df,
        }


def main():
    parser = argparse.ArgumentParser(description="Generate RailwayCascadeAI synthetic dataset")
    parser.add_argument("--seed", type=int, default=RNG_SEED_DEFAULT)
    parser.add_argument("--out", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--n-days", type=int, default=60)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    gen = RailwayDatasetGenerator(seed=args.seed, n_days=args.n_days)
    tables = gen.generate_all()

    for name, df in tables.items():
        path = os.path.join(args.out, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  wrote {path:60s} rows={len(df):6d} cols={len(df.columns)}")

    print("\nSummary:")
    print(f"  Stations           : {len(tables['stations'])}")
    print(f"  Tracks (edges)      : {len(tables['tracks'])}")
    print(f"  Trains              : {len(tables['trains'])}")
    print(f"  Route legs          : {len(tables['routes'])}")
    print(f"  Locomotives         : {len(tables['locomotives'])}")
    print(f"  Rolling stock legs  : {len(tables['rolling_stock'])}")
    print(f"  Weather records     : {len(tables['weather'])}")
    print(f"  Delay events        : {len(tables['delay_events'])}")
    print(f"  Simulation rows     : {len(tables['simulation_dataset'])}")
    total = sum(len(df) for df in tables.values())
    print(f"  TOTAL RECORDS       : {total}")


if __name__ == "__main__":
    main()

"""
Module 3 — Cascade Propagation Simulator (implements SRS FR-04)
=================================================================

Takes a single primary delay event (a train, delayed by N minutes, at a
station) and walks the Module 2 graph OUTWARD from it, computing how far
and how strongly that delay propagates, until it's too small to matter.

Scope decision (documented, same convention as Module 2):
    Propagation follows ONLY SCHED_DEP edges (same train, next stop) and
    RS_DEP edges (same locomotive, next train) — not INFRA edges. INFRA
    congestion spreading to OTHER trains on the same track would require
    full timetable-aware simulation (knowing exactly which train uses that
    track section next, and when), which is a real additional feature, not
    an extension of this one. SCHED_DEP + RS_DEP are exactly the project's
    two core contributions (schedule chains + hidden rolling-stock chains),
    so this scope is deliberate, not a shortcut around something needed now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from multiplex_graph import MultiplexGraph


# A delay under this many minutes is considered absorbed by normal
# operational slack and stops propagating further, per FR-04's
# "continuing while above threshold" condition.
CASCADE_STOP_THRESHOLD_MINS = 5.0

# Safety cap on hops to guarantee termination even if the graph ever
# contains an unexpected cycle (e.g. an unusual locomotive rotation).
MAX_HOPS = 25


@dataclass
class CascadeResult:
    """One simulated cascade, in the same shape as simulation_dataset.csv's
    scenario-level columns, so it can be compared directly against it."""

    train_number: int
    station_id: str
    initial_delay_minutes: float
    cascade_depth: int = 0
    affected_trains: set = field(default_factory=set)
    delay_spread_minutes: float = 0.0
    cross_zone_propagation: bool = False
    path_log: list = field(default_factory=list)  # human-readable trace, for manual checking
    traversed_edges: list = field(default_factory=list)  # structured hop data, for programmatic use (e.g. the dashboard)
    frontier_checks: list = field(default_factory=list)  # (station, p_cross, alert, actually_crossed) tuples, filled only if a frontier_detector is passed in

    def as_dict(self) -> dict:
        return {
            "train_number": self.train_number,
            "station_id": self.station_id,
            "initial_delay_minutes": self.initial_delay_minutes,
            "cascade_depth": self.cascade_depth,
            "affected_trains_count": len(self.affected_trains),
            "delay_spread_minutes": round(self.delay_spread_minutes, 1),
            "cross_zone_propagation": int(self.cross_zone_propagation),
        }


class CascadeSimulator:
    def __init__(self, mg: MultiplexGraph):
        self.mg = mg

    def simulate_event(
        self, train_number: int, station_id: str, delay_minutes: float,
        frontier_detector=None, blocked_hops: Optional[set] = None,
    ) -> CascadeResult:
        """Runs the weighted BFS for a single primary delay event.

        If frontier_detector (Module 5's CrossZoneFrontierDetector) is
        passed in, every zone-crossing SCHED_DEP/RS_DEP edge encountered
        during propagation is also evaluated for a frontier alert, and the
        result recorded in result.frontier_checks alongside whether the
        BFS itself determined the delay actually continued past that edge
        -- this is what lets run_frontier_detector.py compute precision/
        recall for the detector against the simulator's own ground truth.

        blocked_hops (Module 7's counterfactual mechanism): an optional
        set of (src_station, dst_station, edge_type, from_train) tuples.
        Any matching hop is treated as fully absorbed (new_delay forced to
        0, propagation stops there) -- this simulates "what if a mitigation
        action broke this specific dependency," and is how the
        recommendation engine measures estimated delay recovery: run once
        normally, run again with one hop blocked, compare
        delay_spread_minutes.
        """
        blocked_hops = blocked_hops or set()
        result = CascadeResult(
            train_number=train_number,
            station_id=station_id,
            initial_delay_minutes=delay_minutes,
        )
        result.delay_spread_minutes += delay_minutes  # the primary delay itself counts

        if station_id not in self.mg.graph:
            result.path_log.append(f"Station {station_id} not found in graph — no propagation.")
            return result

        # Queue items: (train_number, station_id, incoming_delay, hop)
        queue = [(train_number, station_id, delay_minutes, 0)]
        # Guards against reprocessing the same (train, station) with a
        # delay no larger than one already handled — prevents loops and
        # wasted work, mirrors a visited-set in standard BFS.
        best_seen: dict[tuple, float] = {}

        while queue:
            cur_train, cur_station, cur_delay, hop = queue.pop(0)

            key = (cur_train, cur_station)
            if best_seen.get(key, -1) >= cur_delay:
                continue
            best_seen[key] = cur_delay

            if hop >= MAX_HOPS:
                continue

            for _, dst, data in self.mg.graph.out_edges(cur_station, data=True):
                edge_type = data.get("edge_type")

                # Only follow edges that actually belong to this train's
                # journey or this locomotive's next assignment.
                if edge_type == "SCHED_DEP" and data.get("train_number") == cur_train:
                    next_train = cur_train
                elif edge_type == "RS_DEP" and data.get("from_train") == cur_train:
                    next_train = data.get("to_train")
                else:
                    continue

                buffer = data.get("buffer_time_mins", 0.0)
                weight = data.get("propagation_weight", 0.0)
                new_delay = max(0.0, (cur_delay - buffer) * weight)

                hop_key = (cur_station, dst, edge_type, cur_train)
                if hop_key in blocked_hops:
                    new_delay = 0.0  # simulates a mitigation action fully absorbing the delay here

                actually_continues = new_delay >= CASCADE_STOP_THRESHOLD_MINS

                if frontier_detector is not None and data.get("zone_crossing"):
                    evaluation = frontier_detector.evaluate(cur_station, cur_delay)
                    if evaluation is not None:
                        result.frontier_checks.append(
                            {
                                "station_id": cur_station,
                                "p_cross": evaluation.p_cross,
                                "alert": evaluation.alert,
                                "actually_crossed": actually_continues,
                            }
                        )

                if not actually_continues:
                    result.path_log.append(
                        f"hop {hop+1}: {edge_type} {cur_station}->{dst} "
                        f"(train {cur_train}->{next_train}): "
                        f"{cur_delay:.1f}min -> {new_delay:.1f}min (below threshold, stops)"
                    )
                    continue

                result.path_log.append(
                    f"hop {hop+1}: {edge_type} {cur_station}->{dst} "
                    f"(train {cur_train}->{next_train}): "
                    f"{cur_delay:.1f}min -> {new_delay:.1f}min (continues)"
                )
                result.traversed_edges.append(
                    {
                        "hop": hop + 1,
                        "src": cur_station,
                        "dst": dst,
                        "edge_type": edge_type,
                        "from_train": cur_train,
                        "to_train": next_train,
                        "delay_before": round(cur_delay, 1),
                        "delay_after": round(new_delay, 1),
                        "zone_crossing": bool(data.get("zone_crossing")),
                    }
                )

                result.cascade_depth = max(result.cascade_depth, hop + 1)
                result.delay_spread_minutes += new_delay
                if data.get("zone_crossing"):
                    result.cross_zone_propagation = True
                if next_train != train_number:
                    result.affected_trains.add(next_train)

                queue.append((next_train, dst, new_delay, hop + 1))

        return result

    def simulate_from_delay_events(self, delay_events_path: str, frontier_detector=None) -> pd.DataFrame:
        """Batch-runs simulate_event() over every row in delay_events.csv and
        returns a DataFrame shaped like simulation_dataset.csv's scenario
        columns, but graph-grounded rather than formula-generated."""
        df = pd.read_csv(delay_events_path)
        rows = []
        for _, row in df.iterrows():
            result = self.simulate_event(
                train_number=row["train_number"],
                station_id=row["station_id"],
                delay_minutes=float(row["delay_minutes"]),
                frontier_detector=frontier_detector,
            )
            record = result.as_dict()
            record["event_id"] = row["event_id"]
            record["frontier_checks"] = result.frontier_checks
            rows.append(record)
        return pd.DataFrame(rows)
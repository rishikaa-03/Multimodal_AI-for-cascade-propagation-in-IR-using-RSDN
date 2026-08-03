"""
Module 7 — Recommendation Engine (implements SRS FR-09)
============================================================

"The system shall generate ranked rerouting and scheduling recommendations
using a heuristic optimiser, presenting each recommendation with the
proposed action, estimated delay recovery, and confidence score."

How estimated delay recovery is computed (not guessed):
    For every hop where a delay actually propagated during a cascade, this
    asks a genuine counterfactual question via the simulator itself: "if a
    mitigation action fully absorbed the delay at this one specific hop,
    how much total downstream delay would that prevent?" It answers this
    by re-running CascadeSimulator.simulate_event() with that hop blocked
    (see blocked_hops in cascade_simulator.py) and measuring the drop in
    delay_spread_minutes versus the original run. This is a real
    re-simulation, not an estimate pulled from nowhere.

How confidence score is computed (documented as a heuristic, per the
SRS's own wording -- NOT a trained/calibrated probability):
    confidence = 0.5 + 0.4 * (recovery / baseline_delay_spread_minutes)
    capped at 0.95. Larger relative recovery -> higher confidence the
    action is worth taking. This deliberately mirrors a human dispatcher's
    intuition ("the bigger the payoff, the more confident I am it's worth
    doing") rather than claiming statistical calibration this project's
    data can't actually support.

Action type by edge type:
    - RS_DEP hop  -> "Reallocate locomotive" (breaking a hidden
      rolling-stock chain is exactly this project's core contribution)
    - SCHED_DEP hop -> "Add recovery buffer / hold connection" (can't
      truly "reroute" a train's own scheduled stops without a full
      timetable-replanning system, which is out of scope -- this is the
      honest, achievable action for this edge type)
    - Any zone-crossing hop that fired a Module 5 alert -> an advisory
      "Notify zone controller" recommendation, independent of the
      recovery-ranked list, since this is about giving lead time rather
      than preventing the delay itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from cascade_simulator import CascadeSimulator, CascadeResult

MIN_RECOVERY_TO_RECOMMEND = 2.0  # ignore hops where blocking them barely helps


@dataclass
class Recommendation:
    action: str
    detail: str
    estimated_recovery_minutes: float
    confidence: float
    kind: str  # "locomotive_reallocation" | "schedule_buffer" | "zone_advisory"


class RecommendationEngine:
    def __init__(self, sim: CascadeSimulator):
        self.sim = sim

    def generate(
        self,
        train_number: int,
        station_id: str,
        delay_minutes: float,
        baseline: CascadeResult,
        frontier_detector=None,
        top_n: int = 3,
    ) -> list[Recommendation]:
        recommendations: list[Recommendation] = []

        if baseline.delay_spread_minutes <= 0:
            return recommendations

        # ---- Recovery-ranked action recommendations ------------------------
        # Only consider each unique hop once, even if multiple queue paths
        # touched it.
        seen_hops = set()
        for edge in baseline.traversed_edges:
            hop_key = (edge["src"], edge["dst"], edge["edge_type"], edge["from_train"])
            if hop_key in seen_hops:
                continue
            seen_hops.add(hop_key)

            counterfactual = self.sim.simulate_event(
                train_number=train_number,
                station_id=station_id,
                delay_minutes=delay_minutes,
                blocked_hops={hop_key},
            )
            recovery = baseline.delay_spread_minutes - counterfactual.delay_spread_minutes
            if recovery < MIN_RECOVERY_TO_RECOMMEND:
                continue

            confidence = min(0.95, 0.5 + 0.4 * (recovery / baseline.delay_spread_minutes))

            if edge["edge_type"] == "RS_DEP":
                recommendations.append(
                    Recommendation(
                        action="Reallocate locomotive",
                        detail=(
                            f"Assign a different locomotive to train {edge['to_train']} at "
                            f"{edge['dst']} instead of the one currently continuing from "
                            f"train {edge['from_train']} — breaks the hidden rolling-stock "
                            f"chain at this point."
                        ),
                        estimated_recovery_minutes=round(recovery, 1),
                        confidence=round(confidence, 2),
                        kind="locomotive_reallocation",
                    )
                )
            else:  # SCHED_DEP
                recommendations.append(
                    Recommendation(
                        action="Add recovery buffer / hold connection",
                        detail=(
                            f"Prioritize platform/track access for train {edge['from_train']} "
                            f"at {edge['dst']} so it can regain time before its next leg, "
                            f"rather than departing immediately at full delay."
                        ),
                        estimated_recovery_minutes=round(recovery, 1),
                        confidence=round(confidence, 2),
                        kind="schedule_buffer",
                    )
                )

        recommendations.sort(key=lambda r: r.estimated_recovery_minutes, reverse=True)
        recommendations = recommendations[:top_n]

        # ---- Zone advisory recommendations (independent of the ranking above) ----
        if frontier_detector is not None:
            for check in baseline.frontier_checks:
                if check["alert"]:
                    recommendations.append(
                        Recommendation(
                            action="Notify zone controller",
                            detail=(
                                f"Advance notice to the zone controller at {check['station_id']} "
                                f"— crossing risk P(cross)={check['p_cross']:.2f} exceeds the "
                                f"{frontier_detector.threshold} alert threshold."
                            ),
                            estimated_recovery_minutes=0.0,  # advisory, not a delay-reducing action
                            confidence=round(min(0.95, check["p_cross"]), 2),
                            kind="zone_advisory",
                        )
                    )

        return recommendations
# Module 7 — Recommendation Engine (FR-09)

## What this module does

Answers the question your dashboard wasn't answering yet: **not just "how
bad is this cascade," but "what should be done about it."** Implements
FR-09 directly: *"generate ranked rerouting and scheduling recommendations
using a heuristic optimiser, presenting each recommendation with the
proposed action, estimated delay recovery, and confidence score."*

Now visible in the dashboard's Live Simulation tab, right below the
frontier alerts: a **"🛠️ Recommended mitigation actions"** table.

## Files

| File | Purpose |
|---|---|
| `recommendation_engine.py` | `RecommendationEngine` and `Recommendation` |
| `cascade_simulator.py` | **Modified** — `simulate_event()` now accepts `blocked_hops`, the mechanism recommendations are measured against |
| `app.py` | **Modified** — added the recommendations section to Live Simulation |

## How estimated delay recovery is actually computed

This is the part worth understanding, since it's not a guessed number.
For every hop where a delay genuinely propagated in a cascade, the engine
asks: *"if a mitigation action fully absorbed the delay right at this
hop, how much total downstream delay would that prevent?"* It answers
this by literally re-running your Module 3 simulator a second time with
that one hop's delay forced to zero, and measuring the drop in
`delay_spread_minutes` versus the original run. This is a real
counterfactual simulation using your own graph and propagation logic —
not an estimate invented separately from it.

## Confidence score — documented as a heuristic, not a trained probability

```
confidence = min(0.95, 0.5 + 0.4 * (recovery / baseline_delay_spread_minutes))
```

Bigger relative payoff → higher confidence the action is worth taking.
This deliberately mirrors a dispatcher's intuition rather than claiming
statistical calibration — your data doesn't have real historical records
of "action taken → actual outcome" to calibrate against, so honestly
representing this as a heuristic (matching the SRS's own phrase, "heuristic
optimiser") is the correct claim to make, not an inflated one.

## Recommendation types

- **Reallocate locomotive** (RS_DEP hops) — swap the locomotive continuing
  onto the next train, breaking the hidden rolling-stock chain. This maps
  directly onto your project's core contribution.
- **Add recovery buffer / hold connection** (SCHED_DEP hops) — prioritize
  platform/track access so a train can recover time before its next leg.
  Deliberately *not* called "reroute," since a true alternate-route
  recommendation would need full timetable-replanning logic this project
  doesn't build — this is the honest, achievable action for this edge
  type given what the system actually knows.
- **Notify zone controller** (any hop that fired a Module 5 alert) — an
  advisory action with `estimated_recovery_minutes = 0`, since its value
  is lead time for a controller, not direct delay prevention. Kept
  separate from the ranked action list rather than competing with it on
  the same "recovery minutes" scale.

## Verification performed

Tested against a real 90-minute delay on train 12006 at NDLS (a top
frontier station): produced 3 ranked mitigation actions (416.3 min,
331.7 min, and 264.2 min estimated recovery, confidence 0.71–0.83) plus
5 zone-advisory notices. Recovery estimates decrease down the ranking as
expected — blocking an earlier hop in a chain prevents more downstream
delay than blocking a later one, which is exactly the pattern a correct
counterfactual computation should produce. Confirmed the dashboard boots
cleanly with this module wired in (HTTP 200, no errors in server logs).

## Where this leaves the project

All seven core modules (graph, simulator, AI predictors, frontier
detector, dashboard, and now recommendations) are implemented and
validated against your real data end-to-end. Remaining per your Gantt:
Docker packaging and a formal NFR testing pass.
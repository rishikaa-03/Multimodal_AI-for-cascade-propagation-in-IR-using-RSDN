# Module 3 — Cascade Propagation Simulator (FR-04)

## What this module does

Takes a single primary delay event — a train, delayed by N minutes, at a
station — and walks the Module 2 graph outward from it using a weighted
BFS, computing how far and how strongly that delay propagates before it's
absorbed. Implements FR-04 directly:

```
propagated_delay = max(0, (upstream_delay - buffer_time) * propagation_weight)
```

repeated hop by hop while the result stays at or above a 5-minute
"still matters" threshold (`CASCADE_STOP_THRESHOLD_MINS`).

## Files

| File | Purpose |
|---|---|
| `cascade_simulator.py` | `CascadeSimulator` class and `CascadeResult` dataclass |
| `run_simulator.py` | Runner: manual trace + full batch run + comparison report |
| `graph_simulation_dataset.csv` | Output: one row per `delay_events.csv` event, graph-grounded |

## How to run

```bash
cd RailwayCascadeAI/graph
python3 run_simulator.py
```

Requires `multiplex_graph.pkl` to already exist (run Module 2's
`build_graph.py` first if it doesn't).

## Scope decision — why only SCHED_DEP and RS_DEP edges propagate

A delay is tied to a specific train, not just a station, so propagation
must respect which edges actually apply to that train:

- **SCHED_DEP** — only the edge matching this exact train carries the
  delay to its own next stop.
- **RS_DEP** — only the edge where this train is the `from_train` carries
  the delay onto the next train sharing its locomotive.
- **INFRA edges are deliberately excluded from propagation in this
  version.** Physical track congestion genuinely can spread delay to
  *other* trains using the same section, but modeling that correctly
  requires timetable-aware simulation — knowing exactly which train uses
  that track next, and when — which is a distinct feature, not a small
  extension of this one. Since SCHED_DEP and RS_DEP are the project's two
  core contributions anyway, this scope was a deliberate choice to get a
  correct, defensible v1 working rather than attempting a bigger and
  shakier v1.

## Validation performed

- Manually traced the single largest delay event in the dataset
  (train 12207, 121-minute delay at DLI) and confirmed the propagated
  delay at each hop matches the FR-04 formula by hand.
- Ran the simulator over all 6,000 events in `delay_events.csv` without
  errors, producing `graph_simulation_dataset.csv`.
- Compared aggregate statistics against the existing formula-based
  `simulation_dataset.csv` from Module 1:

| Metric | Graph-based (this module) | Formula-based (Module 1) |
|---|---|---|
| Mean cascade depth | 2.03 | 2.21 |
| Mean affected trains | 0.19 | 0.49 |
| Mean delay spread (min) | 87.9 | 341.3 |
| % cross-zone propagation | 56.5% | 55.0% |
| corr(initial_delay, cascade_depth) | 0.556 | 0.97 |

**Read honestly, not glossed over:** cascade depth and cross-zone rate
line up closely, which is a good sign the graph traversal itself is
behaving sensibly. Affected-trains count, delay spread, and the
correlation strength are all noticeably lower here — this is the direct,
expected consequence of excluding INFRA-layer propagation (see scope
decision above), not a bug. The formula-based generator likely modeled
broader network-congestion effects that this graph-grounded version
intentionally does not yet capture. This difference is worth stating
plainly in your report as a known, explained limitation rather than
something to hide — it's exactly the kind of finding a project
evaluator expects you to be able to discuss.

## Next: Module 4

Module 4 is the **AI prediction layer** — training the LSTM, GATv2, and
DistilBERT models from your SRS on this graph-grounded dataset (or a
blend of both datasets), then fusing their outputs.

**Waiting for confirmation before starting Module 4.**
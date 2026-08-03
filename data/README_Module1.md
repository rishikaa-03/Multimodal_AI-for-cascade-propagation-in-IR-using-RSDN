# Module 1 — Data Collection, Railway Dataset Design & Synthetic Dataset Generation

## Why synthetic data, and how it stays realistic

Real, station-level Indian Railways operational data (live NTES feeds, loco
rotation logs, per-incident delay text) isn't available at the granularity
or volume a BE project can obtain and license. So Module 1 generates a
**synthetic-but-structurally-real** dataset:

- **73 real stations** across all **17 official IR zones** (CR, ER, ECR,
  ECoR, NR, NCR, NER, NFR, NWR, SR, SCR, SECR, SER, SWR, WR, WCR, KR), with
  real names, approximate real lat/long, and realistic platform counts.
- The **track graph isn't random** — it's built as a per-zone nearest-neighbour
  chain + a hub pattern around each zonal HQ + a curated list of real
  inter-zone trunk corridors (e.g. BSP↔R, NDLS↔PRYJ, SC↔SBC). This gives the
  graph a realistic degree distribution (a few high-degree junction hubs,
  many low-degree leaf stations) — important because Module 3's cascade
  simulator and the later GNN both depend on topology, not just labels.
- **Rolling stock rotations are true chains**: each locomotive serves 1–4
  trains in sequence with a turnaround buffer between legs. This *is* the
  hidden dependency structure your project's core contribution (the RSDG)
  is built to expose — Module 1 seeds it, Module 3 will trace and simulate
  through it.
- `simulation_dataset.csv` is not randomly-labeled. Each row's cascade
  labels are produced by actually running a small forward propagation
  (`propagated_delay = max(0, (parent_delay − buffer) × 0.75)`) over both
  the infrastructure graph and the rolling-stock chain. That's why
  `initial_delay_minutes` correlates ~0.97 with `cascade_depth` and
  `has_rolling_stock_dependency=1` scenarios average 0.71 affected trains
  vs. 0.0 for chain-terminal trains — the ML models in Module 6 have a real
  signal to learn, not noise.

## Files generated (all in `data/`)

| File | Rows | Key columns | Purpose |
|---|---|---|---|
| `stations.csv` | 73 | station_id, zone_id, lat/lon, platform_count, traffic_category | Graph nodes |
| `tracks.csv` | 139 | source_station, dest_station, track_type, zone_crossing | Graph edges (infrastructure layer) |
| `routes.csv` | 1,403 | train_number, sequence_no, station_id, scheduled times | Per-train stopping pattern |
| `trains.csv` | 320 | train_number, type, origin/destination, priority | Train master |
| `locomotives.csv` | 140 | locomotive_id, loco_class, home_zone | Loco master |
| `rolling_stock.csv` | 320 | locomotive_id, train_number, sequence_no_in_rotation, turnaround_buffer_mins | Loco rotation chains (RSDG seed) |
| `weather.csv` | 4,380 | station_id, date, condition, severity | Daily per-station weather (73 stations × 60 days) |
| `delay_events.csv` | 6,000 | train_number, station_id, delay_minutes, cause_category, reported_text | Raw delay reports (feeds NLP module) |
| `simulation_dataset.csv` | 5,200 | see below | **ML-ready feature/label table** |

### `simulation_dataset.csv` schema

**Features:** `initial_delay_minutes`, `station_degree`, `platform_capacity`,
`track_congestion_index`, `has_rolling_stock_dependency`, `weather_condition`,
`weather_severity`, `delay_cause`, `traffic_category`, `zone_id`

**Labels:** `cascade_depth`, `affected_trains_count`, `delay_spread_minutes`,
`cross_zone_propagation`

Total records across all 9 files: **17,975** (exceeds the 5,000+ requirement
comfortably; `simulation_dataset.csv` alone is 5,200 rows as required).

## How to regenerate

```bash
cd RailwayCascadeAI/data
python3 generate_dataset.py --seed 42          # deterministic, reproducible
python3 generate_dataset.py --seed 7 --n-days 90   # different seed / more weather history
```

## Validation performed

- Referential integrity: every `station_id`, `train_number`, `locomotive_id`
  reference across all 9 files resolves to a valid row in its master table
  (checked programmatically — 100% pass).
- All 17 zones represented; 42/139 tracks are genuine cross-zone edges.
- Label sanity: `corr(initial_delay, cascade_depth) ≈ 0.97`,
  `corr(initial_delay, delay_spread_minutes) ≈ 0.81` — strong but not
  perfect correlation (realistic noise from weather/congestion/RS factors).

## Design decision flagged per your working rules ("explain before coding")

I used a **BFS/nearest-neighbour hybrid** to build the track graph rather
than fully random edges, and a **chain-based rotation generator** rather
than independently random loco↔train assignment. Both choices are what make
the labels in `simulation_dataset.csv` learnable — a purely random graph or
random assignment would produce a dataset where no ML model could beat a
mean-predictor baseline, which would undermine Module 6 (comparing RF vs
XGBoost vs GBDT) since there'd be no real signal to differentiate models on.

## Next: Module 2

Module 2 (per your folder structure) is the **Railway Network Graph**
module (`graph/`) — loading `stations.csv` + `tracks.csv` into a NetworkX
multiplex graph object with the three edge types (INFRA, SCHED_DEP,
RS_DEP) specified in your SRS (FR-02), plus the `compute_criticality_score()`
and `get_adjacent_stations()` methods from your Class Diagram.

**Waiting for your confirmation before starting Module 2.**

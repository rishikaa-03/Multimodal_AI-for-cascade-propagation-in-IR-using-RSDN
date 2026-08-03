# Module 6 — Dashboard (Streamlit + Plotly)

## What this module does

The integration layer — one interactive app tying together every prior
module:

- **Module 2** (graph) → the network map itself
- **Module 3** (cascade simulator) → the "Live Simulation" tab
- **Module 4** (AI predictors) → cause prediction shown alongside a
  simulated event
- **Module 5** (frontier detector) → alert markers (orange) on the map
  and the alert feed table

## Files

| File | Purpose |
|---|---|
| `app.py` | The full Streamlit application |

## How to run

```bash
cd RailwayCascadeAI/graph
pip install streamlit plotly
streamlit run app.py
```

This opens in your browser automatically (usually `http://localhost:8501`).
Needs `multiplex_graph.pkl`, `cause_classifier.pkl`, and
`cascade_predictor.pkl` from earlier modules to already exist. If you've
also run `run_simulator.py` and `run_frontier_detector.py`, the "Network
Overview" tab will show real aggregate charts; if not, it'll just show a
note telling you which script to run first, rather than crashing.

## Design decisions

- **Map tiles: Plotly's built-in `"open-street-map"` style** — this needs
  no API key and has no usage cap, which is the free/no-subscription swap
  for Mapbox decided earlier in the project. If you look in `app.py`,
  that's the `mapbox_style="open-street-map"` line in `build_network_map()`.
- **Caching**: the graph, both trained models, and reference CSVs are all
  loaded once via `st.cache_resource`/`st.cache_data`, not reloaded on
  every button click — this matters because loading them fresh each time
  would make the UI feel sluggish for no benefit.
- **Two tabs, not one long page**: "Live Simulation" (pick or invent one
  event, see its exact cascade) and "Network Overview" (aggregate
  statistics across everything already simulated) serve genuinely
  different user intents — an operator reacting to one specific delay
  versus a planner looking at systemic risk — so splitting them keeps
  each screen focused rather than showing everything at once.

## Verification performed

Since a dashboard's real test is clicking through it in a browser, I
verified what can be checked headlessly:
- `app.py` compiles with no syntax errors.
- The app was actually launched (`streamlit run`) and confirmed to boot
  without exceptions — server started cleanly, the page returned HTTP 200,
  and no traceback appeared in server logs or the rendered page. This
  covers the highest-risk startup path: loading the graph, both trained
  models, and building the initial network map.
- The "Run simulation" button's logic isn't a new implementation — it
  calls the exact same `CascadeSimulator`/`CrossZoneFrontierDetector`
  classes already validated end-to-end in Modules 3 and 5, so its
  correctness rests on that existing validation.

**You should still click through it yourself once running locally** —
try a few different events in "Live Simulation," confirm the map
highlights look right, and check the Network Overview tab renders its
charts once you've generated the batch CSVs.

## What's left

Per your Gantt: **Docker containerization** and a final **NFR testing
pass** (response-time targets, load testing) are the remaining items to
close out the project. The core AI/graph pipeline and its UI are done.
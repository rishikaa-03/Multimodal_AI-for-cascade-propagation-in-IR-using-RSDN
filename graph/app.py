"""
Module 6 — Dashboard (Streamlit + Plotly)
============================================

Run with:
    cd RailwayCascadeAI/graph
    streamlit run app.py

Ties every previous module together into one interactive view:
    - Module 2 (graph)              -> the network map
    - Module 3 (cascade simulator)  -> "Live Simulation" tab
    - Module 4 (AI predictors)      -> cause prediction + predicted vs actual comparison
    - Module 5 (frontier detector)  -> alert markers + alert feed

Design decisions:
    - Map tiles use Plotly's built-in "open-street-map" style, which needs
      NO API key and NO usage limits -- this was the free/no-subscription
      swap for Mapbox decided earlier in the project.
    - All heavy artifacts (graph, trained models, precomputed batch
      results) are loaded ONCE via st.cache_resource / st.cache_data, so
      re-running a simulation in the UI doesn't reload multi-second
      objects on every interaction.
"""

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiplex_graph import MultiplexGraph  # noqa: E402
from cascade_simulator import CascadeSimulator  # noqa: E402
from cross_zone_detector import CrossZoneFrontierDetector  # noqa: E402
from cause_classifier import CauseClassifier  # noqa: E402
from cascade_predictor import CascadePredictor, add_graph_features  # noqa: E402
from recommendation_engine import RecommendationEngine  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))

st.set_page_config(page_title="Railway Cascade AI", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders -- these are the expensive one-time operations
# ---------------------------------------------------------------------------

@st.cache_resource
def load_graph() -> MultiplexGraph:
    return MultiplexGraph.load(os.path.join(HERE, "multiplex_graph.pkl"))


@st.cache_resource
def load_detector(_mg: MultiplexGraph) -> CrossZoneFrontierDetector:
    return CrossZoneFrontierDetector(_mg)


@st.cache_resource
def load_models():
    clf = CauseClassifier.load(os.path.join(HERE, "cause_classifier.pkl"))
    predictor = CascadePredictor.load(os.path.join(HERE, "cascade_predictor.pkl"))
    return clf, predictor


@st.cache_data
def load_reference_data():
    stations = pd.read_csv(os.path.join(DATA_DIR, "stations.csv"))
    delay_events = pd.read_csv(os.path.join(DATA_DIR, "delay_events.csv"))
    return stations, delay_events


@st.cache_data
def load_precomputed_results():
    """Module 3/5's batch outputs, if they've been generated -- used for
    the Network Overview tab. Returns None for anything not found yet."""
    graph_sim_path = os.path.join(HERE, "graph_simulation_dataset.csv")
    frontier_path = os.path.join(HERE, "frontier_alerts_summary.csv")
    graph_sim = pd.read_csv(graph_sim_path) if os.path.exists(graph_sim_path) else None
    frontier = pd.read_csv(frontier_path) if os.path.exists(frontier_path) else None
    return graph_sim, frontier


@st.cache_data
def load_network_risk():
    """Module 8's whole-network risk tally (Modulerun via
    network_risk_summary.py) -- station/edge involvement across ALL
    events, for the Full Network Map tab. Returns (None, None) if it
    hasn't been generated yet."""
    station_path = os.path.join(HERE, "station_risk_summary.csv")
    edge_path = os.path.join(HERE, "edge_risk_summary.csv")
    if os.path.exists(station_path) and os.path.exists(edge_path):
        return pd.read_csv(station_path), pd.read_csv(edge_path)
    return None, None


RISK_COLORS = {"normal": "#2E7D32", "cascade": "#C62828", "cross_zone": "#1565C0"}  # green / red / blue
RISK_LABELS = {"normal": "Normal", "cascade": "Cascade risk", "cross_zone": "Cross-zone cascade risk"}


mg = load_graph()
detector = load_detector(mg)
cause_clf, cascade_predictor = load_models()
stations_df, delay_events_df = load_reference_data()
graph_sim_df, frontier_df = load_precomputed_results()
station_risk_df, edge_risk_df = load_network_risk()


# ---------------------------------------------------------------------------
# Map building
# ---------------------------------------------------------------------------

def build_network_map(highlight_edges=None, highlight_stations=None, alert_stations=None):
    """Base network map (all INFRA edges, all stations), optionally with a
    simulated cascade path and frontier alerts highlighted on top."""
    highlight_edges = highlight_edges or []
    highlight_stations = highlight_stations or set()
    alert_stations = alert_stations or set()

    fig = go.Figure()

    # Base INFRA edges, drawn faint so the highlighted path stands out
    seen_pairs = set()
    edge_lats, edge_lons = [], []
    for u, v, data in mg.graph.edges(data=True):
        if data.get("edge_type") != "INFRA":
            continue
        pair = tuple(sorted((u, v)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        lat1, lon1 = mg.graph.nodes[u]["latitude"], mg.graph.nodes[u]["longitude"]
        lat2, lon2 = mg.graph.nodes[v]["latitude"], mg.graph.nodes[v]["longitude"]
        edge_lats += [lat1, lat2, None]
        edge_lons += [lon1, lon2, None]

    fig.add_trace(
        go.Scattermapbox(
            lat=edge_lats, lon=edge_lons, mode="lines",
            line=dict(width=1, color="lightgray"),
            hoverinfo="none", showlegend=False,
        )
    )

    # Highlighted cascade path, drawn on top in red
    if highlight_edges:
        h_lats, h_lons = [], []
        for e in highlight_edges:
            lat1, lon1 = mg.graph.nodes[e["src"]]["latitude"], mg.graph.nodes[e["src"]]["longitude"]
            lat2, lon2 = mg.graph.nodes[e["dst"]]["latitude"], mg.graph.nodes[e["dst"]]["longitude"]
            h_lats += [lat1, lat2, None]
            h_lons += [lon1, lon2, None]
        fig.add_trace(
            go.Scattermapbox(
                lat=h_lats, lon=h_lons, mode="lines",
                line=dict(width=3, color=RISK_COLORS["cascade"]),
                hoverinfo="none", name="Cascade path",
            )
        )

    # All stations, colored by role in the current view
    colors, sizes, texts = [], [], []
    for _, row in stations_df.iterrows():
        sid = row["station_id"]
        if sid in highlight_stations:
            colors.append(RISK_COLORS["cascade"]); sizes.append(12)   # red — on this cascade's path
        elif sid in alert_stations:
            colors.append(RISK_COLORS["cross_zone"]); sizes.append(11)  # blue — cross-zone alert fired here
        else:
            colors.append(RISK_COLORS["normal"]); sizes.append(6)     # green — uninvolved
        texts.append(f"{row['station_name']} ({sid}) — {row['zone_name']}")

    fig.add_trace(
        go.Scattermapbox(
            lat=stations_df["latitude"], lon=stations_df["longitude"],
            mode="markers", marker=dict(size=sizes, color=colors),
            text=texts, hoverinfo="text", name="Stations",
        )
    )

    fig.update_layout(
        mapbox_style="open-street-map",  # free, no API key needed
        mapbox_center=dict(lat=22.5, lon=79.0),
        mapbox_zoom=3.8,
        margin=dict(l=0, r=0, t=0, b=0),
        height=550,
        showlegend=True,
    )
    return fig


def build_full_risk_map(station_risk_df: pd.DataFrame, edge_risk_df: pd.DataFrame):
    """The whole-network view: every station and every train/rolling-stock
    dependency link, colored by whether it was EVER part of a cascade
    across all simulated events (green = never, red = cascade at least
    once, blue = a cross-zone cascade at least once). This is a system-
    wide risk map, not tied to any one chosen event -- that's what the
    Live Simulation tab is for.
    """
    fig = go.Figure()

    # Physical track, faint background -- this is the "map", not the
    # thing being colored by risk (that's the dependency edges below).
    seen_pairs = set()
    edge_lats, edge_lons = [], []
    for u, v, data in mg.graph.edges(data=True):
        if data.get("edge_type") != "INFRA":
            continue
        pair = tuple(sorted((u, v)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        lat1, lon1 = mg.graph.nodes[u]["latitude"], mg.graph.nodes[u]["longitude"]
        lat2, lon2 = mg.graph.nodes[v]["latitude"], mg.graph.nodes[v]["longitude"]
        edge_lats += [lat1, lat2, None]
        edge_lons += [lon1, lon2, None]
    fig.add_trace(
        go.Scattermapbox(
            lat=edge_lats, lon=edge_lons, mode="lines",
            line=dict(width=1, color="lightgray"),
            hoverinfo="none", showlegend=False,
        )
    )

    # Train/rolling-stock dependency links (SCHED_DEP + RS_DEP), colored
    # by risk. Drawn in order normal -> cascade -> cross_zone so the more
    # severe colors sit on top and aren't hidden underneath green lines.
    for level in ["normal", "cascade", "cross_zone"]:
        subset = edge_risk_df[edge_risk_df["risk_level"] == level]
        lats, lons = [], []
        for row in subset.itertuples():
            if row.src not in mg.graph or row.dst not in mg.graph:
                continue
            lat1, lon1 = mg.graph.nodes[row.src]["latitude"], mg.graph.nodes[row.src]["longitude"]
            lat2, lon2 = mg.graph.nodes[row.dst]["latitude"], mg.graph.nodes[row.dst]["longitude"]
            lats += [lat1, lat2, None]
            lons += [lon1, lon2, None]
        fig.add_trace(
            go.Scattermapbox(
                lat=lats, lon=lons, mode="lines",
                line=dict(width=2 if level == "normal" else 2.5, color=RISK_COLORS[level]),
                opacity=0.5 if level == "normal" else 0.85,
                hoverinfo="none", name=f"{RISK_LABELS[level]} (link)",
            )
        )

    # Stations, colored by the most severe risk level touching them.
    merged = stations_df.merge(station_risk_df, on="station_id", how="left")
    merged["risk_level"] = merged["risk_level"].fillna("normal")
    for level in ["normal", "cascade", "cross_zone"]:
        subset = merged[merged["risk_level"] == level]
        fig.add_trace(
            go.Scattermapbox(
                lat=subset["latitude"], lon=subset["longitude"],
                mode="markers",
                marker=dict(size=7 if level == "normal" else 10, color=RISK_COLORS[level]),
                text=[f"{r.station_name} ({r.station_id}) — {r.zone_name}" for r in subset.itertuples()],
                hoverinfo="text", name=f"{RISK_LABELS[level]} (station)",
            )
        )

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center=dict(lat=22.5, lon=79.0),
        mapbox_zoom=3.8,
        margin=dict(l=0, r=0, t=0, b=0),
        height=650,
        showlegend=True,
        legend=dict(bgcolor="rgba(255,255,255,0.8)"),
    )
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🚆 Railway Cascade AI — Cross-Zone Cascade Frontier Detection")
st.caption(
    "Multimodal AI for Cascade Propagation and Cross-Zone Frontier Detection "
    "Using Rolling Stock Dependency Graphs in Indian Railways"
)

tab_map, tab_live, tab_overview = st.tabs(["🗺️ Full Network Map", "🔴 Live Simulation", "📊 Network Overview"])

# ---- Tab 0: Full Network Map ------------------------------------------------
with tab_map:
    st.subheader("System-wide cascade risk — every station and train link")
    st.caption(
        "Green = never part of a cascade across all 6,000 simulated events. "
        "Red = part of a cascade at least once. Blue = part of a cascade that "
        "crossed a zone boundary at least once. This is the whole-network "
        "picture — pick one specific event to trace in the Live Simulation tab."
    )
    if station_risk_df is not None and edge_risk_df is not None:
        n_normal = (station_risk_df["risk_level"] == "normal").sum()
        n_cascade = (station_risk_df["risk_level"] == "cascade").sum()
        n_cross = (station_risk_df["risk_level"] == "cross_zone").sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 Normal stations", n_normal)
        c2.metric("🔴 Cascade-risk stations", n_cascade)
        c3.metric("🔵 Cross-zone-risk stations", n_cross)

        st.plotly_chart(build_full_risk_map(station_risk_df, edge_risk_df), use_container_width=True)
    else:
        st.info("Run `python3 network_risk_summary.py` first to generate the whole-network risk data.")

# ---- Tab 1: Live Simulation ------------------------------------------------
with tab_live:
    st.subheader("Simulate a delay event")

    col1, col2 = st.columns([1, 2])

    with col1:
        source = st.radio("Event source", ["Pick a real event", "Custom event"], horizontal=True)

        if source == "Pick a real event":
            sample_events = delay_events_df.sample(50, random_state=7).reset_index(drop=True)
            options = [
                f"{r.event_id} — train {r.train_number} at {r.station_id} ({r.delay_minutes} min, {r.cause_category})"
                for r in sample_events.itertuples()
            ]
            choice = st.selectbox("Event", options)
            idx = options.index(choice)
            chosen = sample_events.iloc[idx]
            train_number = int(chosen["train_number"])
            station_id = chosen["station_id"]
            delay_minutes = float(chosen["delay_minutes"])
            reported_text = chosen["reported_text"]
        else:
            station_options = sorted(mg.graph.nodes())
            station_id = st.selectbox("Station", station_options)
            train_number = st.number_input("Train number", value=12006, step=1)
            delay_minutes = st.slider("Initial delay (minutes)", 1, 150, 30)
            reported_text = st.text_input("Reported cause (free text, optional)", "")

        run = st.button("▶ Run simulation", type="primary")

    if run:
        sim = CascadeSimulator(mg)
        result = sim.simulate_event(
            train_number=train_number, station_id=station_id,
            delay_minutes=delay_minutes, frontier_detector=detector,
        )

        with col2:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Cascade depth", result.cascade_depth)
            m2.metric("Delay spread (min)", f"{result.delay_spread_minutes:.0f}")
            m3.metric("Trains affected", len(result.affected_trains))
            m4.metric("Crosses a zone?", "Yes" if result.cross_zone_propagation else "No")

            if reported_text:
                predicted_cause = cause_clf.predict(pd.Series([reported_text]))[0]
                st.info(f"**Predicted cause (from reported text):** {predicted_cause}")

            alerts = [c for c in result.frontier_checks if c["alert"]]
            if alerts:
                st.warning(f"⚠️ {len(alerts)} cross-zone frontier alert(s) fired during this cascade:")
                st.dataframe(pd.DataFrame(alerts), use_container_width=True, hide_index=True)
            else:
                st.success("No cross-zone frontier alerts fired for this event.")

        st.subheader("🛠️ Recommended mitigation actions")
        st.caption(
            "Each action's estimated recovery is computed by actually re-running the "
            "simulator with that dependency broken and measuring the reduction in total "
            "downstream delay — not a guessed number."
        )
        engine = RecommendationEngine(sim)
        recommendations = engine.generate(
            train_number=train_number, station_id=station_id, delay_minutes=delay_minutes,
            baseline=result, frontier_detector=detector,
        )
        if recommendations:
            action_recs = [r for r in recommendations if r.kind != "zone_advisory"]
            advisory_recs = [r for r in recommendations if r.kind == "zone_advisory"]

            if action_recs:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Action": r.action,
                                "Detail": r.detail,
                                "Est. recovery (min)": r.estimated_recovery_minutes,
                                "Confidence": r.confidence,
                            }
                            for r in action_recs
                        ]
                    ),
                    use_container_width=True, hide_index=True,
                )
            if advisory_recs:
                st.caption("Advisory (no direct delay recovery, but gives controllers lead time):")
                st.dataframe(
                    pd.DataFrame(
                        [{"Notify": r.detail, "Confidence": r.confidence} for r in advisory_recs]
                    ),
                    use_container_width=True, hide_index=True,
                )
        else:
            st.caption("No mitigation actions meet the minimum recovery threshold for this event.")


        st.subheader("Cascade path")
        highlight_stations = {station_id} | {e["dst"] for e in result.traversed_edges}
        alert_stations = {c["station_id"] for c in alerts} if alerts else set()
        st.plotly_chart(
            build_network_map(result.traversed_edges, highlight_stations, alert_stations),
            use_container_width=True,
        )

        if result.traversed_edges:
            st.subheader("Hop-by-hop detail")
            st.dataframe(pd.DataFrame(result.traversed_edges), use_container_width=True, hide_index=True)
        else:
            st.caption("Delay was absorbed immediately — no propagation past the origin station.")
    else:
        with col2:
            st.plotly_chart(build_network_map(), use_container_width=True)
        st.caption("Pick an event and click **Run simulation** to see its cascade.")

# ---- Tab 2: Network Overview ------------------------------------------------
with tab_overview:
    st.subheader("Network-wide statistics")

    if graph_sim_df is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Events simulated", len(graph_sim_df))
        c2.metric("Mean cascade depth", f"{graph_sim_df['cascade_depth'].mean():.2f}")
        c3.metric("Mean affected trains", f"{graph_sim_df['affected_trains_count'].mean():.2f}")
        c4.metric("Cross-zone rate", f"{100 * graph_sim_df['cross_zone_propagation'].mean():.1f}%")

        st.plotly_chart(
            go.Figure(
                data=[go.Histogram(x=graph_sim_df["cascade_depth"], nbinsx=10)],
                layout=dict(title="Distribution of cascade depth across all simulated events", height=350),
            ),
            use_container_width=True,
        )
    else:
        st.info("Run `run_simulator.py` first to generate `graph_simulation_dataset.csv`.")

    st.subheader("Top frontier stations")
    ranked = sorted(detector.frontiers.items(), key=lambda x: x[1]["crossing_edge_count"], reverse=True)[:15]
    frontier_table = pd.DataFrame(
        [
            {
                "station_id": sid,
                "station_name": mg.graph.nodes[sid]["station_name"],
                "crossing_edges": info["crossing_edge_count"],
                "typical_boundary_buffer_min": round(info["typical_boundary_buffer"], 1),
                "adjacent_zones": ", ".join(sorted(z for z in info["adjacent_zones"] if z)),
            }
            for sid, info in ranked
        ]
    )
    st.dataframe(frontier_table, use_container_width=True, hide_index=True)

    if frontier_df is not None:
        st.subheader("Frontier detector performance (from last full run)")
        st.caption(
            "Computed against the simulator's own ground truth of whether a delay "
            "actually propagated across a zone boundary."
        )
        st.dataframe(frontier_df.head(20), use_container_width=True, hide_index=True)
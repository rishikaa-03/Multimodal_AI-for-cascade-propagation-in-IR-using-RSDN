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
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.exceptions import NotFittedError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiplex_graph import MultiplexGraph  # noqa: E402
from cascade_simulator import CascadeSimulator  # noqa: E402
from cross_zone_detector import CrossZoneFrontierDetector  # noqa: E402
from cause_classifier import CauseClassifier  # noqa: E402
from cascade_predictor import CascadePredictor, add_graph_features  # noqa: E402
from recommendation_engine import RecommendationEngine  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))

st.set_page_config(
    page_title="Railway Cascade AI System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Read the active Streamlit theme only for Plotly-specific styling.
# The page itself is themed by Streamlit's native theme system in
# .streamlit/config.toml, rather than forcing a background from CSS.
try:
    _theme_info = st.context.theme or {}
    _is_dark_theme = _theme_info.get("type", "light") == "dark"
except Exception:
    _is_dark_theme = False

PLOT_PAPER_BG = "#101b2b" if _is_dark_theme else "#ffffff"
PLOT_TEXT = "#f1f5f9" if _is_dark_theme else "#172033"
PLOT_LEGEND_BG = "rgba(7,24,45,0.94)" if _is_dark_theme else "rgba(255,255,255,0.96)"
PLOT_LEGEND_BORDER = "#3a4b60" if _is_dark_theme else "#cbd5e1"


# ---------------------------------------------------------------------------
# Railway Operations Dashboard Theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ================================================================
   Theme: Streamlit owns Light/Dark mode, so this CSS intentionally
   does not paint a fixed page background or global text color —
   native Streamlit surfaces switch correctly with the user's
   Light/Dark setting.

   Chrome (header/tabs/buttons/sidebar) is kept to navy/grey/white so
   it doesn't compete with the red/green/blue in RISK_COLORS, which is
   the only place severity is encoded. Operational data (train
   numbers, station codes, timings, log entries) uses a monospace
   face for tabular alignment.
   ================================================================ */

@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap');

:root {
    --rc-navy: #33475b;
    --rc-navy-deep: #14212e;
    --rc-accent: #3f7cac;
    --rc-alert: #b5812f;
    --rc-red: #b8302f;
    --rc-mono: 'IBM Plex Mono', 'Courier New', monospace;
}

/* ---------- Page/layout ---------- */
.block-container {
    max-width: 100%;
    padding-top: 3.2rem;
    padding-bottom: 3rem;
    padding-left: 2.2rem;
    padding-right: 2.2rem;
}

/* ---------- Operational data readouts: monospace, tabular ---------- */
[data-testid="stMetricValue"],
[data-testid="stDataFrame"],
.stTextInput input,
.stNumberInput input,
code {
    font-family: var(--rc-mono) !important;
}

/* ---------- Severity ticker ---------- */
.rc-ticker-wrap {
    width: 100%;
    overflow: hidden;
    background: var(--rc-navy-deep);
    border-radius: 6px;
    padding: 7px 0;
    margin-bottom: 14px;
    border: 1px solid rgba(255,255,255,.08);
}
.rc-ticker-label {
    display: inline-block;
    font-family: var(--rc-mono);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    color: #ffffff;
    background: var(--rc-alert);
    padding: 3px 10px;
    margin-left: 10px;
    border-radius: 3px;
    vertical-align: middle;
}
.rc-ticker-track {
    display: inline-block;
    white-space: nowrap;
    font-family: var(--rc-mono);
    font-size: 12.5px;
    color: #e8edf5;
    padding-left: 14px;
    vertical-align: middle;
    animation: rc-scroll 32s linear infinite;
}
@keyframes rc-scroll {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-100%); }
}

/* ---------- Login gate ---------- */

/* Keep the Streamlit app shell native-theme controlled. */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.main {
    transition: color .2s ease;
}

/* ---------- Header ---------- */
.railway-header {
    background: linear-gradient(135deg, #071b30 0%, #0b3159 55%, #7a1c1c 100%);
    padding: 16px 26px;
    border-radius: 8px;
    margin-bottom: 14px;
    border-bottom: 3px solid var(--rc-red);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
    position: relative;
    overflow: hidden;
}

.railway-header::after {
    content: "";
    position: absolute;
    right: -70px;
    top: -90px;
    width: 230px;
    height: 230px;
    border-radius: 50%;
    border: 34px solid rgba(255,255,255,.05);
    pointer-events: none;
}

.railway-header-left,
.railway-header-right {
    position: relative;
    z-index: 1;
}

.railway-header-left {
    display: flex;
    align-items: center;
    gap: 14px;
}

.railway-emblem {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    border: 1.5px solid rgba(255,255,255,.5);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.railway-emblem svg {
    width: 22px;
    height: 22px;
}

.railway-title {
    color: #ffffff !important;
    font-size: 21px;
    font-weight: 800;
    margin: 0;
    letter-spacing: .3px;
}

.railway-subtitle {
    color: #b9c8db !important;
    font-size: 12px;
    margin-top: 2px;
    font-family: var(--rc-mono);
    letter-spacing: .2px;
}

.railway-header-right {
    text-align: right;
    font-family: var(--rc-mono);
}

.status-pill {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 3px;
    background: rgba(46,125,50,.22);
    color: #8fe39a !important;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .6px;
    border: 1px solid rgba(143,227,154,.35);
}

.railway-clock {
    color: #dce8f5 !important;
    font-size: 11px;
    margin-top: 6px;
    opacity: .85;
}

/* ---------- Tabs: native Streamlit handles theme colors ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 5px;
    padding: 6px;
    border-radius: 12px;
    border: 1px solid currentColor;
    opacity: .95;
}

.stTabs [data-baseweb="tab"] {
    height: 45px;
    padding: 0 19px;
    border-radius: 8px;
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    background: var(--rc-navy-deep) !important;
    color: #ffffff !important;
    box-shadow: inset 0 -3px 0 var(--rc-red);
}

.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] span {
    color: #ffffff !important;
}

/* ---------- Headings ---------- */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    font-weight: 750;
}

.stApp h3 {
    border-left: 4px solid var(--rc-accent);
    padding-left: 10px;
    margin-top: 18px;
    font-family: var(--rc-mono);
    letter-spacing: .2px;
}

/* ---------- Native metric cards: only shape/shadow, no fixed colors ---------- */
[data-testid="stMetric"] {
    border-radius: 12px;
    padding: 15px 17px;
    box-shadow: 0 4px 13px rgba(0,0,0,.07);
    min-height: 102px;
}

[data-testid="stMetricLabel"] {
    font-size: 12px !important;
    font-weight: 750 !important;
    text-transform: uppercase;
    letter-spacing: .35px;
}

[data-testid="stMetricValue"] {
    font-size: 26px !important;
    font-weight: 800 !important;
}

/* ---------- Custom stat cards with a colored status dot ---------- */
.rc-stat-card {
    border-radius: 12px;
    padding: 15px 17px;
    box-shadow: 0 4px 13px rgba(0,0,0,.07);
    min-height: 102px;
    border: 1px solid rgba(120,120,120,.18);
}

.rc-stat-label {
    display: flex;
    align-items: center;
    font-size: 12px;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: .35px;
    opacity: .85;
    margin-bottom: 8px;
}

.rc-dot {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    margin-right: 8px;
    flex-shrink: 0;
}

.rc-stat-value {
    font-family: var(--rc-mono);
    font-size: 26px;
    font-weight: 800;
}

/* ---------- Inputs: keep native theme background/text ---------- */
.stSelectbox > div > div,
.stNumberInput > div > div,
.stTextInput > div > div,
.stTextArea > div > div {
    border-radius: 8px !important;
}

/* Don't force input text to a light/dark color. */
.stSelectbox input,
.stNumberInput input,
.stTextInput input,
.stTextArea textarea {
    color: inherit !important;
    -webkit-text-fill-color: currentColor !important;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: linear-gradient(135deg, #3a4453, #262e3a) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px;
    min-height: 42px;
    font-weight: 700;
    transition: .2s ease;
}

.stButton > button p,
.stButton > button span {
    color: #ffffff !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #464f5f, #2e3743) !important;
    transform: translateY(-1px);
    box-shadow: 0 5px 14px rgba(0,0,0,.18);
}

/* Primary action (e.g. Run Simulation) stands out in red */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #c0392f, #96271f) !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #d1453a, #a52d24) !important;
    box-shadow: 0 5px 14px rgba(192,57,47,.30);
}

/* ---------- Radio / slider / checkbox: inherit native theme ---------- */
.stRadio label,
.stCheckbox label,
.stSlider label {
    font-weight: 600;
}

/* ---------- Alerts ---------- */
.stAlert {
    border-radius: 10px !important;
}

/* ---------- Tables: native Streamlit handles colors ---------- */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

/* ---------- Plotly container ---------- */
.stPlotlyChart {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 14px rgba(0,0,0,.08);
}

/* ---------- Dividers / scrollbar ---------- */
hr {
    margin: 20px 0;
}

::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-thumb {
    background: #8793a1;
    border-radius: 8px;
}

/* ---------- Responsive ---------- */
@media (max-width: 900px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .railway-title {
        font-size: 24px;
    }
    .railway-subtitle {
        font-size: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0 9px;
        font-size: 12px;
    }
}
</style>
""", unsafe_allow_html=True)


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


def render_stat_card(color: str, label: str, value: str) -> None:
    """A metric-style card with a colored status dot ahead of the label,
    matching the same green/red/blue used on the risk map."""
    st.markdown(
        f"""
        <div class="rc-stat-card">
            <div class="rc-stat-label"><span class="rc-dot" style="background:{color};"></span>{label.upper()}</div>
            <div class="rc-stat-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        height=610,
        paper_bgcolor=PLOT_PAPER_BG,
        plot_bgcolor=PLOT_PAPER_BG,
        font=dict(family="Arial, sans-serif", color=PLOT_TEXT),
        legend=dict(
            bgcolor=PLOT_LEGEND_BG,
            bordercolor=PLOT_LEGEND_BORDER,
            borderwidth=1,
            font=dict(size=11, color=PLOT_TEXT),
            x=0.015,
            y=0.025,
            xanchor="left",
            yanchor="bottom",
            orientation="v",
        ),
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
                hoverinfo="none", name=f"{RISK_LABELS[level]} link",
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
                hoverinfo="text", name=f"{RISK_LABELS[level]} station",
            )
        )

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center=dict(lat=22.5, lon=79.0),
        mapbox_zoom=3.8,
        margin=dict(l=0, r=0, t=0, b=0),
        height=680,
        paper_bgcolor=PLOT_PAPER_BG,
        plot_bgcolor=PLOT_PAPER_BG,
        font=dict(family="Arial, sans-serif", color=PLOT_TEXT),
        showlegend=True,
        legend=dict(
            bgcolor=PLOT_LEGEND_BG,
            bordercolor=PLOT_LEGEND_BORDER,
            borderwidth=1,
            font=dict(size=11, color=PLOT_TEXT),
            x=0.015,
            y=0.025,
            xanchor="left",
            yanchor="bottom",
            orientation="v",
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.markdown(f"""
<div class="railway-header">
    <div class="railway-header-left">
        <div class="railway-emblem">
            <svg viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
                <rect x="5" y="4" width="14" height="12" rx="3"></rect>
                <line x1="5" y1="10" x2="19" y2="10"></line>
                <line x1="9" y1="4" x2="9" y2="10"></line>
                <path d="M7 16 L5.5 19"></path>
                <path d="M17 16 L18.5 19"></path>
                <circle cx="8" cy="19" r="1.2" fill="#ffffff" stroke="none"></circle>
                <circle cx="16" cy="19" r="1.2" fill="#ffffff" stroke="none"></circle>
            </svg>
        </div>
        <div>
            <div class="railway-title">Railway Cascade AI System</div>
            <div class="railway-subtitle">Cross-Zone Cascade Frontier Detection &amp; Network Risk Intelligence</div>
        </div>
    </div>
    <div class="railway-header-right">
        <div class="status-pill">System Operational</div>
        <div class="railway-clock">{datetime.now().strftime('%d %b %Y, %H:%M')} IST</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Frontier alert ticker
# ---------------------------------------------------------------------------
if frontier_df is not None and len(frontier_df) > 0 and "alert" in frontier_df.columns:
    recent_alerts = frontier_df[frontier_df["alert"] == True].tail(12)  # noqa: E712
    if len(recent_alerts) > 0:
        items = []
        for r in recent_alerts.itertuples():
            sid = getattr(r, "station_id", "—")
            items.append(f"CROSS-ZONE ALERT · STATION {sid}")
        ticker_text = "    •    ".join(items) * 2  # duplicate for seamless loop
        st.markdown(
            f"""
            <div class="rc-ticker-wrap">
                <span class="rc-ticker-label">LIVE ALERTS</span>
                <span class="rc-ticker-track">{ticker_text}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

tab_map, tab_live = st.tabs([
    "Full Network Map", ":red[●] Live Simulation",
])

# ---- Tab 0: Full Network Map ------------------------------------------------
with tab_map:
    st.subheader("System-Wide Railway Network Risk")
    st.caption(
        "Station and dependency-link risk across the complete railway network. "
        "Green = normal, red = cascade risk, blue = cross-zone cascade risk. "
        "Select an event in Live Simulation to trace its propagation path."
    )
    if station_risk_df is not None and edge_risk_df is not None:
        n_normal = (station_risk_df["risk_level"] == "normal").sum()
        n_cascade = (station_risk_df["risk_level"] == "cascade").sum()
        n_cross = (station_risk_df["risk_level"] == "cross_zone").sum()
        c1, c2, c3 = st.columns(3)
        with c1:
            render_stat_card(RISK_COLORS["normal"], "Normal Stations", f"{n_normal:,}")
        with c2:
            render_stat_card(RISK_COLORS["cascade"], "Cascade-Risk Stations", f"{n_cascade:,}")
        with c3:
            render_stat_card(RISK_COLORS["cross_zone"], "Cross-Zone Risk", f"{n_cross:,}")

        st.plotly_chart(
            build_full_risk_map(station_risk_df, edge_risk_df),
            use_container_width=True,
        )
    else:
        st.info("Run `python3 network_risk_summary.py` first to generate the whole-network risk data.")

# ---- Tab 1: Live Simulation ------------------------------------------------
with tab_live:
    st.subheader("Live Cascade Simulation")
    st.caption(
        "Select a real railway delay or create a custom event to simulate "
        "delay propagation through the dependency network."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Event Control Panel")
        st.caption("Configure the disruption scenario and launch the simulation.")
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

        run = st.button("Run Simulation", type="primary", use_container_width=True)

    if run:
        sim = CascadeSimulator(mg)
        result = sim.simulate_event(
            train_number=train_number, station_id=station_id,
            delay_minutes=delay_minutes, frontier_detector=detector,
        )

        with col2:
            st.markdown("### Simulation Status")
            st.caption("Impact summary for the simulated cascade.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Cascade Depth", result.cascade_depth)
            m2.metric("Delay Spread", f"{result.delay_spread_minutes:.0f} min")
            m3.metric("Trains Affected", len(result.affected_trains))
            m4.metric(
                "Zone Boundary",
                "CROSSED" if result.cross_zone_propagation else "Contained"
            )

            if reported_text and reported_text.strip():
                try:
                    predicted_cause = cause_clf.predict(pd.Series([reported_text]))[0]
                    st.info(f"**Predicted cause (from reported text):** {predicted_cause}")
                except NotFittedError:
                    st.warning(
                        "Cause prediction is currently unavailable because the "
                        "loaded cause-classifier pipeline is not fitted. The cascade "
                        "simulation itself can continue normally. Retrain the cause "
                        "classifier and regenerate `cause_classifier.pkl`."
                    )

            alerts = [c for c in result.frontier_checks if c["alert"]]
            if alerts:
                st.warning(f"{len(alerts)} cross-zone frontier alert(s) fired during this cascade:")
                st.dataframe(pd.DataFrame(alerts), use_container_width=True, hide_index=True)
            else:
                st.success("No cross-zone frontier alerts fired for this event.")

    
        st.subheader("Recommended Mitigation Actions")
        st.caption(
            "Recommendations are generated by testing dependency-breaking actions "
            "against the simulated cascade and measuring estimated downstream delay recovery."
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


        st.subheader("Cascade Propagation Path")
        st.caption(
            "Highlighted stations and links show how the simulated delay propagated "
            "through the railway dependency network."
        )
        highlight_stations = {station_id} | {e["dst"] for e in result.traversed_edges}
        alert_stations = {c["station_id"] for c in alerts} if alerts else set()
        st.plotly_chart(
            build_network_map(result.traversed_edges, highlight_stations, alert_stations),
            use_container_width=True,
        )

        if result.traversed_edges:
            st.subheader("Hop-by-Hop Propagation Detail")
            st.dataframe(pd.DataFrame(result.traversed_edges), use_container_width=True, hide_index=True)
        else:
            st.caption("Delay was absorbed immediately — no propagation past the origin station.")
    else:
        with col2:
            st.plotly_chart(build_network_map(), use_container_width=True)
        st.caption("Pick an event and click **Run simulation** to see its cascade.")
"""
Module 4b — Cascade Outcome Predictor (implements SRS FR-05/FR-06's
prediction targets, using Random Forest / Gradient Boosting instead of
LSTM + GATv2)

Scope decision (documented, same convention as Modules 2 and 3):
    The SRS specifies an LSTM (time-series) and a GATv2 (graph-structured)
    model, fused together. Real time-series modelling needs many delay
    readings per station over time; this dataset has ~6,000 events spread
    across only 62 stations, which is too sparse for a deep sequence model
    to learn real temporal patterns from. Similarly, GATv2 expects
    meaningful per-scenario subgraphs, not one flat feature row per event.
    Given the data is fundamentally tabular at this stage (one row per
    event, mixed numeric/categorical features), tree-ensemble models are
    the honest and efficient match for it -- this is a standard, defensible
    substitution at this data scale, not a workaround. Predicts three
    targets from FR-05/FR-06:
        - cascade_depth            (regression)
        - delay_spread_minutes     (regression)
        - cross_zone_propagation   (classification, this is what FR-06's
          Cross-Zone Frontier Detector consumes)
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

NUMERIC_FEATURES = [
    "initial_delay_minutes",
    "out_degree",
    "sched_dep_out_count",
    "rs_dep_out_count",
    "criticality_score",
    "has_matching_sched_dep",   # does THIS train have a SCHED_DEP edge out of THIS station
    "has_matching_rs_dep",      # does THIS train have an RS_DEP edge out of THIS station
    "next_edge_buffer_mins",    # buffer on the edge the simulator would actually follow next
    "next_edge_weight",         # propagation_weight on that same edge
]
CATEGORICAL_FEATURES = ["station_id"]  # station identity still kept as a fallback signal


def add_graph_features(df: pd.DataFrame, mg) -> pd.DataFrame:
    """Attaches per-event graph-derived features computed from the Module 2
    MultiplexGraph. Two kinds:
      - station-level aggregates (degree, fan-out, criticality) -- general
        "how connected is this station" signal.
      - event-specific mechanism features (does THIS train, at THIS station,
        have a matching SCHED_DEP/RS_DEP edge, and what's its buffer/weight)
        -- this directly encodes what cascade_simulator.py actually checks
        when deciding whether propagation continues, so it should explain
        cascade_depth far better than station identity alone, since two
        different trains passing through the same station can have very
        different onward dependencies.
    """
    out_degree, sched_out, rs_out, criticality = [], [], [], []
    has_sched, has_rs, next_buffer, next_weight = [], [], [], []

    for station_id, train_number in zip(df["station_id"], df["train_number"]):
        if station_id not in mg.graph:
            out_degree.append(0); sched_out.append(0); rs_out.append(0); criticality.append(0.0)
            has_sched.append(0); has_rs.append(0); next_buffer.append(0.0); next_weight.append(0.0)
            continue

        edges = list(mg.graph.out_edges(station_id, data=True))
        out_degree.append(len(edges))
        sched_out.append(sum(1 for _, _, d in edges if d.get("edge_type") == "SCHED_DEP"))
        rs_out.append(sum(1 for _, _, d in edges if d.get("edge_type") == "RS_DEP"))
        criticality.append(mg.compute_criticality_score(station_id))

        matching_sched = [
            d for _, _, d in edges
            if d.get("edge_type") == "SCHED_DEP" and d.get("train_number") == train_number
        ]
        matching_rs = [
            d for _, _, d in edges
            if d.get("edge_type") == "RS_DEP" and d.get("from_train") == train_number
        ]
        match = (matching_sched or matching_rs)
        chosen = matching_sched[0] if matching_sched else (matching_rs[0] if matching_rs else None)

        has_sched.append(1 if matching_sched else 0)
        has_rs.append(1 if matching_rs else 0)
        next_buffer.append(chosen["buffer_time_mins"] if chosen else 0.0)
        next_weight.append(chosen["propagation_weight"] if chosen else 0.0)

    df = df.copy()
    df["out_degree"] = out_degree
    df["sched_dep_out_count"] = sched_out
    df["rs_dep_out_count"] = rs_out
    df["criticality_score"] = criticality
    df["has_matching_sched_dep"] = has_sched
    df["has_matching_rs_dep"] = has_rs
    df["next_edge_buffer_mins"] = next_buffer
    df["next_edge_weight"] = next_weight
    return df


class CascadePredictor:
    def __init__(self):
        preprocessor = ColumnTransformer(
            [
                ("num", "passthrough", NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ]
        )

        self.depth_model = Pipeline(
            [("prep", preprocessor), ("reg", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))]
        )
        self.spread_model = Pipeline(
            [("prep", preprocessor), ("reg", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))]
        )
        self.crosszone_model = Pipeline(
            [
                ("prep", preprocessor),
                ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")),
            ]
        )

    def train(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict:
        features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        X = df[features]

        results = {}

        for target, model, kind in [
            ("cascade_depth", self.depth_model, "regression"),
            ("delay_spread_minutes", self.spread_model, "regression"),
            ("cross_zone_propagation", self.crosszone_model, "classification"),
        ]:
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state,
                stratify=y if kind == "classification" else None,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            if kind == "regression":
                results[target] = {
                    "mae": mean_absolute_error(y_test, preds),
                    "r2": r2_score(y_test, preds),
                }
            else:
                results[target] = {
                    "accuracy": accuracy_score(y_test, preds),
                    "f1": f1_score(y_test, preds),
                }

        return results

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        X = df[features]
        return pd.DataFrame(
            {
                "predicted_cascade_depth": self.depth_model.predict(X),
                "predicted_delay_spread_minutes": self.spread_model.predict(X),
                "predicted_cross_zone_propagation": self.crosszone_model.predict(X),
                "cross_zone_probability": self.crosszone_model.predict_proba(X)[:, 1],
            }
        )

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "depth_model": self.depth_model,
                    "spread_model": self.spread_model,
                    "crosszone_model": self.crosszone_model,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "CascadePredictor":
        obj = cls()
        with open(path, "rb") as f:
            models = pickle.load(f)
        obj.depth_model = models["depth_model"]
        obj.spread_model = models["spread_model"]
        obj.crosszone_model = models["crosszone_model"]
        return obj
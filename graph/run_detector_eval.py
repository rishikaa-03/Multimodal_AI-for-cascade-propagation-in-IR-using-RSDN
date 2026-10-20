"""
Held-out evaluation of the cross-zone frontier detector with baselines
(addresses reviewer point 4).

    cd RailwayCascadeAI/graph
    python3 run_detector_eval.py

Events are split 60/20/20 into train/validation/test. All thresholds are
chosen on validation events only; all reported numbers are on test events.
Writes detector_eval_methods.csv and detector_eval_thresholds.csv.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiplex_graph import MultiplexGraph  # noqa: E402
from cascade_simulator import CascadeSimulator  # noqa: E402
from cross_zone_detector import CrossZoneFrontierDetector  # noqa: E402

SEED = 42
TARGET_PRECISION = 0.80
TARGET_RECALL = 0.75
TAU_GRID = np.round(np.arange(0.05, 2.0001, 0.05), 2)
REPORT_TAUS = [0.65, 0.75, 0.90, 1.00]
N_BOOT = 1000


def collect_checks(mg, events_path):
    detector = CrossZoneFrontierDetector(mg)
    df = CascadeSimulator(mg).simulate_from_delay_events(events_path, frontier_detector=detector)
    rows = []
    for event_id, checks in zip(df["event_id"], df["frontier_checks"]):
        for c in checks:
            info = detector.frontiers[c["station_id"]]
            buf = info["typical_boundary_buffer"]
            rows.append(
                {
                    "event_id": event_id,
                    "station_id": c["station_id"],
                    "p_cross": c["p_cross"],
                    "delay": c["p_cross"] * buf,
                    "buffer": buf,
                    "crossing_edges": info["crossing_edge_count"],
                    "y": int(c["actually_crossed"]),
                }
            )
    return pd.DataFrame(rows)


def split_by_event(checks):
    rng = np.random.default_rng(SEED)
    events = np.array(sorted(checks["event_id"].unique()))
    rng.shuffle(events)
    n = len(events)
    train_e = set(events[: int(0.6 * n)])
    val_e = set(events[int(0.6 * n): int(0.8 * n)])
    part = checks["event_id"].map(lambda e: "train" if e in train_e else ("val" if e in val_e else "test"))
    return checks[part == "train"], checks[part == "val"], checks[part == "test"]


def confusion(y, pred):
    y = np.asarray(y).astype(bool)
    pred = np.asarray(pred).astype(bool)
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else float("nan")
    return precision, recall, specificity, f1


def best_f1_threshold(score, y):
    candidates = np.unique(np.quantile(score, np.linspace(0.01, 0.99, 99)))
    best_t, best_f = candidates[0], -1.0
    for t in candidates:
        f = confusion(y, score >= t)[3]
        if f == f and f > best_f:
            best_t, best_f = t, f
    return best_t


def paper_rule_threshold(val):
    for tau in TAU_GRID:
        p, r, _, _ = confusion(val["y"], val["p_cross"] >= tau)
        if p >= TARGET_PRECISION and r >= TARGET_RECALL:
            return tau
    return None


def bootstrap_ci(test, tau):
    groups = test.groupby("event_id").indices
    keys = list(groups.keys())
    y = test["y"].to_numpy()
    alert = (test["p_cross"].to_numpy() >= tau)
    rng = np.random.default_rng(SEED)
    ps, rs = [], []
    for _ in range(N_BOOT):
        sampled = rng.choice(len(keys), size=len(keys), replace=True)
        idx = np.concatenate([groups[keys[i]] for i in sampled])
        p, r, _, _ = confusion(y[idx], alert[idx])
        ps.append(p)
        rs.append(r)
    return np.nanpercentile(ps, [2.5, 97.5]), np.nanpercentile(rs, [2.5, 97.5])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(here, "..", "data"))
    mg = MultiplexGraph.build_from_csv(data_dir)
    checks = collect_checks(mg, os.path.join(data_dir, "delay_events.csv"))
    train, val, test = split_by_event(checks)

    print(f"checks: total={len(checks)} train={len(train)} val={len(val)} test={len(test)}")
    print(f"events: train={train['event_id'].nunique()} val={val['event_id'].nunique()} test={test['event_id'].nunique()}")
    print(f"positive rate: train={train['y'].mean():.3f} val={val['y'].mean():.3f} test={test['y'].mean():.3f}")

    tau = paper_rule_threshold(val)
    print(f"\ntau selected on validation with the original rule (precision>={TARGET_PRECISION}, recall>={TARGET_RECALL}): {tau}")
    if tau is None:
        tau = best_f1_threshold(val["p_cross"].to_numpy(), val["y"].to_numpy())
        print(f"no tau met both targets on validation; using best-F1 tau={tau:.3f}")

    p, r, s, f = confusion(test["y"], test["p_cross"] >= tau)
    (p_lo, p_hi), (r_lo, r_hi) = bootstrap_ci(test, tau)
    print(f"\nRatio rule on TEST at tau={tau}: precision={p:.3f} [{p_lo:.3f}, {p_hi:.3f}] "
          f"recall={r:.3f} [{r_lo:.3f}, {r_hi:.3f}] specificity={s:.3f} F1={f:.3f}")

    rows = []
    for t in REPORT_TAUS:
        p_t, r_t, s_t, f_t = confusion(test["y"], test["p_cross"] >= t)
        rows.append({"tau": t, "precision": p_t, "recall": r_t, "specificity": s_t, "f1": f_t})
    thr_table = pd.DataFrame(rows)
    print("\nTEST metrics at fixed thresholds")
    print(thr_table.round(3).to_string(index=False))
    thr_table.to_csv(os.path.join(here, "detector_eval_thresholds.csv"), index=False)

    feats_station = ["buffer", "crossing_edges"]
    feats_all = ["delay", "buffer", "crossing_edges"]
    models = {
        "station_only_logreg": (feats_station, make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
        "logreg": (feats_all, make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
        "random_forest": (feats_all, RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=SEED, n_jobs=-1)),
    }

    scores_val = {
        "always_alert": np.ones(len(val)),
        "delay_only": val["delay"].to_numpy(),
        "ratio_rule": val["p_cross"].to_numpy(),
    }
    scores_test = {
        "always_alert": np.ones(len(test)),
        "delay_only": test["delay"].to_numpy(),
        "ratio_rule": test["p_cross"].to_numpy(),
    }
    for name, (feats, model) in models.items():
        model.fit(train[feats], train["y"])
        scores_val[name] = model.predict_proba(val[feats])[:, 1]
        scores_test[name] = model.predict_proba(test[feats])[:, 1]

    methods = []
    for name in ["always_alert", "station_only_logreg", "delay_only", "logreg", "random_forest", "ratio_rule"]:
        t = best_f1_threshold(scores_val[name], val["y"].to_numpy())
        p_m, r_m, s_m, f_m = confusion(test["y"], scores_test[name] >= t)
        ap = average_precision_score(test["y"], scores_test[name])
        auc = roc_auc_score(test["y"], scores_test[name])
        methods.append(
            {"method": name, "pr_auc": ap, "roc_auc": auc, "precision": p_m, "recall": r_m,
             "specificity": s_m, "f1": f_m}
        )
    methods_df = pd.DataFrame(methods)
    print("\nTEST comparison (operating point = threshold maximising F1 on validation)")
    print(methods_df.round(3).to_string(index=False))
    methods_df.to_csv(os.path.join(here, "detector_eval_methods.csv"), index=False)

    print(f"\nTest positive rate (= PR-AUC of always_alert): {test['y'].mean():.3f}")


if __name__ == "__main__":
    main()
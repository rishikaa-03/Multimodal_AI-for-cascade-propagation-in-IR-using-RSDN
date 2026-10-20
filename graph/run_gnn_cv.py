"""
Repeated k-fold comparison of the GATv2 station-risk model against
non-graph baselines (addresses reviewer point 5).

    cd RailwayCascadeAI/graph
    python3 run_gnn_cv.py

Target: mean cascade depth of events originating at each station
(graph_simulation_dataset.csv). Every labelled station is predicted
out-of-fold in each of 10 repeats of 5-fold CV, so R2 is computed on all
labelled stations rather than on a ~13-station split.
Needs torch and torch_geometric for the MLP and GATv2 rows; without them
only the mean, Ridge and RandomForest rows are produced.
Writes gnn_cv_results.csv.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiplex_graph import MultiplexGraph  # noqa: E402

try:
    import torch
    import torch.nn as nn
    from gnn_cascade_model import CascadeGATv2, build_pyg_data

    HAS_TORCH = True
except ImportError as exc:
    HAS_TORCH = False
    TORCH_ERROR = str(exc)

N_SPLITS = 5
N_REPEATS = 10
SEED = 42
EPOCHS = 300
LR = 0.01
WEIGHT_DECAY = 5e-4


def node_features(mg, station_ids):
    rows = []
    for sid in station_ids:
        node = mg.graph.nodes[sid]
        rows.append(
            [
                node.get("platform_count", 0),
                node.get("capacity_trains_per_hour", 0),
                mg.graph.in_degree(sid),
                mg.graph.out_degree(sid),
                mg.compute_criticality_score(sid),
            ]
        )
    return np.array(rows, dtype=float)


if HAS_TORCH:

    class NodeMLP(nn.Module):
        def __init__(self, in_channels, hidden=32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_channels, hidden), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    def fit_predict_torch(kind, data, train_idx, seed):
        torch.manual_seed(seed)
        if kind == "gatv2":
            model = CascadeGATv2(in_channels=data.x.shape[1], edge_dim=data.edge_attr.shape[1])
            forward = lambda: model(data.x, data.edge_index, data.edge_attr)
        else:
            model = NodeMLP(data.x.shape[1])
            forward = lambda: model(data.x)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fn = nn.MSELoss()
        train_t = torch.tensor(train_idx, dtype=torch.long)
        for _ in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(forward()[train_t], data.y[train_t])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            return forward().numpy()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(here, "..", "data"))
    mg = MultiplexGraph.build_from_csv(data_dir)
    sim = pd.read_csv(os.path.join(here, "graph_simulation_dataset.csv"))
    targets = sim.groupby("station_id")["cascade_depth"].mean().to_dict()

    station_ids = sorted(mg.graph.nodes())
    y_all = np.array([targets.get(s, np.nan) for s in station_ids], dtype=float)
    labelled = np.where(~np.isnan(y_all))[0]
    X_all = node_features(mg, station_ids)
    print(f"{len(labelled)} labelled stations out of {len(station_ids)}")

    data = None
    if HAS_TORCH:
        data, _, _ = build_pyg_data(mg, targets)
    else:
        print(f"torch / torch_geometric unavailable ({TORCH_ERROR}); skipping MLP and GATv2 rows")

    methods = ["mean_baseline", "ridge", "random_forest"]
    if HAS_TORCH:
        methods += ["mlp_no_graph", "gatv2"]

    oof = {m: np.full((N_REPEATS, len(labelled)), np.nan) for m in methods}
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

    for i, (tr, te) in enumerate(rkf.split(labelled)):
        rep = i // N_SPLITS
        train_idx, test_idx = labelled[tr], labelled[te]
        y_train = y_all[train_idx]

        oof["mean_baseline"][rep, te] = y_train.mean()

        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X_all[train_idx], y_train)
        oof["ridge"][rep, te] = ridge.predict(X_all[test_idx])

        rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
        rf.fit(X_all[train_idx], y_train)
        oof["random_forest"][rep, te] = rf.predict(X_all[test_idx])

        if HAS_TORCH:
            for kind, name in [("mlp", "mlp_no_graph"), ("gatv2", "gatv2")]:
                pred = fit_predict_torch(kind, data, train_idx, SEED + i)
                oof[name][rep, te] = pred[test_idx]

        if (i + 1) % N_SPLITS == 0:
            print(f"repeat {rep + 1}/{N_REPEATS} done")

    y_lab = y_all[labelled]
    rows = []
    for m in methods:
        r2s = [r2_score(y_lab, oof[m][r]) for r in range(N_REPEATS)]
        maes = [mean_absolute_error(y_lab, oof[m][r]) for r in range(N_REPEATS)]
        rows.append(
            {"method": m, "r2_mean": np.mean(r2s), "r2_std": np.std(r2s),
             "mae_mean": np.mean(maes), "mae_std": np.std(maes)}
        )
    out = pd.DataFrame(rows)
    print(f"\nOut-of-fold results over {N_REPEATS} x {N_SPLITS}-fold CV on {len(labelled)} stations")
    print(out.round(3).to_string(index=False))
    print(f"\nTarget mean={np.mean(y_lab):.3f} std={np.std(y_lab):.3f}")
    out.to_csv(os.path.join(here, "gnn_cv_results.csv"), index=False)


if __name__ == "__main__":
    main()
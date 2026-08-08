"""
Module 4c — GATv2 Graph Neural Network (PyTorch Geometric)
==============================================================

Your guide asked specifically for PyTorch Geometric / DGL for GNN-based
cascade prediction. Module 4's original CascadePredictor (Random Forest)
was a deliberate substitution documented as a scope decision — this module
replaces that reasoning with an actual GNN, and does it in the way that
genuinely fits your data rather than forcing PyTorch Geometric onto
something it isn't suited for.

Why NODE-LEVEL prediction, not per-event:
    An event-level GNN would need a separate graph "snapshot" per delay
    event (6,000 of them), which is heavy machinery for what your data
    actually supports. What your graph DOES have that's genuinely
    GNN-shaped is its own structure: 73 stations connected by three kinds
    of dependency edges. So instead, this model does what GNNs are
    actually built for: message passing across the REAL network topology,
    learning each station's cascade risk from its neighbours' structure —
    not just its own attributes. Concretely, it predicts, for each
    station: the average cascade_depth of delays originating there,
    using Module 3's own simulation output (graph_simulation_dataset.csv)
    as ground truth.

Why GATv2Conv specifically:
    This is the literal "GATv2" your original SRS names. GATv2Conv also
    supports edge features (edge_dim), which lets it directly use each
    edge's buffer_time_mins, propagation_weight, and edge_type as part of
    the message-passing computation — not just the raw graph shape.

Honest caveat (documented, same convention as every other module here):
    73 nodes, of which only ~60 have any simulated events at all, is a
    genuinely small dataset for a neural network. A held-out test set here
    is maybe a dozen stations — results should be read as a demonstration
    that the architecture is wired correctly and learns *something*
    structurally sensible, not as a production-grade accuracy claim. This
    is explicitly the trade-off already flagged when Random Forest was
    chosen originally: small data favours simpler models, and that
    remains true here. It's included because it's the SRS-specified
    architecture, and it's a genuine, correct use of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

from multiplex_graph import MultiplexGraph

EDGE_TYPES = ["INFRA", "SCHED_DEP", "RS_DEP"]


def build_pyg_data(mg: MultiplexGraph, station_targets: dict[str, float]) -> tuple[Data, list[str], np.ndarray]:
    """Converts the Module 2 MultiplexGraph into a PyTorch Geometric Data
    object: node features from station attributes, edge_index + edge_attr
    from all three edge layers, and a target vector (NaN where a station
    has no simulated events, masked out of the loss)."""
    station_ids = sorted(mg.graph.nodes())
    idx_of = {sid: i for i, sid in enumerate(station_ids)}

    # ---- Node features: platform capacity, degree, criticality ----
    node_features = []
    for sid in station_ids:
        node = mg.graph.nodes[sid]
        in_deg = mg.graph.in_degree(sid)
        out_deg = mg.graph.out_degree(sid)
        criticality = mg.compute_criticality_score(sid)
        node_features.append(
            [
                node.get("platform_count", 0),
                node.get("capacity_trains_per_hour", 0),
                in_deg,
                out_deg,
                criticality,
            ]
        )
    x = torch.tensor(node_features, dtype=torch.float)
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)  # standardize, small dataset needs this for stable training

    # ---- Edges: all three layers, with type/buffer/weight as edge_attr ----
    src, dst, edge_attrs = [], [], []
    for u, v, data in mg.graph.edges(data=True):
        src.append(idx_of[u])
        dst.append(idx_of[v])
        edge_type_onehot = [1.0 if data.get("edge_type") == t else 0.0 for t in EDGE_TYPES]
        edge_attrs.append(
            edge_type_onehot + [data.get("buffer_time_mins", 0.0) / 60.0, data.get("propagation_weight", 0.0)]
        )
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    # ---- Targets + mask ----
    y = np.array([station_targets.get(sid, np.nan) for sid in station_ids], dtype=np.float32)
    mask = ~np.isnan(y)
    y = np.nan_to_num(y, nan=0.0)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(y, dtype=torch.float))
    return data, station_ids, mask


class CascadeGATv2(nn.Module):
    """2-layer GATv2 network: message-passes over the real multiplex graph,
    using edge features, to predict one risk score per station."""

    def __init__(self, in_channels: int, edge_dim: int, hidden_channels: int = 16, heads: int = 2):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, edge_dim=edge_dim, dropout=0.2)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, edge_dim=edge_dim, dropout=0.2)
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr):
        h = self.conv1(x, edge_index, edge_attr).relu()
        h = self.conv2(h, edge_index, edge_attr).relu()
        return self.head(h).squeeze(-1)


def train_gnn(data: Data, mask: np.ndarray, epochs: int = 300, lr: float = 0.01, seed: int = 42):
    """Full-batch training (the whole 73-node graph every step -- there's
    no need to mini-batch something this small). Splits the masked
    (station-has-data) nodes 80/20 for train/test."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    masked_idx = np.where(mask)[0]
    rng.shuffle(masked_idx)
    split = int(0.8 * len(masked_idx))
    train_idx = torch.tensor(masked_idx[:split], dtype=torch.long)
    test_idx = torch.tensor(masked_idx[split:], dtype=torch.long)

    model = CascadeGATv2(in_channels=data.x.shape[1], edge_dim=data.edge_attr.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index, data.edge_attr)
        loss = loss_fn(pred[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index, data.edge_attr)
        train_mae = (pred[train_idx] - data.y[train_idx]).abs().mean().item()
        test_mae = (pred[test_idx] - data.y[test_idx]).abs().mean().item()

        y_test = data.y[test_idx].numpy()
        pred_test = pred[test_idx].numpy()
        ss_res = np.sum((y_test - pred_test) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        test_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return model, {
        "train_mae": train_mae,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }
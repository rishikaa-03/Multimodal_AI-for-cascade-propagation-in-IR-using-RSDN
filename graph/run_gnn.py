"""
Run this file to train and evaluate the GATv2 GNN:
    cd RailwayCascadeAI/graph
    python3 run_gnn.py

Requires torch and torch_geometric (pip install torch torch_geometric --
this is a large download, ~1-2GB; Google Colab / Kaggle Notebooks have
both preinstalled, which is why they were chosen as this project's free
GPU option).
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiplex_graph import MultiplexGraph  # noqa: E402
from gnn_cascade_model import build_pyg_data, train_gnn  # noqa: E402


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(here, "multiplex_graph.pkl")
    graph_sim_path = os.path.join(here, "graph_simulation_dataset.csv")

    print("Loading graph and Module 3's simulation output...")
    mg = MultiplexGraph.load(graph_path)
    graph_sim = pd.read_csv(graph_sim_path)

    # Target: mean cascade_depth of events originating at each station
    station_targets = graph_sim.groupby("station_id")["cascade_depth"].mean().to_dict()
    print(f"  {len(station_targets)} of {mg.graph.number_of_nodes()} stations have simulated events")

    data, station_ids, mask = build_pyg_data(mg, station_targets)
    print(f"  Graph: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} directed edges, "
          f"{data.x.shape[1]} node features, {data.edge_attr.shape[1]} edge features")

    print("\nTraining GATv2 (2 layers, full-batch, 300 epochs)...")
    model, metrics = train_gnn(data, mask)

    print(f"\nResults:")
    print(f"  Train MAE: {metrics['train_mae']:.3f}  (n={metrics['n_train']})")
    print(f"  Test MAE:  {metrics['test_mae']:.3f}  (n={metrics['n_test']})")
    print(f"  Test R2:   {metrics['test_r2']:.3f}")
    print(f"\n  Note: test set is only {metrics['n_test']} stations -- read this as a")
    print(f"  correctness check on the architecture, not a robust accuracy claim.")

    import torch
    torch.save(model.state_dict(), os.path.join(here, "gnn_model.pt"))
    print(f"\nModel saved to: {os.path.join(here, 'gnn_model.pt')}")


if __name__ == "__main__":
    main()
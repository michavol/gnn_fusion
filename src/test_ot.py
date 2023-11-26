import os
import random
import torch
import dgl
import hydra
from omegaconf import DictConfig, OmegaConf
import ot
# from ot_fusion.ot.graph_cost import GraphCost
# from ot_fusion.ot.ground_cost import GroundCost
from ot_fusion.ot.cost_matrix import CostMatrix
from ot_fusion.ot.transport_map import TransportMap
import ott
from ott import utils
from ott.math import utils as mu
from ott.geometry import geometry, pointcloud
from ott.geometry.graph import Graph
from ott.solvers.linear import sinkhorn, sinkhorn_lr
from ott.problems.linear import linear_problem
from ott.problems.quadratic import quadratic_problem
from ott.solvers.quadratic import gromov_wasserstein

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import numpy as np
import tqdm
import torch
import jax
import jax.numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt
from dgl import AddReverse


def create_sample_data(n_data_graphs=5, X_size=5, Y_size=5):
    data_graphs = []
    X = [] # Activations for source
    Y = [] # Activations for target
    a = torch.ones(X_size)/X_size
    b = torch.ones(Y_size)/Y_size

    for i in range(n_data_graphs):
        num_nodes = random.randint(5,10)
        edge_indices_1 = [j for j in range(num_nodes)]
        edge_indices_2 = [random.randint(0,num_nodes-1) for j in range(num_nodes)]
        graph = dgl.graph((edge_indices_1, edge_indices_2), num_nodes=num_nodes)
        data_graphs.append(graph)

    for i in range(X_size):
        x = []
        for graph in data_graphs:
            graph_X = graph.clone()
            num_nodes = graph_X.num_nodes()
            #graph_X.ndata["Feature"] = torch.randn(num_nodes,1)
            graph_X.ndata["Feature"] = torch.ones(num_nodes,1)*i
            #graph_X.ndata["Feature"] = torch.linspace(0,1,num_nodes)
            x.append(graph_X)
        X.append(x)

    for i in range(Y_size):
        y = []
        for graph in data_graphs:
            graph_Y = graph.clone()
            num_nodes = graph_Y.num_nodes()
            #graph_Y.ndata["Feature"] = torch.randn(num_nodes,1)
            graph_Y.ndata["Feature"] = torch.ones(num_nodes,1)*i*1.2
            #graph_Y.ndata["Feature"] = torch.linspace(0,1,num_nodes)
            y.append(graph_Y)
        Y.append(y)

    return X, Y, a, b


@hydra.main(config_path="conf_ot", config_name="config_costs", version_base=None)
def main(cfg: DictConfig):
    
    # Create sample data
    X, Y, a, b = create_sample_data(n_data_graphs=1, X_size=1, Y_size=1)

    # Create cost function according to config
    # graph_cost_fn = GraphCosts(cfg).get_graph_cost_fn()
    # ground_cost_fn = GroundCost(cfg).get_cost_fn()
    # cost_matrix = CostMatrix(cfg).get_cost_matrix(X, Y)
    transport_map = TransportMap(cfg).get_current_transport_map(X, Y, a, b)

if __name__ == '__main__':
    main()
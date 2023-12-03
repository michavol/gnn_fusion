from ot_fusion.ot.optimal_transport import OptimalTransport

import os
import random
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf

import dgl
os.environ["DGLBACKEND"] = "pytorch"
import torch

def create_sample_data(n_data_graphs=5, X_size=5, Y_size=5):
    data_graphs = []
    X = [] # Activations for source
    Y = [] # Activations for target
    a = torch.ones(X_size)/X_size
    b = torch.ones(Y_size)/Y_size

    for i in range(n_data_graphs):
        num_nodes = random.randint(8,10)
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
            graph_Y.ndata["Feature"] = torch.ones(num_nodes,1)*i*2
            #graph_Y.ndata["Feature"] = torch.linspace(0,1,num_nodes)
            y.append(graph_Y)
        Y.append(y)

    return X, Y, a, b


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    
    # Demonstrate GCN layer OT
    X, Y, a, b = create_sample_data(n_data_graphs=2, X_size=1, Y_size=1)
    transport_map = OptimalTransport(cfg.conf_ot).get_current_transport_map(X, Y, a, b, layer_type="gcn")
    #plt.imshow(transport_map)
    #plt.show()

    # Demonstrate MLP layer OT
    # X, Y, a, b = torch.rand(5,10), torch.rand(5,10), a, b
    # transport_map = OptimalTransport(cfg.conf_ot).get_current_transport_map(X, Y, a, b, layer_type="mlp")

if __name__ == '__main__':
    main()
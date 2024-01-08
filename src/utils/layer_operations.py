from enum import Enum

import torch

LayerType = Enum('LayerType', ['embedding', 'mlp', 'gcn', 'bn', 'dropout'])


def get_layer_type(layer_name: str):
    """Maps layer name to its type."""
    layer_map = {
        'embedding': LayerType.embedding,
        'MLP': LayerType.mlp,
        'mlp': LayerType.mlp,
        'conv': LayerType.gcn,
        'batchnorm': LayerType.bn,
        'dropout': LayerType.dropout,
    }
    for k, v in layer_map.items():
        if k in layer_name:
            return v


# (Weighted) Averaging of weight matrices of a layer
def get_avg_parameters(parameters: list, weights=None):
    """Averages the contents of parameters list."""
    if weights is not None:
        weighted_par_group = [par * weights[i] for i, par in enumerate(parameters)]
        avg_par = torch.sum(torch.stack(weighted_par_group), dim=0)
    else:
        avg_par = torch.mean(torch.stack(parameters), dim=0)
    return avg_par

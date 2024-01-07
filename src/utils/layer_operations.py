from enum import Enum

import torch
import torch.nn.functional as F

LayerType = Enum('LayerType', ['embedding', 'mlp', 'gcn', 'bn', 'dropout'])

def get_layer_type(layer_name: str):
    layer_map = {
        'embedding': LayerType.embedding,
        'MLP': LayerType.mlp,
        'mlp': LayerType.mlp,
        'conv': LayerType.gcn,
        'batchnorm': LayerType.bn,
        'dropout': LayerType.dropout,
    }
    print('layer name', layer_name)
    for k, v in layer_map.items():
        if k in layer_name:
            return v


# (Weighted) Averaging of weight matrices of a layer
def get_avg_parameters(parameters, weights=None):
    #avg_pars = []
    #print(parameters.shape)
    if weights is not None:
        weighted_par_group = [par * weights[i] for i, par in enumerate(parameters)]
        avg_par = torch.sum(torch.stack(weighted_par_group), dim=0)
    else:
        # print("shape of stacked params is ", torch.stack(par_group).shape) # (2, 400, 784)
        avg_par = torch.mean(torch.stack(parameters), dim=0)
    #print(type(avg_par))
    #avg_pars.append(avg_par)
    return avg_par
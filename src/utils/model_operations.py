import argparse
import sys

from omegaconf import DictConfig
import torch
import torch.nn as nn
import yaml
from typing import List

PATH_TO_BENCHMARK = "./gnn_benchmarking/"
sys.path.append(PATH_TO_BENCHMARK)

from nets.molecules_graph_regression.load_net import gnn_model # import all GNNS

def get_models(args: DictConfig) -> List[nn.Module]:
    # TODO: Make it more general when we have unified config plan
    models = []
    model = gnn_model(args.model_name, net_params=args.model_params1)
    model.load_state_dict(
        torch.load(args.model_path1))
    model.eval()
    models.append(model)
    model = gnn_model(args.model_name, net_params=args.model_params2)
    model.load_state_dict(
        torch.load(args.model_path2))
    model.eval()
    models.append(model)
    return models


## Alternative to current config structure
def get_models_from_paths(args: List) -> List[nn.Module]:
    models = []

    for path in args:
        with open(path + '/config.yaml', 'r') as file:
            loaded_data = yaml.safe_load(file)

        model_name = loaded_data['Model']
        net_params = loaded_data['net_params']

        model = gnn_model(model_name, net_params)
        model.load_state_dict(
            torch.load(path + '/final.pkl'))
        model.eval()
        models.append(model)

    return models
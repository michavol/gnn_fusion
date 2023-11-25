import argparse
import sys

from omegaconf import DictConfig
import torch
import torch.nn as nn
from typing import List

PATH_TO_BENCHMARK = "./gnn_benchmarking/"
sys.path.append(PATH_TO_BENCHMARK)

from gnn_benchmarking.nets.molecules_graph_regression.load_net import gnn_model # import all GNNS

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

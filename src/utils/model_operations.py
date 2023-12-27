import argparse
import sys

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
import torch.nn as nn
import yaml
from typing import List

PATH_TO_BENCHMARK = "./gnn_benchmarking/"
sys.path.append(PATH_TO_BENCHMARK)

from nets.molecules_graph_regression.load_net import gnn_model # import all GNNS

def get_models(args: DictConfig,path) -> List[nn.Module]:
    models = []
    for model, params in args.items():
        trained_model = gnn_model(params["Model"], net_params=params["net_params"])
        trained_model.load_state_dict(torch.load(Path(path + params["model_path"])))
        trained_model.eval()
        models.append(trained_model)
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

def get_models_from_raw_config(args):
    """Gets models without the need to extract model configs."""
    models_conf = OmegaConf.to_container(
        args.models.individual_models, resolve=True, throw_on_missing=True
    )
    return get_models(models_conf, args.individual_models_dir)
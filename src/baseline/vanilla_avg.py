import argparse
import sys
import copy
import torch
import torch.nn as nn
from typing import List

PATH_TO_ROOT = "../"
sys.path.append(PATH_TO_ROOT)

PATH_TO_BENCHMARK = PATH_TO_ROOT + "/gnn_benchmarking/"
sys.path.append(PATH_TO_BENCHMARK)

from utils.layer_operations import get_avg_parameters


def vanilla_avg(args, models: List[nn.Module]):
    """Performs vanilla averaging of model weights and creates a model."""
    # Copy of first model as template 
    avg_model = copy.deepcopy(models[0])
    avg_model_state_dict = avg_model.state_dict()

    for layer_name in avg_model_state_dict.keys():
        if layer_name.endswith('num_batches_tracked'):
            continue
        with torch.no_grad():
            # Get parameters of all models
            parameters = [model.state_dict()[layer_name].to(args.device) for model in models]
            avg_parameters = get_avg_parameters(parameters)
            # Set parameters of new model - param[1] accesses the parameters
            avg_model_state_dict[layer_name] = avg_parameters
    avg_model.load_state_dict(avg_model_state_dict)

    return avg_model
    # Save vanilla averaged model
    # torch.save(avg_model.state_dict(), fileName)


def compose_models(args: argparse.Namespace, models: List):
    naive_model = vanilla_avg(args, models)
    return naive_model

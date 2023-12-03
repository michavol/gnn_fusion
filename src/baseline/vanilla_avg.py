###
# 1. Load models
# 2. Iterate layerwise and average weights
# 3. Store averaged model
###
import argparse
import os
import sys
import copy
import torch
import torch.nn as nn
from typing import List
from torch.utils.data import DataLoader

PATH_TO_ROOT = "../"
sys.path.append(PATH_TO_ROOT)

PATH_TO_BENCHMARK = PATH_TO_ROOT + "/gnn_benchmarking/"
sys.path.append(PATH_TO_BENCHMARK)

from utils.model_operations import get_models_from_paths
from utils.layer_operations import get_avg_parameters

def average_weights():
    return

def vanilla_avg(models: List[nn.Module],fileName: str):
    # Copy of first model as template 
    avg_model = copy.deepcopy(models[0])

    # TO DO: Add assert here that they have the same architecture
    n = len(list(avg_model.named_parameters()))
    
    for i, param in enumerate (avg_model.named_parameters()):
        with torch.no_grad():
        # Get parameters of all models
            parameters = [list(model.parameters())[i] for model in models]

            # TO DO: add assert for same shapes here
            #assert x == "goodbye", "x should be 'hello'"            
            
            avg_parameters = get_avg_parameters(parameters)
            # Set parameters of new model - param[1] accesses the parameters
            param[1].copy_(avg_parameters)

    # Save vanilla averaged model
    torch.save(avg_model.state_dict(), fileName + ".pkl")

def compose_models(args: argparse.Namespace, models: List, test_loader: DataLoader) -> float:
    pass

def main():
    assert len(sys.argv) == 2, "Need to specify averaged model name"

    name = sys.argv[1]
    print(name)

    folderPath = os.path.abspath(PATH_TO_ROOT) + "/models/"
    modelsFolder = folderPath + "individual_models/"
    list = os.listdir(modelsFolder)
    files = [os.path.join(modelsFolder, file) for file in list]
    models = get_models_from_paths(files)

    save_folder = folderPath + "fused_models/vanilla_avg/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    vanilla_avg(models, save_folder + name)


if __name__ == '__main__':
    main()
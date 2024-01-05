import argparse
import sys
import copy

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from typing import List
from tqdm import tqdm

PATH_TO_BENCHMARK = "./gnn_benchmarking/"
sys.path.append(PATH_TO_BENCHMARK)

from nets.molecules_graph_regression.load_net import gnn_model # import all GNNS
# from nets.superpixels_graph_classification.load_net import gnn_model # import all GNNS #uncomment for MNIST
from train.train_molecules_graph_regression import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network

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

def finetune_models(cfg, model,train_loader, val_loader, epochs,save_step):
    # List of finetuned models at different epochs
    models = []
    epoch_save_step = save_step

    model.train()
    model = model.to(cfg['device'])

    best_current_model = copy.deepcopy(model)
    _, best_val_mae = evaluate_network(best_current_model, cfg['device'], val_loader, 0)

    optimizer = optim.Adam(model.parameters(), lr=cfg['params']['init_lr'], weight_decay=cfg['params']['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=cfg['params']['lr_reduce_factor'],
                                                     patience=cfg['params']['lr_schedule_patience'],
                                                     verbose=True)
    with tqdm(range(epochs)) as t:

        for epoch in range(epochs):

            epoch_train_loss, epoch_train_mae, optimizer = train_epoch(model, optimizer, cfg['device'], train_loader, epoch)
                
            epoch_val_loss, epoch_val_mae = evaluate_network(model, cfg['device'], val_loader, epoch)

            scheduler.step(epoch_val_loss)

            print('Epoch: {:03d}, Test Loss: {:.5f}, Test MAE: {:.5f}'.format(epoch, epoch_train_loss,epoch_train_mae))
            print('Validation Loss: {:.5f}, Validation MAE: {:.5f}'.format(epoch_val_loss, epoch_val_mae))

            if epoch_val_mae < best_val_mae:
                best_val_mae = epoch_val_mae
                best_current_model = copy.deepcopy(model)

            # Save current best model
            if epoch % epoch_save_step == 0:
                models.append(best_current_model)

    models.append(model)

    return models
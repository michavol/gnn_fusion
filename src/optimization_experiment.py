import os
import wandb
import torch
import hydra
import yaml
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from utils import params, model_operations, data_operations, activation_operations
from evaluation.evaluate_ZINC import evalModel


# Load model with respect to sweep params
# Evalue and log metrics

def get_args(cfg: DictConfig) -> DictConfig:
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    # print(OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(config_path="conf", config_name="config_optimization_experiment")
def main(cfg: DictConfig):
    args = get_args(cfg)

    wandb.config = OmegaConf.to_container(
        args, resolve=True, throw_on_missing=True
    )
    wandb.init(entity=args.wandb_entity, project=args.wandb_project)

    print(os.getcwd())

    # Get all files in dir
    file_list = os.listdir(args.config_dir)

    # Load test data from first yaml in file_list
    with open(args.config_dir + file_list[0], 'r') as file:
        loaded_data = yaml.safe_load(file)

    loaded_data["device"] = args.device
    train_loader, test_loader = data_operations.get_train_test_loaders(loaded_data, args.dataset_dir)

    for i, file_name in enumerate(file_list):
        with open(args.config_dir + file_name, 'r') as file:
            loaded_data = yaml.safe_load(file)

        # Evaluate model
        test_MAE = evalModel(loaded_data, args.models_dir, args.device, test_loader)

        # Log metrics
        log_key = "MAE_" + file_name[:-5]
        wandb.log({log_key: test_MAE})
        wandb.log({"MAE": test_MAE}, step=i)

        print(test_MAE)
        wandb.finish()


if __name__ == '__main__':
    main()

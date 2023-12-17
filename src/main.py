import os
import wandb
import torch
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from baseline import vanilla_avg, ensemble
from ot_fusion import wasserstein_ensemble
from utils import params, model_operations, data_operations, activation_operations

def get_args(cfg: DictConfig) -> DictConfig:
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    #print(OmegaConf.to_yaml(cfg))
    return cfg

@hydra.main(config_path="conf", config_name="config_fusion")
def main(cfg: DictConfig):
    args = get_args(cfg)

    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project)
    #wandb.log({"loss": loss})

    if args.deterministic:
        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # TODO: set numpy seed here as well ?


    models_conf = OmegaConf.to_container(
        cfg.models.individual_models, resolve=True, throw_on_missing=True
    )
    models = model_operations.get_models(models_conf, args.individual_models_dir)
    train_loader, test_loader = data_operations.get_train_test_loaders(models_conf, args.dataset_dir)


    # run geometric aka wasserstein ensembling
    print("------- Geometric Ensembling -------")
    ot_fusion_model = wasserstein_ensemble.compose_models(args, models, train_loader, test_loader)
    torch.save(ot_fusion_model.state_dict(), args.otFused_model_save_path)

    print("------- Naive ensembling of weights -------")
    naive_model = vanilla_avg.compose_models(args, models)
    torch.save(naive_model.state_dict(), args.vanilla_model_save_path)


if __name__ == '__main__':
    main()

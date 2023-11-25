import os

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from baseline import vanilla_avg, ensemble
from ot_fusion import wasserstein_ensemble
from utils import params, model_operations, data_operations, activation_operations

def get_args(cfg: DictConfig) -> DictConfig:
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    print(OmegaConf.to_yaml(cfg))
    return cfg

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)

    if args.deterministic:
        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # TODO: set numpy seed here as well ?


    models = model_operations.get_models(args)
    train_loader, test_loader = data_operations.get_train_test_loaders(args)
    activations = activation_operations.compute_activations(args, models[0], train_loader)
    # print(activations)
    print(type(activations))
    # print(activations[0]['layers.0.conv'])
    print(activations[0]['layers.0.conv'][0].shape)
    print(activations[0]['layers.0.conv'][1].shape)



    # run geometric aka wasserstein ensembling
    print("------- Geometric Ensembling -------")
    ot_fusion_model = wasserstein_ensemble.compose_models(args, models, train_loader, test_loader, activations)

    print("------- Prediction based ensembling -------")
    ensemble_acc = ensemble.compose_models(args, models, test_loader)

    print("------- Naive ensembling of weights -------")
    naive_model = vanilla_avg.compose_models(args, models, test_loader)


if __name__ == '__main__':
    main()

import os
import wandb
import torch
import hydra
import yaml
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

    fused_model_config = next(iter(models_conf.values()))

    # run geometric aka wasserstein ensembling
    print("------- Geometric Ensembling -------")
    ot_fusion_model = wasserstein_ensemble.compose_models(args, models, train_loader, test_loader)

    # Save corresponding config and model
    ot_fusion_model_config = fused_model_config.copy()
    ot_fusion_model_file_name = "ot_fusion_" + ot_fusion_model_config["Dataset"] + "_" + args.ot.optimal_transport["solver_type"] + "_" + str(int(args.ot.optimal_transport["epsilon"] * 1000)) + "_" + args.ot.costs.graph_cost["graph_cost_type"]
    ot_fusion_model_config["model_path"] = args.fused_models_dir + ot_fusion_model_file_name + ".pkl"
    yaml_data = yaml.dump(ot_fusion_model_config, default_flow_style=False)

    print(os.getcwd())
    with open(args.models_conf_dir + ot_fusion_model_file_name + '.yaml', 'w') as f:
        f.write(yaml_data)

    print(ot_fusion_model_file_name)
    torch.save(ot_fusion_model.state_dict(), ot_fusion_model_config["model_path"])

    print("------- Naive ensembling of weights -------")
    naive_model = vanilla_avg.compose_models(args, models)

    # Save corresponding config and model
    naive_model_config = fused_model_config.copy()
    naive_model_file_name = "vanilla_fusion_" + naive_model_config["Dataset"] + "_" + args.ot.optimal_transport["solver_type"] + "_" + str(int(args.ot.optimal_transport["epsilon"] * 1000)) + "_" + args.ot.costs.graph_cost["graph_cost_type"]
    naive_model_config["model_path"] = args.fused_models_dir + naive_model_file_name + ".pkl"
    yaml_data = yaml.dump(naive_model_config, default_flow_style=False)

    with open(args.models_conf_dir + naive_model_file_name + '.yaml', 'w') as f:
        f.write(yaml_data)

    torch.save(naive_model.state_dict(), naive_model_config["model_path"])


if __name__ == '__main__':
    main()

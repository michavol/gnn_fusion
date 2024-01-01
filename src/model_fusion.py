import os
import wandb
import torch
import hydra
import yaml
from omegaconf import DictConfig, OmegaConf
import numpy as np

from baseline import vanilla_avg
from ot_fusion import wasserstein_ensemble
from utils import model_operations, data_operations


# TO DO: Change file name
def get_args(cfg: DictConfig) -> DictConfig:
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    # print(OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(config_path="conf", config_name="config_fusion")
def main(cfg: DictConfig):
    args = get_args(cfg)

    if args.wandb:
        wandb.config = OmegaConf.to_container(
            args, resolve=True, throw_on_missing=True
        )
        wandb.init(entity=args.wandb_entity, project=args.wandb_project)
        # wandb.log({"loss": loss})

    if args.deterministic:
        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed=0)

    models_conf = OmegaConf.to_container(
        args.models.individual_models, resolve=True, throw_on_missing=True
    )

    models = model_operations.get_models(models_conf, args.individual_models_dir)
    model_0_config = next(iter(models_conf.values()))
    model_0_config["device"] = args.device
    train_loader, test_loader = data_operations.get_train_test_loaders(model_0_config, args.dataset_dir, args)

    print("------- Naive ensembling of weights -------")
    naive_model_config = model_0_config.copy()
    naive_model_file_name = "vanilla_fusion_" + naive_model_config["Dataset"]
    naive_model_config["model_path"] = args.experiment_models_dir + naive_model_file_name + ".pkl"

    if (not os.path.exists(naive_model_config["model_path"])):
        naive_model = vanilla_avg.compose_models(args, models)

        naive_model_config["model_path"] = args.experiment_models_dir + naive_model_file_name + ".pkl"
        yaml_data = yaml.dump(naive_model_config, default_flow_style=False)

        with open(args.models_conf_dir + naive_model_file_name + '.yaml', 'w') as f:
            f.write(yaml_data)

        torch.save(naive_model.state_dict(), args.models_dir + naive_model_config["model_path"])

    # # run geometric aka wasserstein ensembling
    print("------- Geometric Ensembling -------")
    ot_fusion_model, aligned_ot_fusion_models = wasserstein_ensemble.compose_models(args, models, train_loader)

    # Save corresponding config and model
    ot_fusion_model_config = model_0_config.copy()
    ot_fusion_model_file_name = "ot_fusion_" + ot_fusion_model_config["Dataset"] + "_" + args.optimal_transport[
        "solver_type"] + "_" + str(int(args.optimal_transport["relative_epsilon"] * 1000)) + "_" + args.graph_cost[
                                    "graph_cost_type"]
    ot_fusion_model_config["model_path"] = args.experiment_models_dir + ot_fusion_model_file_name + ".pkl"

    yaml_data = yaml.dump(ot_fusion_model_config, default_flow_style=False)

    # print(os.getcwd())
    with open(args.models_conf_dir + ot_fusion_model_file_name + '.yaml', 'w') as f:
        f.write(yaml_data)
    torch.save(ot_fusion_model.state_dict(), args.models_dir + ot_fusion_model_config["model_path"])

    print("------- Save individual aligned models to experiment folders -------")
    for i, model in enumerate(models_conf.keys()):
        # Skipping the first models since it's the target
        if i == 0:
            continue

        ot_aligned_model_config = model_0_config.copy()
        ot_aligned_model_file_name = model + "_aligned_" + args.optimal_transport[
            "solver_type"] + "_" + str(int(args.optimal_transport["relative_epsilon"] * 1000)) + "_" + args.graph_cost[
                                         "graph_cost_type"]
        ot_aligned_model_config["model_path"] = args.experiment_models_dir + ot_aligned_model_file_name + ".pkl"

        ot_aligned_model_config["model_path"] = args.experiment_models_dir + ot_aligned_model_file_name + ".pkl"

        if (not os.path.exists(ot_aligned_model_config["model_path"])):
            yaml_data = yaml.dump(ot_aligned_model_config, default_flow_style=False)
            with open(args.models_conf_dir + ot_aligned_model_file_name + '.yaml', 'w') as f:
                f.write(yaml_data)

            torch.save(aligned_ot_fusion_models[i - 1].state_dict(),
                       args.models_dir + ot_aligned_model_config["model_path"])

    if args.wandb:
        wandb.finish()

    print("------- Save individual models to experiment folders -------")
    for i, (model, params) in enumerate(models_conf.items()):
        model_file_name = model
        params["model_path"] = args.experiment_models_dir + model_file_name + ".pkl"

        if (not os.path.exists(params["model_path"])):
            yaml_data = yaml.dump(params, default_flow_style=False)
            with open(args.models_conf_dir + model_file_name + '.yaml', 'w') as f:
                f.write(yaml_data)

            torch.save(models[i].state_dict(), args.models_dir + params["model_path"])

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()

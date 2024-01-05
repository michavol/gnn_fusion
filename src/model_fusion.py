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
    # cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
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
        wandb.init(entity=args.wandb_entity, project=args.wandb_project, settings=wandb.Settings(start_method="thread"))
        # wandb.log({"loss": loss})

    if args.deterministic:
        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed=0)
        torch.manual_seed(0)

    models_conf = OmegaConf.to_container(
        args.models.individual_models, resolve=True, throw_on_missing=True
    )

    models = model_operations.get_models(models_conf, args.individual_models_dir)

    model_0 = next(iter(models_conf.values()))
    model_0["device"] = args.device
    args.dataset = model_0["Dataset"]

    # Fused gw can achieve same performance with less samples
    if args.graph_cost["graph_cost_type"] == "fused_gw":
        args.batch_size = int(args.batch_size / args.gw_batch_scaling)
        assert args.batch_size > 0, "Batch size must be greater than 0"

    # for finetuning get the 'original' train and val loader
    if args.fine_tune:
        train_fintune_loader, val_finetune_loader = data_operations.get_finetune_test_val_loaders(model_0,
                                                                                                  args.dataset_dir)
    first_model_config = next(iter(models_conf.values()))

    # # run geometric aka wasserstein ensembling
    print("------- Geometric Ensembling -------")
    for trial in range(args.ot_fusions_per_model):
        # train_loader is the 'original' val loader
        train_loader, _ = data_operations.get_train_test_loaders(model_0, args.dataset_dir, args, shuffle_val=True)


        ot_fusion_model, aligned_ot_fusion_models = wasserstein_ensemble.compose_models(args, models, train_loader)

        # Save corresponding config and model
        ot_fusion_model_config = first_model_config.copy()
        model_file_name_base = ot_fusion_model_config["Dataset"] + "_" + args.geom_ensemble_type + "_" + str(args.fast_l2) + "_" + args.optimal_transport[
            "solver_type"] + "_" + str(int(args.optimal_transport["epsilon"] * 100000)) + "_" + args.graph_cost[
                                   "file_name_suffix"] + "_" + "tau" + "_" + str(
            args.optimal_transport["tau"] * 100) + "_" + str(args.batch_size * args.num_batches) + "samples_" + str(
            trial) + "trial"
        ot_fusion_model_file_name = ot_fusion_model_file_name = "ot_fusion_" + model_file_name_base

        ot_fusion_model_config["model_path"] = args.experiment_models_dir + ot_fusion_model_file_name + ".pkl"

        yaml_data = yaml.dump(ot_fusion_model_config, default_flow_style=False)

        # print(os.getcwd())
        with open(args.models_conf_dir + ot_fusion_model_file_name + '.yaml', 'w') as f:
            f.write(yaml_data)
        torch.save(ot_fusion_model.state_dict(), args.models_dir + ot_fusion_model_config["model_path"])

        if args.fine_tune:
            print("------- Finetuning Geometric Ensemble -------")
            ot_fusion_finetuned_models = model_operations.finetune_models(ot_fusion_model_config, ot_fusion_model,
                                                                          train_fintune_loader, val_finetune_loader,
                                                                          args.fine_tune_epochs,
                                                                          args.fine_tune_save_step)  # list of finetuned models
            ot_finetuned_models_configs = first_model_config.copy()  # list of finetuned models configs
            for i, model in enumerate(ot_fusion_finetuned_models):
                ot_finetuned_model_file_name = ot_fusion_model_file_name + "_" + str(
                    i * args.fine_tune_save_step) + "finetuned"
                ot_finetuned_models_configs[
                    "model_path"] = args.experiment_models_dir + ot_finetuned_model_file_name + ".pkl"
                ot_finetuned_models_configs["fine_tune_step"] = i * args.fine_tune_save_step

                yaml_data = yaml.dump(ot_finetuned_models_configs, default_flow_style=False)

                with open(args.models_conf_dir + ot_finetuned_model_file_name + '.yaml', 'w') as f:
                    f.write(yaml_data)

                torch.save(model.state_dict(), args.models_dir + ot_finetuned_models_configs["model_path"])

        if args.save_aligned:
            print("------- Save individual aligned models to experiment folders -------")
            for i, model in enumerate(models_conf.keys()):
                # Skipping the first models since it's the target
                if i == 0:
                    continue

                ot_aligned_model_config = first_model_config.copy()
                ot_aligned_model_file_name = model + "_aligned_" + model_file_name_base
                ot_aligned_model_config["model_path"] = args.experiment_models_dir + ot_aligned_model_file_name + ".pkl"

                if (not os.path.exists(ot_aligned_model_config["model_path"])):
                    yaml_data = yaml.dump(ot_aligned_model_config, default_flow_style=False)
                    with open(args.models_conf_dir + ot_aligned_model_file_name + '.yaml', 'w') as f:
                        f.write(yaml_data)

                    torch.save(aligned_ot_fusion_models[i - 1].state_dict(),
                               args.models_dir + ot_aligned_model_config["model_path"])

    print("------- Naive ensembling of weights -------")
    torch.manual_seed(0)

    naive_model_config = first_model_config.copy()
    naive_model_file_name = "vanilla_fusion_" + naive_model_config["Dataset"]
    naive_model_config["model_path"] = args.experiment_models_dir + naive_model_file_name + ".pkl"

    if (not os.path.exists(naive_model_config["model_path"])):
        naive_model = vanilla_avg.compose_models(args, models)

        naive_model_config["model_path"] = args.experiment_models_dir + naive_model_file_name + ".pkl"
        yaml_data = yaml.dump(naive_model_config, default_flow_style=False)

        with open(args.models_conf_dir + naive_model_file_name + '.yaml', 'w') as f:
            f.write(yaml_data)

        torch.save(naive_model.state_dict(), args.models_dir + naive_model_config["model_path"])

    if args.fine_tune:
        print("------- Finetuning Naive Ensemble -------")
        naive_finetuned_models = model_operations.finetune_models(naive_model_config, naive_model, train_fintune_loader,
                                                                  val_finetune_loader, args.fine_tune_epochs,
                                                                  args.fine_tune_save_step)  # list of finetuned models
        naive_finetuned_models_configs = first_model_config.copy()  # list of finetuned models configs
        for i, model in enumerate(naive_finetuned_models):
            naive_finetuned_model_file_name = "vanilla_fusion_" + naive_model_config["Dataset"] + "_" + str(
                i * args.fine_tune_save_step) + "finetuned"
            naive_finetuned_models_configs[
                "model_path"] = args.experiment_models_dir + naive_finetuned_model_file_name + ".pkl"
            naive_finetuned_models_configs["fine_tune_step"] = i * args.fine_tune_save_step

            yaml_data = yaml.dump(naive_finetuned_models_configs, default_flow_style=False)

            with open(args.models_conf_dir + naive_finetuned_model_file_name + '.yaml', 'w') as f:
                f.write(yaml_data)

            torch.save(model.state_dict(), args.models_dir + naive_finetuned_models_configs["model_path"])

    print("------- Save indiviudal models to experiment folders -------")
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

# TODO: Why no 4s in training doesn't degrade performance
# TODO: Why is it slower for smaller batches
# TODO: Why are activations changing?

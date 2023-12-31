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
    #cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
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
    print(models_conf)
    models = model_operations.get_models(models_conf, args.individual_models_dir)
    model_0 = next(iter(models_conf.values()))
    model_0["device"] = args.device

    # This is not solvable in hydra - change in code to have multiple bacthes fused_gw cost
    if args.graph_cost == "fused_gw":
        args.num_batches = args.num_batches_gw
        args.batch_size = args.batch_size / args.num_batches

    # train_loader is the 'original' val loader
    train_loader, test_loader = data_operations.get_train_test_loaders(model_0, args.dataset_dir, args)

    # for finetuning get the 'original' train and val loader
    if args.fine_tune:
        train_fintune_loader, val_finetune_loader = data_operations.get_finetune_test_val_loaders(model_0, args.dataset_dir)

    first_model_config = next(iter(models_conf.values()))

    # # run geometric aka wasserstein ensembling
    print("------- Geometric Ensembling -------")
    ot_fusion_model = wasserstein_ensemble.compose_models(args, models, train_loader, test_loader)

    # Save corresponding config and model
    ot_fusion_model_config = first_model_config.copy()
    ot_fusion_model_file_name = "ot_fusion_" + ot_fusion_model_config["Dataset"] + "_" + args.optimal_transport[
        "solver_type"] + "_" + str(int(args.optimal_transport["epsilon"] * 1000)) + "_" + args.graph_cost[
                                    "graph_cost_type"] + "_" + str(args.batch_size * args.num_batches) + "samples"
    ot_fusion_model_config["model_path"] = args.experiment_models_dir + ot_fusion_model_file_name + ".pkl"

    yaml_data = yaml.dump(ot_fusion_model_config, default_flow_style=False)

    # print(os.getcwd())
    with open(args.models_conf_dir + ot_fusion_model_file_name + '.yaml', 'w') as f:
        f.write(yaml_data)

    print(ot_fusion_model_file_name)
    torch.save(ot_fusion_model.state_dict(), args.models_dir + ot_fusion_model_config["model_path"])
    
    if args.fine_tune:
        print("------- Finetuning Geometric Ensemble -------")
        ot_fusion_finetuned_models = model_operations.finetune_models(ot_fusion_model_config, ot_fusion_model,train_fintune_loader, val_finetune_loader, args.fine_tune_epochs,args.fine_tune_save_step)# list of finetuned models
        ot_finetuned_models_configs = first_model_config.copy() # list of finetuned models configs
        for i,model in enumerate(ot_fusion_finetuned_models):
            ot_finetuned_model_file_name = "ot_fusion_" + ot_finetuned_models_configs["Dataset"] + "_" + args.optimal_transport[
            "solver_type"] + "_" + str(int(args.optimal_transport["epsilon"] * 1000)) + "_" + args.graph_cost[
                                        "graph_cost_type"] + "_" + str(args.batch_size * args.num_batches) + "samples_" + str(i*args.fine_tune_save_step) + "finetuned"
            ot_finetuned_models_configs["model_path"] = args.experiment_models_dir + ot_finetuned_model_file_name + ".pkl"
            ot_finetuned_models_configs["fine_tune_step"] = i*args.fine_tune_save_step

            yaml_data = yaml.dump(ot_finetuned_models_configs, default_flow_style=False)

            with open(args.models_conf_dir + ot_finetuned_model_file_name + '.yaml', 'w') as f:
                f.write(yaml_data)

            torch.save(model.state_dict(), args.models_dir + ot_finetuned_models_configs["model_path"])


    print("------- Naive ensembling of weights -------")
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
        naive_finetuned_models = model_operations.finetune_models(naive_model_config, naive_model,train_fintune_loader, val_finetune_loader, args.fine_tune_epochs, args.fine_tune_save_step) # list of finetuned models
        naive_finetuned_models_configs = first_model_config.copy() # list of finetuned models configs
        for i,model in enumerate(naive_finetuned_models):
            naive_finetuned_model_file_name = "vanilla_fusion_" + naive_model_config["Dataset"] + "_" +str(i*args.fine_tune_save_step) + "finetuned"
            naive_finetuned_models_configs["model_path"] = args.experiment_models_dir + naive_finetuned_model_file_name + ".pkl"
            naive_finetuned_models_configs["fine_tune_step"] = i*args.fine_tune_save_step

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

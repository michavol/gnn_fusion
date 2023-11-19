import torch

from baseline import vanilla_avg, ensemble
from ot_fusion import wasserstein_ensemble
from utils import params, model_operations, data_operations

def main():
    # TODO: Change print to logging
    print("------- Setting up parameters -------")
    args = params.get_parameters()
    print("The parameters are: \n", args)

    if args.deterministic:
        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # TODO: set numpy seed here as well ?


    models = model_operations.get_models(args)
    activations = model_operations.get_model_activations(args, models)
    test_loader = data_operations.get_test_loader(args)
    train_loader = data_operations.get_train_loader(args)


    # run geometric aka wasserstein ensembling
    print("------- Geometric Ensembling -------")
    ot_fusion_model = wasserstein_ensemble.compose_models(args, models, train_loader, test_loader, activations)

    print("------- Prediction based ensembling -------")
    ensemble_acc = ensemble.compose_models(args, models, test_loader)

    print("------- Naive ensembling of weights -------")
    naive_model = vanilla_avg.compose_models(args, models, test_loader)


if __name__ == '__main__':
    main()

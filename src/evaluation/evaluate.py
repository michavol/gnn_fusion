import torch
import sys

PATH_TO_BENCHMARK = "./gnn_benchmarking/"
sys.path.append(PATH_TO_BENCHMARK)

from nets.molecules_graph_regression.load_net import gnn_model as molecule_gnn_model
from nets.superpixels_graph_classification.load_net import gnn_model as superpixel_gnn_model
import train.train_molecules_graph_regression as train_molecules
import train.train_superpixels_graph_classification as train_superpixels


def evalModel(model_yaml, models_path, device, test_loader):
    if model_yaml["Dataset"] == "ZINC":
        train = train_molecules
        gnn_model = molecule_gnn_model
    elif model_yaml["Dataset"] == "MNIST":
        train = train_superpixels
        gnn_model = superpixel_gnn_model
    else:
        raise NotImplementedError(f"Dataset {model_yaml['Dataset']} not known.")

    model = gnn_model(model_yaml["Model"], model_yaml["net_params"])
    model.load_state_dict(torch.load(models_path + model_yaml["model_path"], map_location=torch.device(device)))
    model.eval()

    _, test_mae = train.evaluate_network_sparse(model, device, test_loader, epoch=0)
    # print("Test MAE: {:.4f}".format(test_mae))

    return model, test_mae


def evalModelRaw(model, device, test_loader, data):
    if data == "ZINC":
        train = train_molecules
    elif data == "MNIST":
        train = train_superpixels
    else:
        raise NotImplementedError(f"Dataset {data} not known.")

    model.eval()

    _, test_mae = train.evaluate_network_sparse(model, device, test_loader, epoch=0)
    # print("Test MAE: {:.4f}".format(test_mae))

    return test_mae

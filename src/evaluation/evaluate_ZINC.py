import torch
import sys

PATH_TO_BENCHMARK = "./gnn_benchmarking/"
sys.path.append(PATH_TO_BENCHMARK)

from nets.molecules_graph_regression.load_net import gnn_model # import all GNNS
from data.data import LoadData # import dataset
import train.train_molecules_graph_regression as train

def evalModel(model_yaml, models_path, device, test_loader):
    model = gnn_model(model_yaml["Model"], model_yaml["net_params"])
    model.load_state_dict(torch.load(models_path + model_yaml["model_path"], map_location=torch.device(device)))
    model.eval()

    _, test_mae = train.evaluate_network_sparse(model, device, test_loader)
    #print("Test MAE: {:.4f}".format(test_mae))
    
    return test_mae
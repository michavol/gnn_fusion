import argparse
import sys

from pathlib import Path
from omegaconf import DictConfig
from torch.utils.data import DataLoader

PATH_TO_BENCHMARK = "./gnn_benchmarking/"
sys.path.append(PATH_TO_BENCHMARK)

from data.data import LoadData


def get_train_test_loaders(model_yaml, path):
    # take dataset name from first model
    dataset = LoadData(model_yaml["Dataset"], custom_data_dir=path)
    # dataset = LoadData(args.dataSet_name, custom_data_dir='/home/weronika/Documents/masters/sem3/deep_learning/gnn_fusion/gnn_benchmarking/data/molecules/')
    # For now, we perform the fusion based on validation dataset and testing on the test dataset.
    valset, testset = dataset.val, dataset.test
    val_loader = DataLoader(valset, batch_size=model_yaml["net_params"]["batch_size"], shuffle=False, drop_last=False,
                            collate_fn=dataset.collate)
    test_loader = DataLoader(testset, batch_size=model_yaml["net_params"]["batch_size"], shuffle=False, drop_last=False,
                             collate_fn=dataset.collate)
    return val_loader, test_loader

import sys
from typing import Optional

from omegaconf import DictConfig
from torch.utils.data import DataLoader
import jax.numpy as jnp

PATH_TO_BENCHMARK = "./gnn_benchmarking/"
sys.path.append(PATH_TO_BENCHMARK)

from data.data import LoadData


def get_train_test_loaders(model_yaml, path, fusion_cfg: Optional[DictConfig] = None, shuffle_val=False):
    """Gets dataloader for OT-fusion."""
    # take dataset name from first model
    dataset = LoadData(model_yaml["Dataset"], custom_data_dir=path)
    # For now, we perform the fusion based on validation dataset and testing on the test dataset.
    valset, testset = dataset.val, dataset.test
    if fusion_cfg is not None:
        val_batch_size = fusion_cfg.batch_size
    else:
        val_batch_size = model_yaml["net_params"]["batch_size"]
    val_loader = DataLoader(valset, batch_size=val_batch_size, shuffle=shuffle_val, drop_last=False,
                            collate_fn=dataset.collate)
    test_loader = DataLoader(testset, batch_size=model_yaml["net_params"]["batch_size"], shuffle=False, drop_last=False,
                             collate_fn=dataset.collate)
    return val_loader, test_loader


def get_finetune_test_val_loaders(model_yaml, path):
    """Gets dataloaders for finetuning."""
    # take dataset name from first model
    dataset = LoadData(model_yaml["Dataset"], custom_data_dir=path)
    trainset, valset = dataset.train, dataset.val

    train_loader = DataLoader(trainset, model_yaml["net_params"]["batch_size"], shuffle=False, drop_last=False,
                              collate_fn=dataset.collate)
    val_loader = DataLoader(valset, model_yaml["net_params"]["batch_size"], shuffle=False, drop_last=False,
                            collate_fn=dataset.collate)

    return train_loader, val_loader


def torch2jnp(x):
    return jnp.array(x.cpu())

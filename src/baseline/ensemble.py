# TODO: Implement classical ensembling

import argparse
from typing import List

from torch.utils.data import DataLoader

def compose_models(args: argparse.Namespace, models: List, test_loader: DataLoader) -> float:
    pass
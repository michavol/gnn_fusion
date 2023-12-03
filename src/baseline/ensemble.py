# TODO: Implement classical ensembling

import argparse
from typing import List

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, g, h, e):
        outputs = [model.forward(g, h, e) for model in self.models]
        print(outputs)
        return torch.mean(torch.stack(outputs), dim=0)
    
    def loss(self,scores, targets):
        outputs = [model.loss(scores, targets) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

def compose_models(args: argparse.Namespace, models: List, test_loader: DataLoader) -> float:
    pass
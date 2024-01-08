import torch
import torch.nn as nn


class Ensemble(nn.Module):
    """Class that ensembles the predictions of the provided models."""

    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, g, h, e):
        outputs = [model.forward(g, h, e) for model in self.models]
        print(outputs)
        return torch.mean(torch.stack(outputs), dim=0)

    def loss(self, scores, targets):
        outputs = [model.loss(scores, targets) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

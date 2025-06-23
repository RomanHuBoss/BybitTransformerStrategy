# Models/direction/losses.py

import torch
import torch.nn as nn

class BCELossWithLogits(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, outputs, targets):
        return self.loss(outputs, targets)

import torch
import torch.nn as nn
import torch.nn.functional as F

class CostSensitiveFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=None, label_smoothing=0.0):
        """
        alpha — веса классов (тензор размерности [num_classes])
        gamma — фокусирующий параметр (либо скаляр, либо тензор размерности [num_classes])
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', weight=self.alpha, label_smoothing=self.label_smoothing
        )
        ce_loss = torch.nan_to_num(ce_loss, nan=0.0, posinf=1e6, neginf=-1e6)

        pt = torch.exp(-ce_loss)
        if isinstance(self.gamma, torch.Tensor):
            gamma_factor = self.gamma[targets]
        else:
            gamma_factor = self.gamma

        focal_loss = ((1 - pt) ** gamma_factor) * ce_loss
        focal_loss = torch.nan_to_num(focal_loss, nan=0.0, posinf=1e6, neginf=-1e6)

        return focal_loss.mean()

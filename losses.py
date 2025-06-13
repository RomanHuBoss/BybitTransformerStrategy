import torch
import torch.nn as nn
import torch.nn.functional as F

class CostSensitiveFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        """
        Cost-Sensitive Focal Loss with class weights and optional per-class gamma.

        :param alpha: Tensor of class weights (shape [num_classes])
        :param gamma: Scalar or tensor per-class focusing parameter
        :param label_smoothing: Label smoothing factor (default=0.0)
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

        pt = torch.exp(-ce_loss).clamp(min=1e-8, max=1.0)
        if isinstance(self.gamma, torch.Tensor):
            gamma_factor = self.gamma[targets]
        else:
            gamma_factor = self.gamma

        focal_loss = ((1 - pt) ** gamma_factor) * ce_loss
        focal_loss = torch.nan_to_num(focal_loss, nan=0.0, posinf=1e6, neginf=-1e6)

        return focal_loss.mean()

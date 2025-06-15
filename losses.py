import torch
import torch.nn as nn
import torch.nn.functional as F


# === Directional Loss ===

class CostSensitiveFocalLoss(nn.Module):
    """
    Классический focal loss с поддержкой cost-sensitive обучения.
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        return focal_loss.mean()


# === Amplitude Loss ===

class QuantileLoss(nn.Module):
    """
    Классический quantile loss для обучения p10 и p90.
    """
    def __init__(self, quantile):
        super().__init__()
        self.quantile = quantile

    def forward(self, preds, target):
        error = target - preds
        return torch.max((self.quantile - 1) * error, self.quantile * error).mean()


class AmplitudeLoss(nn.Module):
    """
    Общий лосс для амплитудной модели с поддержкой весов.
    """
    def __init__(self, weights=None, device=None):
        super().__init__()
        self.up_p10_loss = QuantileLoss(0.1)
        self.up_p90_loss = QuantileLoss(0.9)
        self.down_p10_loss = QuantileLoss(0.1)
        self.down_p90_loss = QuantileLoss(0.9)

        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0]

        self.weights = torch.tensor(weights, dtype=torch.float32)
        if device is not None:
            self.weights = self.weights.to(device)

    def forward(self, preds, targets):
        up_p10_pred, up_p90_pred, down_p10_pred, down_p90_pred = preds

        up_p10_target = targets[:, 0:1]
        up_p90_target = targets[:, 1:2]
        down_p10_target = targets[:, 2:3]
        down_p90_target = targets[:, 3:4]

        loss = 0
        loss += self.weights[0] * self.up_p10_loss(up_p10_pred, up_p10_target)
        loss += self.weights[1] * self.up_p90_loss(up_p90_pred, up_p90_target)
        loss += self.weights[2] * self.down_p10_loss(down_p10_pred, down_p10_target)
        loss += self.weights[3] * self.down_p90_loss(down_p90_pred, down_p90_target)

        return loss

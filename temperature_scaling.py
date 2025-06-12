import torch
import torch.nn as nn
import torch.optim as optim

class TemperatureScaling(nn.Module):
    def __init__(self, init_temp=1.0, min_temp=0.5, max_temp=5.0, reg_lambda=0.01):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.reg_lambda = reg_lambda

    def forward(self, logits):
        T = self.temperature.to(logits.device)
        return logits / T

    def loss(self, logits, labels):
        labels = labels.to(logits.device)
        scaled_logits = self.forward(logits)
        return nn.functional.cross_entropy(scaled_logits, labels)

    def fit(self, logits, labels, lr=0.01, max_iter=500):
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = self.loss(logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature.data = torch.clamp(self.temperature.data, self.min_temp, self.max_temp)

    def get_temperature(self):
        return self.temperature.item()

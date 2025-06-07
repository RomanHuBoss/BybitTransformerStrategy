import torch
import torch.nn as nn
from config import CFG

class MultiPairDirectionalClassifier(nn.Module):
    def __init__(self, model_config, num_pairs):
        super(MultiPairDirectionalClassifier, self).__init__()

        self.input_proj = nn.Linear(model_config.input_dim, model_config.hidden_dim)
        self.model_config = model_config

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_config.hidden_dim,
            nhead=model_config.n_heads,
            dim_feedforward=model_config.dim_feedforward,
            activation=model_config.activation,
            layer_norm_eps=model_config.layer_norm_eps,
            dropout=model_config.dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=model_config.n_layers
        )

        self.output_head = nn.Sequential(
            nn.Linear(model_config.hidden_dim, model_config.hidden_dim),
            nn.ReLU(),
            nn.Linear(model_config.hidden_dim, num_pairs * 3)  # по 3 класса на каждую пару
        )

        self.num_pairs = num_pairs

        # Добавляем слой для анализа важности признаков
        self.feature_projection = nn.Linear(CFG.default_model_config.input_dim, CFG.default_model_config.hidden_dim)

    def feature_importances(self):
        """Возвращает важность признаков на основе весов первого слоя"""
        if hasattr(self, 'input_proj'):
            # Для трансформеров с input_proj слоем
            weights = self.input_proj.weight.abs().mean(dim=0)
        else:
            # Для линейных проекций
            weights = self.feature_projection.weight.abs().mean(dim=0)

        return weights.cpu().detach().numpy()

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # последний временной шаг
        x = self.output_head(x)
        return x.view(x.size(0), self.num_pairs, 3)  # [B, num_pairs, 3]

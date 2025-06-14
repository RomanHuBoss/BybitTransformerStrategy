import torch.nn as nn

# ======================================
# Directional Model (Transformer Head)
# ======================================

class DirectionalModel(nn.Module):
    def __init__(self, model_config):
        super(DirectionalModel, self).__init__()

        self.input_proj = nn.Linear(model_config.input_dim, model_config.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_config.hidden_dim,
            nhead=model_config.n_heads,
            dim_feedforward=model_config.dim_feedforward,
            activation=model_config.activation,
            layer_norm_eps=model_config.layer_norm_eps,
            dropout=model_config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=model_config.n_layers)

        self.output_head = nn.Sequential(
            nn.Linear(model_config.hidden_dim, model_config.hidden_dim),
            nn.ReLU(),
            nn.Linear(model_config.hidden_dim, 3)  # 3 класса: short, neutral, long
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # берём последнее значение окна
        x = self.output_head(x)
        return x


# ======================================
# Amplitude Model (двухголовый регрессор)
# ======================================

class AmplitudeModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.up_head = nn.Linear(128, 1)
        self.down_head = nn.Linear(128, 1)

    def forward(self, x):
        shared_out = self.shared(x)
        up_out = self.up_head(shared_out)
        down_out = self.down_head(shared_out)
        return up_out, down_out

import torch.nn as nn

class DirectionalModel(nn.Module):
    def __init__(self, model_config, num_pairs):
        super(DirectionalModel, self).__init__()

        self.model_config = model_config
        self.num_pairs = num_pairs

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
            nn.Linear(model_config.hidden_dim, num_pairs * 3)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        x = self.output_head(x)
        return x.view(x.size(0), 3)


# AmplitudeModel — регрессия амплитуды
class AmplitudeModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

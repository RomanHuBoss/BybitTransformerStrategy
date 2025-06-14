import torch.nn as nn

class DirectionalModel(nn.Module):
    def __init__(self, model_config):
        super(DirectionalModel, self).__init__()

        self.model_config = model_config

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
            nn.Linear(model_config.hidden_dim, 3)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        x = self.output_head(x)
        return x

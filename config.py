import numpy as np
import math
import torch
from types import SimpleNamespace
import pandas as pd


class Paths:
    raw_data_path = "historical_data/BTCUSDT/30m/monthly/combined_csv.csv"
    model_path = "artifacts/model.pth"
    meta_path = "artifacts/model_meta.json"


class TrainConfig:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timeframe = 30
    lr = 1e-5
    batch_size = 256
    epochs = 300
    patience = 30
    val_ratio = 0.2
    window_size = {
        15: 60,
        30: 100
    }
    lookahead = {
        15: 10,
        30: 8,
    }
    auto_gamma_search = True
    gamma_values = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]
    gamma_search_epochs = 5

    threshold = 0.75
    calibrate_logits = True
    use_selected_features_only = True
    feature_importance_threshold = "mean"
    label_smoothing = 0.05

    scheduler = dict(
        mode="max", factor=0.5, patience=7, cooldown=3, min_lr=1e-6
    )

    loader = staticmethod(lambda: pd.read_csv(Paths.raw_data_path))



class DefaultModelConfig:
    input_dim = 69  # будет обновлено автоматически, если use_selected_features_only=True
    hidden_dim = 192
    n_heads = 16
    n_layers = 4
    dropout = 0.3
    n_classes = 3
    dim_feedforward = 768
    activation = 'gelu'
    layer_norm_eps = 1e-5

class SL_TP_Config:
    tp_sl_levels = [
        (0.02, 0.01), (0.025, 0.01), (0.03, 0.01), (0.035, 0.01), (0.04, 0.01),
        (0.03, 0.015), (0.04, 0.015), (0.045, 0.015), (0.05, 0.015), (0.06, 0.015),
        (0.04, 0.02), (0.05, 0.02), (0.06, 0.02), (0.07, 0.02), (0.08, 0.02),
        (0.06, 0.03), (0.075, 0.03), (0.09, 0.03), (0.105, 0.03), (0.120, 0.03)
    ]

CFG = SimpleNamespace(
    paths=Paths(),
    train=TrainConfig(),
    tp_sl=SL_TP_Config(),
    default_model_config=DefaultModelConfig(),
    label2action={0: "short", 1: "no-trade", 2: "long"},
    action2label={"short": 0, "no-trade": 1, "long": 2},
)

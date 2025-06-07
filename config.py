import numpy as np
import math
import torch
from types import SimpleNamespace
import pandas as pd


class Paths:
    raw_data_path = "historical_data/BTCUSDT/15m/monthly/combined_csv.csv"
    model_path = "artifacts/model.pth"
    meta_path = "artifacts/model_meta.json"


class TrainConfig:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timeframe = 15  # таймфрейм в минутах
    lr = 1e-4
    batch_size = 512
    epochs = 300
    patience = 30
    val_ratio = 0.2
    window_size = 60     # сколько баров в одном обучающем примере
    lookahead = 24       # горизонт TP/SL в барах

    # Дополнительные параметры:
    threshold = 0.7  # порог вероятности для генерации сигнала
    calibrate_logits = True  # если True — калибровать вероятности (например, temperature scaling)
    use_selected_features_only = True  # использовать только важные признаки
    feature_importance_threshold = "mean"  # "mean", "median" или float

    scheduler = dict(  # ReduceLROnPlateau
        mode="max", factor=0.5, patience=5, cooldown=3, min_lr=1e-6
    )

    loader = staticmethod(lambda: pd.read_csv(Paths.raw_data_path))  # загрузка CSV


class DefaultModelConfig:
    input_dim = 61  # будет обновлено автоматически, если use_selected_features_only=True
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
        (0.02, 0.01),
        (0.03, 0.01),
        (0.03, 0.015),
        (0.045, 0.015),
        (0.04, 0.02),
        (0.06, 0.02),
        (0.05, 0.025),
        (0.075, 0.025),
    ]

CFG = SimpleNamespace(
    paths=Paths(),
    train=TrainConfig(),
    tp_sl=SL_TP_Config(),
    default_model_config=DefaultModelConfig(),
)

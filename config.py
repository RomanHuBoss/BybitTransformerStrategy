from pathlib import Path

class CFG:

    class paths:
        base = Path("./")
        train_csv = base / "historical_data" / "LEARN" / "combined_csv.csv"
        train_features_csv = base / "artifacts" / "model_30m" / "train_features.csv"
        scaler_path = base / "artifacts" / "model_30m" / "direction_scaler.joblib"
        direction_model_path = base / "artifacts" / "model_30m" / "direction_model.pth"
        amplitude_model_path = base / "artifacts" / "model_30m" / "amplitude_model.pth"
        hit_order_model_path = base / "artifacts" / "model_30m" / "hit_order_model.pth"
        temperature_path = base / "artifacts" / "model_30m" / "temperature.joblib"
        feature_columns_path = base / "artifacts" / "model_30m" / "features.joblib"

    class train:
        lr = 3e-4
        batch_size = 512
        epochs = 500
        early_stopping_patience = 100
        val_size = 0.1
        focal_gamma = 2.0

    class inference:
        update_interval = 120
        api_port = 8000

    class assets:
        symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT"]
        timeframe = 30
        limit = 1000

    class hybrid:
        min_amplitude = 0.002
        spread_threshold = 0.05
        dynamic_threshold_alpha = 1.0

    class labels:
        lookahead = 10
        rr_min = 2
        sl_min = 0.005
        sl_max = 0.02
        sl_step = 0.001

    class label_generation:
        direction_threshold = 0.015
        amplitude_min_sl = 0.005
        amplitude_max_sl = 0.02
        amplitude_max_tp = 0.15
        hit_order_sl_min = 0.005
        hit_order_sl_max = 0.02
        hit_order_rr_min = 2.0
        hit_order_rr_max = 6.0
        hit_order_sl_step = 0.005
        hit_order_rr_step = 0.5

    class feature_engineering:
        default_shift = 1
        window_size = 30

    class snapshot:
        history_depth = 100

    class action2label:
        mapping = {
            "short": 0,
            "no-trade": 1,
            "long": 2
        }

    class amplitude:
        log_eps = 1e-6
        loss_weights = [0.4, 0.1, 0.4, 0.1]

    class ModelConfig:
        input_dim = None
        hidden_dim = 128
        n_layers = 2
        n_heads = 4
        dim_feedforward = 256
        activation = 'gelu'
        dropout = 0.1
        layer_norm_eps = 1e-5

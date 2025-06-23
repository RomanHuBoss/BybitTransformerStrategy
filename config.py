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
        models_dir = base / "artifacts" / "model_30m"

        data_path = train_csv  # или укажи путь если другой файл
        feature_dataset_path = base / "artifacts" / "model_30m" / "full_dataset.parquet"

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

    class hitorder:
        lr = 9e-2
        batch_size = 2048
        epochs = 500
        early_stopping_patience = 200
        device = "cuda"  # или "cpu"
        val_size = 0.2

    class label_generation:
        direction_lookahead = 10  # в свечах
        amplitude_lookahead = 10  # в свечах
        hitorder_lookahead = 10  # в свечах

        direction_threshold = 0.015

        amplitude_min_sl = 0.005
        amplitude_max_sl = 0.02
        amplitude_max_tp = 0.15

        hitorder_sl_list = [0.005, 0.01, 0.015, 0.02]
        hitorder_rr_list = [1.0, 1.5, 2.0, 2.5, 3.0]

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

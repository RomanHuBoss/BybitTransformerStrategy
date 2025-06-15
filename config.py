from pathlib import Path

class CFG:

    # Пути к данным и моделям
    class paths:
        base = Path("./")
        train_csv = base / "historical_data" / "BTCUSDT" / "30m" / "monthly" / "combined_csv.csv"

        # новые файлы:
        train_features_csv = base / "artifacts" / "model_30m" / "train_features.csv"
        train_labels_direction = base / "artifacts" / "model_30m" / "train_labels_direction.npy"
        train_labels_amplitude = base / "artifacts" / "model_30m" / "train_labels_amplitude.npy"
        train_labels_hitorder = base / "artifacts" / "model_30m" / "train_labels_hitorder.npy"

        scaler_path = base / "artifacts" / "model_30m" / "direction_scaler.joblib"

        direction_model_path = base / "artifacts" / "model_30m" / "direction_model.pth"
        amplitude_model_path = base / "artifacts" / "model_30m" / "amplitude_model.pth"
        hit_order_model_path = base / "artifacts" / "model_30m" / "hit_order_model.pth"

        temperature_path = base / "artifacts" / "model_30m" / "temperature.joblib"
        feature_columns_path = base / "artifacts" / "model_30m" / "features.joblib"


    # Параметры обучения (общие для direction и amplitude)
    class train:
        lr = 3e-4
        batch_size = 512
        epochs = 100
        early_stopping_patience = 10
        val_size = 0.1
        focal_gamma = 2.0

    # Параметры инференса
    class inference:
        update_interval = 120  # секунд между обновлениями snapshot
        api_port = 8000

    # Параметры актива и данных
    class assets:
        symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT"]
        timeframe = 30
        limit = 1000

    # Параметры hybrid логики
    class hybrid:
        min_amplitude = 0.002  # минимальный предсказанный amplitude для фильтрации
        spread_threshold = 0.05  # порог ширины интервала (spread) для входа
        dynamic_threshold_alpha = 1.0  # коэффициент динамической корректировки threshold в зависимости от spread

    # Параметры label генерации
    class labels:
        lookahead = 20
        rr_min = 2
        sl_min = 0.005
        sl_max = 0.02
        sl_step = 0.001

    class label_generation:
        direction_threshold = 0.015

        amplitude_min_sl = 0.005
        amplitude_max_sl = 0.02
        amplitude_max_tp = 0.15

        # параметры под HitOrder
        hit_order_sl_min = 0.005  # 0.5% минимальный SL
        hit_order_sl_max = 0.02  # 2% максимальный SL
        hit_order_rr_min = 2.0  # минимальный RR: TP >= 2×SL
        hit_order_rr_max = 6.0  # максимальный RR (чтобы учесть разные сценарии)

    # Режимы генерации признаков
    class feature_engineering:
        default_shift = 1  # смещение меток
        window_size = 60  # окно

    # Параметры сохранения генератора
    class snapshot:
        history_depth = 100  # глубина хранения snapshot для бэктестинга

    class action2label:
        mapping = {
            "short": 0,
            "no-trade": 1,
            "long": 2
        }

    class amplitude:
        log_eps = 1e-6
        loss_weights = [0.4, 0.1, 0.4, 0.1]

    class DirectionModelConfig:
        input_dim = None  # обязательно задаётся позже (динамически)
        hidden_dim = 128
        n_layers = 2
        n_heads = 4
        dim_feedforward = 256
        activation = 'gelu'
        dropout = 0.1
        layer_norm_eps = 1e-5
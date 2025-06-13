from pathlib import Path

class CFG:

    # Пути к данным и моделям
    class paths:
        base = Path("./")
        train_csv = base / "data" / "train_data.csv"
        scaler_path = base / "artifacts" / "direction_scaler.joblib"
        amplitude_scaler_path = base / "artifacts" / "amplitude_scaler.joblib"
        amplitude_target_scaler_path = base / "artifacts" / "amplitude_target_scaler.joblib"
        direction_model_path = base / "artifacts" / "direction_model.pth"
        amplitude_model_path = base / "artifacts" / "amplitude_model.pth"
        temperature_path = base / "artifacts" / "temperature.joblib"
        thresholds_path = base / "artifacts" / "thresholds.joblib"

    # Параметры обучения (общие для direction и amplitude)
    class train:
        lr = 3e-4
        batch_size = 512
        epochs = 10
        val_size = 0.1
        focal_gamma = 2.0

    # Параметры инференса
    class inference:
        update_interval = 120  # секунд между обновлениями snapshot
        api_port = 8000

    # Параметры актива и данных
    class assets:
        symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT"]
        timeframe = "30"
        limit = 500

    # Параметры hybrid логики
    class hybrid:
        min_amplitude = 0.002  # минимальный предсказанный amplitude для фильтрации

    # Параметры label генерации
    class labels:
        tp_multiplier = 1.0
        sl_multiplier = 1.0
        slippage = 0.0005

    # Режимы генерации признаков
    class feature_engineering:
        default_shift = 1  # смещение меток

    # Параметры сохранения генератора
    class snapshot:
        history_depth = 100  # глубина хранения snapshot для бэктестинга


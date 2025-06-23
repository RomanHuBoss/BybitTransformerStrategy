# Models/direction/config.py

class DirectionConfig:
    """
    Конфиг для модели direction — прогноз пробития вверх / вниз в ближайшем lookahead окне
    """

    # ❗ Размер lookahead окна (в барах) — сколько баров смотрим вперёд для проверки пробития
    LOOKAHEAD_BARS = 10  # т.е. 10 x 30m баров = 5 часов

    # ❗ Пороговое значение для определения пробития TP (в процентах от текущей цены)
    MIN_TP_THRESHOLD_PERCENT = 1.5  # например, 1.5% вверх или вниз

    # ❗ Размер окна для генерации признаков (history window для модели)
    FEATURE_WINDOW = 10  # сколько последних баров брать в фичи

    # ❗ Путь к данным
    DATA_PATH = "data/binance_30m.csv"

    # ❗ Размер обучающего теста и валидации (в процентах)
    TRAIN_SIZE = 0.8
    VAL_SIZE = 0.1  # остальное на тест

    # ❗ Batch size
    BATCH_SIZE = 512

    # ❗ Learning rate
    LEARNING_RATE = 0.0005

    # ❗ Количество эпох
    EPOCHS = 100

    # ❗ Early stopping
    EARLY_STOPPING_PATIENCE = 10

    # ❗ Путь для сохранения модели
    SAVE_PATH = "Models/direction/model.pth"

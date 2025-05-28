import os
import numpy as np
from typing import List, Tuple


# Конфигурационные параметры проекта
class Config:
    # Параметры данных
    DATA_PATH = os.path.join('historical_data', 'BTCUSDT', '5m', 'monthly', 'combined_csv.csv')
    LOOKBACK_WINDOW = 36  # Количество свечей для анализа
    LOOKAHEAD_WINDOW = 24  # Количество свечей для прогноза

    ORIGINAL_COLUMNS = [
        'open_time', 'open', 'high', 'low', 'close', 'volume'
    ]
    ORIGINAL_NUMERIC_COLUMNS = [
        'open', 'high', 'low', 'close', 'volume'
    ]


    # Параметры стоп-лосса и тейк-профита
    SL_PERCENTAGES = np.arange(0.005, 0.03, 0.005)  # значения SL
    TP_MULTIPLIERS = np.arange(1.5, 4, 0.1)  # Множители для TP

    # Параметры модели
    BATCH_SIZE = 128
    EPOCHS = 1
    LEARNING_RATE = 0.0005
    MODEL_FILE_NAME = 'trading_transformer.pth'
    MODEL_SAVE_PATH = os.path.join('models', MODEL_FILE_NAME)

    # Параметры логирования
    LOG_DIR = 'logs/'
    LOG_LEVEL = 'INFO'

    @staticmethod
    def get_sl_tp_pairs() -> List[Tuple[float, float]]:
        """Генерирует все возможные пары SL/TP на основе конфигурации."""
        pairs = []
        for sl in Config.SL_PERCENTAGES:
            for multiplier in Config.TP_MULTIPLIERS:
                tp = sl * multiplier
                pairs.append((sl, tp))
        return pairs
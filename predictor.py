import os

import torch
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime
from config import Config
from data_processor import DataProcessor
from model import TransformerModel

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class TradingPredictor:
    """Класс для загрузки модели и выполнения прогнозов на новых данных."""

    def __init__(self, model_folder: str, threshold: float = 0.5):
        """
        Инициализация прогнозирующего модуля.

        Args:
            model_folder: Путь к папке с моделью
            threshold: Порог вероятности для фильтрации результатов
        """
        self.model_folder = model_folder
        self.threshold = threshold
        self.model = None
        self.data_processor = DataProcessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Загружаем модель при инициализации
        self.load_model()

    def load_model(self):
        """Загружает обученную модель из файла."""
        try:
            model_path = os.path.join(self.model_folder, Config.MODEL_FILE_NAME)
            logger.info(f"Попытка загрузки модели из {model_path}")

            self.model = TransformerModel.load(model_path).to(self.device)
            self.model.eval()

            logger.info("Модель успешно загружена и переведена в режим оценки")
            logger.debug(f"Архитектура модели: {self.model}")

        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {str(e)}")
            logger.exception("Трассировка стека:")
            raise

    def prepare_input_data(self, candles_data: List[Dict]) -> pd.DataFrame:
        """
        Подготавливает входные данные для модели из сырых свечей.

        Args:
            candles_data: Список словарей с данными свечей от Bybit

        Returns:
            pd.DataFrame: Обработанный DataFrame с фичами
        """
        logger.info("Начало подготовки входных данных")
        start_time = datetime.now()

        try:
            # Конвертируем в DataFrame
            df = pd.DataFrame(candles_data)

            # Конвертируем время
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')

            # Сортируем по времени
            df.sort_values('open_time', inplace=True)

            # Добавляем технические индикаторы и временные фичи
            df = self.data_processor.add_technical_indicators(df)
            df = self.data_processor.add_time_features(df)

            # Очищаем данные
            df = self.data_processor.clean_data(df)

            logger.info(f"Данные подготовлены. Форма DataFrame: {df.shape}")
            logger.debug(f"Колонки в DataFrame: {df.columns.tolist()}")
            logger.info(f"Подготовка данных заняла: {datetime.now() - start_time}")

            return df

        except Exception as e:
            logger.error(f"Ошибка подготовки данных: {str(e)}")
            logger.exception("Трассировка стека:")
            raise

    def create_sequences(self, df: pd.DataFrame) -> np.ndarray:
        """
        Создает последовательности для модели из подготовленных данных.

        Args:
            df: DataFrame с подготовленными данными

        Returns:
            np.ndarray: Массив последовательностей формы (n_samples, lookback_window, n_features)
        """
        logger.info("Создание последовательностей для модели")

        try:
            # Выбираем только фичи (исключаем исходные колонки)
            feature_columns = [col for col in df.columns
                               if col not in Config.ORIGINAL_COLUMNS]
            features = df[feature_columns].values

            # Создаем последовательности с учетом lookback_window
            X = []
            for i in range(Config.LOOKBACK_WINDOW, len(features)):
                X.append(features[i - Config.LOOKBACK_WINDOW:i])

            X = np.array(X)

            logger.info(f"Создано {len(X)} последовательностей формы {X.shape}")
            return X

        except Exception as e:
            logger.error(f"Ошибка создания последовательностей: {str(e)}")
            logger.exception("Трассировка стека:")
            raise

    def predict(self, candles_data: List[Dict], threshold: Optional[float] = None) -> Dict:
        """
        Выполняет прогноз на основе входных данных свечей.

        Args:
            candles_data: Список словарей с данными свечей от Bybit
            threshold: Опциональный порог для фильтрации результатов

        Returns:
            Dict: Словарь с прогнозами для всех пар SL/TP
        """
        logger.info("Начало процесса прогнозирования")
        start_time = datetime.now()

        try:
            # Устанавливаем порог, если он передан
            current_threshold = threshold if threshold is not None else self.threshold

            # 1. Подготовка данных
            df = self.prepare_input_data(candles_data)

            # 2. Создание последовательностей
            X = self.create_sequences(df)

            # 3. Прогнозирование
            with torch.no_grad():
                inputs = torch.FloatTensor(X).to(self.device)
                outputs = self.model(inputs)
                probabilities = torch.sigmoid(outputs).cpu().numpy()

            # Берем последнюю последовательность (самые свежие данные)
            last_probabilities = probabilities[-1]

            # 4. Формирование результата
            result = self._format_predictions(last_probabilities, current_threshold)

            logger.info(f"Прогнозирование завершено за {datetime.now() - start_time}")
            return result

        except Exception as e:
            logger.error(f"Ошибка прогнозирования: {str(e)}")
            logger.exception("Трассировка стека:")
            raise

    def _format_predictions(self, probabilities: np.ndarray, threshold: float) -> Dict:
        """
        Форматирует предсказания модели в JSON-подобный словарь.

        Args:
            probabilities: Массив вероятностей от модели
            threshold: Порог для фильтрации результатов

        Returns:
            Dict: Отформатированные предсказания с фильтрацией по порогу
        """
        logger.info(f"Форматирование предсказаний с порогом {threshold}")

        try:
            # Получаем все пары SL/TP из конфига
            sl_tp_pairs = Config.get_sl_tp_pairs()

            # Проверяем соответствие количества пар и предсказаний
            if len(probabilities) != len(sl_tp_pairs) * 2:
                raise ValueError(
                    f"Несоответствие размеров: модель вернула {len(probabilities)} предсказаний, "
                    f"ожидалось {len(sl_tp_pairs) * 2}"
                )

            result = {
                "timestamp": datetime.now().isoformat(),
                "threshold": threshold,
                "predictions": []
            }

            # Формируем предсказания для каждой пары
            for i, (sl, tp) in enumerate(sl_tp_pairs):
                long_prob = probabilities[i * 2]
                short_prob = probabilities[i * 2 + 1]

                # Добавляем только если вероятность выше порога
                if long_prob >= threshold or short_prob >= threshold:
                    prediction = {
                        "sl_percent": round(sl * 100, 2),
                        "tp_percent": round(tp * 100, 2),
                        "long_probability": round(float(long_prob), 4),
                        "short_probability": round(float(short_prob), 4),
                        "long_signal": long_prob >= threshold,
                        "short_signal": short_prob >= threshold
                    }
                    result["predictions"].append(prediction)

            logger.info(f"Сформировано {len(result['predictions'])} предсказаний "
                        f"(отфильтровано по порогу {threshold})")
            return result

        except Exception as e:
            logger.error(f"Ошибка форматирования предсказаний: {str(e)}")
            logger.exception("Трассировка стека:")
            raise
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import pandas as pd
import numpy as np
import talib
import logging
from numba import njit, prange
from config import Config

# Настройка логирования
logging.basicConfig(level=Config.LOG_LEVEL)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@njit(parallel=True, fastmath=True)
def calculate_labels_numba(close, high, low, sl_values, tp_multipliers,
                           lookback_window, lookahead_window):
    """
    Ускоренный расчёт меток для бинарной классификации (0/1).
    Теперь возвращает только 0 (не сработал TP) или 1 (сработал TP).

    Параметры:
        close: массив цен закрытия
        high: массив максимальных цен
        low: массив минимальных цен
        sl_values: массив значений стоп-лосса (в долях от цены)
        tp_multipliers: массив множителей для тейк-профита
        lookback_window: окно исторических данных
        lookahead_window: окно для прогноза

    Возвращает:
        Массив меток формы (n_samples, n_pairs*2), где:
        - 0: TP не достигнут (либо сработал SL, либо ничего)
        - 1: TP достигнут
    """
    n = len(close)
    n_pairs = len(sl_values) * len(tp_multipliers)
    labels = np.zeros((n, n_pairs * 2), dtype=np.int8)

    for i in prange(lookback_window, n):
        current_price = close[i]

        for pair_idx in prange(len(sl_values) * len(tp_multipliers)):
            sl = sl_values[pair_idx // len(tp_multipliers)]
            tp_mult = tp_multipliers[pair_idx % len(tp_multipliers)]
            tp = sl * tp_mult

            # LONG позиция
            long_tp = current_price * (1 + tp)
            long_sl = current_price * (1 - sl)
            long_result = 0  # По умолчанию 0 (TP не достигнут)

            # SHORT позиция
            short_tp = current_price * (1 - tp)
            short_sl = current_price * (1 + sl)
            short_result = 0  # По умолчанию 0 (TP не достигнут)

            # Проверяем следующие lookahead_window свечей
            for j in range(i + 1, min(i + 1 + lookahead_window, n)):
                # Для LONG позиции
                if long_result == 0:
                    if high[j] >= long_tp:  # Сработал TP
                        long_result = 1
                    elif low[j] <= long_sl:  # Сработал SL - остаётся 0
                        pass

                # Для SHORT позиции
                if short_result == 0:
                    if low[j] <= short_tp:  # Сработал TP
                        short_result = 1
                    elif high[j] >= short_sl:  # Сработал SL - остаётся 0
                        pass

                # Прерываем если оба направления уже определены
                if long_result != 0 and short_result != 0:
                    break

            # Сохраняем результаты (только 0 или 1)
            labels[i, pair_idx * 2] = long_result
            labels[i, pair_idx * 2 + 1] = short_result

    return labels


def log_progress(current, total, start_time):
    """Функция для отображения прогресс-бара."""
    elapsed = datetime.now() - start_time
    percent = (current / total) * 100
    eta = elapsed * (total - current) / max(1, current)

    logger.info(
        f"Прогресс: {percent:.1f}% | "
        f"Обработано: {current}/{total} | "
        f"Затрачено: {str(elapsed).split('.')[0]} | "
        f"ETA: {str(eta).split('.')[0]}"
    )


class DataProcessor:
    def calculate_labels(self, df):
        """Расчёт меток для обучения модели с логированием прогресса."""
        start_time = datetime.now()
        logger.info("Начало расчёта бинарных меток (TP=1, иначе=0)...")

        # Проверка входных данных
        if df.empty:
            logger.error("Передан пустой DataFrame!")
            raise ValueError("Пустой DataFrame")

        # Подготовка данных
        close = df['close'].values.astype(np.float64)
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)

        logger.info(f"Всего свечей: {len(close)}, SL значений: {len(Config.SL_PERCENTAGES)}, "
                    f"TP множителей: {len(Config.TP_MULTIPLIERS)}")

        # Вызов ускоренной функции
        labels_array = calculate_labels_numba(
            close, high, low,
            Config.SL_PERCENTAGES, Config.TP_MULTIPLIERS,
            Config.LOOKBACK_WINDOW, Config.LOOKAHEAD_WINDOW
        )

        unique_values = np.unique(labels_array)
        if len(unique_values) < 2:
            raise ValueError(f"Метки содержат только один класс: {unique_values}. Проверьте параметры SL/TP.")

        # Проверка результатов
        unique_values = np.unique(labels_array)
        logger.info(f"Расчёт завершён за {datetime.now() - start_time}")
        logger.info(f"Уникальные значения меток: {unique_values}")
        logger.info(f"Распределение меток: 0 - {(labels_array == 0).sum()}, 1 - {(labels_array == 1).sum()}")

        unique_values = np.unique(labels_array)
        if not np.array_equal(unique_values, [0, 1]):
            raise ValueError(f"Метки должны содержать только 0 и 1. Найдены значения: {unique_values}")

        return self._create_result_df(df, labels_array)

    def _create_result_df(self, df, labels_array):
        """Создает DataFrame с метками."""
        column_names = []
        for sl in Config.SL_PERCENTAGES:
            for tp_mult in Config.TP_MULTIPLIERS:
                tp = sl * tp_mult
                column_names.extend([
                    f'LONG_sl_{sl * 100:.1f}%_tp_{tp * 100:.1f}%',
                    f'SHORT_sl_{sl * 100:.1f}%_tp_{tp * 100:.1f}%'
                ])

        return pd.concat([
            df,
            pd.DataFrame(
                labels_array[Config.LOOKBACK_WINDOW:],
                columns=column_names,
                index=df.index[Config.LOOKBACK_WINDOW:]
            )
        ], axis=1)

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Загружает данные из CSV файла."""
        logger.info(f"Загрузка данных из {file_path}")
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')

            df.sort_values('open_time', inplace=True)
            df.reset_index(drop=True, inplace=True)

            logger.info(f"Успешно загружено {len(df)} записей из файла {file_path}")
            return df
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            raise

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет технические индикаторы в DataFrame."""
        logger.info("Добавление технических индикаторов")

        # Проверка на пустые данные
        if df.empty:
            logger.warning("Пустой DataFrame, индикаторы не добавлены")
            return df

        # Обязательные колонки
        for col in Config.ORIGINAL_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Отсутствует обязательная колонка: {col}")

        # удаляем лишние колонки, кроме обязательных
        df = df[Config.ORIGINAL_COLUMNS]

        # Копируем DataFrame для избежания предупреждений
        df = df.copy()

        # Преобразование типов данных
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in Config.ORIGINAL_NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)

        # Удаление строк с NaN после преобразования
        df = df.dropna(subset=numeric_cols)

        # Ценовые данные
        open = df['open'].values
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        # Трендовые индикаторы
        df['SMA_20'] = talib.SMA(close, timeperiod=20)
        df['SMA_50'] = talib.SMA(close, timeperiod=50)
        df['EMA_12'] = talib.EMA(close, timeperiod=12)
        df['EMA_26'] = talib.EMA(close, timeperiod=26)

        # MACD
        macd, macdsignal, macdhist = talib.MACD(close)
        df['MACD'] = macd
        df['MACD_signal'] = macdsignal
        df['MACD_hist'] = macdhist

        # Осцилляторы
        df['RSI_14'] = talib.RSI(close, timeperiod=14)
        df['RSI_28'] = talib.RSI(close, timeperiod=28)
        df['STOCH_k'], df['STOCH_d'] = talib.STOCH(high, low, close)
        df['CCI'] = talib.CCI(high, low, close, timeperiod=20)

        # Волатильность
        df['ATR_14'] = talib.ATR(high, low, close, timeperiod=14)
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(
            close, timeperiod=20)

        # Объемные индикаторы
        df['OBV'] = talib.OBV(close, volume)

        # Свечные паттерны (пример нескольких паттернов)
        df['CDLDOJI'] = talib.CDLDOJI(open, high, low, close)
        df['CDLENGULFING'] = talib.CDLENGULFING(open, high, low, close)
        df['CDLHAMMER'] = talib.CDLHAMMER(open, high, low, close)
        df['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(open, high, low, close)

        # # Разворотные паттерны
        # df['CDLHARAMI'] = talib.CDLHARAMI(open, high, low, close)
        # df['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(open, high, low, close)
        # df['CDLEVENINGSTAR'] = talib.CDLEVENINGSTAR(open, high, low, close)
        # df['CDLHANGINGMAN'] = talib.CDLHANGINGMAN(open, high, low, close)
        # df['CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(open, high, low, close)
        # df['CDL3WHITESOLDIERS'] = talib.CDL3WHITESOLDIERS(open, high, low, close)
        # df['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(open, high, low, close)
        #
        # # Продолжающие паттерны
        # df['CDLMARUBOZU'] = talib.CDLMARUBOZU(open, high, low, close)
        # df['CDLPIERCING'] = talib.CDLPIERCING(open, high, low, close)
        # df['CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(open, high, low, close)
        #
        # # Другие паттерны
        # df['CDLSPINNINGTOP'] = talib.CDLSPINNINGTOP(open, high, low, close)
        # df['CDL3LINESTRIKE'] = talib.CDL3LINESTRIKE(open, high, low, close)
        # df['CDLABANDONEDBABY'] = talib.CDLABANDONEDBABY(open, high, low, close)
        # df['CDLADVANCEBLOCK'] = talib.CDLADVANCEBLOCK(open, high, low, close)
        # df['CDLIDENTICAL3CROWS'] = talib.CDLIDENTICAL3CROWS(open, high, low, close)
        #

        # Разница между SMA
        df['SMA_diff'] = df['SMA_20'] - df['SMA_50']

        # Процентное изменение цены
        df['pct_change'] = df['close'].pct_change()

        logger.info(f"Добавлено {len(df.columns) - len(Config.ORIGINAL_COLUMNS)} индикаторов")

        # Исключаем временные и оригинальные колонки из нормализации
        cols_to_normalize = [col for col in df.columns
                             if col not in Config.ORIGINAL_COLUMNS
                             and not col.startswith(('time_', 'day_', 'is_', 'month', 'season'))]

        scaler = MinMaxScaler(feature_range=(0, 1))
        df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
        logger.info(f"Нормализовано {len(cols_to_normalize)} колонок")

        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет временные фичи в DataFrame."""
        logger.info("Добавление временных фич")

        if df.empty:
            logger.warning("Пустой DataFrame, временные фичи не добавлены")
            return df

        df = df.copy()
        df['time_hour'] = df['open_time'].dt.hour
        df['time_minute'] = df['open_time'].dt.minute
        df['day_of_week'] = df['open_time'].dt.dayofweek
        df['day_of_month'] = df['open_time'].dt.day
        df['month'] = df['open_time'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Время года (сезон)
        df['season'] = df['month'] % 12 // 3 + 1

        # Бинарные фичи для времени суток
        df['is_night'] = ((df['time_hour'] >= 0) & (df['time_hour'] < 6)).astype(int)
        df['is_morning'] = ((df['time_hour'] >= 6) & (df['time_hour'] < 12)).astype(int)
        df['is_afternoon'] = ((df['time_hour'] >= 12) & (df['time_hour'] < 18)).astype(int)
        df['is_evening'] = ((df['time_hour'] >= 18) & (df['time_hour'] < 24)).astype(int)

        logger.info(f"Добавлено {len(df.columns) - len(Config.ORIGINAL_COLUMNS)} временных фич")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очищает данные от NaN и бесконечных значений."""
        logger.info("Очистка данных")

        if df.empty:
            logger.warning("Пустой DataFrame, очистка не выполнена")
            return df

        # Заполняем NaN (новый рекомендуемый синтаксис)
        df = df.bfill().ffill()  # или df.ffill().bfill()

        # Удаляем оставшиеся NaN (если есть)
        df = df.dropna()

        # Заменяем бесконечные значения
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        logger.info(f"После очистки осталось {len(df)} записей")
        return df

    def prepare_dataset(self, file_path: str) -> pd.DataFrame:
        """Полный процесс подготовки данных."""
        df = self.load_data(file_path)
        df = self.add_technical_indicators(df)
        df = self.add_time_features(df)
        df = self.calculate_labels(df)
        df = self.clean_data(df)
        return df
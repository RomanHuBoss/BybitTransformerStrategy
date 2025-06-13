import pandas as pd
import numpy as np
import logging
from numba import njit
import ta
import ta.trend
import ta.momentum
import ta.volatility
from ta.volatility import AverageTrueRange
from sklearn.preprocessing import StandardScaler
from config import CFG
from scipy.stats import linregress, skew, kurtosis

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, timeframe_minutes=15):
        self.feature_columns = None
        self.scaler = None
        self.timeframe_minutes = timeframe_minutes
        self.tf_ratio = self.timeframe_minutes / 15

    def load_scaler(self, mean, scale):
        self.scaler = StandardScaler()
        self.scaler.mean_ = mean
        self.scaler.scale_ = scale

    def _adjust(self, base):
        return max(1, int(base * self.tf_ratio))

    def _get_linreg_angle(self, series, window):
        y = series[-window:]
        x = np.arange(len(y))
        slope = linregress(x, y).slope
        return np.degrees(np.arctan(slope))

    def generate_features(self, df: pd.DataFrame, fit: bool = False, use_logging: bool = True) -> pd.DataFrame:
        required_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

        df = df.copy()
        df['open_time'] = pd.to_datetime(df['open_time'])
        shift = CFG.train.lookahead[CFG.train.timeframe]

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        features = {}

        # Временные признаки (seasonality encoding)
        features['hour_sin'] = np.sin(2 * np.pi * df['open_time'].dt.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * df['open_time'].dt.hour / 24)
        features['minute_sin'] = np.sin(2 * np.pi * df['open_time'].dt.minute / 60)
        features['minute_cos'] = np.cos(2 * np.pi * df['open_time'].dt.minute / 60)
        features['dow_sin'] = np.sin(2 * np.pi * df['open_time'].dt.dayofweek / 7)
        features['dow_cos'] = np.cos(2 * np.pi * df['open_time'].dt.dayofweek / 7)

        # Логарифмические доходности
        returns = np.log(df['close'] / df['close'].shift(1))
        features['log_return_1'] = returns.shift(shift)
        features['returns_mean'] = returns.rolling(self._adjust(10)).mean().shift(shift)
        features['returns_std'] = returns.rolling(self._adjust(10)).std().shift(shift)
        features['returns_skew'] = returns.rolling(self._adjust(20)).apply(lambda x: skew(x)).shift(shift)
        features['returns_kurtosis'] = returns.rolling(self._adjust(20)).apply(lambda x: kurtosis(x)).shift(shift)

        # Range
        features['range_pct'] = (df['high'] - df['low']) / df['close'].shift(1)

        # ATR (volatility at multiple windows)
        for label, w in [('short', 3), ('mid', 8), ('long', 20)]:
            window = self._adjust(w)
            atr = AverageTrueRange(df['high'], df['low'], df['close'], window=window)
            features[f'atr_{label}_pct'] = (atr.average_true_range() / df['close'].shift(window - 1)).shift(shift + window - 1)

        # VWAP deviation
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        features['vwap_deviation'] = (df['close'] / df['vwap'] - 1).shift(shift)

        # RSI short
        rsi_window = self._adjust(6)
        rsi = ta.momentum.RSIIndicator(close=df['close'], window=rsi_window)
        features['rsi_short'] = rsi.rsi().shift(shift + rsi_window - 1)

        # Stochastic oscillator (сигнал перепроданности)
        stoch_window = self._adjust(5)
        stoch = ta.momentum.StochasticOscillator(
            high=df['high'], low=df['low'], close=df['close'],
            window=stoch_window, smooth_window=3)
        features['stoch_short'] = stoch.stoch().shift(shift + stoch_window - 1)

        # MACD fast (ускоренное расхождение средних)
        macd_fast = ta.trend.MACD(close=df['close'], window_slow=10, window_fast=3, window_sign=5)
        features['macd_fast'] = macd_fast.macd_diff().shift(shift + 9)

        # Trend Acceleration (склон тренда + ускорение)
        accel_window = self._adjust(10)
        trend_slope = df['close'].rolling(accel_window).apply(
            lambda x: linregress(np.arange(len(x)), x).slope).shift(shift + accel_window - 1)
        features['trend_slope'] = trend_slope
        features['trend_acceleration'] = trend_slope.diff().shift(shift)

        # EMA Cross Momentum (классический кросс скользящих)
        ema_5 = df['close'].ewm(span=self._adjust(5), min_periods=self._adjust(5)).mean()
        ema_10 = df['close'].ewm(span=self._adjust(10), min_periods=self._adjust(10)).mean()
        features['ema_cross_momentum'] = ema_5 - ema_10

        # Micro ATR (быстрая локальная волатильность)
        micro_window = self._adjust(5)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['micro_atr'] = tr.rolling(micro_window).mean().shift(shift + micro_window - 1)

        # Volume spike (объёмный буст)
        volume_window = self._adjust(20)
        rolling_mean_vol = df['volume'].rolling(volume_window).mean()
        features['volume_spike_boost'] = (df['volume'] / (rolling_mean_vol + 1e-8)).shift(shift + volume_window - 1)

        # Volatility breakout (пробой уровней)
        breakout_window = self._adjust(20)
        rolling_high = df['high'].rolling(breakout_window).max()
        rolling_low = df['low'].rolling(breakout_window).min()
        features['volatility_breakout_up_boost'] = (df['close'] > rolling_high.shift(1)).astype(int).shift(shift)
        features['volatility_breakout_down_boost'] = (df['close'] < rolling_low.shift(1)).astype(int).shift(shift)

        # Shadow ratios (соотношение теней к телу свечи)
        body = (df['close'] - df['open']).abs()
        upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
        lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
        features['upper_shadow_ratio'] = (upper_shadow / (body + 1e-6)).shift(shift)
        features['lower_shadow_ratio'] = (lower_shadow / (body + 1e-6)).shift(shift)

        # Entropy of returns (энтропия доходностей)
        entropy_window = self._adjust(15)
        features['entropy_returns'] = returns.rolling(entropy_window).apply(
            lambda x: -np.sum((p := np.histogram(x, bins=10, density=True)[0]) * np.log1p(p + 1e-6))
            if len(x.dropna()) == entropy_window else np.nan
        ).shift(shift + entropy_window - 1)

        # Скрытая дивергенция по RSI (твоя логика)
        if use_logging: logging.info("Генерация скрытых дивергенций...")
        features['rsi_divergence'] = (
            (df['close'] > df['close'].shift(2)) &
            (features['rsi_short'] < features['rsi_short'].shift(2))
        ).astype(int).shift(shift)

        # Candle Body Size (в % от полного диапазона свечи)
        body_size = (df['close'] - df['open']).abs()
        total_range = (df['high'] - df['low']).abs() + 1e-6
        features['body_size_pct'] = (body_size / total_range).shift(shift)
        # Интерпретация: как сильно "начинена" свеча — тело относительно полной длины.

        # Candle Direction (бинарная индикаторная фича)
        features['candle_direction'] = (df['close'] > df['open']).astype(int).shift(shift)
        # 1 — бычья свеча, 0 — медвежья.

        # High-Low / Open-Close ratio (стабилизированное соотношение диапазона к телу)
        hl_range = (df['high'] - df['low']).abs() + 1e-6
        oc_range = (df['close'] - df['open']).abs() + 1e-6
        features['hl_oc_ratio'] = (hl_range / oc_range).shift(shift)
        # Показывает степень "раздутости" тени относительно тела.

        # Open vs VWAP deviation (отклонение открытия от VWAP)
        features['open_vwap_deviation'] = ((df['open'] / df['vwap']) - 1).shift(shift)
        # Работает как индикатор гэпов и early momentum.

        # Momentum over N bars (абсолютное приращение закрытия за последние N баров)
        momentum_window = self._adjust(10)
        features['momentum_10'] = (df['close'] - df['close'].shift(momentum_window)).shift(shift)
        # Очень полезно как общий индикатор силы движения.

        # Volume Z-score (аномальные всплески объёма)
        vol_window = self._adjust(20)
        rolling_vol_mean = df['volume'].rolling(vol_window).mean()
        rolling_vol_std = df['volume'].rolling(vol_window).std() + 1e-6
        features['volume_zscore'] = ((df['volume'] - rolling_vol_mean) / rolling_vol_std).shift(shift)
        # Превращаем объём в нормированный score — позволяет лучше ловить выбросы.

        # Rolling Max-Min Stretch (расширение диапазона за N баров)
        stretch_window = self._adjust(20)
        rolling_max = df['high'].rolling(stretch_window).max()
        rolling_min = df['low'].rolling(stretch_window).min()
        features['stretch_range'] = (rolling_max - rolling_min).shift(shift)
        # Работает как хороший early breakout detector

        # Volatility Ratio (отношение краткосрочной и долгосрочной ATR)
        short_window = self._adjust(5)
        long_window = self._adjust(20)
        atr_short = AverageTrueRange(df['high'], df['low'], df['close'], window=short_window).average_true_range()
        atr_long = AverageTrueRange(df['high'], df['low'], df['close'], window=long_window).average_true_range()
        features['volatility_ratio'] = ((atr_short / (atr_long + 1e-6))).shift(shift + long_window - 1)
        # Прекрасный индикатор смены волатильностных режимов

        # Body Trend Consistency (число подряд идущих однонаправленных свечей)
        trend_direction = (df['close'] > df['open']).astype(int) * 2 - 1  # 1 если рост, -1 если падение
        trend_switch = trend_direction != trend_direction.shift(1)
        trend_group = trend_switch.cumsum()
        features['body_trend_consistency'] = trend_group.groupby(trend_group).cumcount().shift(shift)
        # Даёт маркер устойчивости текущей серии роста/падения

        # Volatility Regime Change (стабильность текущей std против rolling std)
        std_window = self._adjust(20)
        rolling_std = returns.rolling(std_window).std()
        features['volatility_regime_change'] = (rolling_std / (rolling_std.rolling(std_window).mean() + 1e-6)).shift(
            shift + std_window - 1)
        # Маркер смены турбулентности на рынке

        # Range Compression (степень сжатия диапазона)
        rolling_range = (df['high'] - df['low']).rolling(std_window).mean()
        features['range_compression'] = (rolling_std / (rolling_range + 1e-6)).shift(shift + std_window - 1)
        # Сильное сжатие часто предшествует выбросам

        # Opening Gap vs Range (гэп открытия относительно среднего диапазона)
        rolling_range_gap = (df['high'] - df['low']).rolling(stretch_window).mean()
        features['opening_gap_vs_range'] = (
                    (df['open'] - df['close'].shift(1)).abs() / (rolling_range_gap + 1e-6)).shift(
            shift + stretch_window - 1)
        # Хороший маркер силы утренних гэпов на больших таймфреймах

        # Финальный сбор признаков
        features_df = pd.DataFrame(features, index=df.index)

        # Очистка: убираем бесконечности и NaN
        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        features_df.dropna(inplace=True)

        # Сохраняем список колонок
        self.feature_columns = features_df.columns.tolist()

        # Масштабирование (стандартизация)
        X = features_df
        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return pd.DataFrame(X_scaled, columns=self.feature_columns, index=features_df.index)


    def to_sequences(self, df: pd.DataFrame, window_size: int = 40):
        """Преобразует датафрейм фичей в последовательности для подачи в модели"""
        arr = df[self.feature_columns].values
        X = []
        for i in range(len(arr) - window_size):
            X.append(arr[i:i + window_size])
        return np.array(X)
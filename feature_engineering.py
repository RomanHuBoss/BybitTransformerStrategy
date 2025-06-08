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
from scipy.stats import linregress

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.feature_columns = None
        self.scaler = None

    def load_scaler(self, mean, scale):
        self.scaler = StandardScaler()
        self.scaler.mean_ = mean
        self.scaler.scale_ = scale

    def _get_ema_angle(self, series, window=5):
        """Вычисляет угол наклона EMA в градусах."""
        y = series.values[-window:]
        x = np.arange(len(y))
        slope = linregress(x, y).slope
        return np.degrees(np.arctan(slope))

    def generate_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        # Проверка данных
        required_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

        df = df.copy()
        df['open_time'] = pd.to_datetime(df['open_time'])
        shift = CFG.train.lookahead

        # === 1. Временные фичи (Session + Microtime) ===
        df['hour'] = df['open_time'].dt.hour
        df['minute'] = df['open_time'].dt.minute
        df['is_high_impact_time'] = ((df['hour'].isin([14, 15])) & (df['minute'] == 0)).astype(int)  # Открытие NY/London
        df['day_of_week'] = df['open_time'].dt.dayofweek  # 0-6 (пн-вс)
        df['month'] = df['open_time'].dt.month  # 1-12
        df['season'] = (df['month'] % 12 + 3) // 3  # 1-4 (зима, весна, лето, осень)

        # === 2. Ценовые производные (Ultra-Short Term) ===
        df['log_return_1'] = np.log(df['close'] / df['close'].shift(1)).shift(shift)
        df['range_pct'] = (df['high'] - df['low']) / df['close'].shift(1)

        # Гэпы (разрыв между свечами)
        df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
        df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)

        # === 3. Микротренды (3-5 свечей) ===
        for window in [3, 5, 8]:
            df[f'linreg_angle_{window}'] = df['close'].rolling(window).apply(
                lambda x: np.degrees(np.arctan(linregress(np.arange(window), x).slope))
            ).shift(shift + window - 1)

        # === 4. Уровни (Smart Money Concepts) ===
        for window in [5, 15]:
            df[f'high_{window}'] = df['high'].rolling(window).max().shift(shift + 1)
            df[f'low_{window}'] = df['low'].rolling(window).min().shift(shift + 1)
            df[f'close_near_high_{window}'] = (df['close'] >= 0.997 * df[f'high_{window}']).astype(int)
            df[f'close_near_low_{window}'] = (df['close'] <= 1.003 * df[f'low_{window}']).astype(int)

        # === 5. Объемный анализ (Volume Profile) ===
        df['volume_zscore_10'] = (
                (df['volume'] - df['volume'].rolling(10).mean()) /
                (df['volume'].rolling(10).std() + 1e-6)
        ).shift(shift + 9)

        df['volume_ma_ratio_5_20'] = (
                df['volume'].rolling(5).mean() /
                df['volume'].rolling(20).mean()
        ).shift(shift + 1)

        # === 6. Волатильность (Intraday ATR) ===
        for window in [3, 8, 20]:
            atr = AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=window,
                fillna=False
            )
            df[f'atr_{window}_pct'] = (atr.average_true_range() / df['close'].shift(window - 1)).shift(
                shift + window - 1)

        # === 7. Осцилляторы (Оптимизированные для 15M) ===
        rsi = ta.momentum.RSIIndicator(close=df['close'], window=6, fillna=False)
        df['rsi_6'] = rsi.rsi().shift(shift + 5)

        df['stoch_5_3'] = ta.momentum.StochasticOscillator(
            df['high'], df['low'], df['close'], 5, 3
        ).stoch().shift(shift + 4)

        # MACD с короткими окнами
        df['macd_3_10_5'] = ta.trend.MACD(
            df['close'], window_slow=10, window_fast=3, window_sign=5, fillna=False
        ).macd_diff().shift(shift + 9)

        # === 8. Паттерны (Продвинутые) ===
        # 1. Пин-бар (Pinbar)
        df['is_pinbar'] = (
                ((df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low']) > 0.67) |
                ((df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low']) > 0.67)
        ).astype(int).shift(shift + 1)

        # 2. Внутридневной Inside Bar
        df['is_inside_bar'] = (
                (df['high'] < df['high'].shift(1)) &
                (df['low'] > df['low'].shift(1))
        ).astype(int).shift(shift + 1)

        # === 9. Кластерные фичи (Имитация Order Flow) ===
        df['buy_volume_ratio'] = (
                (df['close'] > df['open']).astype(int) * df['volume']
        ).rolling(5).mean().shift(shift)

        # === 10. Скрытые дивергенции ===
        df['rsi_divergence'] = (
                (df['close'] > df['close'].shift(2)) &
                (df['rsi_6'] < df['rsi_6'].shift(2))
        ).astype(int).shift(shift)

        # 11. Buy/Sell Volume Imbalance (Сдвиг + нормировка)
        df['buy_volume'] = (df['volume'] * (df['close'] > df['open'])).shift(1)  # Сдвиг!
        df['sell_volume'] = (df['volume'] * (df['close'] < df['open'])).shift(1)  # Сдвиг!
        df['volume_imbalance'] = (
                (df['buy_volume'].rolling(5, min_periods=1).sum() -
                 df['sell_volume'].rolling(5, min_periods=1).sum()
                 ) / (df['volume'].rolling(5, min_periods=1).sum() + 1e-6))

        # 22. Volume Spike Z-Score (Сдвиг + проверка на аномалии)
        df['volume_z'] = (
                (df['volume'] - df['volume'].rolling(50, min_periods=1).mean()) / (df['volume'].rolling(50, min_periods=1).std() + 1e-6)).shift(1)
        df['volume_spike_5'] = (df['volume_z'] > 3).astype(int)

        # 23. Volume Spikes (Pump/Dump Detection)
        df['volume_z'] = (df['volume'] - df['volume'].rolling(50).mean()) / (df['volume'].rolling(50).std() + 1e-6)
        df['volume_spike_5'] = (df['volume_z'] > 3).astype(int).shift(1)

        # 24. Crypto ATR (True Range с защитой от look-ahead)
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        ).shift(1)  # Сдвиг!
        df['atr_crypto_10'] = df['true_range'].rolling(10, min_periods=1).mean() / df['close'].shift(1)

        # 25. Liquidity Shock (Изменение ликвидности)
        df['liquidity_ratio'] = (
                (df['close'] - df['low']) /
                (df['high'] - df['low'] + 1e-6)
        ).shift(1)  # Сдвиг!
        df['liquidity_shock'] = (df['liquidity_ratio'].pct_change() > 0.5).astype(int)

        # 26. V-Shape Recovery (Паттерн "дно")
        df['v_shape_recovery'] = (
                (df['close'].shift(2) > df['close'].shift(1)) &  # 2 свечи назад > 1 свеча назад
                (df['close'].shift(1) > df['close'].shift(0)) &  # 1 свеча назад > текущая (сдвиг в будущее!)
                (df['volume'].shift(1) > df['volume'].rolling(10, min_periods=1).mean().shift(1))
        ).astype(int)

        # 27. Fakeout (Ложный пробой)
        df['fakeout_high_volume'] = (
                (df['high'].shift(1) > df['high'].shift(2)) &  # Сравниваем только прошлые данные
                (df['close'].shift(1) < df['open'].shift(1)) &
                (df['volume'].shift(1) > df['volume'].rolling(20, min_periods=1).mean().shift(1) * 1.5)
        ).astype(int)

        # 28. Whale Volume Clusters (Крупные ордера)
        df['whale_volume'] = (
            df['volume'].rolling(5, min_periods=1)
            .apply(lambda x: np.sum(x > x.mean() * 2))
            .shift(1)
        )

        # 29. Cumulative Delta (Накопленный дисбаланс)
        df['delta'] = ((df['close'] - df['open']) * df['volume']).shift(1)  # Сдвиг!
        df['cumulative_delta_15'] = df['delta'].rolling(15, min_periods=1).sum()

        # 30. Сессии (Asia, US, EU)
        df['asia_session'] = df['hour'].isin([2, 3, 4]).astype(int)  # Только текущий час
        df['us_session'] = df['hour'].isin([14, 15, 16]).astype(int)
        df['eu_session'] = df['hour'].isin([8, 9, 10]).astype(int)

        # 31. Weekend Effect (Сдвиг не нужен, это внешний фактор)
        df['is_weekend'] = (df['open_time'].dt.dayofweek >= 5).astype(int)

        # 32. Twitter Pump Signal (Аномалии объема и цены)
        df['twitter_pump_signal'] = (
                (df['volume'].shift(1) > df['volume'].rolling(50, min_periods=1).mean().shift(1) * 2) &
                (df['close'].shift(1).pct_change() > 0.05)
        ).astype(int)

        # 33. FOMO Indicator (Быстрое движение цены)
        df['fomo_5m'] = (
            df['close'].pct_change(3).abs()
            .rolling(5, min_periods=1).mean()
            .shift(1)
        )

        # 34. BTC Dominance Effect (Для альткоинов)
        df['btc_dominance_effect'] = (
                df['close'] / df['close'].rolling(20, min_periods=1).mean() - 1
        ).shift(1)

        # 35. Stablecoin Inflows Proxy
        df['usdt_volume_ratio'] = (
                df['volume'] / df['volume'].rolling(50, min_periods=1).mean()
        ).shift(1)

        # 36. Flash Crash Detection
        df['flash_crash'] = (
                (df['low'].pct_change() < -0.05) &
                (df['volume'] > df['volume'].rolling(20, min_periods=1).mean())
        ).astype(int).shift(1)

        # 37. Volatility Regime Switch
        df['volatility_regime'] = (
                df['atr_crypto_10'] > df['atr_crypto_10'].rolling(50, min_periods=1).mean() * 1.5
        ).astype(int).shift(1)


        # === Новые фичи (без lookahead, краткосрочные) ===

        # 38. Сила тела свечи
        df['body_strength'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-6)

        # 39. Смещение цены к верхней половине
        df['bias_up'] = (df['close'] > (df['high'] + df['low']) / 2).astype(int)

        # 41. EMA cross momentum
        ema_5 = df['close'].ewm(span=5, min_periods=5).mean()
        ema_10 = df['close'].ewm(span=10, min_periods=10).mean()
        df['ema_cross_momentum'] = ema_5 - ema_10

        # 42. Пробой волатильности (уровни в прошлом!)
        df['volatility_breakout'] = (
            (df['high'] > df['high'].rolling(20, min_periods=20).max().shift(1)) |
            (df['low'] < df['low'].rolling(20, min_periods=20).min().shift(1))
        ).astype(int)

        # 43. Отношения теней к телу (аналитика силы свечи)
        body = (df['close'] - df['open']).abs()
        upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
        lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
        df['upper_shadow_ratio'] = upper_shadow / (body + 1e-6)
        df['lower_shadow_ratio'] = lower_shadow / (body + 1e-6)

        # === 💡 Новые продвинутые фичи из Kaggle (2024–2025) ===
        # 44. Entropy of log returns (информационная насыщенность движения цены)
        returns = np.log(df['close'] / df['close'].shift(1))
        df['entropy_returns_15'] = returns.rolling(15).apply(
            lambda x: -np.sum(
                (p := np.histogram(x, bins=10, density=True)[0]) * np.log1p(p + 1e-6)
            ) if len(x.dropna()) == 15 else np.nan
        ).shift(CFG.train.lookahead + 14)

        # 45. Hurst Exponent (трендовость или шум)
        @njit
        def fast_hurst(ts):
            lags = np.arange(2, 20)
            tau = np.empty(len(lags))
            for i in range(len(lags)):
                lag = lags[i]
                tau[i] = np.std(ts[lag:] - ts[:-lag])

            log_lags = np.log(lags)
            log_tau = np.log(tau)

            # Оценка наклона (slope) вручную
            n = len(log_lags)
            sum_x = np.sum(log_lags)
            sum_y = np.sum(log_tau)
            sum_xx = np.sum(log_lags * log_lags)
            sum_xy = np.sum(log_lags * log_tau)

            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
            return slope

        df['hurst_30'] = df['close'].rolling(30).apply(fast_hurst, raw=True).shift(shift + 29)

        # 46. Bar Strength Ratio (текущая сила свечи к предыдущим)
        body = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-6)
        body_mean = (df['close'].rolling(5).mean() - df['open'].rolling(5).mean() + 1e-6)
        df['bar_strength_5'] = (body / body_mean).shift(CFG.train.lookahead + 4)

        # 47. DMI (Directional Movement Index) — трендовая сила
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx_14'] = adx.adx().shift(CFG.train.lookahead + 13)
        df['plus_di_14'] = adx.adx_pos().shift(CFG.train.lookahead + 13)
        df['minus_di_14'] = adx.adx_neg().shift(CFG.train.lookahead + 13)

        # 48. Real Body to Range Ratio (альтернатива body_strength)
        df['body_to_range'] = (
            (df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-6)
        ).shift(CFG.train.lookahead)

        # 49. Noise-to-Signal Ratio (волатильность vs истинный разброс)
        atr = ta.volatility.AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=20,
            fillna=False
        ).average_true_range()
        df['signal_noise_ratio'] = (
            df['close'].rolling(20).std() / (atr + 1e-6)
        ).shift(CFG.train.lookahead + 19)

        # === Финализация ===
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        self.feature_columns = [col for col in df.columns if col not in [
            'open_time', 'open', 'high', 'low', 'close', 'volume', 'hour', 'minute'
        ]]

        # Масштабирование
        X = df[self.feature_columns]
        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return pd.DataFrame(X_scaled, columns=self.feature_columns, index=df.index)


    def to_sequences(self, df: pd.DataFrame, window_size: int = 40):
        """Преобразует в последовательности для LSTM/CNN."""
        arr = df[self.feature_columns].values
        X = []
        for i in range(len(arr) - window_size):
            X.append(arr[i:i + window_size])
        return np.array(X)
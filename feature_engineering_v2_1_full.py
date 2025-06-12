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

        # Приведение типов
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        # Временные признаки
        if use_logging: logging.info("Генерация временных фичей...")
        df['hour'] = df['open_time'].dt.hour
        df['minute'] = df['open_time'].dt.minute
        df['day_of_week'] = df['open_time'].dt.dayofweek
        df['month'] = df['open_time'].dt.month
        df['season'] = (df['month'] % 12 + 3) // 3

        # Hour (24 часа)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Minute (60 минут)
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

        # Day of week (7 дней)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Month (12 месяцев)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Season (4 сезона)
        df['season_sin'] = np.sin(2 * np.pi * df['season'] / 4)
        df['season_cos'] = np.cos(2 * np.pi * df['season'] / 4)

        # Базовые ценовые производные
        if use_logging: logging.info("Генерация базовых производных...")
        df['log_return_1'] = np.log(df['close'] / df['close'].shift(1)).shift(shift)
        df['range_pct'] = (df['high'] - df['low']) / df['close'].shift(1)
        df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
        df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)

        # Осцилляторы
        if use_logging: logging.info("Генерация осцилляторов...")
        rsi_window = self._adjust(6)
        rsi = ta.momentum.RSIIndicator(close=df['close'], window=rsi_window)
        df['rsi_short'] = rsi.rsi().shift(shift + rsi_window - 1)

        stoch_window = self._adjust(5)
        df['stoch_short'] = ta.momentum.StochasticOscillator(
            high=df['high'], low=df['low'], close=df['close'],
            window=stoch_window, smooth_window=3
        ).stoch().shift(shift + stoch_window - 1)

        df['macd_fast'] = ta.trend.MACD(
            close=df['close'], window_slow=10, window_fast=3, window_sign=5
        ).macd_diff().shift(shift + 9)

        # Скрытые дивергенции (теперь безопасно вызываем rsi_short)
        if use_logging: logging.info("Генерация скрытых дивергенций...")
        df['rsi_divergence'] = (
            (df['close'] > df['close'].shift(2)) &
            (df['rsi_short'] < df['rsi_short'].shift(2))
        ).astype(int).shift(shift)


        # Микротренды (линейная регрессия углов)
        if use_logging: logging.info("Генерация микротрендов...")
        for label, w in [('short', 3), ('mid', 5), ('long', 8)]:
            window = self._adjust(w)
            df[f'linreg_angle_{label}'] = df['close'].rolling(window).apply(
                lambda x: self._get_linreg_angle(x, window)
            ).shift(shift + window - 1)

        # Уровни поддержки и сопротивления
        if use_logging: logging.info("Генерация уровней...")
        for label, w in [('short', 5), ('mid', 15)]:
            window = self._adjust(w)
            df[f'high_{label}'] = df['high'].rolling(window).max().shift(shift + 1)
            df[f'low_{label}'] = df['low'].rolling(window).min().shift(shift + 1)
            df[f'close_near_high_{label}'] = (df['close'] >= 0.997 * df[f'high_{label}']).astype(int)
            df[f'close_near_low_{label}'] = (df['close'] <= 1.003 * df[f'low_{label}']).astype(int)

        # Кластерные объемные фичи
        if use_logging: logging.info("Генерация кластерных объемных фич...")
        vol_window_10 = self._adjust(10)
        vol_window_5 = self._adjust(5)
        vol_window_20 = self._adjust(20)

        df['volume_zscore_short'] = (
            (df['volume'] - df['volume'].rolling(vol_window_10).mean()) /
            (df['volume'].rolling(vol_window_10).std() + 1e-6)
        ).shift(shift + vol_window_10 - 1)

        df['volume_ma_ratio_short'] = (
            df['volume'].rolling(vol_window_5).mean() /
            df['volume'].rolling(vol_window_20).mean()
        ).shift(shift + 1)

        # Волатильность (ATR)
        if use_logging: logging.info("Генерация волатильности (ATR)...")
        for label, w in [('short', 3), ('mid', 8), ('long', 20)]:
            window = self._adjust(w)
            atr = AverageTrueRange(df['high'], df['low'], df['close'], window=window)
            df[f'atr_{label}_pct'] = (atr.average_true_range() / df['close'].shift(window - 1)).shift(shift + window - 1)

        # Booster Pack: VWAP deviation
        if use_logging: logging.info("Booster Pack: VWAP deviation...")
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['vwap_deviation'] = (df['close'] / df['vwap'] - 1).shift(shift)

        # Booster Pack: Rolling skewness
        if use_logging: logging.info("Booster Pack: Rolling skewness...")
        returns = np.log(df['close'] / df['close'].shift(1))
        skew_window = self._adjust(20)
        df['returns_skew'] = returns.rolling(skew_window).apply(lambda x: skew(x)).shift(shift + skew_window - 1)

        # Booster Pack: Rolling kurtosis
        if use_logging: logging.info("Booster Pack: Rolling kurtosis...")
        kurt_window = self._adjust(20)
        df['returns_kurtosis'] = returns.rolling(kurt_window).apply(lambda x: kurtosis(x)).shift(shift + kurt_window - 1)

        # Booster Pack: Trend Acceleration
        if use_logging: logging.info("Booster Pack: Trend Acceleration...")
        accel_window = self._adjust(10)
        df['trend_slope'] = df['close'].rolling(accel_window).apply(
            lambda x: linregress(np.arange(len(x)), x).slope
        ).shift(shift + accel_window - 1)

        df['trend_acceleration'] = df['trend_slope'].diff().shift(shift)

        # Booster Pack: Fractal proxy
        if use_logging: logging.info("Booster Pack: Fractal proxy...")
        fractal_window = self._adjust(15)
        df['fractal_dimension_proxy'] = returns.rolling(fractal_window).std().shift(shift + fractal_window - 1)

        # Паттерны свечей
        if use_logging: logging.info("Генерация паттернов свечей...")
        df['is_pinbar'] = (
            ((df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low']) > 0.67) |
            ((df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low']) > 0.67)
        ).astype(int).shift(shift + 1)

        df['is_inside_bar'] = (
            (df['high'] < df['high'].shift(1)) &
            (df['low'] > df['low'].shift(1))
        ).astype(int).shift(shift + 1)

        # Order Flow Imbalance
        if use_logging: logging.info("Генерация объёмного имбаланса (order flow)...")
        df['buy_volume'] = (df['volume'] * (df['close'] > df['open'])).shift(1)
        df['sell_volume'] = (df['volume'] * (df['close'] < df['open'])).shift(1)

        df['volume_imbalance'] = (
            (df['buy_volume'].rolling(self._adjust(5), min_periods=1).sum() -
             df['sell_volume'].rolling(self._adjust(5), min_periods=1).sum()) /
            (df['volume'].rolling(self._adjust(5), min_periods=1).sum() + 1e-6)
        )

        # Volume spike
        if use_logging: logging.info("Генерация volume spike...")
        vol_roll_50 = self._adjust(50)
        df['volume_z'] = (
            (df['volume'] - df['volume'].rolling(vol_roll_50).mean()) /
            (df['volume'].rolling(vol_roll_50).std() + 1e-6)
        ).shift(1)
        df['volume_spike'] = (df['volume_z'] > 3).astype(int)

        # Cumulative Delta
        if use_logging: logging.info("Генерация Cumulative Delta...")
        delta_window = self._adjust(15)
        df['delta'] = ((df['close'] - df['open']) * df['volume']).shift(1)
        df['cumulative_delta'] = df['delta'].rolling(delta_window, min_periods=1).sum()

        # True Range и ATR (альтернативный)
        if use_logging: logging.info("Генерация true range и atr_crypto...")
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        ).shift(1)

        atr_crypto_window = self._adjust(10)
        df['atr_crypto_short'] = (
            df['true_range'].rolling(atr_crypto_window, min_periods=1).mean() / df['close'].shift(1)
        )

        # Liquidity Shock
        if use_logging: logging.info("Генерация liquidity shock...")
        df['liquidity_ratio'] = (
            (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-6)
        ).shift(1)
        df['liquidity_shock'] = (df['liquidity_ratio'].pct_change() > 0.5).astype(int)

        # V-Shape Recovery
        if use_logging: logging.info("Генерация V-Shape Recovery...")
        df['v_shape_recovery'] = (
            (df['close'].shift(2) > df['close'].shift(1)) &
            (df['close'].shift(1) > df['close']) &
            (df['volume'].shift(1) > df['volume'].rolling(self._adjust(10)).mean().shift(1))
        ).astype(int)

        # Fakeout
        if use_logging: logging.info("Генерация Fakeout...")
        df['fakeout_high_volume'] = (
            (df['high'].shift(1) > df['high'].shift(2)) &
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['volume'].shift(1) > df['volume'].rolling(self._adjust(20)).mean().shift(1) * 1.5)
        ).astype(int)

        # Whale Volume Clusters
        if use_logging: logging.info("Генерация Whale Volume Clusters...")
        whale_window = self._adjust(5)
        df['whale_volume'] = (
            df['volume'].rolling(whale_window, min_periods=1)
            .apply(lambda x: np.sum(x > x.mean() * 2))
            .shift(1)
        )

        # Flash Crash Detection
        if use_logging: logging.info("Генерация Flash Crash...")
        df['flash_crash'] = (
            (df['low'].pct_change() < -0.05) &
            (df['volume'] > df['volume'].rolling(self._adjust(20), min_periods=1).mean())
        ).astype(int)

        # Volatility Regime Switch
        if use_logging: logging.info("Генерация Volatility Regime Switch...")
        df['volatility_regime'] = (
            df['atr_crypto_short'] > df['atr_crypto_short'].rolling(self._adjust(50), min_periods=1).mean() * 1.5
        ).astype(int)

        # Body Strength
        if use_logging: logging.info("Генерация body strength...")
        df['body_strength'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-6)
        df['bias_up'] = (df['close'] > (df['high'] + df['low']) / 2).astype(int)

        # EMA Cross Momentum
        if use_logging: logging.info("Генерация EMA cross momentum...")
        ema_5 = df['close'].ewm(span=self._adjust(5), min_periods=self._adjust(5)).mean()
        ema_10 = df['close'].ewm(span=self._adjust(10), min_periods=self._adjust(10)).mean()
        df['ema_cross_momentum'] = ema_5 - ema_10

        # Volatility Breakout
        if use_logging: logging.info("Генерация volatility breakout...")
        df['volatility_breakout'] = (
            (df['high'] > df['high'].rolling(self._adjust(20), min_periods=20).max().shift(1)) |
            (df['low'] < df['low'].rolling(self._adjust(20), min_periods=20).min().shift(1))
        ).astype(int)

        # Shadow Ratios
        if use_logging: logging.info("Генерация shadow ratios...")
        body = (df['close'] - df['open']).abs()
        upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
        lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
        df['upper_shadow_ratio'] = upper_shadow / (body + 1e-6)
        df['lower_shadow_ratio'] = lower_shadow / (body + 1e-6)

        # Entropy of log returns
        if use_logging: logging.info("Генерация entropy_returns...")
        entropy_window = self._adjust(15)
        df['entropy_returns'] = returns.rolling(entropy_window).apply(
            lambda x: -np.sum((p := np.histogram(x, bins=10, density=True)[0]) * np.log1p(p + 1e-6))
            if len(x.dropna()) == entropy_window else np.nan
        ).shift(shift + entropy_window - 1)

        # Hurst Exponent
        if use_logging: logging.info("Генерация Hurst exponent...")
        @njit
        def fast_hurst(ts):
            lags = np.arange(2, 20)
            tau = np.empty(len(lags))
            for i in range(len(lags)):
                lag = lags[i]
                tau[i] = np.std(ts[lag:] - ts[:-lag])
            log_lags = np.log(lags)
            log_tau = np.log(tau)
            n = len(log_lags)
            sum_x = np.sum(log_lags)
            sum_y = np.sum(log_tau)
            sum_xx = np.sum(log_lags * log_lags)
            sum_xy = np.sum(log_lags * log_tau)
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
            return slope

        hurst_window = self._adjust(30)
        df['hurst_exponent'] = df['close'].rolling(hurst_window).apply(fast_hurst, raw=True).shift(shift + hurst_window - 1)

        # Bar Strength Ratio
        if use_logging: logging.info("Генерация Bar Strength Ratio...")
        body_mean = (
            df['close'].rolling(self._adjust(5)).mean() - df['open'].rolling(self._adjust(5)).mean() + 1e-6
        )
        df['bar_strength'] = (body / body_mean).shift(shift + self._adjust(5) - 1)

        # ADX (DMI)
        if use_logging: logging.info("Генерация DMI (ADX)...")
        adx_window = self._adjust(14)
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=adx_window)
        df['adx'] = adx.adx().shift(shift + adx_window - 1)
        df['plus_di'] = adx.adx_pos().shift(shift + adx_window - 1)
        df['minus_di'] = adx.adx_neg().shift(shift + adx_window - 1)

        # Body-to-Range Ratio
        if use_logging: logging.info("Генерация Body-to-Range Ratio...")
        df['body_to_range'] = ((df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-6)).shift(shift)

        # Noise-to-Signal Ratio
        if use_logging: logging.info("Генерация Noise-to-Signal Ratio...")
        atr_final = AverageTrueRange(df['high'], df['low'], df['close'], window=self._adjust(20)).average_true_range()
        df['signal_noise_ratio'] = (
            df['close'].rolling(self._adjust(20)).std() / (atr_final + 1e-6)
        ).shift(shift + self._adjust(20) - 1)

        if use_logging: logging.info("Генерация micro_atr...")
        micro_window = self._adjust(5)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['micro_atr'] = tr.rolling(micro_window).mean().shift(shift + micro_window - 1)

        if use_logging: logging.info("Генерация volatility_burst...")
        fast_window = self._adjust(5)
        slow_window = self._adjust(20)
        high_low = df['high'] - df['low']
        fast_vol = high_low.rolling(fast_window).mean()
        slow_vol = high_low.rolling(slow_window).mean()
        df['volatility_burst'] = (fast_vol / (slow_vol + 1e-8)).shift(shift + slow_window - 1)

        if use_logging: logging.info("Генерация shadow_ratios (precision boost)...")
        body = (df['close'] - df['open']).abs()
        upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
        lower_shadow = df[['close', 'open']].min(axis=1) - df['low']

        df['upper_shadow_ratio_boost'] = (upper_shadow / (body + 1e-8)).shift(shift)
        df['lower_shadow_ratio_boost'] = (lower_shadow / (body + 1e-8)).shift(shift)
        df['body_to_range_ratio_boost'] = (body / (df['high'] - df['low'] + 1e-8)).shift(shift)

        if use_logging: logging.info("Генерация volume_spike_boost...")
        volume_window = self._adjust(20)
        rolling_mean_vol = df['volume'].rolling(volume_window).mean()
        df['volume_spike_boost'] = (df['volume'] / (rolling_mean_vol + 1e-8)).shift(shift + volume_window - 1)

        if use_logging: logging.info("Генерация volatility_breakout_boost...")
        breakout_window = self._adjust(20)
        rolling_high = df['high'].rolling(breakout_window).max()
        rolling_low = df['low'].rolling(breakout_window).min()

        df['volatility_breakout_up_boost'] = (df['close'] > rolling_high.shift(1)).astype(int).shift(shift)
        df['volatility_breakout_down_boost'] = (df['close'] < rolling_low.shift(1)).astype(int).shift(shift)

        if use_logging: logging.info("V2.2: Генерация swing high/low...")
        swing_window = self._adjust(5)
        df['swing_high'] = df['high'].rolling(swing_window, center=True).max()
        df['swing_low'] = df['low'].rolling(swing_window, center=True).min()
        df['is_swing_high'] = (df['high'] >= df['swing_high']).astype(int).shift(shift)
        df['is_swing_low'] = (df['low'] <= df['swing_low']).astype(int).shift(shift)

        if use_logging: logging.info("V2.2: Генерация price squeeze zones...")
        boll_window = self._adjust(20)
        boll = ta.volatility.BollingerBands(close=df['close'], window=boll_window, window_dev=2)
        df['bollinger_width'] = (boll.bollinger_hband() - boll.bollinger_lband()) / df['close']
        df['price_squeeze'] = (df['bollinger_width'] < df['bollinger_width'].rolling(boll_window).mean() * 0.7).astype(
            int).shift(shift)

        if use_logging: logging.info("V2.2: Генерация 2-bar/3-bar reversal patterns...")
        df['two_bar_bullish'] = (
                (df['close'].shift(2) < df['open'].shift(2)) &
                (df['close'].shift(1) > df['open'].shift(1)) &
                (df['close'].shift(1) > df['close'].shift(2))
        ).astype(int).shift(shift)

        df['two_bar_bearish'] = (
                (df['close'].shift(2) > df['open'].shift(2)) &
                (df['close'].shift(1) < df['open'].shift(1)) &
                (df['close'].shift(1) < df['close'].shift(2))
        ).astype(int).shift(shift)

        df['three_bar_reversal'] = (
                (df['high'].shift(2) < df['high'].shift(1)) &
                (df['high'] > df['high'].shift(1)) &
                (df['low'].shift(2) > df['low'].shift(1)) &
                (df['low'] < df['low'].shift(1))
        ).astype(int).shift(shift)

        if use_logging: logging.info("V2.2: Генерация candle body position...")
        body = (df['close'] - df['open'])
        df['body_position'] = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-6)).shift(shift)

        if use_logging: logging.info("V2.2: Генерация trend duration...")
        trend = np.sign(df['close'] - df['open'])
        df['trend_duration'] = trend.groupby((trend != trend.shift()).cumsum()).cumcount() + 1
        df['trend_duration'] = df['trend_duration'].shift(shift)

        # Финальная очистка и нормализация
        if use_logging: logging.info("Финализация фичей, очистка данных...")
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        cols_to_drop = ['hour', 'minute', 'day_of_week', 'month', 'season']
        df = df.drop(columns=cols_to_drop)

        self.feature_columns = [
            col for col in df.columns if col not in ['open_time', 'open', 'high', 'low', 'close', 'volume']
        ]

        X = df[self.feature_columns]
        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return pd.DataFrame(X_scaled, columns=self.feature_columns, index=df.index)

    def to_sequences(self, df: pd.DataFrame, window_size: int = 40):
        """Преобразует датафрейм фичей в последовательности для подачи в модели"""
        arr = df[self.feature_columns].values
        X = []
        for i in range(len(arr) - window_size):
            X.append(arr[i:i + window_size])
        return np.array(X)
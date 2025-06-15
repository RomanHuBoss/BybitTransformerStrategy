import joblib
import pandas as pd
import numpy as np
import ta
from scipy.stats import kurtosis, skew, linregress
from sklearn.preprocessing import StandardScaler
from ta.volatility import AverageTrueRange
from ta.momentum import StochasticOscillator
from config import CFG
import logging
import os


class FeatureEngineer:
    def __init__(self):
        self.feature_columns = None
        self.scaler = None

        # Попробуем сразу загрузить признаки из файла, если он существует
        if os.path.exists(CFG.paths.feature_columns_path):
            self.feature_columns = joblib.load(CFG.paths.feature_columns_path)

        # Загружаем scaler, если есть
        if os.path.exists(CFG.paths.scaler_path):
            self.scaler = joblib.load(CFG.paths.scaler_path)

    def _adjust(self, base_window):
        return max(2, int(base_window * CFG.assets.timeframe / 30))

    def generate_features(self, df: pd.DataFrame, fit: bool = False, use_logging: bool = True) -> pd.DataFrame:
        required_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

        df = df.copy()
        df['open_time'] = pd.to_datetime(df['open_time'])
        shift = CFG.feature_engineering.default_shift

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

        # Returns & Volatility Core
        returns = np.log(df['close'] / df['close'].shift(1))
        features['log_return_1'] = returns.shift(shift)
        features['returns_mean'] = returns.rolling(self._adjust(10)).mean().shift(shift)
        features['returns_std'] = returns.rolling(self._adjust(10)).std().shift(shift)
        features['returns_skew'] = returns.rolling(self._adjust(20)).apply(lambda x: skew(x)).shift(shift)
        features['returns_kurtosis'] = returns.rolling(self._adjust(20)).apply(lambda x: kurtosis(x)).shift(shift)
        features['range_pct'] = (df['high'] - df['low']) / df['close'].shift(1)

        # ATR (short, mid, long)
        for label, w in [('short', 3), ('mid', 8), ('long', 20)]:
            window = self._adjust(w)
            atr = AverageTrueRange(df['high'], df['low'], df['close'], window=window)
            features[f'atr_{label}_pct'] = (atr.average_true_range() / df['close'].shift(window - 1)).shift(shift + window - 1)

        # VWAP deviation
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        features['vwap_deviation'] = (df['close'] / df['vwap'] - 1).shift(shift)

        # Momentum over N bars
        momentum_window = self._adjust(10)
        features['momentum_10'] = (df['close'] - df['close'].shift(momentum_window)).shift(shift)

        # Entropy of returns
        entropy_window = self._adjust(15)
        features['entropy_returns'] = returns.rolling(entropy_window).apply(
            lambda x: -np.sum((p := np.histogram(x, bins=10, density=True)[0]) * np.log1p(p + 1e-6))
            if len(x.dropna()) == entropy_window else np.nan
        ).shift(shift + entropy_window - 1)

        # Volume z-score
        vol_window = self._adjust(20)
        rolling_vol_mean = df['volume'].rolling(vol_window).mean()
        rolling_vol_std = df['volume'].rolling(vol_window).std() + 1e-6
        features['volume_zscore'] = ((df['volume'] - rolling_vol_mean) / rolling_vol_std).shift(shift)

        # Breakout detectors
        breakout_window = self._adjust(20)
        rolling_high = df['high'].rolling(breakout_window).max()
        rolling_low = df['low'].rolling(breakout_window).min()
        features['volatility_breakout_up_boost'] = (df['close'] > rolling_high.shift(1)).astype(int).shift(shift)
        features['volatility_breakout_down_boost'] = (df['close'] < rolling_low.shift(1)).astype(int).shift(shift)
        features['stretch_range'] = (rolling_high - rolling_low).shift(shift)

        # Volatility regime switching (ATR ratio)
        short_window = self._adjust(5)
        long_window = self._adjust(20)
        atr_short = AverageTrueRange(df['high'], df['low'], df['close'], window=short_window).average_true_range()
        atr_long = AverageTrueRange(df['high'], df['low'], df['close'], window=long_window).average_true_range()
        features['volatility_ratio'] = (atr_short / (atr_long + 1e-6)).shift(shift + long_window - 1)

        # Compression regimes
        std_window = self._adjust(20)
        rolling_std = returns.rolling(std_window).std()
        rolling_range = (df['high'] - df['low']).rolling(std_window).mean()
        features['range_compression'] = (rolling_std / (rolling_range + 1e-6)).shift(shift + std_window - 1)

        # Opening gap
        rolling_range_gap = (df['high'] - df['low']).rolling(breakout_window).mean()
        features['opening_gap_vs_range'] = ((df['open'] - df['close'].shift(1)).abs() / (rolling_range_gap + 1e-6)).shift(shift + breakout_window - 1)

        # Trend slope & acceleration
        accel_window = self._adjust(10)
        trend_slope = df['close'].rolling(accel_window).apply(
            lambda x: linregress(np.arange(len(x)), x).slope).shift(shift + accel_window - 1)
        features['trend_slope'] = trend_slope
        features['trend_acceleration'] = trend_slope.diff().shift(shift)

        # EMA Cross Momentum
        ema_5 = df['close'].ewm(span=self._adjust(5), min_periods=self._adjust(5)).mean()
        ema_10 = df['close'].ewm(span=self._adjust(10), min_periods=self._adjust(10)).mean()
        features['ema_cross_momentum'] = (ema_5 - ema_10).shift(shift)

        # Micro ATR
        micro_window = self._adjust(5)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['micro_atr'] = tr.rolling(micro_window).mean().shift(shift + micro_window - 1)

        # Shadow ratios
        body = (df['close'] - df['open']).abs()
        upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
        lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
        features['upper_shadow_ratio'] = (upper_shadow / (body + 1e-6)).shift(shift)
        features['lower_shadow_ratio'] = (lower_shadow / (body + 1e-6)).shift(shift)

        # Fractal Swing High/Low
        swing_high = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        swing_low = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
        fractal_high = df['high'].where(swing_high).ffill()
        fractal_low = df['low'].where(swing_low).ffill()
        features['fractal_width'] = (fractal_high - fractal_low).shift(shift)

        # Feature crossing interaction (V4.0 meta-cross)
        features['volatility_momentum_cross'] = (features['returns_std'] * features['momentum_10']).shift(shift)
        features['atr_vwap_cross'] = (features['atr_long_pct'] * features['vwap_deviation']).shift(shift)
        features['volatility_regime_cross'] = (features['volatility_ratio'] * features['range_compression']).shift(shift)

        upper_tail = df['high'] - df[['close', 'open']].max(axis=1)
        lower_tail = df[['close', 'open']].min(axis=1) - df['low']
        features['tail_asymmetry'] = ((upper_tail - lower_tail) / (upper_tail + lower_tail + 1e-6)).shift(shift)
        features['shadow_tail_cross'] = (features['upper_shadow_ratio'] - features['lower_shadow_ratio']) * features['tail_asymmetry']

        # Directional Bias Regime Filter
        rolling_mean_window = self._adjust(20)
        rolling_mean_price = df['close'].rolling(rolling_mean_window).mean()
        features['directional_bias'] = ((df['close'] - rolling_mean_price) / (rolling_mean_price + 1e-6)).shift(shift)

        # Volatility state regime
        features['volatility_state'] = (features['returns_std'] > features['returns_std'].rolling(100).median()).astype(int).shift(shift)
        features['volume_state'] = (df['volume'] > df['volume'].rolling(100).median()).astype(int).shift(shift)
        features['risk_imbalance'] = (features['atr_long_pct'] / (features['fractal_width'] + 1e-6)).shift(shift)

        # RSI и Stochastic
        rsi_window = self._adjust(6)
        rsi = ta.momentum.RSIIndicator(close=df['close'], window=rsi_window)
        features['rsi_short'] = rsi.rsi().shift(shift + rsi_window - 1)

        stoch_window = self._adjust(5)
        stoch = ta.momentum.StochasticOscillator(
            high=df['high'], low=df['low'], close=df['close'], window=stoch_window, smooth_window=3)
        features['stoch_short'] = stoch.stoch().shift(shift + stoch_window - 1)

        # Hidden RSI Divergence
        if use_logging: logging.info("Генерация скрытых дивергенций...")
        features['rsi_divergence'] = (
            (df['close'] > df['close'].shift(2)) & (features['rsi_short'] < features['rsi_short'].shift(2))
        ).astype(int).shift(shift)

        # Candlestick engulfing patterns
        features['bullish_engulfing'] = (
            (df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1)) &
            (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))
        ).astype(int).shift(shift)

        features['bearish_engulfing'] = (
            (df['close'] < df['open']) & (df['close'].shift(1) > df['open'].shift(1)) &
            (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1))
        ).astype(int).shift(shift)

        # === FEATURE BOOSTING BLOCK ===

        # Return cascade (multi-lag returns)
        for lag in [3, 5, 10, 20, 30]:
            features[f'return_lag_{lag}'] = np.log(df['close'] / df['close'].shift(lag)).shift(shift)

        # Volatility clusters
        for window in [5, 10, 20, 50]:
            features[f'vol_std_{window}'] = df['close'].rolling(window).std().shift(shift)

        features['vol_ratio_5_20'] = (features['vol_std_5'] / (features['vol_std_20'] + 1e-8)).shift(shift)
        features['vol_ratio_10_50'] = (features['vol_std_10'] / (features['vol_std_50'] + 1e-8)).shift(shift)

        # Momentum indicators
        features['momentum_10'] = (df['close'] - df['close'].shift(10)).shift(shift)

        # RSI + StochRSI
        rsi_window = 14
        stoch_window = 14
        features['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=rsi_window).rsi().shift(shift + rsi_window - 1)
        features['stoch_rsi'] = ta.momentum.stochrsi(df['close'], window=stoch_window).shift(shift + stoch_window - 1)

        # CCI, Williams %R
        features['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20).shift(shift + 20 - 1)
        features['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14).shift(shift + 14 - 1)

        # EMA cross
        ema_fast = df['close'].ewm(span=10).mean()
        ema_slow = df['close'].ewm(span=50).mean()
        features['ema_fast'] = ema_fast.shift(shift)
        features['ema_slow'] = ema_slow.shift(shift)
        features['ema_diff'] = (ema_fast - ema_slow).shift(shift)
        features['ema_ratio'] = (ema_fast / (ema_slow + 1e-8)).shift(shift)

        # Candle structure
        body = (df['close'] - df['open']).shift(shift)
        upper_shadow = (df['high'] - df[['close', 'open']].max(axis=1)).shift(shift)
        lower_shadow = (df[['close', 'open']].min(axis=1) - df['low']).shift(shift)
        features['candle_body'] = body
        features['upper_shadow'] = upper_shadow
        features['lower_shadow'] = lower_shadow
        features['body_to_range'] = (body / (df['high'] - df['low'] + 1e-8)).shift(shift)

        # Cross interaction terms
        features['rsi_over_vol'] = (features['rsi_14'] / (features['vol_std_10'] + 1e-8)).shift(shift)
        features['momentum_over_vol'] = (features['momentum_10'] / (features['vol_std_10'] + 1e-8)).shift(shift)
        features['ema_diff_over_vol'] = (features['ema_diff'] / (features['vol_std_10'] + 1e-8)).shift(shift)

        # Hidden Hilbert phase (если scipy доступен)
        try:
            from scipy.signal import hilbert
            analytic_signal = hilbert(df['close'].ffill().values)
            hilbert_phase = np.angle(analytic_signal)
            features['hilbert_phase'] = pd.Series(hilbert_phase, index=df.index).shift(shift)
        except Exception as e:
            features['hilbert_phase'] = np.nan  # fallback

        # Финальный сбор признаков в DataFrame
        features_df = pd.DataFrame(features, index=df.index)

        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        features_df.dropna(inplace=True)

        # === На этапе fit: сохраняем оригинальные фичи ===
        if fit:
            self.feature_columns = features_df.columns.tolist()
            joblib.dump(self.feature_columns, CFG.paths.feature_columns_path)
        else:
            # === На инференсе: восстанавливаем отсутствующие фичи ===
            for col in self.feature_columns:
                if col not in features_df.columns:
                    features_df[col] = np.nan

            features_df = features_df[self.feature_columns]
            features_df.fillna(0, inplace=True)

        # Стандартизация
        X = features_df
        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return pd.DataFrame(X_scaled, columns=self.feature_columns, index=features_df.index)

    def forward(self, X: pd.DataFrame, window_size: int = 100) -> np.ndarray:
        data = []
        for i in range(window_size, len(X)):
            window = X.iloc[i - window_size:i].values
            data.append(window)
        return np.array(data)

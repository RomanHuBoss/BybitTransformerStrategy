# amplitude_label_generator.py
import numpy as np
import pandas as pd

class AmplitudeLabelGenerator:
    """
    Генератор амплитудных таргетов:
    """
    def __init__(self, lookahead: int):
        self.lookahead = lookahead

    def generate_labels(self, df: pd.DataFrame) -> np.ndarray:
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        N = len(close)
        amplitudes = np.full(N, np.nan)

        for i in range(N - self.lookahead):
            entry_price = close[i]
            future_high = np.max(high[i+1:i+1+self.lookahead])
            future_low = np.min(low[i+1:i+1+self.lookahead])

            amplitude = max(abs(future_high - entry_price), abs(future_low - entry_price))
            amplitude_pct = amplitude / entry_price
            amplitudes[i] = amplitude_pct

        return amplitudes

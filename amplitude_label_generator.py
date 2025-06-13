import numpy as np
import pandas as pd

class AmplitudeLabelGenerator:
    """
    Генератор амплитудных таргетов для amplitude регрессии.
    """

    def __init__(self, lookahead: int):
        self.lookahead = lookahead

    def generate_amplitude_labels(self, df: pd.DataFrame) -> np.ndarray:
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        N = len(close)
        amplitudes = np.full(N, np.nan)

        for i in range(N - self.lookahead):
            entry_price = close[i]
            high_future = high[i+1 : i+1+self.lookahead]
            low_future = low[i+1 : i+1+self.lookahead]

            max_future_move = max(np.max(high_future) - entry_price, entry_price - np.min(low_future))
            amplitudes[i] = max_future_move / entry_price

        return amplitudes

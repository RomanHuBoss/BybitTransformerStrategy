import numpy as np
import pandas as pd

class AmplitudeLabelGenerator:
    """
    Генератор раздельных амплитудных меток: вверх и вниз.
    """
    def __init__(self, lookahead: int):
        self.lookahead = lookahead

    def generate_amplitude_labels(self, df: pd.DataFrame):
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        N = len(close)
        up_targets = np.full(N, np.nan)
        down_targets = np.full(N, np.nan)

        for i in range(N - self.lookahead):
            entry_price = close[i]
            high_future = high[i+1 : i+1+self.lookahead]
            low_future = low[i+1 : i+1+self.lookahead]

            up_move = np.max(high_future) - entry_price
            down_move = entry_price - np.min(low_future)

            up_targets[i] = up_move / entry_price
            down_targets[i] = down_move / entry_price

        return up_targets, down_targets

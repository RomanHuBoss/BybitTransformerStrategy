import numpy as np
import pandas as pd
from config import CFG

class DirectionalLabelGenerator:
    """
    Генерация меток {0=short, 1=no-trade, 2=long} для каждой TP/SL пары.
    """

    def __init__(self, tp_sl_levels, lookahead):
        if not tp_sl_levels:
            raise RuntimeError("Нужен список TP/SL.")
        if lookahead is None:
            raise RuntimeError("Параметр lookahead обязателен.")

        self.tp_sl_levels = tp_sl_levels
        self.lookahead = lookahead

    def generate_labels(self, df: pd.DataFrame):
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        N = len(close)
        num_pairs = len(self.tp_sl_levels)
        Y = np.full((N, num_pairs), CFG.action2label["no-trade"], dtype=int)

        for i in range(N - self.lookahead):
            entry = close[i]
            high_future = high[i + 1 : i + 1 + self.lookahead]
            low_future = low[i + 1 : i + 1 + self.lookahead]

            for j, (tp, sl) in enumerate(self.tp_sl_levels):
                tp_long, sl_long = entry * (1 + tp), entry * (1 - sl)
                tp_short, sl_short = entry * (1 - tp), entry * (1 + sl)

                long_hit = self._check_hit(high_future, low_future, tp_long, sl_long, mode="long")
                short_hit = self._check_hit(high_future, low_future, tp_short, sl_short, mode="short")

                if long_hit == "tp":
                    Y[i, j] = CFG.action2label["long"]
                elif short_hit == "tp":
                    Y[i, j] = CFG.action2label["short"]

        return Y

    @staticmethod
    def _check_hit(high_future, low_future, tp_level, sl_level, mode):
        for hi, lo in zip(high_future, low_future):
            if mode == "long":
                if lo <= sl_level: return "sl"
                if hi >= tp_level: return "tp"
            else:  # short
                if hi >= sl_level: return "sl"
                if lo <= tp_level: return "tp"
        return None

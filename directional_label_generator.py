import numpy as np
import pandas as pd
from config import CFG
import logging

class DirectionalLabelGenerator:
    """
    Генерация меток {0=short, 1=no-trade, 2=long}
    с динамическими TP/SL уровнями по соотношению RR >= 2.5.
    Параметры берутся из конфигурационного файла CFG.
    """

    def __init__(self, lookahead):
        if lookahead is None:
            raise RuntimeError("Параметр lookahead обязателен.")

        self.lookahead = lookahead
        self.rr_min = CFG.labels.rr_min
        self.sl_min = CFG.labels.sl_min
        self.sl_max = CFG.labels.sl_max
        self.sl_step = CFG.labels.sl_step

        self.sl_levels = np.arange(self.sl_min, self.sl_max + self.sl_step, self.sl_step)
        self.tp_levels = self.sl_levels * self.rr_min

    def generate_labels(self, df: pd.DataFrame):
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        N = len(close)
        Y = np.full(N, CFG.action2label.mapping["no-trade"], dtype=int)
        generated = 0

        for i in range(N - self.lookahead):
            entry = close[i]
            high_future = high[i + 1: i + 1 + self.lookahead]
            low_future = low[i + 1: i + 1 + self.lookahead]

            for sl, tp in zip(self.sl_levels, self.tp_levels):
                tp_long = entry * (1 + tp)
                sl_long = entry * (1 - sl)

                if self._check_hit(high_future, low_future, tp_long, sl_long, mode="long") == "tp":
                    Y[i] = CFG.action2label.mapping["long"]
                    generated += 1
                    break

                tp_short = entry * (1 - tp)
                sl_short = entry * (1 + sl)

                if self._check_hit(high_future, low_future, tp_short, sl_short, mode="short") == "tp":
                    Y[i] = CFG.action2label.mapping["short"]
                    generated += 1
                    break

        logging.info(f"Сгенерировано {generated} трейдовых меток из {N} ({generated / N:.2%})")
        return Y

    @staticmethod
    def _check_hit(high_future, low_future, tp_level, sl_level, mode):
        for hi, lo in zip(high_future, low_future):
            if mode == "long":
                if lo <= sl_level:
                    return "sl"
                if hi >= tp_level:
                    return "tp"
            else:  # short
                if hi >= sl_level:
                    return "sl"
                if lo <= tp_level:
                    return "tp"
        return None

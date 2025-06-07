import numpy as np
import pandas as pd


class DirectionalLabelGenerator:
    """
    Генерирует метки {0=short, 1=no-trade, 2=long} для каждой TP/SL пары.
    """

    def __init__(self, tp_sl_levels, lookahead):
        if not tp_sl_levels:
            raise RuntimeError("Нужен непустой список tp_sl_pairs.")
        if lookahead is None:
            raise RuntimeError("Параметр lookahead обязателен.")

        self.tp_sl_levels = tp_sl_levels
        self.lookahead = lookahead

    def generate_labels(self, df: pd.DataFrame):
        """
        Возвращает массив меток размерности (N, num_pairs), где:
            - 0: short
            - 1: no-trade
            - 2: long
        """
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        N = len(close)
        num_pairs = len(self.tp_sl_levels)
        Y = np.ones((N, num_pairs), dtype=int)  # по умолчанию — no-trade

        for i in range(N - self.lookahead):
            entry_price = close[i]
            high_future = high[i + 1 : i + 1 + self.lookahead]
            low_future = low[i + 1 : i + 1 + self.lookahead]

            for j, (tp, sl) in enumerate(self.tp_sl_levels):
                # === LONG позиция ===
                tp_long_level = entry_price * (1 + tp)
                sl_long_level = entry_price * (1 - sl)

                # === SHORT позиция ===
                tp_short_level = entry_price * (1 - tp)
                sl_short_level = entry_price * (1 + sl)

                # === Проверка порядка срабатывания LONG ===
                long_hit = None
                for hi, lo in zip(high_future, low_future):
                    if lo <= sl_long_level:
                        long_hit = "sl"
                        break
                    elif hi >= tp_long_level:
                        long_hit = "tp"
                        break

                # === Проверка порядка срабатывания SHORT ===
                short_hit = None
                for hi, lo in zip(high_future, low_future):
                    if hi >= sl_short_level:
                        short_hit = "sl"
                        break
                    elif lo <= tp_short_level:
                        short_hit = "tp"
                        break

                # === Назначение метки ===
                if long_hit == "tp":
                    Y[i, j] = 2  # long
                elif short_hit == "tp":
                    Y[i, j] = 0  # short
                # иначе — no-trade (1)

        return Y

def generate_binary_labels(Y: np.ndarray) -> np.ndarray:
    """
    Преобразует многоклассовые метки в бинарные:
    1 — сигнал (SHORT или LONG), 0 — NO TRADE
    """
    return np.isin(Y, [0, 2]).astype(int)
# Models/direction/labeler.py

from config import CFG
import pandas as pd
import numpy as np

def create_labels(df: pd.DataFrame) -> pd.Series:
    """
    Разметка: будет ли пробой вверх на MIN_TP_THRESHOLD_PERCENT
    в течение следующих LOOKAHEAD_BARS свечей.
    """
    lookahead = CFG.market.lookahead_bars
    threshold = CFG.market.min_tp_percent / 100

    high_prices = df['high'].to_numpy()
    close_prices = df['close'].to_numpy()

    labels = np.zeros(len(df), dtype=int)

    for i in range(len(df) - lookahead):
        current_price = close_prices[i]
        future_high = np.max(high_prices[i+1:i+1+lookahead])
        if (future_high - current_price) / current_price >= threshold:
            labels[i] = 1

    return pd.Series(labels, index=df.index)
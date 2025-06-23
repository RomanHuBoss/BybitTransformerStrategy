# Models/direction/labeler.py

import pandas as pd

def create_labels(df: pd.DataFrame) -> pd.Series:
    """
    Классическая разметка для Direction модели.
    """
    df['label'] = (df['close'] > df['open']).astype(int)
    return df['label']

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import CFG

# === Универсальная загрузка полного датафрейма ===

def load_full_dataframe():
    return pd.read_csv(CFG.paths.train_features_csv)

# === Разделение признаков и лейблов ===

def load_train_features():
    df = load_full_dataframe()
    feature_cols = df.columns.difference([
        'direction_label',
        'amp_up_p10', 'amp_up_p90', 'amp_down_p10', 'amp_down_p90'
    ])
    return df[feature_cols].values

def load_train_labels_direction():
    df = load_full_dataframe()
    return df['direction_label'].values

def load_train_labels_amplitude():
    df = load_full_dataframe()
    return df[['amp_up_p10', 'amp_up_p90', 'amp_down_p10', 'amp_down_p90']].values

# === Direction Dataset (последовательный) ===

class SequenceDataset(Dataset):
    def __init__(self, X, y, window_size):
        self.X = X
        self.y = y
        self.window_size = window_size

    def __len__(self):
        return len(self.X) - self.window_size + 1

    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.window_size]
        y_label = self.y[idx + self.window_size - 1]
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_label, dtype=torch.long)

# === Amplitude Dataset (регрессия) ===

class AmplitudeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

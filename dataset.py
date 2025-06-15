import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import CFG

# === Унифицированные функции загрузки данных ===

def load_train_features():
    """Загрузка признаков для всех моделей"""
    return pd.read_csv(CFG.paths.train_features_csv).values

def load_train_labels_direction():
    """Загрузка меток для Direction модели"""
    return np.load(CFG.paths.train_labels_direction)

def load_train_labels_amplitude():
    """Загрузка меток для Amplitude модели"""
    return np.load(CFG.paths.train_labels_amplitude)

def load_train_labels_hitorder():
    """Загрузка меток для HitOrder модели"""
    return np.load(CFG.paths.train_labels_hitorder)

# === PyTorch Dataset для Direction модели (последовательности) ===

class SequenceDataset(Dataset):
    """
    Датасет для Direction модели (последовательный input)
    """
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

# === PyTorch Dataset для Amplitude модели (табличные данные) ===

class AmplitudeDataset(Dataset):
    """
    Датасет для Amplitude модели (регрессия)
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === PyTorch Dataset для HitOrder модели (бинарная классификация) ===

class HitOrderDataset(Dataset):
    """
    Датасет для HitOrder модели (бинарная классификация)
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

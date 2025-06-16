import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import CFG

# === Универсальная загрузка признаков ===

def load_train_features():
    return pd.read_csv(CFG.paths.train_features_csv).values

def load_train_labels_direction():
    return np.load(CFG.paths.train_labels_direction)

def load_train_labels_amplitude():
    return np.load(CFG.paths.train_labels_amplitude)

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

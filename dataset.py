import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import CFG


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

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


class HitOrderDataset(Dataset):
    def __init__(self):
        # Загружаем фичи
        features = pd.read_csv(CFG.paths.train_features_csv).values
        self.features = features.astype(np.float32)

        # Загружаем метки
        labels = np.load(CFG.paths.train_labels_hitorder)
        self.labels = labels[:, 2].astype(np.float32)  # берём только колонку hit (0 или 1)

        # Синхронизация длины (если вдруг не совпали)
        min_len = min(len(self.features), len(self.labels))
        self.features = self.features[:min_len]
        self.labels = self.labels[:min_len]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return torch.tensor(x), torch.tensor(y)
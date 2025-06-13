import torch
from torch.utils.data import Dataset

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

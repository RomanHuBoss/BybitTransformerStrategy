import pandas as pd
import numpy as np
import joblib
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from feature_engineering import FeatureEngineer
from dataset import SequenceDataset
from model import DirectionalModel
from losses import CostSensitiveFocalLoss
from config import CFG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class DirectionalTrainer:
    def __init__(self):
        logging.info("Загрузка признаков и лейблов...")
        X = pd.read_csv(CFG.paths.train_features_csv).values
        y = np.load(CFG.paths.train_labels_direction)

        self.engineer = FeatureEngineer()
        self.engineer.scaler = joblib.load(CFG.paths.scaler_path)
        self.engineer.feature_columns = joblib.load(CFG.paths.feature_columns_path)

        self.dataset = SequenceDataset(X, y, CFG.train.direction_window_size)
        self.dataloader = DataLoader(self.dataset, batch_size=CFG.train.batch_size, shuffle=True)

        model_cfg = CFG.DirectionModelConfig()
        model_cfg.input_dim = len(self.engineer.feature_columns)
        self.model = DirectionalModel(model_cfg)

        self.criterion = CostSensitiveFocalLoss(alpha=None, gamma=2.0, label_smoothing=0.0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=CFG.train.lr)

    def train(self):
        logging.info("Обучение Direction модели...")
        self.model.train()
        for epoch in range(CFG.train.epochs):
            total_loss = 0
            for X_batch, y_batch in self.dataloader:
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            logging.info(f"Эпоха {epoch + 1}: Loss {total_loss / len(self.dataloader):.6f}")

        torch.save(self.model.state_dict(), CFG.paths.direction_model_path)
        logging.info("Direction модель сохранена")

if __name__ == '__main__':
    trainer = DirectionalTrainer()
    trainer.train()

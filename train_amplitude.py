import pandas as pd
import numpy as np
import joblib
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from feature_engineering import FeatureEngineer
from dataset import SequenceDataset
from model import AmplitudeModel
from losses import AmplitudeLoss
from config import CFG

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class AmplitudeTrainer:
    def __init__(self):
        logging.info("🚀 Запуск обучения Amplitude модели")

        # Загружаем признаки и таргеты
        X = pd.read_csv(CFG.paths.train_features_csv).values
        y = np.load(CFG.paths.train_labels_amplitude)
        logging.info(f"✅ Данные загружены: {X.shape[0]} примеров, {X.shape[1]} признаков")

        # Загружаем scaler и feature columns
        self.engineer = FeatureEngineer()
        self.engineer.scaler = joblib.load(CFG.paths.scaler_path)
        self.engineer.feature_columns = joblib.load(CFG.paths.feature_columns_path)

        # Формируем датасет и dataloader
        self.dataset = SequenceDataset(X, y, CFG.train.amplitude_window_size)
        self.dataloader = DataLoader(self.dataset, batch_size=CFG.train.batch_size, shuffle=True)

        input_size = len(self.engineer.feature_columns)
        self.model = AmplitudeModel(input_size)

        self.criterion = AmplitudeLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=CFG.train.lr)

    def train(self):
        logging.info("🚀 Старт цикла обучения")

        self.model.train()
        for epoch in range(CFG.train.epochs):
            total_loss = 0
            for X_batch, y_batch in self.dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            logging.info(f"🧮 Эпоха {epoch + 1}/{CFG.train.epochs} — Loss: {avg_loss:.6f}")

        # Сохраняем модель
        torch.save(self.model.state_dict(), CFG.paths.amplitude_model_path)
        logging.info(f"✅ Модель сохранена: {CFG.paths.amplitude_model_path}")

if __name__ == '__main__':
    trainer = AmplitudeTrainer()
    trainer.train()

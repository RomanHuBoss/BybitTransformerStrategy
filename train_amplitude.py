import pandas as pd
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
        logging.info("Загрузка обучающих данных...")
        self.df = pd.read_csv(CFG.paths.train_csv)

        logging.info("Загрузка scaler и списка признаков...")
        self.engineer = FeatureEngineer()
        self.engineer.scaler = joblib.load(CFG.paths.scaler_path)
        self.engineer.feature_columns = joblib.load(CFG.paths.feature_columns_path)

        logging.info("Генерация признаков для обучения...")
        self.df_feat = self.engineer.generate_features(self.df, fit=False)

        X = self.df_feat[self.engineer.feature_columns].values
        y = self.df_feat[['amplitude_up', 'amplitude_down']].values  # multi-target для амплитуд

        self.dataset = SequenceDataset(X, y, CFG.train.amplitude_window_size)
        self.dataloader = DataLoader(self.dataset, batch_size=CFG.train.batch_size, shuffle=True)

        input_size = len(self.engineer.feature_columns)
        self.model = AmplitudeModel(input_size)

        self.criterion = AmplitudeLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=CFG.train.lr)

    def train(self):
        logging.info("Начало обучения Amplitude модели...")
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
            logging.info(f"Эпоха {epoch + 1}/{CFG.train.epochs} — Loss: {total_loss / len(self.dataloader):.6f}")

        torch.save(self.model.state_dict(), CFG.paths.amplitude_model_path)
        logging.info("Amplitude модель успешно сохранена.")

if __name__ == '__main__':
    trainer = AmplitudeTrainer()
    trainer.train()

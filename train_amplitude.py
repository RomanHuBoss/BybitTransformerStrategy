import pandas as pd
import numpy as np
import joblib
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from feature_engineering import FeatureEngineer
from dataset import TabularDataset
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

        X = pd.read_csv(CFG.paths.train_features_csv).values
        y = np.load(CFG.paths.train_labels_amplitude)
        logging.info(f"✅ Данные загружены: {X.shape[0]} примеров, {X.shape[1]} признаков")

        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]

        self.engineer = FeatureEngineer()
        self.engineer.scaler = joblib.load(CFG.paths.scaler_path)
        self.engineer.feature_columns = joblib.load(CFG.paths.feature_columns_path)

        self.dataset = TabularDataset(X, y)
        self.dataloader = DataLoader(self.dataset, batch_size=CFG.train.batch_size, shuffle=True)

        input_size = len(self.engineer.feature_columns)
        self.model = AmplitudeModel(input_size)

        self.criterion = AmplitudeLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=CFG.train.lr)

    def train(self):
        logging.info("🚀 Старт цикла обучения")

        best_loss = float('inf')
        epochs_no_improve = 0
        early_stop_patience = CFG.train.early_stopping_patience

        self.model.train()
        for epoch in range(CFG.train.epochs):
            total_loss = 0

            # для MAE накопления
            mae_up_p10 = []
            mae_up_p90 = []
            mae_down_p10 = []
            mae_down_p90 = []

            for X_batch, y_batch in self.dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                # разложим головы
                up_p10_pred, up_p90_pred, down_p10_pred, down_p90_pred = outputs
                up_p10_target, up_p90_target, down_p10_target, down_p90_target = torch.chunk(y_batch, 4, dim=1)

                mae_up_p10.append(torch.mean(torch.abs(up_p10_pred - up_p10_target)).item())
                mae_up_p90.append(torch.mean(torch.abs(up_p90_pred - up_p90_target)).item())
                mae_down_p10.append(torch.mean(torch.abs(down_p10_pred - down_p10_target)).item())
                mae_down_p90.append(torch.mean(torch.abs(down_p90_pred - down_p90_target)).item())

            avg_loss = total_loss / len(self.dataloader)
            mean_up_p10 = np.mean(mae_up_p10)
            mean_up_p90 = np.mean(mae_up_p90)
            mean_down_p10 = np.mean(mae_down_p10)
            mean_down_p90 = np.mean(mae_down_p90)

            logging.info(
                f"🧮 Эпоха {epoch + 1}: Train Loss={avg_loss:.6f} | "
                f"MAE up_p10={mean_up_p10:.5f}, up_p90={mean_up_p90:.5f}, "
                f"down_p10={mean_down_p10:.5f}, down_p90={mean_down_p90:.5f}"
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), CFG.paths.amplitude_model_path)
                logging.info(f"🎯 Новый лучший результат: Train Loss {avg_loss:.6f}. Модель сохранена.")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stop_patience:
                logging.info(f"🛑 Ранняя остановка: {epochs_no_improve} эпох без улучшения.")
                break

        logging.info("✅ Обучение завершено.")

if __name__ == '__main__':
    trainer = AmplitudeTrainer()
    trainer.train()

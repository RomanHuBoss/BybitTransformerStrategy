import pandas as pd
import numpy as np
import joblib
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from feature_engineering import FeatureEngineer
from dataset import SequenceDataset
from model import DirectionalModel
from losses import CostSensitiveFocalLoss
from config import CFG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class DirectionalTrainer:
    def __init__(self):
        logging.info("🚀 Начало обучения Direction модели")
        X = pd.read_csv(CFG.paths.train_features_csv).values
        y = np.load(CFG.paths.train_labels_direction)

        assert len(X) == len(y), f"Длины X ({len(X)}) и y ({len(y)}) не совпадают!"
        logging.info(f"✅ Данные загружены: {len(X)} примеров, {X.shape[1]} признаков")

        self.engineer = FeatureEngineer()
        self.engineer.scaler = joblib.load(CFG.paths.scaler_path)
        self.engineer.feature_columns = joblib.load(CFG.paths.feature_columns_path)

        full_dataset = SequenceDataset(X, y, CFG.train.direction_window_size)
        val_size = int(len(full_dataset) * CFG.train.val_size)
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        self.train_loader = DataLoader(self.train_dataset, batch_size=CFG.train.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=CFG.train.batch_size, shuffle=False)

        model_cfg = CFG.DirectionModelConfig()
        model_cfg.input_dim = len(self.engineer.feature_columns)
        self.model = DirectionalModel(model_cfg)

        self.criterion = CostSensitiveFocalLoss(alpha=None, gamma=2.0, label_smoothing=0.0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=CFG.train.lr)

    def train(self):
        best_val_loss = float('inf')
        epochs_no_improve = 0
        patience = CFG.train.early_stopping_patience

        for epoch in range(CFG.train.epochs):
            self.model.train()
            total_loss = 0
            for X_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            val_loss, acc, bal_acc, macro_f1 = self.validate()

            logging.info(f"🧮 Эпоха {epoch + 1}: Train Loss {avg_loss:.6f}, Val Loss {val_loss:.6f}, "
                         f"Accuracy {acc:.4f}, Balanced Acc {bal_acc:.4f}, Macro F1 {macro_f1:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), CFG.paths.direction_model_path)
                logging.info(f"🎯 Новый лучший результат: Val Loss {val_loss:.6f}. Модель сохранена.")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logging.info(f"🛑 Ранняя остановка: {epochs_no_improve} эпох без улучшения.")
                break

        logging.info("✅ Обучение завершено.")

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                total_loss += loss.item()

                preds = torch.argmax(output, dim=1).cpu().numpy()
                targets = y_batch.cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets)

        avg_loss = total_loss / len(self.val_loader)
        acc = accuracy_score(all_targets, all_preds)
        bal_acc = balanced_accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(all_targets, all_preds, average='macro')
        return avg_loss, acc, bal_acc, macro_f1

if __name__ == '__main__':
    trainer = DirectionalTrainer()
    trainer.train()

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.calibration import calibration_curve
import joblib
import logging

from feature_engineering import FeatureEngineer
from model import DirectionalModel
from losses import CostSensitiveFocalLoss
from label_generator import LabelGenerator
from margin_calibrator import MarginCalibrator
from config import CFG


class CryptoDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DirectionalTrainer:
    def __init__(self):
        self.df = pd.read_csv(CFG.paths.train_csv)
        logging.info(f"Загружено {len(self.df)} строк.")
        self.engineer = FeatureEngineer()

        # Генерация признаков
        logging.info("Генерация признаков...")
        self.df_feat = self.engineer.generate_features(self.df, fit=True)

        # Сохраняем скейлер
        joblib.dump(self.engineer.scaler, CFG.paths.scaler_path)

        # Генерация меток
        logging.info("Генерация меток SL/TP...")
        generator = LabelGenerator()
        self.df_feat['label'] = generator.generate_labels(self.df)

        # Балансировка классов
        self._log_class_balance()

        # Train / Validation split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.df_feat.drop(columns=['label']),
            self.df_feat['label'],
            test_size=CFG.train.val_size,
            shuffle=False
        )

        # PyTorch Dataset
        self.train_ds = CryptoDataset(self.X_train.values, self.y_train.values)
        self.val_ds = CryptoDataset(self.X_val.values, self.y_val.values)

        # Model init
        self.model = DirectionalModel(input_size=self.X_train.shape[1])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=CFG.train.lr)
        self.criterion = CostSensitiveFocalLoss(gamma=CFG.train.focal_gamma)

    def _log_class_balance(self):
        counts = self.df_feat['label'].value_counts()
        total = len(self.df_feat)
        for cls, cnt in counts.items():
            pct = cnt / total * 100
            logging.info(f"Класс {cls}: {cnt} примеров ({pct:.2f}%)")

    def train(self):
        best_profit_f1 = -np.inf

        train_loader = DataLoader(self.train_ds, batch_size=CFG.train.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_ds, batch_size=CFG.train.batch_size, shuffle=False)

        for epoch in range(1, CFG.train.epochs + 1):
            self.model.train()
            total_loss = 0

            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            val_loss, profit_f1, metrics = self.validate(val_loader)

            logging.info(
                f"Эпоха {epoch} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} "
                f"| Profit F1: {profit_f1:.4f}")

            for cls, m in metrics.items():
                logging.info(f"Класс {cls}: precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f}")

            if profit_f1 > best_profit_f1:
                best_profit_f1 = profit_f1
                torch.save(self.model.state_dict(), CFG.paths.direction_model_path)
                logging.info(f"Сохранена новая лучшая модель (Profit F1={profit_f1:.4f})")

        self.calibrate_and_save()

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_logits, all_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                total_loss += loss.item()

                all_logits.append(logits.cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())

        logits = np.concatenate(all_logits)
        labels = np.concatenate(all_labels)
        probs = self.softmax(logits)
        preds = np.argmax(probs, axis=1)

        cls_report = classification_report(labels, preds, output_dict=True, zero_division=0)
        metrics = {cls: {
            'precision': cls_report[str(cls)]['precision'],
            'recall': cls_report[str(cls)]['recall'],
            'f1': cls_report[str(cls)]['f1-score']
        } for cls in [0, 1, 2]}

        profit_f1 = np.mean([metrics[0]['f1'], metrics[2]['f1']])

        return total_loss / len(val_loader), profit_f1, metrics

    def calibrate_and_save(self):
        logging.info("Калибровка модели (temperature scaling)...")
        calibrator = MarginCalibrator(self.model, self.val_ds)
        T = calibrator.fit_temperature()
        joblib.dump(T, CFG.paths.temperature_path)

        # Пороговая оптимизация
        thresholds = calibrator.find_best_thresholds()
        joblib.dump(thresholds, CFG.paths.thresholds_path)
        logging.info(f"Оптимальные пороги: {thresholds}")

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    trainer = DirectionalTrainer()
    trainer.train()

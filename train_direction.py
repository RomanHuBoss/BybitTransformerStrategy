import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import logging
import os

from feature_engineering import FeatureEngineer
from model import DirectionalModel  # <-- теперь строго централизованный импорт
from losses import CostSensitiveFocalLoss
from directional_label_generator import DirectionalLabelGenerator
from dataset import SequenceDataset
from config import CFG


# Конфигурация модели трансформера (локальная для direction)
class ModelConfig:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.hidden_dim = 128
        self.n_layers = 2
        self.n_heads = 4
        self.dim_feedforward = 256
        self.activation = 'gelu'
        self.dropout = 0.1
        self.layer_norm_eps = 1e-5


class DirectionalTrainer:
    def __init__(self):
        self.window_size = CFG.feature_engineering.window_size

        self.df = pd.read_csv(CFG.paths.train_csv)
        logging.info(f"Загружено {len(self.df)} строк.")

        self.engineer = FeatureEngineer()
        logging.info("Генерация признаков...")
        self.df_feat = self.engineer.generate_features(self.df, fit=True)

        os.makedirs(os.path.dirname(CFG.paths.scaler_path), exist_ok=True)
        joblib.dump(self.engineer.scaler, CFG.paths.scaler_path)

        logging.info("Генерация directional меток...")
        generator = DirectionalLabelGenerator(lookahead=CFG.labels.lookahead)
        labels = generator.generate_labels(self.df)

        labels_series = pd.Series(labels, index=self.df.index)

        # Безопасная синхронизация по индексам
        self.df_feat = self.df_feat.merge(
            labels_series.rename("label"),
            left_index=True, right_index=True, how="inner"
        )

        # Удаляем потенциальные строки с NaN
        self.df_feat = self.df_feat.dropna(subset=["label"])

        self._log_class_balance()

        X = self.df_feat.drop(columns=['label']).values
        y = self.df_feat['label'].astype(int).values

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=CFG.train.val_size, shuffle=False
        )

        assert len(self.X_train) == len(self.y_train), "Train: длины X и y не совпадают"
        assert len(self.X_val) == len(self.y_val), "Val: длины X и y не совпадают"

        self.train_ds = SequenceDataset(self.X_train, self.y_train, self.window_size)
        self.val_ds = SequenceDataset(self.X_val, self.y_val, self.window_size)

        input_dim = self.X_train.shape[1]
        model_cfg = ModelConfig(input_dim=input_dim)
        self.model = DirectionalModel(model_config=model_cfg)

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

            val_loss, profit_f1 = self.validate(val_loader)

            logging.info(
                f"Эпоха {epoch} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Profit F1: {profit_f1:.4f}"
            )

            if profit_f1 > best_profit_f1:
                best_profit_f1 = profit_f1
                torch.save(self.model.state_dict(), CFG.paths.direction_model_path)
                logging.info(f"Сохранена новая лучшая модель (Profit F1={profit_f1:.4f})")

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_logits, all_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                total_loss += loss.item()
                all_logits.append(logits.numpy())
                all_labels.append(y_batch.numpy())

        logits = np.concatenate(all_logits)
        labels = np.concatenate(all_labels)
        probs = self._softmax(logits)
        preds = np.argmax(probs, axis=1)

        labels_str = list(map(str, labels))
        preds_str = list(map(str, preds))
        cls_report = classification_report(labels_str, preds_str, output_dict=True, zero_division=0)

        profit_f1 = np.mean([cls_report['0']['f1-score'], cls_report['2']['f1-score']])
        return total_loss / len(val_loader), profit_f1

    @staticmethod
    def _softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    trainer = DirectionalTrainer()
    trainer.train()

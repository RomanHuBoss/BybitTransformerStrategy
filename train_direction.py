import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import logging
import os
from feature_engineering import FeatureEngineer
from model import DirectionalModel
from losses import CostSensitiveFocalLoss
from directional_label_generator import DirectionalLabelGenerator
from config import CFG


# ------------------------------
# Конфигурация модели
# ------------------------------

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


# ------------------------------
# PyTorch Dataset
# ------------------------------

class CryptoDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ------------------------------
# Калибровка логитов и порогов
# ------------------------------

class MarginCalibrator:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def fit_temperature(self):
        loader = DataLoader(self.dataset, batch_size=CFG.train.batch_size, shuffle=False)
        logits_list, targets_list = [], []

        self.model.eval()
        with torch.no_grad():
            for X_batch, y_batch in loader:
                logits = self.model(X_batch)
                logits_list.append(logits.numpy())
                targets_list.append(y_batch.numpy())

        logits = np.concatenate(logits_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)

        best_T, best_loss = 1.0, float("inf")
        for T in np.linspace(0.1, 3.0, 30):
            loss = self._nll_loss(logits / T, targets)
            if loss < best_loss:
                best_T, best_loss = T, loss

        return best_T

    def find_best_thresholds(self):
        loader = DataLoader(self.dataset, batch_size=CFG.train.batch_size, shuffle=False)
        logits_list, targets_list = [], []

        self.model.eval()
        with torch.no_grad():
            for X_batch, y_batch in loader:
                logits = self.model(X_batch)
                logits_list.append(logits.numpy())
                targets_list.append(y_batch.numpy())

        logits = np.concatenate(logits_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        probs = self._softmax(logits)

        thresholds = {}
        for cls in [0, 2]:
            best_th, best_f1 = 0.5, 0
            for th in np.linspace(0.4, 0.9, 20):
                preds = self._apply_thresholds(probs, cls, th)
                cls_targets = (targets == cls).astype(int)
                cls_preds = (preds == cls).astype(int)
                tp = (cls_targets & cls_preds).sum()
                fp = ((1 - cls_targets) & cls_preds).sum()
                fn = (cls_targets & (1 - cls_preds)).sum()

                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                if f1 > best_f1:
                    best_th, best_f1 = th, f1
            thresholds[cls] = best_th

        return thresholds

    def _apply_thresholds(self, probs, target_class, threshold):
        preds = np.argmax(probs, axis=1)
        preds_adjusted = preds.copy()

        for i in range(len(probs)):
            if probs[i][target_class] < threshold:
                preds_adjusted[i] = 1  # no-trade fallback
        return preds_adjusted

    def _nll_loss(self, logits, targets):
        probs = self._softmax(logits)
        log_probs = np.log(probs + 1e-8)
        nll = -log_probs[np.arange(len(targets)), targets].mean()
        return nll

    @staticmethod
    def _softmax(logits):
        e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)


# ------------------------------
# Тренировочный класс
# ------------------------------

class DirectionalTrainer:
    def __init__(self):
        self.df = pd.read_csv(CFG.paths.train_csv)
        logging.info(f"Загружено {len(self.df)} строк.")
        self.engineer = FeatureEngineer()

        logging.info("Генерация признаков...")
        self.df_feat = self.engineer.generate_features(self.df, fit=True)

        os.makedirs(os.path.dirname(CFG.paths.scaler_path), exist_ok=True)
        joblib.dump(self.engineer.scaler, CFG.paths.scaler_path)

        logging.info("Генерация меток SL/TP...")
        generator = DirectionalLabelGenerator(tp_sl_levels=CFG.labels.tp_sl_levels, lookahead=CFG.labels.lookahead)
        labels = generator.generate_labels(self.df)
        self.df_feat['label'] = labels.flatten()

        self._log_class_balance()

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.df_feat.drop(columns=['label']),
            self.df_feat['label'],
            test_size=CFG.train.val_size,
            shuffle=False
        )

        self.train_ds = CryptoDataset(self.X_train.values, self.y_train.values)
        self.val_ds = CryptoDataset(self.X_val.values, self.y_val.values)

        input_dim = self.X_train.shape[1]
        model_cfg = ModelConfig(input_dim=input_dim)
        self.model = DirectionalModel(model_config=model_cfg, num_pairs=len(CFG.assets.symbols))

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
                all_logits.append(logits.numpy())
                all_labels.append(y_batch.numpy())

        logits = np.concatenate(all_logits)
        labels = np.concatenate(all_labels)
        probs = self._softmax(logits)
        preds = np.argmax(probs, axis=1)

        # Приводим к строкам чтобы стабилизировать classification_report
        labels_str = list(map(str, labels))
        preds_str = list(map(str, preds))
        cls_report = classification_report(labels_str, preds_str, output_dict=True, zero_division=0)

        profit_f1 = np.mean([cls_report['0']['f1-score'], cls_report['2']['f1-score']])
        return total_loss / len(val_loader), profit_f1

    def calibrate_and_save(self):
        logging.info("Калибровка temperature scaling и порогов...")
        calibrator = MarginCalibrator(self.model, self.val_ds)
        T = calibrator.fit_temperature()

        os.makedirs(os.path.dirname(CFG.paths.temperature_path), exist_ok=True)
        joblib.dump(T, CFG.paths.temperature_path)

        thresholds = calibrator.find_best_thresholds()

        os.makedirs(os.path.dirname(CFG.paths.thresholds_path), exist_ok=True)
        joblib.dump(thresholds, CFG.paths.thresholds_path)

    @staticmethod
    def _softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    trainer = DirectionalTrainer()
    trainer.train()

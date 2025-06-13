import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from feature_engineering import FeatureEngineer
from config import CFG
from amplitude_label_generator import AmplitudeLabelGenerator
from sklearn.model_selection import train_test_split
import numpy as np
from types import SimpleNamespace
import json
import os
import logging
from amplitude_regressor import AmplitudeRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def save_meta(cfg):
    os.makedirs(os.path.dirname(CFG.paths.meta_path), exist_ok=True)
    meta_dict = vars(cfg).copy()
    for key, value in meta_dict.items():
        if hasattr(value, 'tolist'):
            meta_dict[key] = value.tolist()
        elif isinstance(value, (np.generic, np.ndarray)):
            meta_dict[key] = value.item() if value.size == 1 else value.tolist()
    with open(CFG.paths.meta_path, "w") as f:
        json.dump(meta_dict, f, indent=4)

class AmplitudeTrainer:
    def __init__(self):
        self.device = CFG.train.device

        logging.info("Загрузка данных из CSV...")
        df = CFG.train.loader()
        logging.info(f"Загружено {len(df)} строк.")

        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
        ohlcv_data = df[['open', 'high', 'low', 'close', 'volume']].copy()

        logging.info("Генерация признаков...")
        self.engineer = FeatureEngineer()
        df_feat = self.engineer.generate_features(df, fit=True)
        self.feature_columns = df_feat.columns
        logging.info(f"Сгенерировано {len(self.feature_columns)} признаков на {len(df_feat)} строках.")

        logging.info("Генерация амплитудных таргетов...")
        lookahead = CFG.train.lookahead[CFG.train.timeframe]
        labeler = AmplitudeLabelGenerator(lookahead=lookahead)
        amplitudes = labeler.generate_labels(ohlcv_data.loc[df_feat.index])

        valid_idx = ~np.isnan(amplitudes)
        df_feat = df_feat.loc[valid_idx]
        amplitudes = amplitudes[valid_idx]

        # === Вот тут главное изменение: нормализация на ATR ===
        df_feat_valid = df_feat.copy()
        atr = df_feat_valid['atr_long_pct'].values
        atr[atr < 1e-6] = 1e-6  # защита от деления на ноль

        norm_amplitudes = amplitudes / atr

        # Обучаем модель на нормализованных амплитудах
        X = self.engineer.to_sequences(df_feat, CFG.train.window_size[CFG.train.timeframe])
        norm_amplitudes = norm_amplitudes[-len(X):]

        X_train, X_val, y_train, y_val = train_test_split(
            X, norm_amplitudes, test_size=CFG.train.val_ratio, shuffle=False)

        self.atr_train = atr[-len(X):]  # сохраним atr для валидации

        self.train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.float32)),
            batch_size=CFG.train.batch_size,
            shuffle=False
        )
        self.val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                          torch.tensor(y_val, dtype=torch.float32)),
            batch_size=CFG.train.batch_size,
            shuffle=False
        )

        model_config = SimpleNamespace(
            input_dim=X.shape[2],
            hidden_dim=CFG.default_model_config.hidden_dim,
            n_heads=CFG.default_model_config.n_heads,
            n_layers=CFG.default_model_config.n_layers,
            dropout=CFG.default_model_config.dropout,
            n_classes=CFG.default_model_config.n_classes,
            dim_feedforward=CFG.default_model_config.dim_feedforward,
            activation=CFG.default_model_config.activation,
            layer_norm_eps=CFG.default_model_config.layer_norm_eps
        )

        self.model = AmplitudeRegressor(model_config=model_config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=CFG.train.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=7, cooldown=3, min_lr=1e-6)
        self.criterion = nn.MSELoss()

    def train(self):
        best_loss = np.inf
        epochs_no_improve = 0

        for epoch in range(1, CFG.train.epochs + 1):
            train_loss = self._train_one_epoch()
            val_loss = self._validate()

            self.scheduler.step(val_loss)
            logging.info(f"Эпоха {epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0

                torch.save(self.model.state_dict(), CFG.paths.model_path)
                logging.info(f"📦 Сохранена новая лучшая модель (Val Loss={val_loss:.6f}) на эпохе {epoch}")

                meta_args = SimpleNamespace(
                    input_dim=self.model.input_proj.in_features,
                    timeframe=CFG.train.timeframe,
                    window_size=CFG.train.window_size[CFG.train.timeframe],
                    lookahead=CFG.train.lookahead[CFG.train.timeframe],
                    feature_columns=self.feature_columns,
                    scaler_params={
                        "mean": self.engineer.scaler.mean_.tolist(),
                        "scale": self.engineer.scaler.scale_.tolist()
                    },
                    hidden_dim=CFG.default_model_config.hidden_dim,
                    n_heads=CFG.default_model_config.n_heads,
                    n_layers=CFG.default_model_config.n_layers,
                    dropout=CFG.default_model_config.dropout,
                    dim_feedforward=CFG.default_model_config.dim_feedforward,
                    activation=CFG.default_model_config.activation,
                    layer_norm_eps=CFG.default_model_config.layer_norm_eps
                )

                save_meta(meta_args)

            else:
                epochs_no_improve += 1
                if epochs_no_improve >= CFG.train.patience:
                    logging.info(f"⏹ Early stopping после {epoch} эпох")
                    break

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0.0

        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(X_batch)
            loss = self.criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        return total_loss / len(self.train_loader.dataset)

    def _validate(self):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)
                total_loss += loss.item() * X_batch.size(0)

                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)

        mae = np.mean(np.abs(y_pred - y_true))
        mape = np.mean(np.abs((y_pred - y_true) / (y_true + 1e-8))) * 100
        r2 = 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

        logging.info(f"🔬 Val MAE (norm ATR units): {mae:.6f} | MAPE: {mape:.2f}% | R²: {r2:.4f}")

        return total_loss / len(self.val_loader.dataset)

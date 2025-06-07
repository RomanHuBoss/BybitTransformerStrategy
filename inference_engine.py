import torch
import numpy as np
import pandas as pd
import json
from config import CFG
from feature_engineering import FeatureEngineer
from model import MultiPairDirectionalClassifier
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class InferenceEngine:
    def __init__(self, model_folder=None, scaler_mean=None, scaler_scale=None):
        meta_path = CFG.paths.meta_path if model_folder is None else os.path.join(model_folder, "model_meta.json")
        model_path = CFG.paths.model_path if model_folder is None else os.path.join(model_folder, "model.pth")

        with open(meta_path, 'r') as f:
            self.meta = json.load(f)

        # Использовать переданные значения или из meta
        self.scaler_mean = np.array(scaler_mean) if scaler_mean is not None else np.array(self.meta['scaler_params']['mean'])
        self.scaler_scale = np.array(scaler_scale) if scaler_scale is not None else np.array(self.meta['scaler_params']['scale'])

        self.device = CFG.train.device
        self.engineer = FeatureEngineer()
        self.engineer.load_scaler(self.scaler_mean, self.scaler_scale)
        self.feature_columns = self.meta['feature_columns']
        self.window_size = self.meta['window_size']
        self.lookahead = self.meta['lookahead']
        self.temperature = self.meta.get('temperature', 1.0)

        model_config = CFG.default_model_config
        model_config.input_dim = self.meta['input_dim']
        model_config.n_heads = self.meta['n_heads']
        model_config.n_layers = self.meta['n_layers']
        model_config.hidden_dim = self.meta['hidden_dim']
        model_config.dropout = self.meta['dropout']
        model_config.n_classes = self.meta['n_classes']
        model_config.dim_feedforward = self.meta['dim_feedforward']
        model_config.activation = self.meta['activation']
        model_config.layer_norm_eps = self.meta['layer_norm_eps']

        self.model = MultiPairDirectionalClassifier(model_config=model_config, num_pairs=len(self.meta['tp_sl_pairs']))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, df: pd.DataFrame):
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
        df_feat = self.engineer.generate_features(df, fit=False)
        df_feat = df_feat[self.feature_columns]
        X = self.engineer.to_sequences(df_feat, window_size=self.window_size)

        if len(X) == 0:
            logging.warning("Недостаточно данных для формирования входной последовательности")
            return None

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            logits = logits[:, -1, :] / self.temperature  # Калибровка логитов
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)

        final_class = preds[-1]
        final_conf = confs[-1]

        # Возвращаем всё: классы, вероятности, флаг трейда
        all_preds = preds.tolist()
        all_confs = confs.tolist()
        all_probs = probs.tolist()

        return {
            "final_class": int(final_class),
            "final_confidence": float(final_conf),
            "classes": preds.tolist(),
            "confidences": confs.tolist(),
            "probabilities": probs.tolist(),
            "tp_sl_pairs": self.meta["tp_sl_pairs"]
        }

    def predict_logits(self, df: pd.DataFrame):
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
        df_feat = self.engineer.generate_features(df, fit=False)
        df_feat = df_feat[self.feature_columns]
        X = self.engineer.to_sequences(df_feat, window_size=self.window_size)

        if len(X) == 0:
            logging.warning("Недостаточно данных для формирования входной последовательности")
            return None

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            logits = logits[:, -1, :] / self.temperature  # Калибровка логитов

        return logits.cpu().numpy()

    def predict_proba(self, df: pd.DataFrame):
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
        df_feat = self.engineer.generate_features(df, fit=False)
        df_feat = df_feat[self.feature_columns]
        X = self.engineer.to_sequences(df_feat, window_size=self.window_size)

        if len(X) == 0:
            logging.warning("Недостаточно данных для формирования входной последовательности")
            return None

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            logits = logits[:, -1, :] / self.temperature
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return probs

    def predict_confidence(self, df: pd.DataFrame):
        probs = self.predict_proba(df)
        if probs is None:
            return None
        return np.max(probs, axis=1)

    def predict_class_distribution(self, df: pd.DataFrame):
        probs = self.predict_proba(df)
        if probs is None:
            return None
        return np.mean(probs, axis=0)




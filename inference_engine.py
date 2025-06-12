import torch
import numpy as np
import pandas as pd
import json
from config import CFG
from feature_engineering_v2_1_full import FeatureEngineer
from model import MultiPairDirectionalClassifier
import logging
import os
from threshold_tuner import ThresholdTuner
from margin_calibrator import MarginCalibrator


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
        self.timeframe = self.meta['timeframe']
        self.feature_columns = self.meta['feature_columns']
        self.window_size = self.meta['window_size']
        self.lookahead = self.meta['lookahead']
        self.temperature = self.meta.get('temperature', 1.0)
        self.thresholds = self.meta.get('thresholds', None)
        self.tuner = ThresholdTuner.from_dict(self.thresholds)

        self.margins = self.meta.get('margins', None)
        print("DEBUG LOADED MARGINS", self.margins)  # <-- ВРЕМЕННЫЙ ОТЛАДОЧНЫЙ ВЫВОД
        if self.margins is not None:
            self.margins = {int(k): v for k, v in self.margins.items()}
        print("DEBUG PARSED MARGINS", self.margins)  # <-- ВРЕМЕННЫЙ ОТЛАДОЧНЫЙ ВЫВОД
        self.margin_calibrator = MarginCalibrator(margins=self.margins)
        self.num_pairs =  self.meta["num_pairs"]

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

        self.model = MultiPairDirectionalClassifier(model_config=model_config, num_pairs=self.num_pairs)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, df: pd.DataFrame):
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
        df_feat = self.engineer.generate_features(df, fit=False, use_logging=False)
        df_feat = df_feat[self.feature_columns]
        X = self.engineer.to_sequences(df_feat, window_size=self.window_size)

        if len(X) == 0:
            logging.warning("Недостаточно данных для формирования входной последовательности")
            return None

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            logits = logits[:, -1, :] / self.temperature  # Temperature Scaling
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            if self.margins:
                batch_size = probs.shape[0]
                probs = probs.reshape(-1, 3)
                probs = self.margin_calibrator.calibrate_probs(np.log(probs + 1e-8))
                probs = probs.reshape(batch_size, self.num_pairs, 3)

        # ✅ Здесь теперь применяем Threshold Tuner вместо argmax
        preds = self.tuner.apply_thresholds(probs)

        # Корректный confidence: берем вероятность предсказанного класса
        confs = np.array([probs[i, cls] for i, cls in enumerate(preds)])

        final_class = preds[-1]
        final_conf = confs[-1]

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




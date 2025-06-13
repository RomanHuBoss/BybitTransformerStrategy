import torch
import numpy as np
import pandas as pd
import json
from config import CFG
from feature_engineering import FeatureEngineer
from amplitude_regressor import AmplitudeRegressor
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class AmplitudeInferenceEngine:
    def __init__(self, model_folder=None, scaler_mean=None, scaler_scale=None):
        meta_path = CFG.paths.meta_path if model_folder is None else os.path.join(model_folder, "model_meta.json")
        model_path = CFG.paths.model_path if model_folder is None else os.path.join(model_folder, "model.pth")

        with open(meta_path, 'r') as f:
            self.meta = json.load(f)

        self.scaler_mean = np.array(scaler_mean) if scaler_mean is not None else np.array(self.meta['scaler_params']['mean'])
        self.scaler_scale = np.array(scaler_scale) if scaler_scale is not None else np.array(self.meta['scaler_params']['scale'])

        self.device = CFG.train.device
        self.engineer = FeatureEngineer()
        self.engineer.load_scaler(self.scaler_mean, self.scaler_scale)
        self.timeframe = self.meta['timeframe']
        self.feature_columns = self.meta['feature_columns']
        self.window_size = self.meta['window_size']
        self.lookahead = self.meta['lookahead']

        model_config = CFG.default_model_config
        model_config.input_dim = self.meta['input_dim']
        model_config.n_heads = self.meta['n_heads']
        model_config.n_layers = self.meta['n_layers']
        model_config.hidden_dim = self.meta['hidden_dim']
        model_config.dropout = self.meta['dropout']
        model_config.dim_feedforward = self.meta['dim_feedforward']
        model_config.activation = self.meta['activation']
        model_config.layer_norm_eps = self.meta['layer_norm_eps']

        self.model = AmplitudeRegressor(model_config=model_config)
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

        # Попробуем взять более стабильную колонку ATR:
        if 'atr_14' in df_feat.columns:
            atr_now_pct = df_feat['atr_14'].iloc[-1]
        else:
            atr_now_pct = df_feat['atr_long_pct'].iloc[-1]

        # В случае если всё равно NaN — защищаемся
        if np.isnan(atr_now_pct) or atr_now_pct <= 0:
            atr_now_pct = 1e-3  # временно ставим маленький стабилизатор вместо 1e-6

        close_now = df['close'].iloc[-1]
        atr_now = atr_now_pct * close_now
        atr_now = max(atr_now, 1e-6)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds_norm = self.model(X_tensor).cpu().numpy()

        predicted_norm = preds_norm[-1]
        predicted_amplitude = predicted_norm * atr_now

        return {
            "predicted_norm": float(predicted_norm),
            "atr": float(atr_now),
            "predicted_amplitude": float(predicted_amplitude)
        }


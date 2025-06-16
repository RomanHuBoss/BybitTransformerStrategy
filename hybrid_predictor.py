import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn.functional as F
from model import DirectionalModel, AmplitudeModel, HitOrderClassifier
from config import CFG

class HybridPredictor:
    def __init__(self):
        # Загружаем scaler и список feature columns
        self.scaler = joblib.load(CFG.paths.scaler_path)
        self.feature_columns = joblib.load(CFG.paths.feature_columns_path)

        # Загружаем temperature
        self.temperature = joblib.load(CFG.paths.temperature_path)

        # Загружаем модели
        self.direction_model = self._load_direction_model()
        self.amplitude_model = self._load_amplitude_model()
        self.hitorder_model = self._load_hitorder_model()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_direction_model(self):
        model_cfg = CFG.ModelConfig()
        model_cfg.input_dim = len(self.feature_columns)
        model = DirectionalModel(model_cfg)
        model.load_state_dict(torch.load(CFG.paths.direction_model_path, map_location="cpu"))
        model.eval()
        return model

    def _load_amplitude_model(self):
        model = AmplitudeModel(input_size=len(self.feature_columns))
        model.load_state_dict(torch.load(CFG.paths.amplitude_model_path, map_location="cpu"))
        model.eval()
        return model

    def _load_hitorder_model(self):
        model = HitOrderClassifier(input_size=len(self.feature_columns))
        model.load_state_dict(torch.load(CFG.paths.hit_order_model_path, map_location="cpu"))
        model.eval()
        return model

    def preprocess(self, df_live):
        """
        Предобработка данных перед инференсом
        """
        features = df_live[self.feature_columns].copy()
        features_scaled = self.scaler.transform(features)
        return torch.tensor(features_scaled, dtype=torch.float32)

    def predict(self, df_live):
        """
        df_live — DataFrame c уже рассчитанными полными признаками.
        """

        # Предобработка
        X = self.preprocess(df_live)

        # Инференс Direction
        with torch.no_grad():
            logits = self.direction_model(X.unsqueeze(0))
            logits_np = logits.numpy() / self.temperature
            probs = F.softmax(torch.tensor(logits_np), dim=1).numpy()[0]

            pred_direction = int(np.argmax(probs))
            confidence = float(np.max(probs))

        # Инференс Amplitude
        with torch.no_grad():
            up_p10, up_p90, down_p10, down_p90 = self.amplitude_model(X)
            amplitude = {
                "up_p10": float(up_p10.item()),
                "up_p90": float(up_p90.item()),
                "down_p10": float(down_p10.item()),
                "down_p90": float(down_p90.item()),
            }

        # Инференс HitOrder
        with torch.no_grad():
            logit_hit = self.hitorder_model(X)
            prob_hit = torch.sigmoid(logit_hit).item()

        return {
            "direction": pred_direction,
            "confidence": confidence,
            "amplitude": amplitude,
            "hitorder_prob": prob_hit
        }

import numpy as np
import torch
import joblib

from feature_engineering import FeatureEngineer
from model import DirectionalModel, AmplitudeModel, HitOrderClassifier
from config import CFG

class HybridPredictor:
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.feature_engineer.scaler = joblib.load(CFG.paths.scaler_path)
        self.feature_engineer.feature_columns = joblib.load(CFG.paths.feature_columns_path)

        input_dim = len(self.feature_engineer.feature_columns)
        model_cfg = CFG.DirectionModelConfig()
        model_cfg.input_dim = input_dim

        self.direction_model = DirectionalModel(model_cfg)
        self.direction_model.load_state_dict(torch.load(CFG.paths.direction_model_path))
        self.direction_model.eval()

        self.amplitude_model = AmplitudeModel(input_size=input_dim)
        self.amplitude_model.load_state_dict(torch.load(CFG.paths.amplitude_model_path))
        self.amplitude_model.eval()

        self.hit_order_model = HitOrderClassifier(input_size=input_dim)
        self.hit_order_model.load_state_dict(torch.load(CFG.paths.hit_order_model_path))
        self.hit_order_model.eval()

        self.temperature = joblib.load(CFG.paths.temperature_path)
        self._validate_model_dimensions()

    def _validate_model_dimensions(self):
        feature_dim = len(self.feature_engineer.feature_columns)
        assert self.direction_model.input_proj.in_features == feature_dim
        assert self.amplitude_model.shared[0].in_features == feature_dim
        assert self.hit_order_model.net[0].in_features == feature_dim
        print("[HybridPredictor] Все модели успешно валидированы по размерностям.")

    def predict(self, df):
        features = self.feature_engineer.generate_features(df, fit=False)

        window_size = CFG.train.direction_window_size
        X_input = features.iloc[-window_size:].values.astype(np.float32)
        X_tensor = torch.tensor(X_input).unsqueeze(0)

        X_flat_input = features.iloc[-1:].values.astype(np.float32)
        X_flat_tensor = torch.tensor(X_flat_input)

        with torch.no_grad():
            logits = self.direction_model(X_tensor).numpy() / self.temperature
            probs = self.softmax(logits)
            final_class = int(np.argmax(probs))
            confidence = float(probs[0, final_class])  # просто сохраняем для UI

            up_p10_pred, up_p90_pred, down_p10_pred, down_p90_pred = self.amplitude_model(X_flat_tensor)
            up_p10 = up_p10_pred.item()
            up_p90 = up_p90_pred.item()
            down_p10 = down_p10_pred.item()
            down_p90 = down_p90_pred.item()

            amplitude_pred = max(up_p90, down_p90)
            amplitude_spread = max(up_p90 - up_p10, down_p90 - down_p10)

            hit_order_prob = self.hit_order_model(X_flat_tensor).item()

        hit_order_class = int(hit_order_prob >= 0.5)

        tp = max(up_p90, 0.001)
        sl_raw = max(down_p90, 0.001)
        min_sl_value = CFG.labels.sl_min
        sl = max(sl_raw, min_sl_value)
        rr = tp / sl if sl > 0 else 0.0
        rr = max(0.0, min(rr, 50.0))

        # === Финальный risk-фильтр: теперь БЕЗ confidence ===
        if rr < 2.0 or sl > 0.02 or amplitude_pred < 0.005:
            return None

        return {
            "final_class": final_class,
            "final_confidence": confidence,
            "predicted_up_p10": float(up_p10),
            "predicted_up_p90": float(up_p90),
            "predicted_down_p10": float(down_p10),
            "predicted_down_p90": float(down_p90),
            "predicted_amplitude": float(amplitude_pred),
            "amplitude_spread": float(amplitude_spread),
            "hit_order_prob": float(hit_order_prob),
            "hit_order": hit_order_class,
            "tp": float(tp),
            "sl": float(sl),
            "rr": float(rr)
        }

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

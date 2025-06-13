import numpy as np
import torch
import joblib

from feature_engineering import FeatureEngineer
from model import DirectionalModel, AmplitudeModel
from config import CFG


class HybridPredictor:
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.feature_engineer.scaler = joblib.load(CFG.paths.scaler_path)
        self.feature_engineer.feature_columns = joblib.load(CFG.paths.feature_columns_path)

        input_dim = len(self.feature_engineer.feature_columns)
        model_cfg = CFG.DirectionModelConfig()
        model_cfg.input_dim = input_dim

        self.direction_model = DirectionalModel(model_config=model_cfg)
        self.direction_model.load_state_dict(torch.load(CFG.paths.direction_model_path))
        self.direction_model.eval()

        self.amplitude_model = AmplitudeModel(input_size=input_dim)
        self.amplitude_model.load_state_dict(torch.load(CFG.paths.amplitude_model_path))
        self.amplitude_model.eval()

        self.temperature = joblib.load(CFG.paths.temperature_path)

        self.amplitude_up_scaler = joblib.load(CFG.paths.amplitude_target_scaler_path.with_name("amplitude_up_scaler.joblib"))
        self.amplitude_down_scaler = joblib.load(CFG.paths.amplitude_target_scaler_path.with_name("amplitude_down_scaler.joblib"))

        self._validate_model_dimensions()

    def _validate_model_dimensions(self):
        feature_dim = len(self.feature_engineer.feature_columns)
        assert self.direction_model.input_proj.in_features == feature_dim, (
            f"DirectionalModel input_dim ({self.direction_model.input_proj.in_features}) != features ({feature_dim})")
        assert self.amplitude_model.shared[0].in_features == feature_dim, (
            f"AmplitudeModel input_dim ({self.amplitude_model.shared[0].in_features}) != features ({feature_dim})")
        print("[HybridPredictor] Модели успешно валидированы по размерностям.")

    def predict(self, df):
        features = self.feature_engineer.generate_features(df, fit=False)

        # Подготовка данных для Direction
        window_size = CFG.train.direction_window_size
        X_input = features.iloc[-window_size:].values.astype(np.float32)
        X_tensor = torch.tensor(X_input).unsqueeze(0)

        # Подготовка данных для Amplitude
        X_flat_input = features.iloc[-1:].values.astype(np.float32)
        X_flat_tensor = torch.tensor(X_flat_input)

        # Direction prediction
        with torch.no_grad():
            logits = self.direction_model(X_tensor).numpy() / self.temperature
            probs = self.softmax(logits)

        final_class = int(np.argmax(probs))
        confidence = float(probs[0, final_class])

        # Amplitude prediction (quantile head)
        with torch.no_grad():
            up_p10_pred, up_p90_pred, down_p10_pred, down_p90_pred = self.amplitude_model(X_flat_tensor)

            up_p10_val = up_p10_pred.item()
            up_p90_val = up_p90_pred.item()
            down_p10_val = down_p10_pred.item()
            down_p90_val = down_p90_pred.item()

            up_p10 = self.amplitude_up_scaler.inverse_transform([[up_p10_val]])[0, 0]
            up_p90 = self.amplitude_up_scaler.inverse_transform([[up_p90_val]])[0, 0]

            down_p10 = self.amplitude_down_scaler.inverse_transform([[down_p10_val]])[0, 0]
            down_p90 = self.amplitude_down_scaler.inverse_transform([[down_p90_val]])[0, 0]

            amplitude_pred = max(up_p90, down_p90)
            amplitude_spread = max(up_p90 - up_p10, down_p90 - down_p10)

        return {
            "final_class": final_class,
            "final_confidence": confidence,
            "predicted_up_p10": float(up_p10),
            "predicted_up_p90": float(up_p90),
            "predicted_down_p10": float(down_p10),
            "predicted_down_p90": float(down_p90),
            "predicted_amplitude": float(amplitude_pred),
            "amplitude_spread": float(amplitude_spread)
        }

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

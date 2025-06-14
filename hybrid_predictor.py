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

        # Directional Model
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

        input_dim = len(self.feature_engineer.feature_columns)
        model_cfg = ModelConfig(input_dim=input_dim)

        self.direction_model = DirectionalModel(model_config=model_cfg)
        self.direction_model.load_state_dict(torch.load(CFG.paths.direction_model_path))
        self.direction_model.eval()

        self.amplitude_model = AmplitudeModel(input_size=input_dim)
        self.amplitude_model.load_state_dict(torch.load(CFG.paths.amplitude_model_path))
        self.amplitude_model.eval()

        self.temperature = joblib.load(CFG.paths.temperature_path)
        self.thresholds = joblib.load(CFG.paths.thresholds_path)

        self.amplitude_up_scaler = joblib.load(CFG.paths.amplitude_target_scaler_path.with_name("amplitude_up_scaler.joblib"))
        self.amplitude_down_scaler = joblib.load(CFG.paths.amplitude_target_scaler_path.with_name("amplitude_down_scaler.joblib"))

    def predict(self, df):
        features = self.feature_engineer.generate_features(df, fit=False)
        X_input = features.iloc[-1:].values.astype(np.float32)
        X_tensor = torch.tensor(X_input)

        # Direction prediction
        with torch.no_grad():
            logits = self.direction_model(X_tensor).numpy() / self.temperature
            probs = self.softmax(logits)

        final_class = np.argmax(probs)
        confidence = probs[0, final_class]

        # Amplitude prediction (двухголовая модель)
        with torch.no_grad():
            up_pred_norm, down_pred_norm = self.amplitude_model(X_tensor)
            up_pred_norm = up_pred_norm.numpy()[0, 0]
            down_pred_norm = down_pred_norm.numpy()[0, 0]

            up_pred = self.amplitude_up_scaler.inverse_transform([[up_pred_norm]])[0, 0]
            down_pred = self.amplitude_down_scaler.inverse_transform([[down_pred_norm]])[0, 0]

            amplitude_pred = max(up_pred, down_pred)

        hybrid_class = self._apply_thresholds(probs[0], amplitude_pred)

        return {
            "final_class": int(hybrid_class),
            "final_confidence": float(confidence),
            "predicted_up": float(up_pred),
            "predicted_down": float(down_pred),
            "predicted_amplitude": float(amplitude_pred)
        }

    def _apply_thresholds(self, probs, amplitude):
        prob_0, prob_1, prob_2 = probs

        if prob_0 > self.thresholds[0] and amplitude > CFG.hybrid.min_amplitude:
            return 0  # Short
        if prob_2 > self.thresholds[2] and amplitude > CFG.hybrid.min_amplitude:
            return 2  # Long
        return 1  # No trade

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

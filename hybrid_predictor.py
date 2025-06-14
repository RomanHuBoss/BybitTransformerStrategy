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

        self.amplitude_model = AmplitudeModel(input_size=len(self.feature_engineer.feature_columns))
        self.amplitude_model.load_state_dict(torch.load(CFG.paths.amplitude_model_path))
        self.amplitude_model.eval()

        self.temperature = joblib.load(CFG.paths.temperature_path)
        self.thresholds = joblib.load(CFG.paths.thresholds_path)
        self.amplitude_scaler = joblib.load(CFG.paths.amplitude_target_scaler_path)

    def predict(self, df):
        # Генерация признаков
        features = self.feature_engineer.generate_features(df, fit=False)

        # Берём последний сэмпл
        X_input = features.iloc[-1:].values.astype(np.float32)
        X_tensor = torch.tensor(X_input)

        # Directional prediction
        with torch.no_grad():
            logits = self.direction_model(X_tensor).numpy() / self.temperature
            probs = self.softmax(logits)

        final_class = np.argmax(probs)
        confidence = probs[0, final_class]

        # Amplitude prediction
        with torch.no_grad():
            amp_pred_norm = self.amplitude_model(X_tensor).numpy()[0, 0]
            amp_pred = self.amplitude_scaler.inverse_transform([[amp_pred_norm]])[0, 0]

        # Adaptive hybrid decision
        hybrid_class = self._apply_thresholds(probs[0], amp_pred)

        return {
            "final_class": int(hybrid_class),
            "final_confidence": float(confidence),
            "predicted_norm": float(amp_pred_norm),
            "predicted_amplitude": float(amp_pred)
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

import os
import json
import numpy as np
import pandas as pd
from inference_engine import InferenceEngine


class Predictor:
    def __init__(self, model_folder: str):
        self.model_folder = model_folder

        # Загрузка мета-информации модели
        meta_path = os.path.join(model_folder, "model_meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        scaler_params = meta.get("scaler_params")
        if scaler_params is None:
            raise ValueError("model_meta.json должен содержать scaler_params")

        scaler_mean = np.array(scaler_params["mean"])
        scaler_scale = np.array(scaler_params["scale"])

        # Инициализация движка инференса с прокинутым scaler
        self.engine = InferenceEngine(
            model_folder=model_folder,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale
        )

    def predict(self, df: pd.DataFrame):
        return self.engine.predict(df)

    def predict_with_logits(self, df: pd.DataFrame):
        return self.engine.predict_logits(df)

    def predict_proba(self, df: pd.DataFrame):
        return self.engine.predict_proba(df)

    def predict_confidence(self, df: pd.DataFrame):
        return self.engine.predict_confidence(df)

    def predict_class_distribution(self, df: pd.DataFrame):
        return self.engine.predict_class_distribution(df)

    def predict_high_confidence_signals(self, df: pd.DataFrame, threshold: float = 0.8):
        proba = self.predict_proba(df)
        preds = np.argmax(proba, axis=1)
        confidences = np.max(proba, axis=1)

        confident_indices = confidences >= threshold
        confident_preds = preds[confident_indices]
        confident_probs = proba[confident_indices]

        result = pd.DataFrame({
            "prediction": confident_preds,
            "confidence": confidences[confident_indices],
        })

        for i in range(proba.shape[1]):
            result[f"proba_{i}"] = confident_probs[:, i]

        return result.reset_index(drop=True)

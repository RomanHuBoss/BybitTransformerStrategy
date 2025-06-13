import os
import json
import numpy as np
import pandas as pd
from amplitude_inference import AmplitudeInferenceEngine

class AmplitudePredictor:
    def __init__(self, model_folder: str):
        self.model_folder = model_folder
        meta_path = os.path.join(model_folder, "model_meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        scaler_params = meta.get("scaler_params")
        scaler_mean = np.array(scaler_params["mean"])
        scaler_scale = np.array(scaler_params["scale"])

        self.engine = AmplitudeInferenceEngine(
            model_folder=model_folder,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale
        )

    def predict_amplitude(self, df: pd.DataFrame):
        return self.engine.predict(df)

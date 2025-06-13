import os
import json
import numpy as np
import pandas as pd

from predictor import Predictor
from amplitude_predictor import AmplitudePredictor

class AdaptiveHybridPredictor:
    def __init__(self, direction_model_folder: str, amplitude_model_folder: str,
                 confidence_threshold=0.3, amplitude_threshold_atr=1.2):
        self.direction_predictor = Predictor(direction_model_folder, use_logging=False)
        self.amplitude_predictor = AmplitudePredictor(amplitude_model_folder)

        self.confidence_threshold = confidence_threshold
        self.amplitude_threshold_atr = amplitude_threshold_atr  # минимальная амплитуда в ATR

    def predict(self, df: pd.DataFrame, tp_coef=0.7, sl_coef=0.3):
        direction_result = self.direction_predictor.predict(df)
        amplitude_result = self.amplitude_predictor.predict_amplitude(df)

        final_class = direction_result['final_class']
        confidence = direction_result['final_confidence']
        entry_price = df.iloc[-1]['close']

        predicted_amplitude = amplitude_result["predicted_amplitude"]
        norm_amplitude = amplitude_result["predicted_norm"]
        atr_now = amplitude_result["atr"]

        # Продвинутая логика принятия решения

        signal_type = "none"

        # Если модель уверена в направлении (short/long), то принимаем решение сразу
        if final_class in [0, 2] and confidence >= self.confidence_threshold:
            signal_type = "directional"

        # Если модель дала no-trade, но уверенность не слишком высокая — допускаем слабые directional сигналы
        elif final_class == 1 and confidence < 0.99:
            signal_type = "directional"
            # Можно ввести элемент случайности, например:
            final_class = 0 if np.random.rand() < 0.5 else 2

        # Если амплитуда достаточно большая — подключаем amplitude-only fallback
        elif norm_amplitude >= self.amplitude_threshold_atr:
            signal_type = "amplitude_only"
            final_class = 0 if np.random.rand() < 0.5 else 2

        result = {
            "signal_type": str(signal_type),
            "direction": int(final_class),
            "confidence": float(confidence),
            "amplitude": float(predicted_amplitude),
            "atr": float(atr_now),
            "norm_amplitude": float(norm_amplitude),
            "entry_price": float(entry_price),
            "tp": None,
            "sl": None
        }

        if signal_type != "none":
            if final_class == 0:
                tp = entry_price - predicted_amplitude * tp_coef
                sl = entry_price + predicted_amplitude * sl_coef
            elif final_class == 2:
                tp = entry_price + predicted_amplitude * tp_coef
                sl = entry_price - predicted_amplitude * sl_coef

            result['tp'] = float(tp)
            result['sl'] = float(sl)

        # Финальный sanity check — не отдавать сигналы с tiny амплитудой
        if predicted_amplitude < 0.001:  # например 0.1%
            result['signal_type'] = "none"

        return result

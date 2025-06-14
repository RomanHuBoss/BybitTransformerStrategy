import logging
import pandas as pd
import numpy as np
from config import CFG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class AmplitudeLabelGenerator:
    def __init__(self, shift=CFG.label_generation.amplitude_shift):
        self.shift = shift

    def generate_labels(self, df):
        high_future = df['high'].shift(-self.shift)
        low_future = df['low'].shift(-self.shift)
        current_close = df['close']

        up_amplitude = (high_future - current_close) / current_close
        down_amplitude = (current_close - low_future) / current_close

        up_amplitude = up_amplitude[:-self.shift]
        down_amplitude = down_amplitude[:-self.shift]

        return up_amplitude, down_amplitude

if __name__ == "__main__":
    logging.info("Запуск генератора меток Amplitude...")

    df = pd.read_csv(CFG.paths.train_csv)

    generator = AmplitudeLabelGenerator()
    up_ampl, down_ampl = generator.generate_labels(df)

    df = df.iloc[:-generator.shift].copy()
    df['amplitude_up'] = up_ampl
    df['amplitude_down'] = down_ampl

    df.to_csv(CFG.paths.train_csv, index=False)
    logging.info("✅ Amplitude метки успешно сгенерированы и сохранены.")

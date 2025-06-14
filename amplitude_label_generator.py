import logging
import pandas as pd
import numpy as np
from config import CFG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class AmplitudeQuantileLabelGenerator:
    def __init__(self, shift=CFG.label_generation.amplitude_shift, quantile_window=CFG.label_generation.quantile_window):
        self.shift = shift
        self.quantile_window = quantile_window

    def generate_labels(self, df):
        up_p10_list, up_p90_list = [], []
        down_p10_list, down_p90_list = [], []

        for idx in range(len(df) - self.shift - self.quantile_window + 1):
            window = df.iloc[idx + self.shift : idx + self.shift + self.quantile_window]

            high_future = window['high']
            low_future = window['low']
            current_close = df.iloc[idx]['close']

            up_amplitudes = (high_future - current_close) / current_close
            down_amplitudes = (current_close - low_future) / current_close

            up_p10 = np.percentile(up_amplitudes, 10)
            up_p90 = np.percentile(up_amplitudes, 90)
            down_p10 = np.percentile(down_amplitudes, 10)
            down_p90 = np.percentile(down_amplitudes, 90)

            up_p10_list.append(up_p10)
            up_p90_list.append(up_p90)
            down_p10_list.append(down_p10)
            down_p90_list.append(down_p90)

        return np.column_stack([up_p10_list, up_p90_list, down_p10_list, down_p90_list])

if __name__ == "__main__":
    logging.info("🚀 Запуск генератора квантильных меток Amplitude...")

    df = pd.read_csv(CFG.paths.train_csv)

    generator = AmplitudeQuantileLabelGenerator()
    labels = generator.generate_labels(df)

    np.save(CFG.paths.train_labels_amplitude, labels)
    logging.info(f"✅ Amplitude квантильные метки сгенерированы для {labels.shape[0]} строк и сохранены.")

import logging
import pandas as pd
import numpy as np
from config import CFG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class AmplitudeLabelGenerator:
    def __init__(self, shift=CFG.label_generation.amplitude_shift, quantile_window=CFG.label_generation.quantile_window):
        self.shift = shift
        self.quantile_window = quantile_window

        # Ограничения
        self.min_sl = 0.005  # 0.5%
        self.max_sl = 0.02   # 2%
        self.max_tp = 0.15   # 15%

    def generate_labels(self, df):
        up_p10_list, up_p90_list = [], []
        down_p10_list, down_p90_list = [], []

        for idx in range(len(df) - self.shift - self.quantile_window + 1):
            window = df.iloc[idx + self.shift: idx + self.shift + self.quantile_window]

            high_future = window['high']
            low_future = window['low']
            current_close = df.iloc[idx]['close']

            up_amplitudes = (high_future - current_close) / current_close
            down_amplitudes = (current_close - low_future) / current_close

            # Рассчитываем квантильные оценки
            raw_down_p10 = np.percentile(down_amplitudes, 10)
            raw_down_p90 = np.percentile(down_amplitudes, 90)

            # Ограничиваем SL
            down_p10 = np.clip(raw_down_p10, self.min_sl, self.max_sl)
            down_p90 = np.clip(raw_down_p90, self.min_sl, self.max_sl)

            # Минимальный TP привязан к SL (хотя бы в 2 раза больше SL)
            min_tp_p10 = 2 * down_p10
            min_tp_p90 = 2 * down_p90

            raw_up_p10 = np.percentile(up_amplitudes, 10)
            raw_up_p90 = np.percentile(up_amplitudes, 90)

            # Ограничиваем TP
            up_p10 = np.clip(max(raw_up_p10, min_tp_p10), min_tp_p10, self.max_tp)
            up_p90 = np.clip(max(raw_up_p90, min_tp_p90), min_tp_p90, self.max_tp)

            # Сохраняем
            up_p10_list.append(up_p10)
            up_p90_list.append(up_p90)
            down_p10_list.append(down_p10)
            down_p90_list.append(down_p90)

        return np.column_stack([up_p10_list, up_p90_list, down_p10_list, down_p90_list])

if __name__ == "__main__":
    logging.info("🚀 Запуск генератора нормализованных меток Amplitude...")

    df = pd.read_csv(CFG.paths.train_csv)

    generator = AmplitudeLabelGenerator()
    labels = generator.generate_labels(df)

    np.save(CFG.paths.train_labels_amplitude, labels)
    logging.info(f"✅ Нормализованные Amplitude метки сгенерированы для {labels.shape[0]} строк и сохранены.")

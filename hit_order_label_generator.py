import logging
import pandas as pd
import numpy as np
from config import CFG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class HitOrderLabelGenerator:
    def __init__(self):
        self.shift = CFG.label_generation.hit_order_shift
        self.label_window = CFG.label_generation.hit_order_window
        self.sl_min = CFG.label_generation.hit_order_sl_min
        self.sl_max = CFG.label_generation.hit_order_sl_max
        self.rr_min = CFG.label_generation.hit_order_rr_min
        self.rr_max = CFG.label_generation.hit_order_rr_max

    def generate_labels(self, df):
        labels = []

        for idx in range(len(df) - self.shift - self.label_window + 1):
            window = df.iloc[idx + self.shift: idx + self.shift + self.label_window]

            current_close = df.iloc[idx]['close']

            # случайный SL в заданном диапазоне
            sl_relative = np.random.uniform(self.sl_min, self.sl_max)
            sl_level = current_close * (1 - sl_relative)

            # RR → TP
            rr = np.random.uniform(self.rr_min, self.rr_max)
            tp_relative = sl_relative * rr
            tp_level = current_close * (1 + tp_relative)

            hit = 0  # SL по умолчанию
            for _, row in window.iterrows():
                if row['high'] >= tp_level:
                    hit = 1
                    break
                elif row['low'] <= sl_level:
                    hit = 0
                    break

            labels.append([sl_relative, tp_relative, hit])

        labels = np.array(labels)
        return labels

if __name__ == "__main__":
    logging.info("🚀 Запуск генератора HitOrder...")

    df = pd.read_csv(CFG.paths.train_csv)

    generator = HitOrderLabelGenerator()
    labels = generator.generate_labels(df)

    np.save(CFG.paths.train_labels_hitorder, labels)
    logging.info(f"✅ HitOrder метки сгенерированы: {labels.shape[0]} строк.")

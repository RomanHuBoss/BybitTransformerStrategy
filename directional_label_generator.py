import logging
import pandas as pd
import numpy as np
from config import CFG
from feature_engineering import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class DirectionalLabelGenerator:
    def __init__(self, shift=CFG.label_generation.direction_shift, threshold=CFG.label_generation.direction_threshold):
        self.shift = shift
        self.threshold = threshold

    def generate_labels(self, df):
        future_close = df['close'].shift(-self.shift)
        current_close = df['close']
        returns = (future_close - current_close) / current_close

        labels = np.where(returns > self.threshold, 2,
                  np.where(returns < -self.threshold, 0, 1))

        # Удалим последние shift строк с NaN
        labels = labels[:-self.shift]
        return labels

if __name__ == "__main__":
    logging.info("Запуск генератора меток Direction...")

    df = pd.read_csv(CFG.paths.train_csv)

    generator = DirectionalLabelGenerator()
    labels = generator.generate_labels(df)

    # Обрезаем DataFrame под длину меток
    df = df.iloc[:-generator.shift].copy()
    df['direction_class'] = labels

    df.to_csv(CFG.paths.train_csv, index=False)
    logging.info(f"✅ Direction метки сгенерированы для {len(labels)} строк и сохранены.")

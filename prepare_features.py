import pandas as pd
import numpy as np
import joblib
import logging
from feature_engineering import FeatureEngineer
from config import CFG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def prepare_features():
    logging.info("🚀 Старт генерации признаков...")
    df = pd.read_csv(CFG.paths.train_csv)
    logging.info(f"✅ Данные загружены: {df.shape[0]} строк")

    engineer = FeatureEngineer()
    df_features = engineer.generate_features(df, fit=True)

    joblib.dump(engineer.scaler, CFG.paths.scaler_path)
    joblib.dump(engineer.feature_columns, CFG.paths.feature_columns_path)
    logging.info("✅ Сохранён scaler и список признаков")

    df_features[engineer.feature_columns].to_csv(CFG.paths.train_features_csv, index=False)
    logging.info(f"✅ Features сохранены: {len(engineer.feature_columns)} признаков")

    np.save(CFG.paths.train_labels_direction, df['direction_class'].values)
    np.save(CFG.paths.train_labels_amplitude, df[['amplitude_up', 'amplitude_down']].values)
    logging.info("✅ Labels сохранены")

if __name__ == "__main__":
    prepare_features()

import pandas as pd
import numpy as np
import joblib
import logging
from feature_engineering import FeatureEngineer
from config import CFG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def prepare_features():
    logging.info("Загрузка данных...")
    df = pd.read_csv(CFG.paths.train_csv)

    engineer = FeatureEngineer()
    df_features = engineer.generate_features(df, fit=True)

    # сохраняем scaler и список признаков
    joblib.dump(engineer.scaler, CFG.paths.scaler_path)
    joblib.dump(engineer.feature_columns, CFG.paths.feature_columns_path)

    # сохраняем features
    df_features[engineer.feature_columns].to_csv(CFG.paths.train_features_csv, index=False)
    logging.info(f"Features сохранены: {len(engineer.feature_columns)} колонок")

    # сохраняем labels отдельно
    np.save(CFG.paths.train_labels_direction, df['direction_class'].values)
    np.save(CFG.paths.train_labels_amplitude, df[['amplitude_up', 'amplitude_down']].values)
    logging.info("Labels сохранены")

if __name__ == "__main__":
    prepare_features()

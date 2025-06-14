import pandas as pd
import joblib
import logging
from feature_engineering import FeatureEngineer
from config import CFG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def prepare_features():
    logging.info("Загрузка обучающих данных...")
    df = pd.read_csv(CFG.paths.train_csv)

    logging.info("Генерация признаков и обучение scaler...")
    engineer = FeatureEngineer()
    engineer.generate_features(df, fit=True)

    logging.info("Сохранение scaler и списка признаков...")
    joblib.dump(engineer.scaler, CFG.paths.scaler_path)
    joblib.dump(engineer.feature_columns, CFG.paths.feature_columns_path)

    logging.info("Подготовка признаков успешно завершена.")
    logging.info(f"Количество сгенерированных признаков: {len(engineer.feature_columns)}")

if __name__ == "__main__":
    prepare_features()

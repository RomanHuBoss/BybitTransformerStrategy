import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import logging
from config import CFG
from feature_engineering import FeatureEngineer
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

logging.info("🚀 Старт генерации признаков...")

# Загружаем основной датасет
logging.info("Загрузка основного train_csv...")
df = pd.read_csv(CFG.paths.train_csv)
logging.info(f"✅ Данные загружены: {len(df)} строк")

# Генерация признаков
engineer = FeatureEngineer()

# Проверяем, есть ли сохранённый master feature set
if os.path.exists(CFG.paths.feature_columns_path):
    logging.info("📌 Найден сохранённый features.joblib — загружаем master feature set.")
    master_features = joblib.load(CFG.paths.feature_columns_path)
    features = engineer.generate_features(df, fit=False)
    missing = set(master_features) - set(features.columns)
    if missing:
        raise ValueError(f"Отсутствуют признаки при повторной генерации: {missing}")
    features = features[master_features]
else:
    logging.info("📌 Первый запуск — формируем master feature set.")
    features = engineer.generate_features(df, fit=True)
    master_features = engineer.feature_columns
    joblib.dump(master_features, CFG.paths.feature_columns_path)
    logging.info(f"✅ Сохранён master feature set: {len(master_features)} признаков")

# Стандартизация
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features[master_features])
joblib.dump(scaler, CFG.paths.scaler_path)
logging.info("✅ Сохранён scaler")

# Сохраняем стандартизированные фичи
pd.DataFrame(scaled_features, columns=master_features).to_csv(CFG.paths.train_features_csv, index=False)
logging.info("✅ Сохранён train_features.csv")

# Проверка консистентности
assert scaled_features.shape[1] == len(master_features), "Feature size mismatch после стандартизации"

logging.info("🎯 Полная генерация признаков успешно завершена.")

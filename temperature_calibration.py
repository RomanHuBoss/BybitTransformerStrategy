import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import joblib
import logging

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

from dataset import SequenceDataset
from directional_label_generator import DirectionalLabelGenerator
from feature_engineering import FeatureEngineer
from model import DirectionalModel
from config import CFG

# --- Логирование ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_validation_logits_and_labels():
    logging.info("Загружаем обучающие данные...")
    df = pd.read_csv(CFG.paths.train_csv)

    engineer = FeatureEngineer()
    engineer.scaler = joblib.load(CFG.paths.scaler_path)
    engineer.feature_columns = joblib.load(CFG.paths.feature_columns_path)
    df_feat = engineer.generate_features(df, fit=False)

    generator = DirectionalLabelGenerator(
        shift=CFG.label_generation.direction_shift,
        threshold=CFG.label_generation.direction_threshold
    )
    labels = generator.generate_labels(df)
    labels_series = pd.Series(labels, index=df.index[:-generator.shift])

    # Синхронизация индексов с метками
    df_feat = df_feat.merge(
        labels_series.rename("label"),
        left_index=True, right_index=True, how="inner"
    )
    df_feat = df_feat.dropna(subset=["label"])

    X = df_feat.drop(columns=['label']).values
    y = df_feat['label'].astype(int).values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=CFG.train.val_size, shuffle=False
    )

    window_size = CFG.feature_engineering.window_size

    val_ds = SequenceDataset(X_val, y_val, window_size)
    val_loader = DataLoader(val_ds, batch_size=CFG.train.batch_size, shuffle=False)

    input_dim = X.shape[1]
    model_cfg = CFG.DirectionModelConfig()
    model_cfg.input_dim = input_dim

    model = DirectionalModel(model_config=model_cfg)
    model.load_state_dict(torch.load(CFG.paths.direction_model_path))
    model.eval()

    logits_list = []
    labels_list = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            logits = model(X_batch)
            logits_list.append(logits.numpy())
            labels_list.append(y_batch.numpy())

    logits = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    return logits, labels


def temperature_loss(T, logits, labels):
    T = T[0]
    scaled_logits = logits / T
    probs = F.softmax(torch.tensor(scaled_logits), dim=1).numpy()
    log_probs = np.log(probs + 1e-8)
    nll = -np.mean(log_probs[np.arange(len(labels)), labels])
    return nll


def calibrate_temperature():
    logging.info("🚀 Старт Temperature Calibration...")

    logits, labels = get_validation_logits_and_labels()

    logging.info("Оптимизируем температуру...")
    result = minimize(temperature_loss, [1.0], args=(logits, labels), bounds=[(0.5, 5.0)])
    optimal_T = result.x[0]

    logging.info(f"✅ Температура откалибрована: T = {optimal_T:.4f}")

    joblib.dump(optimal_T, CFG.paths.temperature_path)
    logging.info(f"Температура сохранена в {CFG.paths.temperature_path}")


if __name__ == '__main__':
    calibrate_temperature()

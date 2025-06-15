import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from config import CFG
from model import DirectionalModel  # <-- вот тут теперь всё корректно
from feature_engineering import FeatureEngineer


def load_data():
    features = pd.read_csv(CFG.paths.train_features_csv, index_col=0)
    feature_engineer = FeatureEngineer()
    feature_engineer.feature_columns = joblib.load(CFG.paths.feature_columns_path)
    feature_engineer.scaler = joblib.load(CFG.paths.scaler_path)
    features = feature_engineer.forward(features, window_size=CFG.train.direction_window_size)

    labels = np.load(CFG.paths.train_labels_direction)
    labels = labels[-len(features):]

    return features, labels


def load_model(input_dim):
    model_config = CFG.DirectionModelConfig()
    model_config.input_dim = input_dim
    model = DirectionalModel(model_config)
    state_dict = torch.load(CFG.paths.direction_model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_temperature(model, X_val, y_val):
    logits = []
    with torch.no_grad():
        for i in range(0, len(X_val), 512):
            batch = torch.tensor(X_val[i:i+512]).float()
            output = model(batch)
            logits.append(output)
    logits = torch.cat(logits)

    criterion = nn.CrossEntropyLoss()

    def loss_fn(temp):
        temp_tensor = torch.tensor(temp, dtype=torch.float32)
        scaled_logits = logits / temp_tensor
        loss = criterion(scaled_logits, torch.tensor(y_val))
        return loss.item()

    res = minimize(loss_fn, x0=np.array([1.0]), bounds=[(0.1, 10.0)])
    return res.x[0]


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=CFG.train.val_size, random_state=42)

    input_dim = X.shape[2]
    model = load_model(input_dim)

    temperature = evaluate_temperature(model, X_val, y_val)
    print(f"Лучшее значение температуры: {temperature:.4f}")

    joblib.dump(temperature, CFG.paths.temperature_path)
    print(f"Температура сохранена в {CFG.paths.temperature_path}")

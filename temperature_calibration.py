import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import logging

from config import CFG
from model import DirectionalModel

logging.basicConfig(level=logging.INFO)

# Загрузка данных
logging.info("Загружаем признаки и метки...")
X_full_df = pd.read_csv(CFG.paths.train_features_csv)
feature_columns = list(X_full_df.columns)
X_full = X_full_df.values

y = np.load(CFG.paths.train_labels_direction)

# Разделение на train/val
X_train, X_val, y_train, y_val = train_test_split(X_full, y, test_size=CFG.train.val_size, shuffle=False)

# Загрузка модели
logging.info("Загружаем модель...")
input_dim = X_full.shape[1]
model = DirectionalModel(input_dim=input_dim)
state_dict = torch.load(CFG.paths.direction_model_path)
model.load_state_dict(state_dict)
model.eval()


# Калибровка температуры
def softmax_temperature(logits, temperature):
    logits = logits / temperature
    return F.softmax(torch.tensor(logits), dim=1).numpy()


def nll_loss_temperature(temperature, logits, labels):
    probs = softmax_temperature(logits, temperature[0])
    true_probs = probs[np.arange(len(labels)), labels]
    nll = -np.mean(np.log(true_probs + 1e-12))
    return nll


# Получение логитов на валидации
with torch.no_grad():
    logits = model(torch.tensor(X_val, dtype=torch.float32)).numpy()

# Поиск оптимальной температуры
logging.info("Запускаем оптимизацию температуры...")
res = minimize(nll_loss_temperature, x0=[1.0], args=(logits, y_val), bounds=[(0.5, 10.0)])

optimal_temperature = res.x[0]
logging.info(f"Оптимальная температура: {optimal_temperature:.4f}")

# Сохраняем температуру
joblib.dump(optimal_temperature, CFG.paths.temperature_path)
logging.info("Температура успешно сохранена.")

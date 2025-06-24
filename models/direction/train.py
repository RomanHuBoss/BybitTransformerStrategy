# Models/direction/train.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from dataset import DirectionDataset
from model import DirectionModel
from losses import BCELossWithLogits
from config import CFG
from labeler import create_labels

from sklearn.preprocessing import StandardScaler
import joblib
import logging

# Логгер
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train.log")
    ]
)
log = logging.getLogger(__name__)

# Загрузка и подготовка данных
df = pd.read_csv(CFG.paths.data_path)
labels = create_labels(df)
features = df.drop(columns=CFG.pre.drop_columns, errors='ignore').values

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

os.makedirs(os.path.dirname(CFG.paths.scaler_path), exist_ok=True)
joblib.dump(scaler, CFG.paths.scaler_path)

dataset = DirectionDataset(features_scaled, labels.values)

val_size = int(len(dataset) * CFG.train.val_size)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=CFG.train.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CFG.train.batch_size)

# Модель, лосс, оптимизатор
model = DirectionModel(
    CFG.arch.input_size,
    CFG.arch.hidden_size,
    CFG.arch.num_layers,
    CFG.arch.dropout
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

pos_weight = torch.tensor((labels == 0).sum() / (labels == 1).sum()).to(device)
criterion = BCELossWithLogits(pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=CFG.train.learning_rate)

# Early stopping
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None

# Цикл обучения
for epoch in range(CFG.train.epochs):
    model.train()
    train_losses = []

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    val_losses = []
    model.eval()
    with torch.no_grad():
        for val_X, val_y in val_loader:
            val_X, val_y = val_X.to(device), val_y.to(device)
            outputs = model(val_X)
            loss = criterion(outputs, val_y)
            val_losses.append(loss.item())

    val_loss_mean = np.mean(val_losses)
    log.info(f"Epoch [{epoch+1}/{CFG.train.epochs}], Train Loss: {np.mean(train_losses):.4f}, Val Loss: {val_loss_mean:.4f}")

    if val_loss_mean < best_val_loss:
        best_val_loss = val_loss_mean
        epochs_no_improve = 0
        best_model_state = model.state_dict()
        log.info("Model improved.")
    else:
        epochs_no_improve += 1
        log.info(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= CFG.train.early_stopping_patience:
            log.info("Early stopping triggered.")
            break

# Финальное сохранение лучшей модели
if best_model_state is not None:
    os.makedirs(os.path.dirname(CFG.paths.model_path), exist_ok=True)
    torch.save(best_model_state, CFG.paths.model_path)
    log.info(f"Best model saved to {CFG.paths.model_path}")
else:
    log.warning("No model improvement detected — nothing was saved.")

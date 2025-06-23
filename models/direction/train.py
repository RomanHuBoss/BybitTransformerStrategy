# Models/direction/train.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from dataset import DirectionDataset
from model import DirectionModel
from losses import BCELossWithLogits
from config import config
from labeler import create_labels

from sklearn.preprocessing import StandardScaler
import joblib

# Загружаем данные
df = pd.read_csv(config.DATA_PATH)
labels = create_labels(df)
features = df.drop(columns=['label', 'timestamp', 'symbol'], errors='ignore').values

# Нормализация
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Сохраняем scaler
os.makedirs(os.path.dirname(config.SCALER_PATH), exist_ok=True)
joblib.dump(scaler, config.SCALER_PATH)

# Dataset
dataset = DirectionDataset(features_scaled, labels.values)

val_size = int(len(dataset) * config.VALIDATION_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

# Model
model = DirectionModel(config.INPUT_SIZE, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss & Optimizer
pos_weight = torch.tensor((labels == 0).sum() / (labels == 1).sum()).to(device)
criterion = BCELossWithLogits(pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

# Train loop
for epoch in range(config.EPOCHS):
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

    print(f"Epoch [{epoch+1}/{config.EPOCHS}], Train Loss: {np.mean(train_losses):.4f}, Val Loss: {np.mean(val_losses):.4f}")

# Save model
os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), config.MODEL_PATH)

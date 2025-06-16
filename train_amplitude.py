import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import load_train_features, load_train_labels_amplitude, AmplitudeDataset
from model import AmplitudeRegressor
from losses import AmplitudeLoss
from config import CFG

logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"⚡ Работаем на устройстве: {device}")

logging.info("🚀 Загружаем признаки и амплитудные метки...")
X = load_train_features()
y = load_train_labels_amplitude()

# Разделяем train/val
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=CFG.train.val_size, shuffle=False
)

train_dataset = AmplitudeDataset(X_train, y_train)
val_dataset = AmplitudeDataset(X_val, y_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CFG.train.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CFG.train.batch_size)

model = AmplitudeRegressor(input_size=X.shape[1]).to(device)
loss_fn = AmplitudeLoss(weights=CFG.amplitude.loss_weights, device=device)
optimizer = optim.AdamW(model.parameters(), lr=CFG.train.lr)

best_val_loss = np.inf
patience_counter = 0

logging.info("🧮 Запуск обучения Amplitude модели...")

for epoch in range(1, CFG.train.epochs + 1):
    model.train()
    train_losses = []

    for xb, yb in tqdm(train_loader, desc=f"Эпоха {epoch} [обучение]", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            val_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses)
    avg_val_loss = np.mean(val_losses)

    logging.info(f"📊 Эпоха {epoch}: Train Loss={avg_train_loss:.6f} | Val Loss={avg_val_loss:.6f}")

    if avg_val_loss < best_val_loss - 1e-6:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), str(CFG.paths.amplitude_model_path))
        logging.info("🎯 Новая лучшая модель сохранена.")
    else:
        patience_counter += 1
        if patience_counter >= CFG.train.early_stopping_patience:
            logging.info("⏸ Early stopping — остановка обучения.")
            break

logging.info("✅ Обучение Amplitude модели завершено.")

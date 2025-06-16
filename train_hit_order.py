import pandas as pd
import numpy as np
import torch
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from config import CFG
from model import HitOrderClassifier
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Гиперпараметры из конфига
EPOCHS = CFG.hitorder.epochs
BATCH_SIZE = CFG.hitorder.batch_size
LR = CFG.hitorder.lr

# Загрузка данных
logging.info("🚀 Загружаем признаки...")
df = pd.read_csv(CFG.paths.train_features_csv)

# Загрузка feature_columns — строго то же, что использовалось при обучении scaler!
feature_cols = joblib.load(CFG.paths.feature_columns_path)
scaler = joblib.load(CFG.paths.scaler_path)

# Подготовка нормализованных признаков
X = df[feature_cols]
X_scaled = scaler.transform(X)

# Берём все hitorder профили
hit_cols = [col for col in df.columns if col.startswith("hit_SL")]

# Обучаем по каждому профилю
for profile_column in hit_cols:
    logging.info(f"\n🚀 Обучаем профиль: {profile_column}")
    y = df[profile_column].values

    pos_frac = np.mean(y)
    pos_weight = (len(y) - np.sum(y)) / np.sum(y)
    logging.info(f"✅ Баланс классов: POS={np.sum(y)}, NEG={len(y)-np.sum(y)}, POS%={pos_frac*100:.3f}%")
    logging.info(f"⚖️ Расчетный pos_weight: {pos_weight:.4f}")

    # Разделяем на train/val
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=CFG.train.val_size, shuffle=False)

    train_data = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_data = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE)

    model = HitOrderClassifier(input_dim=X_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    best_val_loss = np.inf

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb).squeeze()
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * xb.size(0)
        train_loss = total_train_loss / len(train_loader.dataset)

        model.eval()
        total_val_loss = 0
        val_preds = []
        val_true = []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb).squeeze()
                loss = criterion(logits, yb)
                total_val_loss += loss.item() * xb.size(0)
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(yb.cpu().numpy())
        val_loss = total_val_loss / len(val_loader.dataset)

        acc = accuracy_score(val_true, val_preds)
        bal_acc = balanced_accuracy_score(val_true, val_preds)
        f1 = f1_score(val_true, val_preds)

        logging.info(f"📊 Эпоха {epoch}: Train Loss={train_loss:.6f} | Val Loss={val_loss:.6f} | Acc={acc:.4f} | Balanced Acc={bal_acc:.4f} | F1={f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            profile_safe = profile_column.replace(".", "_")
            model_path = os.path.join(CFG.paths.models_dir, f"hitorder_{profile_safe}.pth")
            torch.save(model.state_dict(), model_path)
            logging.info(f"🎯 Лучшая модель сохранена: {model_path}")

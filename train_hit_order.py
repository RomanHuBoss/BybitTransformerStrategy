import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import os
import logging

from config import CFG
from model import HitOrderClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# Обновленный FocalLoss с поддержкой pos_weight
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# Датасет
class HitOrderDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Тренировка одного профиля
def train_one_profile(df, profile, feature_cols):
    sl, rr = profile
    label_col = f"hit_SL{str(sl).replace('.', '_')}_RR{str(rr).replace('.', '_')}"
    if label_col not in df.columns:
        logging.warning(f"Пропускаем профиль {label_col}, колонка отсутствует.")
        return

    logging.info(f"\n🚀 Обучаем профиль: {label_col}")

    X = df[feature_cols].values
    y = df[label_col].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=CFG.train.val_size, random_state=42
    )

    train_ds = HitOrderDataset(X_train, y_train)
    val_ds = HitOrderDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=CFG.hitorder.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.hitorder.batch_size, shuffle=False)

    model = HitOrderClassifier(input_dim=X.shape[1]).to(CFG.hitorder.device)

    pos_weight_val = (len(y) - y.sum()) / y.sum()
    pos_weight_tensor = torch.tensor(pos_weight_val, dtype=torch.float32).to(CFG.hitorder.device)

    criterion = FocalLoss(alpha=1.0, gamma=CFG.train.focal_gamma, pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.hitorder.lr)
    best_val_loss = float('inf')
    patience_counter = 0

    logging.info(f"✅ Баланс классов: POS={int(y.sum())}, NEG={int(len(y) - y.sum())}, POS%={y.mean() * 100:.3f}%")
    logging.info(f"⚖️ Расчетный pos_weight: {pos_weight_val:.4f}")

    for epoch in range(1, CFG.hitorder.epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(CFG.hitorder.device), yb.to(CFG.hitorder.device)
            optimizer.zero_grad()
            out = model(xb).squeeze()
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, preds, targets = [], [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(CFG.hitorder.device), yb.to(CFG.hitorder.device)
                out = model(xb).squeeze()
                loss = criterion(out, yb)
                val_losses.append(loss.item())
                preds.append(torch.sigmoid(out).cpu().numpy())
                targets.append(yb.cpu().numpy())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        preds_binary = (preds > 0.5).astype(int)

        acc = (preds_binary == targets).mean()
        precision = precision_score(targets, preds_binary, zero_division=0)
        recall = recall_score(targets, preds_binary, zero_division=0)
        f1 = f1_score(targets, preds_binary, zero_division=0)
        val_loss = np.mean(val_losses)

        logging.info(
            f"📊 Эпоха {epoch}: "
            f"Train Loss={np.mean(train_losses):.6f} | Val Loss={val_loss:.6f} | "
            f"Acc={acc:.4f} | Precision={precision:.4f} | Recall={recall:.4f} | F1={f1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model_path = os.path.join(CFG.paths.models_dir, f"hitorder_{label_col}.pth")
            torch.save(model.state_dict(), model_path)
            logging.info(f"🎯 Лучшая модель сохранена: {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= CFG.hitorder.early_stopping_patience:
                logging.info("⏹ Early stopping сработал.")
                break


# Основной запуск
if __name__ == "__main__":
    df = pd.read_parquet(CFG.paths.feature_dataset_path)
    scaler = joblib.load(CFG.paths.scaler_path)
    feature_cols = joblib.load(CFG.paths.feature_columns_path)
    df[feature_cols] = scaler.transform(df[feature_cols])

    for sl in CFG.label_generation.hitorder_sl_list:
        for rr in CFG.label_generation.hitorder_rr_list:
            train_one_profile(df, (sl, rr), feature_cols)

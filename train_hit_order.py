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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# === Label Smoothing ===
def smooth_labels(y, smoothing=0.01):
    return y * (1 - smoothing) + 0.5 * smoothing


# === Asymmetric Focal Loss (модифицированная) ===
class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_pos=0.5, gamma_neg=1.5, pos_weight=None):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1 - prob) * (1 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight
        )
        focal_weight = (1 - pt) ** gamma
        return (focal_weight * loss).mean()


# === Датасет ===
class HitOrderDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# === Стабильная модель v4_smooth ===
class HitOrderClassifierStable(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


# === Поиск лучшего threshold ===
def find_best_threshold(y_true, y_pred_prob):
    best_thr = 0.5
    best_f1 = 0
    for thr in np.arange(0.1, 0.91, 0.05):
        y_pred_bin = (y_pred_prob > thr).astype(int)
        f1 = f1_score(y_true, y_pred_bin, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1


# === Тренировка одного профиля ===
def train_one_profile(df, profile, feature_cols):
    sl, rr = profile
    label_col = f"hit_SL{str(sl).replace('.', '_')}_RR{str(rr).replace('.', '_')}"
    if label_col not in df.columns:
        logging.warning(f"Пропускаем профиль {label_col}, колонка отсутствует.")
        return

    logging.info(f"\n🚀 Обучаем профиль: {label_col}")

    X = df[feature_cols].values
    y = df[label_col].values

    y_smooth = smooth_labels(y, smoothing=0.01)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_smooth, test_size=CFG.train.val_size, random_state=42, stratify=y
    )

    train_ds = HitOrderDataset(X_train, y_train)
    val_ds = HitOrderDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=CFG.hitorder.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.hitorder.batch_size, shuffle=False)

    model = HitOrderClassifierStable(input_dim=X.shape[1]).to(CFG.hitorder.device)

    pos_weight_val = (len(y) - y.sum()) / y.sum()
    pos_weight_tensor = torch.tensor(pos_weight_val, dtype=torch.float32).to(CFG.hitorder.device)

    criterion = AsymmetricFocalLoss(gamma_pos=0.5, gamma_neg=1.5, pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.hitorder.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )

    best_f1 = 0
    patience_counter = 0

    logging.info(f"✅ Баланс классов: POS={int(y.sum())}, NEG={int(len(y) - y.sum())}, POS%={y.mean() * 100:.3f}%")
    logging.info(f"⚖️ pos_weight: {pos_weight_val:.4f}")

    for epoch in range(1, CFG.hitorder.epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(CFG.hitorder.device), yb.to(CFG.hitorder.device)
            optimizer.zero_grad()
            out = model(xb).squeeze()
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        preds_bin = (preds > 0.5).astype(int)

        true_labels = (targets > 0.5).astype(int)

        acc = (preds_bin == true_labels).mean()
        precision = precision_score(true_labels, preds_bin, zero_division=0)
        recall = recall_score(true_labels, preds_bin, zero_division=0)
        f1 = f1_score(true_labels, preds_bin, zero_division=0)
        val_loss = np.mean(val_losses)

        scheduler.step(val_loss)

        logging.info(
            f"📊 Эпоха {epoch}: Train Loss={np.mean(train_losses):.6f} | "
            f"Val Loss={val_loss:.6f} | Acc={acc:.4f} | Precision={precision:.4f} | Recall={recall:.4f} | F1={f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            model_path = os.path.join(CFG.paths.models_dir, f"hitorder_{label_col}.pth")
            torch.save(model.state_dict(), model_path)
            logging.info(f"🎯 Лучшая модель сохранена: {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= CFG.hitorder.early_stopping_patience:
                logging.info("⏹ Early stopping по F1.")
                break

    # Подбор threshold:
    logging.info("🔎 Подбор threshold...")
    model.eval()
    full_preds, full_targets = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(CFG.hitorder.device), yb.to(CFG.hitorder.device)
            out = model(xb).squeeze()
            full_preds.append(torch.sigmoid(out).cpu().numpy())
            full_targets.append(yb.cpu().numpy())

    full_preds = np.concatenate(full_preds)
    full_targets = np.concatenate(full_targets)
    true_labels = (full_targets > 0.5).astype(int)

    best_thr, best_thr_f1 = find_best_threshold(true_labels, full_preds)
    logging.info(f"✅ Лучший threshold: {best_thr:.3f} (F1={best_thr_f1:.4f})")

    threshold_path = os.path.join(CFG.paths.models_dir, f"hitorder_{label_col}_threshold.joblib")
    joblib.dump(best_thr, threshold_path)


# === Основной запуск ===
if __name__ == "__main__":
    df = pd.read_parquet(CFG.paths.feature_dataset_path)
    scaler = joblib.load(CFG.paths.scaler_path)
    feature_cols = joblib.load(CFG.paths.feature_columns_path)
    df[feature_cols] = scaler.transform(df[feature_cols])

    for sl in CFG.label_generation.hitorder_sl_list:
        for rr in CFG.label_generation.hitorder_rr_list:
            train_one_profile(df, (sl, rr), feature_cols)

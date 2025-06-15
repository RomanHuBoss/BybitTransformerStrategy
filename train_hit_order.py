import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from dataset import load_train_features, load_train_labels_hitorder, HitOrderDataset
from model import HitOrderClassifier
from config import CFG

logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"⚡️ Работаем на устройстве: {device}")

    logging.info("🚀 Загружаем признаки и метки HitOrder...")

    # Загружаем features (инженерные признаки)
    X = load_train_features()

    # Загружаем метки: sl_relative, tp_relative, hit
    sl_tp, y = load_train_labels_hitorder()

    # Немного статистики по классам
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    pos_weight_value = n_neg / (n_pos + 1e-8)
    logging.info(f"✅ Баланс классов: POS={n_pos}, NEG={n_neg}, POS%={n_pos / len(y):.3%}")
    logging.info(f"⚖️ Расчетный pos_weight: {pos_weight_value:.4f}")

    # Сплитим на train/val
    X_train, X_val, sl_tp_train, sl_tp_val, y_train, y_val = train_test_split(
        X, sl_tp, y, test_size=CFG.train.val_size, shuffle=False
    )

    # Создаем датасеты и лоадеры
    train_dataset = HitOrderDataset(X_train, sl_tp_train, y_train)
    val_dataset = HitOrderDataset(X_val, sl_tp_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CFG.train.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CFG.train.batch_size)

    # Создаем модель с новым размером входа (+2 признака)
    input_size = X.shape[1] + 2
    model = HitOrderClassifier(input_size=input_size).to(device)

    # Потери
    pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.train.lr)

    best_val_loss = np.inf
    patience_counter = 0
    logging.info("🧮 Старт обучения HitOrder модели...")

    for epoch in range(1, CFG.train.epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        all_preds, all_targets = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                val_losses.append(loss.item())

                probs = torch.sigmoid(logits)
                preds_class = (probs > 0.5).int().cpu().numpy()
                all_preds.extend(preds_class.flatten())
                all_targets.extend(yb.cpu().numpy().flatten())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        acc = accuracy_score(all_targets, all_preds)
        bal_acc = balanced_accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds)

        logging.info(f"📊 Эпоха {epoch}: Train Loss={avg_train_loss:.6f} | "
                     f"Val Loss={avg_val_loss:.6f} | Acc={acc:.4f} | "
                     f"Balanced Acc={bal_acc:.4f} | F1={f1:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CFG.paths.hit_order_model_path)
            logging.info("🎯 Лучшая модель обновлена.")
        else:
            patience_counter += 1
            if patience_counter >= CFG.train.early_stopping_patience:
                logging.info("⏸ Early stopping — остановка обучения.")
                break

    logging.info("✅ Обучение HitOrder модели завершено.")

if __name__ == "__main__":
    main()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from dataset import load_train_features
from config import CFG

logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"⚡ Работаем на устройстве: {device}")

logging.info("🚀 Загружаем признаки...")
X = load_train_features()

# ❗ Здесь ты должен подставить свои реальные HitOrder метки:
# Вместо этой строчки - загрузка своих сохранённых меток из npy или csv.
y = np.random.randint(0, 2, size=(X.shape[0],))  # <-- заменить на реальные метки

n_pos = np.sum(y == 1)
n_neg = np.sum(y == 0)
pos_weight_value = n_neg / (n_pos + 1e-8)
logging.info(f"✅ Баланс классов: POS={n_pos}, NEG={n_neg}, POS%={n_pos / len(y):.3%}")
logging.info(f"⚖️ Расчетный pos_weight: {pos_weight_value:.4f}")

# Сплит на train/val
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=CFG.train.val_size, shuffle=False
)

# Torch датасеты
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)
val_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32)
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CFG.train.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CFG.train.batch_size)

# Модель
class HitOrderClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

model = HitOrderClassifier(input_size=X.shape[1]).to(device)

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
    val_losses, all_preds, all_targets = [], [], []

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

    logging.info(f"📊 Эпоха {epoch}: Train Loss={avg_train_loss:.6f} | Val Loss={avg_val_loss:.6f} | "
                 f"Acc={acc:.4f} | Balanced Acc={bal_acc:.4f} | F1={f1:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), CFG.paths.hit_order_model_path)
        logging.info("🎯 Лучшая модель сохранена.")
    else:
        patience_counter += 1
        if patience_counter >= CFG.train.early_stopping_patience:
            logging.info("⏸ Early stopping.")
            break

logging.info("✅ Обучение HitOrder модели завершено.")

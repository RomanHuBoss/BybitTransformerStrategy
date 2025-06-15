import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import HitOrderClassifier
from dataset import HitOrderDataset
from config import CFG
import joblib
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class HitOrderTrainer:
    def __init__(self):
        logging.info("🚀 Загружаем датасет HitOrder...")

        self.dataset = HitOrderDataset()
        self.dataloader = DataLoader(self.dataset, batch_size=CFG.train.batch_size, shuffle=True)

        # Загружаем feature_columns, чтобы определить размерность входа модели
        feature_columns = joblib.load(CFG.paths.feature_columns_path)
        input_size = len(feature_columns)

        self.model = HitOrderClassifier(input_size)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=CFG.train.lr)

    def train(self):
        logging.info("🚀 Старт обучения HitOrderClassifier")
        self.model.train()

        best_loss = float('inf')
        epochs_no_improve = 0
        patience = CFG.train.early_stopping_patience

        for epoch in range(CFG.train.epochs):
            total_loss = 0
            y_true, y_pred = [], []

            for X_batch, y_batch in self.dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch).squeeze()
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                preds = (outputs.detach() >= 0.5).int()
                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

            avg_loss = total_loss / len(self.dataloader)
            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            logging.info(f"🧮 Epoch {epoch + 1}: Loss={avg_loss:.6f}, "
                         f"Accuracy={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), CFG.paths.hit_order_model_path)
                logging.info("🎯 Новый лучший результат. Модель сохранена.")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logging.info("🛑 Ранняя остановка обучения.")
                break

        logging.info("✅ Обучение завершено.")


if __name__ == '__main__':
    trainer = HitOrderTrainer()
    trainer.train()

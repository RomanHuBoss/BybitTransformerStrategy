import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib
import logging

from feature_engineering import FeatureEngineer
from model import AmplitudeModel
from amplitude_label_generator import AmplitudeLabelGenerator
from config import CFG


class AmplitudeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AmplitudeTrainer:
    def __init__(self):
        self.df = pd.read_csv(CFG.paths.train_csv)
        logging.info(f"Загружено {len(self.df)} строк.")
        self.engineer = FeatureEngineer()

        logging.info("Генерация признаков...")
        self.df_feat = self.engineer.generate_features(self.df, fit=True)

        joblib.dump(self.engineer.scaler, CFG.paths.amplitude_scaler_path)

        logging.info("Генерация амплитудных меток...")
        generator = AmplitudeLabelGenerator()
        amplitude_targets = generator.generate_amplitude_labels(self.df)

        # Нормализация амплитуды через ATR
        if 'atr_long_pct' in self.df_feat.columns:
            atr = self.df_feat['atr_long_pct'].values
        else:
            raise ValueError("ATR признака не найдено в feature engineering")

        y_norm = amplitude_targets / (atr * self.df['close'].values + 1e-8)
        self.df_feat['label'] = y_norm

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.df_feat.drop(columns=['label']),
            self.df_feat['label'],
            test_size=CFG.train.val_size,
            shuffle=False
        )

        self.scaler_y = RobustScaler()
        self.y_train_scaled = self.scaler_y.fit_transform(self.y_train.values.reshape(-1, 1))
        self.y_val_scaled = self.scaler_y.transform(self.y_val.values.reshape(-1, 1))

        joblib.dump(self.scaler_y, CFG.paths.amplitude_target_scaler_path)

        self.train_ds = AmplitudeDataset(self.X_train.values, self.y_train_scaled)
        self.val_ds = AmplitudeDataset(self.X_val.values, self.y_val_scaled)

        self.model = AmplitudeModel(input_size=self.X_train.shape[1])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=CFG.train.lr)
        self.criterion = nn.MSELoss()

    def train(self):
        best_loss = np.inf

        train_loader = DataLoader(self.train_ds, batch_size=CFG.train.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_ds, batch_size=CFG.train.batch_size, shuffle=False)

        for epoch in range(1, CFG.train.epochs + 1):
            self.model.train()
            total_loss = 0

            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            val_loss = self.validate(val_loader)

            logging.info(
                f"Эпоха {epoch} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), CFG.paths.amplitude_model_path)
                logging.info("Сохранена новая лучшая модель.")

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)
                total_loss += loss.item()

        return total_loss / len(val_loader)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    trainer = AmplitudeTrainer()
    trainer.train()

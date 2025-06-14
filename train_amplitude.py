import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib
import logging
import os

from feature_engineering import FeatureEngineer
from amplitude_label_generator import AmplitudeLabelGenerator
from config import CFG

# Полностью встроенная двухголовая модель
class AmplitudeModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.up_head = nn.Linear(128, 1)
        self.down_head = nn.Linear(128, 1)

    def forward(self, x):
        shared_out = self.shared(x)
        up_out = self.up_head(shared_out)
        down_out = self.down_head(shared_out)
        return up_out, down_out


class AmplitudeDataset(Dataset):
    def __init__(self, X, y_up, y_down):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_up = torch.tensor(y_up, dtype=torch.float32).view(-1, 1)
        self.y_down = torch.tensor(y_down, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_up[idx], self.y_down[idx]


class AmplitudeTrainer:
    def __init__(self):
        self.df = pd.read_csv(CFG.paths.train_csv)
        logging.info(f"Загружено {len(self.df)} строк.")

        self.engineer = FeatureEngineer()
        logging.info("Генерация признаков...")
        self.df_feat = self.engineer.generate_features(self.df, fit=True)

        os.makedirs(os.path.dirname(CFG.paths.amplitude_scaler_path), exist_ok=True)
        joblib.dump(self.engineer.scaler, CFG.paths.amplitude_scaler_path)

        logging.info("Генерация амплитудных меток...")
        generator = AmplitudeLabelGenerator(CFG.labels.lookahead)
        up_targets, down_targets = generator.generate_amplitude_labels(self.df)

        targets_df = pd.DataFrame({
            "up_target": up_targets,
            "down_target": down_targets
        }, index=self.df.index)

        # Синхронизация по индексам
        self.df_feat = self.df_feat.merge(targets_df, left_index=True, right_index=True, how="inner")

        if 'atr_long_pct' not in self.df_feat.columns:
            raise ValueError("ATR признака не найдено в feature engineering")

        atr = self.df_feat['atr_long_pct'].values
        close = self.df.loc[self.df_feat.index, 'close'].values

        up_targets = self.df_feat['up_target'].values
        down_targets = self.df_feat['down_target'].values

        valid_idx = (atr > 1e-8) & (~np.isnan(up_targets)) & (~np.isnan(down_targets))
        atr = atr[valid_idx]
        close = close[valid_idx]
        up_targets = up_targets[valid_idx]
        down_targets = down_targets[valid_idx]
        self.df_feat = self.df_feat.iloc[valid_idx]

        y_up_raw = up_targets / (atr * close + 1e-8)
        y_down_raw = down_targets / (atr * close + 1e-8)

        y_up_norm = np.log1p(y_up_raw)
        y_down_norm = np.log1p(y_down_raw)

        self.df_feat['up_norm'] = y_up_norm
        self.df_feat['down_norm'] = y_down_norm

        self.X_train, self.X_val, self.y_up_train, self.y_up_val, self.y_down_train, self.y_down_val = train_test_split(
            self.df_feat.drop(columns=['up_norm', 'down_norm', 'up_target', 'down_target']),
            self.df_feat['up_norm'],
            self.df_feat['down_norm'],
            test_size=CFG.train.val_size,
            shuffle=False
        )

        self.scaler_up = RobustScaler()
        self.scaler_down = RobustScaler()

        self.y_up_train_scaled = self.scaler_up.fit_transform(self.y_up_train.values.reshape(-1, 1))
        self.y_up_val_scaled = self.scaler_up.transform(self.y_up_val.values.reshape(-1, 1))

        self.y_down_train_scaled = self.scaler_down.fit_transform(self.y_down_train.values.reshape(-1, 1))
        self.y_down_val_scaled = self.scaler_down.transform(self.y_down_val.values.reshape(-1, 1))

        joblib.dump(self.scaler_up, CFG.paths.amplitude_target_scaler_path.with_name("amplitude_up_scaler.joblib"))
        joblib.dump(self.scaler_down, CFG.paths.amplitude_target_scaler_path.with_name("amplitude_down_scaler.joblib"))

        self.train_ds = AmplitudeDataset(self.X_train.values, self.y_up_train_scaled, self.y_down_train_scaled)
        self.val_ds = AmplitudeDataset(self.X_val.values, self.y_up_val_scaled, self.y_down_val_scaled)

        self.model = AmplitudeModel(input_size=self.X_train.shape[1])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=CFG.train.lr)
        self.criterion = nn.HuberLoss(delta=1.0)

    def train(self):
        best_loss = np.inf

        train_loader = DataLoader(self.train_ds, batch_size=CFG.train.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_ds, batch_size=CFG.train.batch_size, shuffle=False)

        for epoch in range(1, CFG.train.epochs + 1):
            self.model.train()
            total_loss = 0

            for X_batch, y_up_batch, y_down_batch in train_loader:
                self.optimizer.zero_grad()
                pred_up, pred_down = self.model(X_batch)
                loss_up = self.criterion(pred_up, y_up_batch)
                loss_down = self.criterion(pred_down, y_down_batch)
                loss = loss_up + loss_down
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            val_loss = self.validate(val_loader)

            logging.info(f"Эпоха {epoch} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), CFG.paths.amplitude_model_path)
                logging.info("Сохранена новая лучшая модель.")

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for X_batch, y_up_batch, y_down_batch in val_loader:
                pred_up, pred_down = self.model(X_batch)
                loss_up = self.criterion(pred_up, y_up_batch)
                loss_down = self.criterion(pred_down, y_down_batch)
                loss = loss_up + loss_down
                total_loss += loss.item()

        return total_loss / len(val_loader)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    trainer = AmplitudeTrainer()
    trainer.train()

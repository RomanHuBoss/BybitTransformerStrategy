import torch
import torch.nn as nn
from imblearn.under_sampling import TomekLinks
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from model import MultiPairDirectionalClassifier
from feature_engineering_v2_1_full import FeatureEngineer
from config import CFG
from label_generator import DirectionalLabelGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from types import SimpleNamespace
import json
import os
import logging
from collections import Counter
from losses import CostSensitiveFocalLoss
from temperature_scaling import TemperatureScaling
from threshold_tuner import ThresholdTuner

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def save_meta(cfg):
    os.makedirs(os.path.dirname(CFG.paths.meta_path), exist_ok=True)
    meta_dict = vars(cfg).copy()
    for key, value in meta_dict.items():
        if hasattr(value, 'tolist'):
            meta_dict[key] = value.tolist()
        elif isinstance(value, (np.generic, np.ndarray)):
            meta_dict[key] = value.item() if value.size == 1 else value.tolist()
    with open(CFG.paths.meta_path, "w") as f:
        json.dump(meta_dict, f, indent=4)

class Trainer:
    def __init__(self):
        self.device = CFG.train.device
        self.calibration_temperature = None
        self.calibration_thresholds = None

        logging.info("Загрузка данных из CSV...")
        df = CFG.train.loader()
        logging.info(f"Загружено {len(df)} строк.")

        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
        ohlcv_data = df[['open', 'high', 'low', 'close', 'volume']].copy()

        logging.info("Генерация признаков...")
        self.engineer = FeatureEngineer()
        df_feat = self.engineer.generate_features(df, fit=True)
        self.feature_columns = df_feat.columns
        logging.info(f"Сгенерировано {len(self.feature_columns)} признаков на {len(df_feat)} строках.")

        logging.info("Генерация меток SL/TP...")
        labeler = DirectionalLabelGenerator(
            tp_sl_levels=CFG.tp_sl.tp_sl_levels,
            lookahead=CFG.train.lookahead[CFG.train.timeframe]
        )
        Y = labeler.generate_labels(ohlcv_data.loc[df_feat.index])
        self.tp_sl_pairs = labeler.tp_sl_levels

        assert len(df_feat) == len(Y)
        assert (df_feat.index == ohlcv_data.loc[df_feat.index].index).all()

        flat_y = Y.ravel()
        counter = Counter(flat_y)
        total = sum(counter.values())
        for cls in sorted(counter):
            logging.info(f"Класс {cls} — {counter[cls]} примеров ({counter[cls]/total:.2%})")

        self.class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(flat_y),
            y=flat_y
        )

        # Логарифмическое сглаживание
        self.class_weights = np.power(self.class_weights, 0.65)
        logging.info(f"Веса классов: {self.class_weights}")

        X = self.engineer.to_sequences(df_feat, CFG.train.window_size[CFG.train.timeframe])
        min_len = min(len(X), len(Y))
        X, Y = X[:min_len], Y[:min_len]
        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=CFG.train.val_ratio, shuffle=False)

        self.train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.long)),
            batch_size=CFG.train.batch_size,
            shuffle=False
        )
        self.val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                          torch.tensor(y_val, dtype=torch.long)),
            batch_size=CFG.train.batch_size,
            shuffle=False
        )

        model_config = SimpleNamespace(
            input_dim=X.shape[2],
            hidden_dim=CFG.default_model_config.hidden_dim,
            n_heads=CFG.default_model_config.n_heads,
            n_layers=CFG.default_model_config.n_layers,
            dropout=CFG.default_model_config.dropout,
            n_classes=CFG.default_model_config.n_classes,
            dim_feedforward=CFG.default_model_config.dim_feedforward,
            activation=CFG.default_model_config.activation,
            layer_norm_eps=CFG.default_model_config.layer_norm_eps
        )

        self.model = MultiPairDirectionalClassifier(model_config=model_config, num_pairs=len(self.tp_sl_pairs)).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=CFG.train.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, **CFG.train.scheduler)

        self.gamma = CFG.train.gamma_values[0]
        if CFG.train.auto_gamma_search:
            self.gamma = self.auto_gamma_search()
            logging.info(f"🔍 Выбрана наилучшая gamma = {self.gamma}")

        # Персонализированные gamma (можно будет тюнить)
        self.gamma_per_class = torch.tensor([1.5, 0.5, 1.5], dtype=torch.float32).to(self.device)

        self.criterion = CostSensitiveFocalLoss(
            alpha=torch.tensor(self.class_weights, dtype=torch.float32).to(self.device),
            gamma=self.gamma_per_class,
            label_smoothing=CFG.train.label_smoothing
        )

    def auto_gamma_search(self):
        best_gamma = None
        best_profit_f1 = -np.inf

        for gamma in CFG.train.gamma_values:
            logging.info(f"🔍 Тестируем gamma = {gamma}")
            model = MultiPairDirectionalClassifier(
                model_config=self.model.model_config, num_pairs=len(self.tp_sl_pairs)
            ).to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.train.lr)

            criterion = CostSensitiveFocalLoss(
                alpha=torch.tensor(self.class_weights, dtype=torch.float32).to(self.device),
                gamma=torch.tensor([gamma, gamma, gamma], dtype=torch.float32).to(self.device),
                label_smoothing=CFG.train.label_smoothing
            )

            # Обучаем несколько эпох (например, 5)
            for epoch in range(CFG.train.gamma_search_epochs):
                model.train()
                for X_batch, y_batch in self.train_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    optimizer.zero_grad()
                    logits = model(X_batch)
                    loss = criterion(logits.view(-1, 3), y_batch.view(-1))
                    loss.backward()
                    optimizer.step()

            # Валидируем
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    logits = model(X_batch)
                    preds = torch.argmax(logits, dim=2).cpu().numpy()
                    val_preds.append(preds)
                    val_targets.append(y_batch.cpu().numpy())
            y_pred = np.concatenate(val_preds).ravel()
            y_true = np.concatenate(val_targets).ravel()
            profit_true = np.isin(y_true, [0, 2]).astype(int)
            profit_pred = np.isin(y_pred, [0, 2]).astype(int)
            profit_f1 = f1_score(profit_true, profit_pred)

            logging.info(f"📊 Gamma {gamma}: Profit F1 = {profit_f1:.4f}")
            if profit_f1 > best_profit_f1:
                best_profit_f1 = profit_f1
                best_gamma = gamma

        return best_gamma

    def train(self):
        best_score = -np.inf
        epochs_no_improve = 0

        for epoch in range(1, CFG.train.epochs + 1):
            train_loss = self._train_one_epoch()
            val_loss, val_score = self._validate()

            self.scheduler.step(val_score["profit_f1"])

            logging.info(f"Эпоха {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | F1_macro: {val_score['f1_macro']:.4f} | Profit Acc: {val_score['profit_accuracy']:.4f} | Profit F1: {val_score['profit_f1']:.4f} | ROC-AUC: {val_score['roc_auc']:.4f}")
            logging.info(f"[Класс {CFG.action2label["short"]} | SHORT]: precision={val_score['precision'][0]:.3f} recall={val_score['recall'][0]:.3f} f1={val_score['f1'][0]:.3f}")
            logging.info(f"[Класс {CFG.action2label["no-trade"]} | no-trade]: precision={val_score['precision'][1]:.3f} recall={val_score['recall'][1]:.3f} f1={val_score['f1'][1]:.3f}")
            logging.info(f"[Класс {CFG.action2label["long"]} | LONG]: precision={val_score['precision'][2]:.3f} recall={val_score['recall'][2]:.3f} f1={val_score['f1'][2]:.3f}")

            if val_score["profit_f1"] > best_score:
                best_score = val_score["profit_f1"]
                epochs_no_improve = 0

                torch.save(self.model.state_dict(), CFG.paths.model_path)
                logging.info(f"📦 Сохранена новая лучшая модель (Profit F1={val_score['profit_f1']:.4f}) на эпохе {epoch}")

                if CFG.train.calibrate_logits:
                    logits_list, targets_list = [], []
                    self.model.eval()
                    with torch.no_grad():
                        for X_batch, y_batch in self.val_loader:
                            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                            logits = self.model(X_batch).reshape(-1, 3)
                            targets = y_batch.reshape(-1)
                            logits_list.append(logits)
                            targets_list.append(targets)
                    logits_all = torch.cat(logits_list, dim=0)
                    targets_all = torch.cat(targets_list, dim=0)

                    # Temperature Scaling
                    ts_model = TemperatureScaling(init_temp=1.0, min_temp=0.5, max_temp=5.0, reg_lambda=0.01)
                    ts_model.fit(logits_all, targets_all)
                    self.calibration_temperature = ts_model.get_temperature()
                    logging.info(
                        f"📏 Найдена температура для калибровки логитов: T = {self.calibration_temperature:.4f}")

                    # Threshold Tuning
                    logits_calibrated = logits_all / self.calibration_temperature
                    probs_val = torch.softmax(logits_calibrated, dim=1).cpu().numpy()
                    targets_val = targets_all.cpu().numpy()

                    tuner = ThresholdTuner()
                    thresholds = tuner.search_optimal_thresholds(probs_val, targets_val)
                    self.calibration_thresholds = thresholds
                    logging.info(f"📐 Найдены оптимальные пороги: {thresholds}")

                meta_args = SimpleNamespace(
                    input_dim=self.model.model_config.input_dim,
                    window_size=CFG.train.window_size,
                    lookahead=CFG.train.lookahead,
                    num_pairs=len(self.tp_sl_pairs),
                    feature_columns=self.feature_columns,
                    tp_sl_pairs=self.tp_sl_pairs,
                    scaler_params={
                        "mean": self.engineer.scaler.mean_.tolist(),
                        "scale": self.engineer.scaler.scale_.tolist()
                    },
                    n_heads=self.model.model_config.n_heads,
                    n_layers=self.model.model_config.n_layers,
                    hidden_dim=self.model.model_config.hidden_dim,
                    dropout=self.model.model_config.dropout,
                    n_classes=self.model.model_config.n_classes,
                    dim_feedforward=self.model.model_config.dim_feedforward,
                    activation=self.model.model_config.activation,
                    layer_norm_eps=self.model.model_config.layer_norm_eps,
                    feature_importances=self.model.feature_importances(),
                    temperature=self.calibration_temperature,
                    gamma=self.gamma,
                    thresholds=self.calibration_thresholds,
                    margins={"0": 0.15, "2": 0.15}
                )

                save_meta(meta_args)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= CFG.train.patience:
                    logging.info(f"⏹ Early stopping после {epoch} эпох")
                    break

    def _train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = self.criterion(logits.view(-1, 3), y_batch.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if torch.isnan(loss):
                logging.error("NaN в функции потерь! Обучение остановлено.")
                raise ValueError("NaN в функции потерь")
            self.optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        return running_loss / len(self.train_loader.dataset)

    def _validate(self):
        from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

        self.model.eval()
        val_preds, val_targets = [], []
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                logits = self.model(X_batch)
                loss = self.criterion(logits.view(-1, 3), y_batch.view(-1))
                val_loss += loss.item() * X_batch.size(0)
                preds = torch.argmax(logits, dim=2).cpu().numpy()
                val_preds.append(preds)
                val_targets.append(y_batch.cpu().numpy())

        y_pred = np.concatenate(val_preds).ravel()
        y_true = np.concatenate(val_targets).ravel()

        f1_macro = f1_score(y_true, y_pred, average="macro")
        accuracy = accuracy_score(y_true, y_pred)
        try:
            present_classes = np.unique(y_true)
            if len(present_classes) > 1:
                y_score = np.zeros((len(y_pred), len(present_classes)))
                for i, cls in enumerate(present_classes):
                    y_score[y_pred == cls, i] = 1
                roc_auc = roc_auc_score(y_true, y_score, multi_class="ovo")
            else:
                roc_auc = 0.0
        except Exception as e:
            logging.warning(f"Не удалось вычислить ROC-AUC: {str(e)}")
            roc_auc = 0.0

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)

        profit_true = np.isin(y_true, [0, 2]).astype(int)
        profit_pred = np.isin(y_pred, [0, 2]).astype(int)
        profit_f1 = f1_score(profit_true, profit_pred)
        profit_accuracy = accuracy_score(profit_true, profit_pred)

        return val_loss / len(self.val_loader.dataset), {
            "f1_macro": f1_macro,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "profit_f1": profit_f1,
            "profit_accuracy": profit_accuracy
        }

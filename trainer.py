from datetime import datetime

from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import logging
from typing import Tuple
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, \
    precision_recall_curve, average_precision_score
from torch import nn

from config import Config
from model import TransformerModel

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class TradingDataset(data.Dataset):
    """Кастомный Dataset для загрузки данных обучения."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.FloatTensor(self.features[idx]),
            torch.FloatTensor(self.labels[idx])
        )


class ModelTrainer:
    """Класс для обучения модели."""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimal_thresholds = None # Будет хранить пороги для всех пар TP/SL

        logger.info(f"Используемое устройство: {self.device}")

        # # Проверка памяти GPU перед обучением
        # if torch.cuda.is_available():
        #     gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        #     required_mem = Config.BATCH_SIZE * 2 * 1024 ** 3  # Эмпирическая оценка
        #     if required_mem > gpu_mem:
        #         logger.warning(
        #             f"Недостаточно GPU памяти! Доступно: {gpu_mem:.1f}GB, Требуется: ~{required_mem / (1024 ** 3):.1f}GB")
        #         Config.BATCH_SIZE = Config.BATCH_SIZE // 2
        #         logger.warning(f"Автоматическое уменьшение BATCH_SIZE до {Config.BATCH_SIZE}")

    def _optimize_thresholds(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Вычисляет оптимальные пороги для каждого SL/TP-класса по F1."""
        thresholds = []
        for i in range(y_pred.shape[1]):  # Для каждой из 274 пар
            precision, recall, threshs = precision_recall_curve(y_true[:, i], y_pred[:, i])
            f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
            thresholds.append(threshs[np.argmax(f1_scores[:-1])])
        return np.array(thresholds)

    def calculate_class_weights(self, df: pd.DataFrame):
        """Автоматический расчет весов на основе дисбаланса меток"""
        label_columns = [col for col in df.columns if col.startswith(('LONG_', 'SHORT_'))]
        positive_counts = df[label_columns].sum().sum()
        total_counts = df[label_columns].count().sum()

        neg_weight = positive_counts / total_counts
        pos_weight = (total_counts - positive_counts) / total_counts

        self.pos_weight = torch.tensor([pos_weight / neg_weight]).to(self.device)
        logger.info(f"Автоматически рассчитанные веса: pos_weight={self.pos_weight.item():.2f}")

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Подготавливает данные для обучения."""
        logger.info("Подготовка данных для обучения")

        # Выбираем только фичи (исключаем исходные колонки и метки)
        feature_columns = [col for col in df.columns
                           if not col.startswith('label_') and
                           col not in Config.ORIGINAL_COLUMNS]
        features = df[feature_columns].values

        # Выбираем метки (все колонки, начинающиеся с 'LONG_' или 'SHORT_')
        label_columns = [col for col in df.columns if col.startswith('LONG_') or col.startswith('SHORT_')]
        labels = df[label_columns].values

        # Создаем последовательности с учетом lookback_window
        X, y = [], []
        for i in range(Config.LOOKBACK_WINDOW, len(features)):
            X.append(features[i - Config.LOOKBACK_WINDOW:i])
            y.append(labels[i])

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Форма X: {X.shape}, форма y: {y.shape}")
        logger.info(f"Проверка меток: min={y.min()}, max={y.max()}")
        assert (y.min() >= 0) and (y.max() <= 1), "Метки должны быть в диапазоне [0, 1]"
        assert not np.isnan(X).any(), "Найдены NaN значения в фичах"
        assert not np.isnan(y).any(), "Найдены NaN значения в метках"
        return X, y

    def train_test_split(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple:
        """Разделяет данные на обучающую и тестовую выборки."""
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        return X_train, X_test, y_train, y_test

    def train(self, df: pd.DataFrame) -> Tuple[list, bool]:
        """Обучает модель на подготовленных данных с детальным логированием прогресса.

        Возвращает:
            Tuple[list, bool]: (метрики по эпохам, флаг успешного сохранения модели)
        """
        logger.info("Начало обучения модели")

        # Вычисление весов классов для несбалансированных данных
        logger.info("Автоматическое вычисление весов классов")
        self.calculate_class_weights(df)

        try:
            # 1. Подготовка данных
            logger.info("Подготовка данных...")
            X, y = self.prepare_data(df)
            X_train, X_test, y_train, y_test = self.train_test_split(X, y)

            # 2. Создание DataLoader
            logger.info("Создание DataLoader...")
            train_dataset = TradingDataset(X_train, y_train)
            test_dataset = TradingDataset(X_test, y_test)

            train_loader = data.DataLoader(
                train_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True
            )

            test_loader = data.DataLoader(
                test_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False
            )

            # 3. Инициализация модели
            logger.info("Инициализация модели...")
            input_dim = X.shape[2]
            num_sl_tp_pairs = y.shape[1]
            model = TransformerModel(input_dim, num_sl_tp_pairs).to(self.device)

            # 4. Оптимизатор и функция потерь
            optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
            criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

            best_test_loss = float('inf')
            epoch_metrics = []  # Для хранения метрик всех эпох
            model_saved = False  # Флаг, указывающий, была ли сохранена модель

            # 5. Цикл обучения
            logger.info(f"Старт обучения на {Config.EPOCHS} эпох...")
            for epoch in range(Config.EPOCHS):
                model.train()
                train_loss = 0.0
                processed_batches = 0

                # Прогресс-бар эпохи
                progress_bar = tqdm(train_loader, desc=f"Эпоха {epoch + 1}/{Config.EPOCHS}", leave=False)

                for batch_X, batch_y in progress_bar:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    # Прямой проход
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    # Обратный проход
                    loss.backward()
                    optimizer.step()

                    # Логирование батча
                    train_loss += loss.item() * batch_X.size(0)
                    processed_batches += 1

                    # Обновление прогресс-бара
                    progress_bar.set_postfix({
                        'loss': f"{loss.item():.6f}",
                        'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                    })

                    # Логирование каждые 10 батчей
                    if processed_batches % 10 == 0:
                        logger.debug(
                            f"Эпоха {epoch + 1} | Батч {processed_batches}/{len(train_loader)} "
                            f"| Loss: {loss.item():.6f}"
                        )

                # 6. Валидация после эпохи
                model.eval()
                test_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)

                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        test_loss += loss.item() * batch_X.size(0)

                # 7. Расчет метрик
                train_loss /= len(train_loader.dataset)
                test_loss /= len(test_loader.dataset)

                # Получение дополнительных метрик
                eval_results = self._evaluate_model(model, test_loader, criterion)
                metrics = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'roc_auc': eval_results['roc_auc'],
                    'f1_score': eval_results['f1_score'],
                    'accuracy': eval_results['accuracy']
                }
                epoch_metrics.append(metrics)

                # 8. Логирование результатов эпохи
                logger.info(
                    f"Эпоха {epoch + 1}/{Config.EPOCHS} завершена:\n"
                    f"  Train Loss: {metrics['train_loss']:.6f}\n"
                    f"  Test Loss: {metrics['test_loss']:.6f}\n"
                    f"  ROC-AUC: {metrics['roc_auc']:.4f}\n"
                    f"  F1-Score: {metrics['f1_score']:.4f}\n"
                    f"  Accuracy: {metrics['accuracy']:.4f}"
                )

                # 9. Сохранение лучшей модели
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    model.save(Config.MODEL_SAVE_PATH)
                    model_saved = True
                    logger.info(f"Новая лучшая модель сохранена (Loss: {best_test_loss:.6f})")

            # 10. Финализация обучения
            logger.info("Обучение завершено. Итоговые метрики:")
            if epoch_metrics:
                best_epoch = min(epoch_metrics, key=lambda x: x['test_loss'])
                logger.info(f"Лучшая эпоха {best_epoch['epoch']}:")
                logger.info(f"  Test Loss: {best_epoch['test_loss']:.6f}")
                logger.info(f"  ROC-AUC: {best_epoch['roc_auc']:.4f}")
                logger.info(f"  F1-Score: {best_epoch['f1_score']:.4f}")
                logger.info(f"  Accuracy: {best_epoch['accuracy']:.4f}")

            return epoch_metrics, model_saved

        except Exception as e:
            logger.error(f"Ошибка при обучении: {str(e)}")
            logger.exception("Трассировка стека:")
            raise

    from sklearn.metrics import average_precision_score, precision_recall_curve

    def _evaluate_model(self, model, data_loader, criterion) -> dict:
        """
        Полная оценка модели с расширенным логированием и метриками.
        Возвращает словарь с метриками качества:
        - roc_auc: Площадь под ROC-кривой (macro-average)
        - f1_score: F1-мера (macro-average)
        - accuracy: Точность классификации
        - avg_precision: Average Precision (площадь под PR-кривой)
        - precision_recall_curve: Данные для построения PR-кривой
        """
        logger.info("Начало оценки модели...")
        start_time = datetime.now()

        # 1. Сбор предсказаний и меток
        model.eval()
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, (batch_X, batch_y) in enumerate(data_loader):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = model(batch_X)
                all_outputs.append(outputs.cpu())
                all_labels.append(batch_y.cpu())

                # Логирование прогресса каждые 10 батчей
                if batch_idx % 10 == 0:
                    logger.debug(f"Обработан батч {batch_idx}/{len(data_loader)}")

        # 2. Подготовка данных для метрик
        try:
            all_outputs = torch.sigmoid(torch.cat(all_outputs)).numpy()
            all_labels = torch.cat(all_labels).numpy()

            # Оптимизация порогов для каждого класса отдельно
            optimized_predictions = np.zeros_like(all_outputs)
            thresholds = []

            for i in range(all_outputs.shape[1]):
                if len(np.unique(all_labels[:, i])) > 1:
                    precision, recall, threshs = precision_recall_curve(all_labels[:, i], all_outputs[:, i])
                    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
                    best_thresh = threshs[np.argmax(f1_scores[:-1])]
                    thresholds.append(best_thresh)
                    optimized_predictions[:, i] = (all_outputs[:, i] > best_thresh).astype(int)
                else:
                    thresholds.append(0.5)  # дефолтный порог
                    optimized_predictions[:, i] = (all_outputs[:, i] > 0.5).astype(int)

            self.optimal_thresholds = np.array(thresholds)

            logger.info(f"Данные подготовлены. Форма outputs: {all_outputs.shape}, labels: {all_labels.shape}")

            # Проверка распределения классов
            unique_labels, counts = np.unique(all_labels, return_counts=True)
            logger.info(f"Распределение меток: {dict(zip(unique_labels, counts))}")

        except Exception as e:
            logger.error(f"Ошибка подготовки данных: {str(e)}")
            raise


        # Добавить расчет loss
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item() * batch_X.size(0)

        test_loss /= len(data_loader.dataset)

        # 3. Инициализация метрик
        metrics = {
            'test_loss': test_loss,  # Добавлено
            'roc_auc': 0.5,
            'f1_score': 0.0,
            'accuracy': 0.0,
            'avg_precision': 0.0,
            'precision_recall_curve': None,
            'thresholds': self.optimal_thresholds.tolist()  # Добавляем пороги в метрики
        }

        # 4. Расчет основных метрик
        try:
            # ROC-AUC с проверкой наличия обоих классов
            if len(np.unique(all_labels)) > 1:
                metrics['roc_auc'] = roc_auc_score(all_labels, all_outputs, average='macro')

            # Используем оптимизированные предсказания для F1 и Accuracy
            metrics['f1_score'] = f1_score(all_labels, optimized_predictions, average='macro', zero_division=0)
            metrics['accuracy'] = accuracy_score(all_labels, optimized_predictions)

            # 5. Расчет Average Precision и PR-кривой
            metrics['avg_precision'] = average_precision_score(
                all_labels, all_outputs, average='macro'
            )
            logger.debug(f"Average Precision рассчитан: {metrics['avg_precision']:.4f}")

            # Полная PR-кривая для анализа
            precision, recall, _ = precision_recall_curve(all_labels.ravel(), all_outputs.ravel())
            metrics['precision_recall_curve'] = {
                'precision': precision.tolist(),
                'recall': recall.tolist()
            }
            logger.info("PR-кривая успешно построена")

        except Exception as e:
            logger.error(f"Ошибка расчета метрик: {str(e)}")
            logger.exception("Трассировка стека:")

        # 6. Финализация оценки
        elapsed_time = datetime.now() - start_time
        logger.info(
            f"Оценка завершена за {elapsed_time}. Итоговые метрики:\n"
            f"  ROC-AUC: {metrics['roc_auc']:.4f}\n"
            f"  F1-Score: {metrics['f1_score']:.4f}\n"
            f"  Accuracy: {metrics['accuracy']:.4f}\n"
            f"  Avg Precision: {metrics['avg_precision']:.4f}"
        )

        return metrics

    def evaluate(self, df: pd.DataFrame, save_results: bool = True) -> dict:
        """Полная оценка модели с оптимизацией порогов для каждого SL/TP-класса."""

        # --- 1. Подготовка данных (существующий код) ---
        logger.info("Подготовка данных для оценки...")
        X, y = self.prepare_data(df)
        X_train, X_val, y_train, y_val = self.train_test_split(X, y, test_size=0.3)  # 30% на валидацию
        X_val, X_test, y_val, y_test = self.train_test_split(X_val, y_val, test_size=0.5)  # 15% тест

        # --- 2. Загрузка модели (если не загружена) ---
        if not hasattr(self, 'model'):
            self.model = TransformerModel.load(Config.MODEL_SAVE_PATH).to(self.device)
        self.model.eval()

        # --- 3. Оптимизация порогов на валидационных данных ---
        logger.info("Оптимизация порогов классификации...")

        # Получаем предсказания для валидации
        val_outputs = []
        with torch.no_grad():
            for batch_X, _ in DataLoader(TradingDataset(X_val, y_val),
                                         batch_size=Config.BATCH_SIZE):
                val_outputs.append(self.model(batch_X.to(self.device)))

        val_probs = torch.sigmoid(torch.cat(val_outputs)).cpu().numpy()

        # Оптимизируем порог для каждого из 274 классов
        self.optimal_thresholds = np.array([self._find_optimal_threshold(y_val[:, i], val_probs[:, i])
                                            for i in range(val_probs.shape[1])])

        logger.info(f"Оптимальные пороги: min={np.min(self.optimal_thresholds):.3f}, "
                    f"max={np.max(self.optimal_thresholds):.3f}")

        # --- 4. Оценка на тестовых данных с оптимизированными порогами ---
        logger.info("Запуск оценки с оптимальными порогами...")
        test_outputs = []

        test_loss = 0.0
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        with torch.no_grad():
            for batch_X, batch_y in DataLoader(TradingDataset(X_test, y_test),
                                               batch_size=Config.BATCH_SIZE):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item() * batch_X.size(0)
                test_outputs.append(outputs)

        test_loss /= len(X_test)  # Добавьте эту строку для нормализации loss

        test_probs = torch.sigmoid(torch.cat(test_outputs)).cpu().numpy()
        predictions = (test_probs > self.optimal_thresholds).astype(int)  # Применяем пороги!

        # --- 5. Расчет метрик (адаптированный код) ---
        metrics = {
            'test_loss': test_loss,  # Добавлено
            'roc_auc': roc_auc_score(y_test, test_probs, average='macro'),
            'f1_score': f1_score(y_test, predictions, average='macro'),
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='macro', zero_division=0),
            'recall': recall_score(y_test, predictions, average='macro'),
            'thresholds': self.optimal_thresholds.tolist()
        }

        # --- 6. Дополнительный анализ по парам SL/TP ---
        pair_metrics = {}
        for i, col in enumerate([c for c in df.columns if c.startswith(('LONG_', 'SHORT_'))]):
            pair_metrics[col] = {
                'threshold': self.optimal_thresholds[i],
                'f1': f1_score(y_test[:, i], predictions[:, i]),
                'precision': precision_score(y_test[:, i], predictions[:, i], zero_division=0),
                'recall': recall_score(y_test[:, i], predictions[:, i])
            }

        # --- 7. Сохранение результатов
        if save_results:
            # Преобразование numpy-типов в стандартные Python-типы
            def convert_numpy_types(obj):
                if isinstance(obj, np.generic):
                    return float(obj) if isinstance(obj, (np.float32, np.float64)) else int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(x) for x in obj]
                return obj

            results = {'metrics': metrics, 'pair_metrics': pair_metrics}
            results = convert_numpy_types(results)  # Применяем преобразование

            with open('evaluation_results.json', 'w') as f:
                json.dump(results, f, indent=4)  # Добавлен indent для читаемости

        return metrics

    def _find_optimal_threshold(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Вспомогательный метод для поиска оптимального порога по F1."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        return thresholds[np.argmax(f1_scores[:-1])]

    def _find_hard_examples(self, outputs: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> dict:
        """Находит сложные для модели примеры (с наибольшей ошибкой)."""
        errors = np.abs(outputs - labels)
        hard_examples = {}

        for i in range(outputs.shape[1]):  # Для каждой пары SL/TP
            worst_idx = np.argmax(errors[:, i])
            hard_examples[i] = {
                'index': int(indices[worst_idx]),
                'prediction': float(outputs[worst_idx, i]),
                'true_label': int(labels[worst_idx, i]),
                'error': float(errors[worst_idx, i])
            }

        return hard_examples

    def _count_false_positives(self, outputs: np.ndarray, labels: np.ndarray) -> dict:
        """Подсчитывает ложноположительные срабатывания по всем парам."""
        fp_counts = {}
        predictions = (outputs > 0.5).astype(int)

        for i in range(outputs.shape[1]):
            fp = np.sum((predictions[:, i] == 1) & (labels[:, i] == 0))
            fp_counts[i] = int(fp)

        return fp_counts

    def _count_false_negatives(self, outputs: np.ndarray, labels: np.ndarray) -> dict:
        """Подсчитывает ложноотрицательные срабатывания по всем парам."""
        fn_counts = {}
        predictions = (outputs > 0.5).astype(int)

        for i in range(outputs.shape[1]):
            fn = np.sum((predictions[:, i] == 0) & (labels[:, i] == 1))
            fn_counts[i] = int(fn)

        return fn_counts

from config import Config
from data_processor import DataProcessor
from trainer import ModelTrainer
import logging
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

def main():
    try:
        # Проверка и создание директории для модели
        os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
        logger.info(f"Проверка директории для модели: {os.path.dirname(Config.MODEL_SAVE_PATH)}")

        # Инициализация
        data_processor = DataProcessor()
        trainer = ModelTrainer()

        # Загрузка и подготовка данных
        logger.info(f"Загрузка данных из {Config.DATA_PATH}")
        df = data_processor.prepare_dataset(Config.DATA_PATH)

        # Анализ распределения меток
        label_columns = [col for col in df.columns if col.startswith(('LONG_', 'SHORT_'))]
        if not label_columns:
            raise ValueError("Не найдены колонки с метками (LONG_*, SHORT_*)")

        label_stats = df[label_columns].sum().sum()
        total_labels = len(label_columns) * len(df)
        logger.info(f"Распределение меток: 1 - {label_stats}, 0 - {total_labels - label_stats}")

        if label_stats == 0 or label_stats == total_labels:
            raise ValueError("Все метки одного класса! Проверьте расчет меток в DataProcessor.")

        # Обучение модели
        logger.info("Начало процесса обучения...")
        metrics, model_saved = trainer.train(df)

        # Оценка модели только если она была сохранена
        if model_saved:
            logger.info("Модель успешно сохранена, начинаем оценку...")
            eval_results = trainer.evaluate(df)
            logger.info(
                f"Финальные метрики модели:\n"
                f"  Test Loss: {eval_results.get('test_loss', 'N/A'):.6f}\n"
                f"  ROC-AUC: {eval_results.get('roc_auc', 'N/A'):.4f}\n"
                f"  F1-Score: {eval_results.get('f1_score', 'N/A'):.4f}\n"
                f"  Accuracy: {eval_results.get('accuracy', 'N/A'):.4f}"
            )
        else:
            logger.warning("Модель не была сохранена в процессе обучения. Оценка невозможна.")

    except Exception as e:
        logger.error(f"Критическая ошибка в основном процессе: {str(e)}")
        logger.exception("Трассировка стека:")
        raise


if __name__ == "__main__":
    main()
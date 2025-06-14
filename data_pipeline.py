import logging
import subprocess
import pandas as pd
from config import CFG

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Обязательные колонки для начала генерации признаков
REQUIRED_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume',
    'direction_class',
    'amplitude_up',
    'amplitude_down'
]

def validate_dataset():
    logging.info("Запуск валидации тренировочного датасета...")
    df = pd.read_csv(CFG.paths.train_csv)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        logging.error(f"❌ Отсутствуют необходимые колонки: {missing_columns}")
        raise ValueError("Валидация не пройдена. Проверьте тренировочный датасет.")
    else:
        logging.info("✅ Все необходимые колонки присутствуют. Датасет готов к генерации признаков.")

def run_script(script_name):
    logging.info(f"Запуск скрипта: {script_name}...")
    result = subprocess.run(["python", script_name])
    if result.returncode != 0:
        raise RuntimeError(f"Ошибка при выполнении скрипта {script_name}")

if __name__ == "__main__":
    logging.info("=== Запуск полного конвейера подготовки данных ===")

    try:
        # 1️⃣ Генерация direction-меток (если еще не сгенерированы)
        run_script("directional_label_generator.py")

        # 2️⃣ Генерация amplitude-меток (если еще не сгенерированы)
        run_script("amplitude_label_generator.py")

        # 3️⃣ Валидация после генерации меток
        validate_dataset()

        # 4️⃣ Генерация признаков и scaler
        run_script("prepare_features.py")

        logging.info("=== ✅ Полная подготовка данных завершена успешно ===")

    except Exception as e:
        logging.error(f"❌ Ошибка в процессе подготовки данных: {e}")

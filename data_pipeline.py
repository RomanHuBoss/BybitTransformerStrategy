import logging
import subprocess
import pandas as pd
from config import CFG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'direction_class', 'amplitude_up', 'amplitude_down']

def validate_dataset():
    logging.info("Валидация...")
    df = pd.read_csv(CFG.paths.train_csv)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Отсутствуют колонки: {missing_columns}")

def run_script(script_name):
    logging.info(f"Запуск {script_name}")
    result = subprocess.run(["python", script_name])
    if result.returncode != 0:
        raise RuntimeError(f"Ошибка: {script_name}")

if __name__ == "__main__":
    try:
        run_script("directional_label_generator.py")
        run_script("amplitude_label_generator.py")
        validate_dataset()
        run_script("prepare_features.py")
        run_script("train_direction.py")
        run_script("train_amplitude.py")
        run_script("temperature_calibration.py")
        logging.info("Pipeline завершён успешно")
    except Exception as e:
        logging.error(e)

import logging
import subprocess
import pandas as pd
from config import CFG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

REQUIRED_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume',
    'direction_class', 'amplitude_up', 'amplitude_down'
]

def validate_dataset():
    logging.info("🔎 Валидация датасета...")
    df = pd.read_csv(CFG.paths.train_csv)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        logging.error(f"❌ Отсутствуют колонки: {missing_columns}")
        raise ValueError("Валидация не пройдена.")
    logging.info("✅ Валидация успешна.")

def run_script(script_name):
    logging.info(f"🚀 Запуск скрипта: {script_name} ...")
    result = subprocess.run(["python", script_name])
    if result.returncode != 0:
        raise RuntimeError(f"❌ Ошибка при выполнении {script_name}")
    logging.info(f"✅ Скрипт {script_name} завершён успешно.")

if __name__ == "__main__":
    logging.info("=== 🔥 Старт полного пайплайна обучения ===")
    try:
        run_script("directional_label_generator.py")
        run_script("amplitude_label_generator.py")
        run_script("hit_order_label_generator.py")
        validate_dataset()
        run_script("prepare_features.py")
        run_script("train_direction.py")
        run_script("train_amplitude.py")
        run_script("train_hit_order.py")
        run_script("temperature_calibration.py")
        logging.info("=== 🎯 Полный пайплайн завершён успешно ===")
    except Exception as e:
        logging.error(f"💥 Ошибка пайплайна: {e}")

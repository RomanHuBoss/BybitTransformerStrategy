import subprocess
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

pipeline_steps = [
    "reset_data.py",
    "data_pipeline.py",
    "train_direction.py",
    "train_amplitude.py",
    "train_hit_order.py",
    "temperature_calibration.py"
]

def run_script(script_name):
    logging.info(f"🚀 Запуск {script_name} ...")
    start_time = time.perf_counter()
    try:
        subprocess.run(["python", script_name], check=True)
        elapsed = time.perf_counter() - start_time
        logging.info(f"✅ {script_name} завершён за {elapsed:.1f} сек.")
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Ошибка в {script_name}: {e}")
        raise

if __name__ == "__main__":
    logging.info("🚀 Старт полного обучения и калибровки...")
    for step in pipeline_steps:
        run_script(step)
    logging.info("🎯 Полный продакшн-пайплайн успешно завершён!")

import subprocess

def run_script(script_name):
    print(f"🚀 Запуск {script_name} ...")
    subprocess.run(["python", script_name], check=True)

if __name__ == "__main__":
    run_script("reset_data.py")
    run_script("data_pipeline.py")
    run_script("train_direction.py")
    run_script("train_amplitude.py")
    run_script("train_hit_order.py")
    run_script("temperature_calibration.py")  # вот тут калибровка температуры
    print("✅ Полный цикл подготовки данных, обучения и калибровки завершён!")
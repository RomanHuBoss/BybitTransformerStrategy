import os
from config import CFG

def safe_remove(path):
    try:
        os.remove(path)
        print(f"✅ Удалено: {path}")
    except FileNotFoundError:
        print(f"⚠️ Не найдено (пропущено): {path}")

def main():
    # Основные файлы пайплайна
    files_to_remove = [
        CFG.paths.train_features_csv,
        CFG.paths.scaler_path,
        CFG.paths.feature_columns_path,
        CFG.paths.direction_model_path,
        CFG.paths.amplitude_model_path,
        CFG.paths.hit_order_model_path,
        CFG.paths.temperature_path
    ]

    print("🚀 Очистка данных пайплайна...")
    for file_path in files_to_remove:
        safe_remove(file_path)

    print("✅ Очистка завершена.")

if __name__ == "__main__":
    main()

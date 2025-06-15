import os
from config import CFG

files_to_remove = [
    CFG.paths.train_features_csv,
    CFG.paths.train_labels_direction,
    CFG.paths.train_labels_amplitude,
    CFG.paths.train_labels_hitorder,
    CFG.paths.scaler_path,
    CFG.paths.feature_columns_path,
    CFG.paths.direction_model_path,
    CFG.paths.amplitude_model_path,
    CFG.paths.hit_order_model_path,
    CFG.paths.temperature_path
]

for file in files_to_remove:
    file_path = str(file)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"✅ Удалён: {file_path}")
    else:
        print(f"⚠ Файл не найден: {file_path}")

print("🎯 Очистка данных завершена.")

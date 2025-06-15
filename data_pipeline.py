import logging
import pandas as pd
import numpy as np
import joblib
from config import CFG
from feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 1️⃣ Загрузка исходных данных
logging.info("Загрузка исходного train_csv...")
df_raw = pd.read_csv(CFG.paths.train_csv)
logging.info(f"✅ Загружено {len(df_raw)} строк исходных данных")

# 2️⃣ Генерация признаков
logging.info("Генерация признаков...")
engineer = FeatureEngineer()
features = engineer.generate_features(df_raw, fit=True)
joblib.dump(engineer.scaler, CFG.paths.scaler_path)
joblib.dump(engineer.feature_columns, CFG.paths.feature_columns_path)
logging.info(f"✅ Сгенерированы признаки: {features.shape}")

# Привязка индексов признаков и исходных данных
df_clean = df_raw.loc[features.index].reset_index(drop=True)
features.reset_index(drop=True, inplace=True)

# 3️⃣ Direction метки
logging.info("Генерация Direction меток...")
lookahead = CFG.label_generation.direction_shift
threshold = CFG.label_generation.direction_threshold

future_close = df_clean['close'].shift(-lookahead)
current_close = df_clean['close']
returns = (future_close - current_close) / current_close
labels_direction = np.where(returns > threshold, 2, np.where(returns < -threshold, 0, 1))

valid_length = len(df_clean) - lookahead
features = features.iloc[:valid_length].reset_index(drop=True)
df_clean = df_clean.iloc[:valid_length].reset_index(drop=True)
labels_direction = labels_direction[:valid_length]

assert len(features) == len(labels_direction), "Рассинхрон после Direction!"
np.save(CFG.paths.train_labels_direction, labels_direction)
logging.info(f"✅ Direction метки: {len(labels_direction)}")

# 4️⃣ Amplitude метки
logging.info("Генерация Amplitude меток...")
shift = CFG.label_generation.amplitude_shift
window = CFG.label_generation.quantile_window
min_sl, max_sl, max_tp = 0.005, 0.02, 0.15

amp_labels = []
for idx in range(len(df_clean) - shift - window + 1):
    win = df_clean.iloc[idx + shift: idx + shift + window]
    close = df_clean.iloc[idx]['close']
    up_ampl = (win['high'] - close) / close
    down_ampl = (close - win['low']) / close

    down_p10 = np.clip(np.percentile(down_ampl, 10), min_sl, max_sl)
    down_p90 = np.clip(np.percentile(down_ampl, 90), min_sl, max_sl)
    up_p10 = np.clip(max(np.percentile(up_ampl, 10), 2 * down_p10), 2 * down_p10, max_tp)
    up_p90 = np.clip(max(np.percentile(up_ampl, 90), 2 * down_p90), 2 * down_p90, max_tp)

    amp_labels.append([up_p10, up_p90, down_p10, down_p90])

labels_amplitude = np.array(amp_labels)

cut_len = len(labels_amplitude)
features = features.iloc[:cut_len].reset_index(drop=True)
df_clean = df_clean.iloc[:cut_len].reset_index(drop=True)
labels_direction = labels_direction[:cut_len]

assert len(features) == len(labels_amplitude) == len(labels_direction), "Рассинхрон после Amplitude!"
np.save(CFG.paths.train_labels_amplitude, labels_amplitude)
logging.info(f"✅ Amplitude метки: {len(labels_amplitude)}")

# 5️⃣ HitOrder метки
logging.info("Генерация HitOrder меток...")
shift = CFG.label_generation.hit_order_shift
window = CFG.label_generation.hit_order_window
sl_min = CFG.label_generation.hit_order_sl_min
sl_max = CFG.label_generation.hit_order_sl_max
rr_min = CFG.label_generation.hit_order_rr_min
rr_max = CFG.label_generation.hit_order_rr_max

hit_labels = []
for idx in range(len(df_clean) - shift - window + 1):
    win = df_clean.iloc[idx + shift: idx + shift + window]
    close = df_clean.iloc[idx]['close']

    sl_relative = np.random.uniform(sl_min, sl_max)
    sl_level = close * (1 - sl_relative)
    rr = np.random.uniform(rr_min, rr_max)
    tp_relative = sl_relative * rr
    tp_level = close * (1 + tp_relative)

    hit = 0
    for _, row in win.iterrows():
        if row['high'] >= tp_level:
            hit = 1
            break
        elif row['low'] <= sl_level:
            hit = 0
            break
    hit_labels.append([sl_relative, tp_relative, hit])

labels_hitorder = np.array(hit_labels)

final_len = len(labels_hitorder)
features = features.iloc[:final_len].reset_index(drop=True)
labels_direction = labels_direction[:final_len]
labels_amplitude = labels_amplitude[:final_len]

assert len(features) == len(labels_hitorder) == len(labels_amplitude) == len(labels_direction), "Финальный рассинхрон!"

np.save(CFG.paths.train_labels_hitorder, labels_hitorder)

# Финальная безопасная запись признаков
features.to_csv(CFG.paths.train_features_csv, index=False)
logging.info(f"✅ Сохранены признаки: {len(features)} строк")

logging.info("🎯 Полный суперстабильный боевой пайплайн завершён!")

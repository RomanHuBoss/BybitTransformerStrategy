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

# 3️⃣ Direction метки (range-based, строго от текущей свечи)
logging.info("Генерация Direction меток (range-based)...")

threshold = CFG.label_generation.direction_threshold

direction_labels = []
for idx in range(len(df_clean) - CFG.labels.lookahead):
    current_close = df_clean.iloc[idx]['close']
    future_window = df_clean.iloc[idx : idx + CFG.labels.lookahead]

    max_return = (future_window['high'].max() - current_close) / current_close
    min_return = (future_window['low'].min() - current_close) / current_close

    if max_return > threshold:
        label = 2  # long
    elif min_return < -threshold:
        label = 0  # short
    else:
        label = 1  # no-trade

    direction_labels.append(label)

labels_direction = np.array(direction_labels)

valid_length = len(labels_direction)
features = features.iloc[:valid_length].reset_index(drop=True)
df_clean = df_clean.iloc[:valid_length].reset_index(drop=True)

assert len(features) == len(labels_direction), "Рассинхрон после Direction!"
logging.info(f"✅ Direction метки (range-based): {len(labels_direction)}")

# 4️⃣ Amplitude метки (60 назад и 20 вперёд)
logging.info("Генерация Amplitude меток...")

min_sl = CFG.label_generation.amplitude_min_sl
max_sl = CFG.label_generation.amplitude_max_sl
max_tp = CFG.label_generation.amplitude_max_tp

amp_labels = []
for idx in range(CFG.feature_engineering.window_size, len(df_clean) - CFG.labels.lookahead):
    current_close = df_clean.iloc[idx]['close']
    future_win = df_clean.iloc[idx : idx + CFG.labels.lookahead]

    up_ampl = (future_win['high'] - current_close) / current_close
    down_ampl = (current_close - future_win['low']) / current_close

    down_p10 = np.clip(np.percentile(down_ampl, 10), min_sl, max_sl)
    down_p90 = np.clip(np.percentile(down_ampl, 90), min_sl, max_sl)
    up_p10 = np.clip(max(np.percentile(up_ampl, 10), 2 * down_p10), 2 * down_p10, max_tp)
    up_p90 = np.clip(max(np.percentile(up_ampl, 90), 2 * down_p90), 2 * down_p90, max_tp)

    amp_labels.append([up_p10, up_p90, down_p10, down_p90])

labels_amplitude = np.array(amp_labels)

features = features.iloc[CFG.feature_engineering.window_size : CFG.feature_engineering.window_size + len(labels_amplitude)].reset_index(drop=True)
df_clean = df_clean.iloc[CFG.feature_engineering.window_size : CFG.feature_engineering.window_size + len(labels_amplitude)].reset_index(drop=True)
labels_direction = labels_direction[CFG.feature_engineering.window_size : CFG.feature_engineering.window_size + len(labels_amplitude)]

assert len(features) == len(labels_amplitude) == len(labels_direction), "Рассинхрон после Amplitude!"
logging.info(f"✅ Amplitude метки: {len(labels_amplitude)}")

# 5️⃣ HitOrder метки (direction-aware режим)
logging.info("Генерация HitOrder меток...")

sl_min = CFG.label_generation.hit_order_sl_min
sl_max = CFG.label_generation.hit_order_sl_max
rr_min = CFG.label_generation.hit_order_rr_min
rr_max = CFG.label_generation.hit_order_rr_max

hit_labels = []

for idx in range(len(df_clean) - CFG.labels.lookahead):
    win = df_clean.iloc[idx : idx + CFG.labels.lookahead]
    close = df_clean.iloc[idx]['close']

    # Берем направление из уже рассчитанных Direction меток
    direction = labels_direction[idx]

    # Пропускаем no-trade
    if direction == 1:
        continue

    sl_relative = np.random.uniform(sl_min, sl_max)
    rr = np.random.uniform(rr_min, rr_max)
    tp_relative = sl_relative * rr

    hit = 0

    if direction == 2:  # Long сценарий
        sl_level = close * (1 - sl_relative)
        tp_level = close * (1 + tp_relative)

        for _, row in win.iterrows():
            if row['high'] >= tp_level:
                hit = 1
                break
            elif row['low'] <= sl_level:
                hit = 0
                break

    elif direction == 0:  # Short сценарий
        sl_level = close * (1 + sl_relative)
        tp_level = close * (1 - tp_relative)

        for _, row in win.iterrows():
            if row['low'] <= tp_level:
                hit = 1
                break
            elif row['high'] >= sl_level:
                hit = 0
                break

    hit_labels.append([direction, sl_relative, tp_relative, hit])

labels_hitorder = np.array(hit_labels)

# Финальная синхронизация
final_len = len(labels_hitorder)
features = features.iloc[:final_len].reset_index(drop=True)
labels_direction = labels_direction[:final_len]
labels_amplitude = labels_amplitude[:final_len]

assert len(features) == len(labels_hitorder) == len(labels_amplitude) == len(labels_direction), "Финальный рассинхрон!"

# 6️⃣ Сохраняем
features.to_csv(CFG.paths.train_features_csv, index=False)
np.save(CFG.paths.train_labels_direction, labels_direction)
np.save(CFG.paths.train_labels_amplitude, labels_amplitude)
np.save(CFG.paths.train_labels_hitorder, labels_hitorder)

logging.info(f"✅ Финально сохранено: {len(features)} строк")
logging.info("🎯 Полный боевой FINISHED pipeline собран.")

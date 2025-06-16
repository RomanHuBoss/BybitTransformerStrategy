import pandas as pd
import numpy as np
import joblib
from feature_engineering import FeatureEngineer
from config import CFG
from sklearn.preprocessing import StandardScaler
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Direction лейблинг
def generate_direction_labels(df, lookahead, threshold):
    labels = []
    close_prices = df["close"].values

    for i in range(len(df)):
        if i + lookahead >= len(df):
            labels.append(2)  # боковик
            continue

        future_close = close_prices[i + lookahead]
        ret = (future_close - close_prices[i]) / close_prices[i]

        if ret > threshold:
            labels.append(1)
        elif ret < -threshold:
            labels.append(0)
        else:
            labels.append(2)

    return labels

# Amplitude лейблинг (ускоренная версия)
def generate_amplitude_labels(df, lookahead):
    highs = df['high'].values
    lows = df['low'].values

    up_moves = []
    down_moves = []

    for i in range(len(df)):
        window_highs = highs[i+1:i+1+lookahead]
        window_lows = lows[i+1:i+1+lookahead]

        if len(window_highs) == 0:
            up_moves.append(np.nan)
            down_moves.append(np.nan)
            continue

        base_price = df['close'].iloc[i]
        ups = (window_highs - base_price) / base_price
        downs = (window_lows - base_price) / base_price

        up_moves.append(ups)
        down_moves.append(downs)

    up_p10 = [np.percentile(m, 10) if isinstance(m, np.ndarray) and len(m) > 0 else 0 for m in up_moves]
    up_p90 = [np.percentile(m, 90) if isinstance(m, np.ndarray) and len(m) > 0 else 0 for m in up_moves]
    down_p10 = [np.percentile(m, 10) if isinstance(m, np.ndarray) and len(m) > 0 else 0 for m in down_moves]
    down_p90 = [np.percentile(m, 90) if isinstance(m, np.ndarray) and len(m) > 0 else 0 for m in down_moves]

    return pd.DataFrame({
        "amp_up_p10": up_p10,
        "amp_up_p90": up_p90,
        "amp_down_p10": down_p10,
        "amp_down_p90": down_p90
    })

# HitOrder лейблинг (ускоренная версия)
def generate_hitorder_labels(df, sl_list, rr_list, lookahead):
    close_prices = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    hit_labels = {}

    for sl in sl_list:
        for rr in rr_list:
            label_col = f"hit_SL{str(sl).replace('.', '_')}_RR{str(rr).replace('.', '_')}"
            hit_labels[label_col] = []

    for i in range(len(df)):
        if i + lookahead >= len(df):
            for label_col in hit_labels:
                hit_labels[label_col].append(0)
            continue

        future_highs = highs[i+1:i+1+lookahead]
        future_lows = lows[i+1:i+1+lookahead]
        entry_price = close_prices[i]

        for sl in sl_list:
            for rr in rr_list:
                sl_threshold = entry_price * (1 - sl)
                tp_threshold = entry_price * (1 + sl * rr)

                hit_sl = np.any(future_lows <= sl_threshold)
                hit_tp = np.any(future_highs >= tp_threshold)

                label_col = f"hit_SL{str(sl).replace('.', '_')}_RR{str(rr).replace('.', '_')}"
                hit_labels[label_col].append(1 if hit_tp and not hit_sl else 0)

    return pd.DataFrame(hit_labels)

# Основной пайплайн
def main():
    logging.info("🚀 Загружаем исходные данные...")
    df = pd.read_csv(CFG.paths.data_path)
    logging.info(f"✅ Загружено {len(df)} строк")

    logging.info("🧪 Генерируем признаки...")
    fe = FeatureEngineer(feature_columns=None)
    df_features = fe.generate_features(df, fit=True)
    logging.info(f"✅ Сгенерировано признаков: {df_features.shape[1]}")

    logging.info("🏷 Генерируем Direction метки...")
    df_features['direction_label'] = generate_direction_labels(
        df_features,
        lookahead=CFG.label.lookahead,
        threshold=CFG.label.direction_threshold
    )

    logging.info("📈 Генерируем Amplitude метки (ускоренная версия)...")
    amp_labels = generate_amplitude_labels(
        df_features,
        lookahead=CFG.label.lookahead
    )
    df_features = pd.concat([df_features, amp_labels], axis=1)

    logging.info("🎯 Генерируем HitOrder метки (ускоренная версия)...")
    hit_labels = generate_hitorder_labels(
        df_features,
        sl_list=CFG.label_generation.hitorder_sl_list,
        rr_list=CFG.label_generation.hitorder_rr_list,
        lookahead=CFG.label_generation.hitorder_lookahead
    )
    df_features = pd.concat([df_features, hit_labels], axis=1)

    df_features.dropna(inplace=True)
    logging.info(f"💾 Сохраняем итоговый датасет: {len(df_features)} строк")

    df_features.to_parquet(CFG.paths.feature_dataset_path, index=False)

    logging.info("⚙️ Обучаем scaler и сохраняем признаки...")
    feature_cols = [col for col in df_features.columns if col not in [
        "direction_label", "amp_up_p10", "amp_up_p90", "amp_down_p10", "amp_down_p90"
    ] and not col.startswith("hit_SL") and col not in [
        "open_time", "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]]

    scaler = StandardScaler()
    scaler.fit(df_features[feature_cols])

    joblib.dump(scaler, CFG.paths.scaler_path)
    joblib.dump(feature_cols, CFG.paths.feature_columns_path)
    logging.info("✅ Готово")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import joblib
import logging
import os
from sklearn.preprocessing import StandardScaler
from config import CFG
from feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Direction разметка (осталась прежней)
def generate_direction_labels(df, threshold, lookahead):
    closes = df["close"].values
    future_closes = np.roll(closes, -lookahead)
    changes = (future_closes - closes) / closes
    changes[-lookahead:] = 0  # хвост

    labels = np.full(len(df), 1)
    labels[changes > threshold] = 2
    labels[changes < -threshold] = 0
    return labels.tolist()


# Ускоренная Amplitude разметка
def generate_amplitude_labels(df, lookahead):
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    n = len(df)
    amp_up_p10, amp_up_p90 = [], []
    amp_down_p10, amp_down_p90 = [], []

    for i in range(n):
        window_highs = highs[i+1:i+1+lookahead]
        window_lows = lows[i+1:i+1+lookahead]

        if len(window_highs) < lookahead:
            amp_up_p10.append(0)
            amp_up_p90.append(0)
            amp_down_p10.append(0)
            amp_down_p90.append(0)
            continue

        up_moves = (window_highs - closes[i]) / closes[i]
        down_moves = (closes[i] - window_lows) / closes[i]

        amp_up_p10.append(np.quantile(up_moves, 0.1))
        amp_up_p90.append(np.quantile(up_moves, 0.9))
        amp_down_p10.append(np.quantile(down_moves, 0.1))
        amp_down_p90.append(np.quantile(down_moves, 0.9))

    return pd.DataFrame({
        'amp_up_p10': amp_up_p10,
        'amp_up_p90': amp_up_p90,
        'amp_down_p10': amp_down_p10,
        'amp_down_p90': amp_down_p90
    })


# HitOrder разметка (ускоренная и проверенная)
def generate_hitorder_labels(df, sl_list, rr_list, lookahead):
    result = {}

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)

    for sl in sl_list:
        for rr in rr_list:
            column_name = f"hit_SL{sl}_RR{rr}"
            labels = np.zeros(n, dtype=int)

            tp_prices = closes * (1 + rr * sl)
            sl_prices = closes * (1 - sl)

            for i in range(n):
                start_idx = i + 1
                end_idx = i + 1 + lookahead

                if end_idx >= n:
                    labels[i] = 0
                    continue

                high_window = highs[start_idx:end_idx]
                low_window = lows[start_idx:end_idx]

                hit_tp = high_window >= tp_prices[i]
                hit_sl = low_window <= sl_prices[i]

                first_tp = np.argmax(hit_tp) if np.any(hit_tp) else lookahead + 1
                first_sl = np.argmax(hit_sl) if np.any(hit_sl) else lookahead + 1

                labels[i] = 1 if first_tp < first_sl else 0

            result[column_name] = labels

    for col, values in result.items():
        df[col] = values

    return df


def main():
    logging.info("🚀 Загружаем исходные данные...")
    df = pd.read_csv(CFG.paths.train_csv)
    logging.info(f"✅ Загружено {len(df)} строк")

    logging.info("🧪 Генерируем признаки...")
    fe = FeatureEngineer(feature_columns=None)
    df_features = fe.generate_features(df, fit=True)
    logging.info(f"✅ Сгенерировано признаков: {df_features.shape[1]}")

    df = df.tail(len(df_features)).reset_index(drop=True)
    df_features = df_features.reset_index(drop=True)

    logging.info("🏷 Генерируем Direction метки...")
    df_features['direction_label'] = generate_direction_labels(
        df,
        threshold=CFG.label_generation.direction_threshold,
        lookahead=CFG.label_generation.direction_lookahead
    )

    logging.info("📈 Генерируем Amplitude метки (ускоренная версия)...")
    amp_labels = generate_amplitude_labels(
        df,
        lookahead=CFG.label_generation.amplitude_lookahead
    )
    df_features = pd.concat([df_features, amp_labels], axis=1)

    logging.info("🎯 Генерируем HitOrder метки (ускоренная версия)...")
    df_features = generate_hitorder_labels(
        df,
        sl_list=CFG.label_generation.hitorder_sl_list,
        rr_list=CFG.label_generation.hitorder_rr_list,
        lookahead=CFG.label_generation.hitorder_lookahead
    )

    df_features.dropna(inplace=True)
    df_features.reset_index(drop=True, inplace=True)
    logging.info(f"💾 Сохраняем итоговый датасет: {len(df_features)} строк")
    df_features.to_csv(CFG.paths.train_features_csv, index=False)

    logging.info("⚙️ Обучаем scaler и сохраняем признаки...")

    non_feature_cols = [
        'open_time', 'close_time', 'timestamp', 'date', 'symbol',
        'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
    ]

    non_label_cols = ['direction_label'] + [
        col for col in df_features.columns if col.startswith("amp_") or col.startswith("hit_SL")
    ]

    feature_cols = [
        col for col in df_features.columns
        if col not in (non_feature_cols + non_label_cols)
    ]

    scaler = StandardScaler()
    scaler.fit(df_features[feature_cols])

    joblib.dump(scaler, CFG.paths.scaler_path)
    joblib.dump(feature_cols, CFG.paths.feature_columns_path)

    logging.info("✅ Сохранили scaler и feature_columns.")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.preprocessing import StandardScaler
from config import CFG
from feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# Direction (остается как есть, он лёгкий)
def generate_direction_labels(df, threshold, lookahead):
    labels = []
    for i in range(len(df)):
        if i + lookahead >= len(df):
            labels.append(1)
            continue
        future_close = df.iloc[i + lookahead]["close"]
        change = (future_close - df.iloc[i]["close"]) / df.iloc[i]["close"]

        if change > threshold:
            labels.append(2)
        elif change < -threshold:
            labels.append(0)
        else:
            labels.append(1)
    return labels


# Ускоренный Amplitude
def generate_amplitude_labels(df, lookahead):
    price = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)

    shifted_highs = np.concatenate([[np.nan], highs[:-1]])
    shifted_lows = np.concatenate([[np.nan], lows[:-1]])

    high_windows = pd.Series(shifted_highs).rolling(window=lookahead, min_periods=1).apply(lambda x: list(x), raw=False).values
    low_windows = pd.Series(shifted_lows).rolling(window=lookahead, min_periods=1).apply(lambda x: list(x), raw=False).values

    up_p10, up_p90, down_p10, down_p90 = [], [], [], []

    for i in range(n):
        if not isinstance(high_windows[i], list):
            up_p10.append(0)
            up_p90.append(0)
            down_p10.append(0)
            down_p90.append(0)
            continue

        highs_i = np.array(high_windows[i])
        lows_i = np.array(low_windows[i])

        up_move = (highs_i - price[i]) / price[i]
        down_move = (price[i] - lows_i) / price[i]

        up_p10.append(np.quantile(up_move, 0.1))
        up_p90.append(np.quantile(up_move, 0.9))
        down_p10.append(np.quantile(down_move, 0.1))
        down_p90.append(np.quantile(down_move, 0.9))

    return pd.DataFrame({
        "amp_up_p10": up_p10,
        "amp_up_p90": up_p90,
        "amp_down_p10": down_p10,
        "amp_down_p90": down_p90
    })


# HitOrder (оптимизированный)
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

                if first_tp < first_sl:
                    labels[i] = 1
                else:
                    labels[i] = 0

            result[column_name] = labels

    for col, values in result.items():
        df[col] = values

    return df


def main():
    logging.info("🚀 Загружаем исходные данные...")
    df = pd.read_csv(CFG.paths.train_csv)
    logging.info(f"✅ Загружено {len(df)} строк")

    logging.info("🧪 Генерируем признаки...")
    fe = FeatureEngineer()
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

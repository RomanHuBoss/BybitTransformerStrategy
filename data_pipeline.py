import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.preprocessing import StandardScaler
from config import CFG
from feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Генерация direction-меток (3 класса: падение, боковик, рост)
def generate_direction_labels(df, threshold, lookahead):
    labels = []
    for i in range(len(df)):
        if i + lookahead >= len(df):
            labels.append(1)  # нет будущих данных → считаем боковик
            continue

        future_close = df.iloc[i + lookahead]["close"]
        change = (future_close - df.iloc[i]["close"]) / df.iloc[i]["close"]

        if change > threshold:
            labels.append(2)  # рост
        elif change < -threshold:
            labels.append(0)  # падение
        else:
            labels.append(1)  # боковик
    return labels

# Генерация amplitude-меток (амплитудные квантили будущих движений)
def generate_amplitude_labels(df, lookahead):
    up_p10, up_p90, down_p10, down_p90 = [], [], [], []

    for i in range(len(df)):
        if i + lookahead >= len(df):
            # Нет будущих данных — считаем как идеальный боковик (нулевая амплитуда)
            up_p10.append(0)
            up_p90.append(0)
            down_p10.append(0)
            down_p90.append(0)
            continue

        window = df.iloc[i + 1: i + lookahead + 1]
        price = df.iloc[i]["close"]

        up = (window["high"] - price) / price
        down = (price - window["low"]) / price

        up_p10.append(up.quantile(0.1))
        up_p90.append(up.quantile(0.9))
        down_p10.append(down.quantile(0.1))
        down_p90.append(down.quantile(0.9))

    return pd.DataFrame({
        "amp_up_p10": up_p10,
        "amp_up_p90": up_p90,
        "amp_down_p10": down_p10,
        "amp_down_p90": down_p90
    })

# Генерация hitorder-меток (срабатывание TP/SL по профилям)
def generate_hitorder_labels(df, sl_list, rr_list, lookahead):
    result = {}

    for sl in sl_list:
        for rr in rr_list:
            column_name = f"hit_SL{sl}_RR{rr}"
            labels = []

            for i in range(len(df)):
                start_idx = i + 1
                end_idx = i + 1 + lookahead

                if end_idx >= len(df):
                    labels.append(0)  # нет будущих данных — считаем как SL
                    continue

                entry_price = df.iloc[i]['close']
                tp_price = entry_price * (1 + rr * sl)
                sl_price = entry_price * (1 - sl)

                window = df.iloc[start_idx:end_idx]
                hit_label = None

                for _, row in window.iterrows():
                    high = row['high']
                    low = row['low']

                    if high >= tp_price and low <= sl_price:
                        hit_label = 0  # оба достигнуты — SL
                        break
                    elif high >= tp_price:
                        hit_label = 1  # TP
                        break
                    elif low <= sl_price:
                        hit_label = 0  # SL
                        break

                if hit_label is None:
                    labels.append(0)  # ничего не достигнуто — считаем SL
                else:
                    labels.append(hit_label)

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

    # Синхронизация с длиной признаков после feature engineering
    df = df.tail(len(df_features)).reset_index(drop=True)
    df_features = df_features.reset_index(drop=True)

    logging.info("🏷 Генерируем Direction метки...")
    df_features['direction_label'] = generate_direction_labels(
        df,
        threshold=CFG.label_generation.direction_threshold,
        lookahead=CFG.label_generation.direction_lookahead
    )

    logging.info("📈 Генерируем Amplitude метки...")
    amp_labels = generate_amplitude_labels(
        df,
        lookahead=CFG.label_generation.amplitude_lookahead
    )
    df_features = pd.concat([df_features, amp_labels], axis=1)

    logging.info("🎯 Генерируем HitOrder метки...")
    df_features = generate_hitorder_labels(
        df,
        sl_list=CFG.label_generation.hitorder_sl_list,
        rr_list=CFG.label_generation.hitorder_rr_list,
        lookahead=CFG.label_generation.hitorder_lookahead
    )

    # Удаляем только строки с пропусками в самих признаках (теоретически их быть не должно)
    df_features.dropna(inplace=True)
    df_features.reset_index(drop=True, inplace=True)

    logging.info(f"💾 Сохраняем итоговый датасет: {len(df_features)} строк")
    df_features.to_csv(CFG.paths.train_features_csv, index=False)

    # ⬇️ АВТОМАТИЧЕСКИЙ scaler и сохранение признаков:

    logging.info("⚙️ Обучаем scaler и сохраняем признаки...")

    # Определяем признаки: всё кроме лейблов
    feature_cols = [col for col in df_features.columns if not col.startswith("direction_label")
                     and not col.startswith("amp_")
                     and not col.startswith("hit_SL")]

    # Обучаем scaler
    scaler = StandardScaler()
    scaler.fit(df_features[feature_cols])

    # Сохраняем scaler и список признаков
    joblib.dump(scaler, CFG.paths.scaler_path)
    pd.Series(feature_cols).to_csv(CFG.paths.feature_columns_path, index=False)

    logging.info("✅ Сохранили scaler и feature_columns.")

if __name__ == "__main__":
    main()

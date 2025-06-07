import os.path
import pandas as pd
from datetime import datetime
from config import CFG

def bybit_candles_to_df(json_data):
    original_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
    data_list = json_data['result']['list']
    df = pd.DataFrame(data_list).iloc[:, :len(original_columns)]
    df.columns = original_columns

    # Приведение типов
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df['open_time'] = pd.to_datetime(pd.to_numeric(df['open_time'], errors='coerce'), unit='ms', errors='coerce')


    df.dropna(subset=['open_time'], inplace=True)

    df.sort_values('open_time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def bybit_candles_to_csv(json):
    """
    Сохраняет полученные с биржи Bybit JSON-данные торговли
    в csv-файл

    Параметры:
    json_data (dict): JSON данные, содержащие массив 'list' в 'result'
    """
    current_utc_time = datetime.utcnow()
    formatted_time = current_utc_time.strftime("%Y-%m-%d-%H-%M")

    filename = f"{json['result']['symbol']}_{json['timeframe']}m_{formatted_time}.csv"
    df = bybit_candles_to_df(json)
    df.to_csv(os.path.join("downloads", filename), index=False)
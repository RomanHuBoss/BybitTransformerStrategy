import os.path
import pandas as pd
from datetime import datetime
from config import Config

def bybit_candles_to_df(json_data):
    """
    Преобразует JSON данные в DataFrame с колонками:
    open_time, open, high, low, close, volume

    Параметры:
    json_data (dict): JSON данные, содержащие массив 'list' в 'result'

    Возвращает:
    pd.DataFrame: DataFrame с торговыми данными
    """
    # Извлекаем массив 'list' из JSON
    data_list = json_data['result']['list']

    # Создаем DataFrame с указанными колонками
    df = pd.DataFrame(data_list).iloc[:, :len(Config.ORIGINAL_COLUMNS)]
    df.columns = Config.ORIGINAL_COLUMNS

    # Сортируем по времени (на всякий случай)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')

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
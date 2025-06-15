import pandas as pd
import os
from dateutil.relativedelta import relativedelta

# Пути к файлам
folder = os.path.join("..", "historical_data", "BTC-ETH-COMBINED")
csv1 = os.path.join(folder, "btc_combined_csv.csv")
csv2 = os.path.join(folder, "eth_combined_csv.csv")
common_csv = os.path.join(folder, "combined_csv.csv")

def safe_shift_year(date, shift):
    """
    Сдвиг года с защитой от високосных дат
    """
    try:
        return date.replace(year=date.year + shift)
    except ValueError:
        if date.month == 2 and date.day == 29:
            return date.replace(year=date.year + shift, day=28)
        else:
            raise

def calculate_shift_years(file1_dates, file2_dates, safety_years=2):
    """
    Вычисляем безопасный сдвиг по годам
    """
    max_time_1 = file1_dates.max()
    min_time_2 = file2_dates.min()
    years_shift = (max_time_1.year - min_time_2.year) + safety_years
    return years_shift

def combine_files(file1_path, file2_path, output_path):
    """
    Объединяет два файла с безопасным сдвигом времени
    """
    # Загрузка данных
    file1 = pd.read_csv(file1_path)
    file2 = pd.read_csv(file2_path)

    # Парсим open_time (в миллисекундах)
    file1['open_time'] = pd.to_datetime(file1['open_time'], unit='ms', errors='coerce')
    file2['open_time'] = pd.to_datetime(file2['open_time'], unit='ms', errors='coerce')

    # Удаляем некорректные строки
    file1 = file1.dropna(subset=['open_time'])
    file2 = file2.dropna(subset=['open_time'])

    # Считаем безопасный сдвиг лет
    years_shift = calculate_shift_years(file1['open_time'], file2['open_time'])

    # Применяем сдвиг
    file2['open_time'] = file2['open_time'].apply(lambda dt: safe_shift_year(dt, years_shift))

    # Объединяем и сортируем
    combined = pd.concat([file1, file2], ignore_index=True).sort_values(by='open_time').reset_index(drop=True)

    # Сохраняем
    combined.to_csv(output_path, index=False)
    print(f"\nОбъединение успешно завершено!\nРезультат сохранён в: {output_path}")

# Запускаем
combine_files(csv1, csv2, common_csv)

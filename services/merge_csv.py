import os
import pandas as pd

# Укажите путь к папке с CSV-файлами
folder_path = os.path.join('..', 'historical_data', 'ETHUSDT', '30m', 'monthly')  # Замените на ваш путь

# Получаем список всех CSV-файлов в папке
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Проверяем, что файлы найдены
if not csv_files:
    print("В указанной папке нет CSV-файлов.")
else:
    # Создаём пустой DataFrame для хранения объединённых данных
    combined_data = pd.DataFrame()

    columns = [
         'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
         'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
    ]

    # Читаем и объединяем все CSV-файлы
    for file in csv_files:
        if file == 'combined_csv.csv':
            continue

        file_path = os.path.join(folder_path, file)

        data = pd.read_csv(file_path, header=None)
        first_row_data = data.iloc[0].tolist()
        is_header = any(value in columns for value in first_row_data)

        if is_header:
            data = data.iloc[1:]

        combined_data = pd.concat([combined_data, data], ignore_index=True)

    combined_data.columns = columns

    # Сохраняем объединённый файл
    output_path = os.path.join(folder_path, 'combined_csv.csv')
    combined_data.to_csv(output_path, index=False)
    print(f"Объединённый файл сохранён как: {output_path}")
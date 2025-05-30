import os
import uvicorn
import json
from pybit.unified_trading import HTTP
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Optional
from services.get_bybit_candles import get_bybit_candles
from services.bybit_candles_handlers import bybit_candles_to_csv, bybit_candles_to_df
from services.bybit_symbols_list import BybitSymbolsList
from predictor import TradingPredictor
import time

"""
Эта балалайка запрашивает с биржи Bybit данные о торговле конкретной криптовалютной парой с текущего момента времени и до момента в прошлом, определяемого
Как это работает:   
    http://127.0.0.1:8000/prognosis/?symbol=BTCUSDT&timeframe=3&candles_num=200
    Единственный эндпоинт /prognosis/ принимает:
        symbol (строка) - торговый символ
        timeframe - таймфрейм (1, 5, 15 минут и т.д.)
        candles_num (целое число) - количество свечей
        save_to_csv (boolean) - надо ли скачанную инфу сохранить в csv-файле в папке downloads
    Валидация параметров:
        Если параметры не соответствуют бизнес-правилам (например, слишком длинный символ), возвращается ошибка 400 с деталями
    Успешный ответ:
        При корректных параметрах возвращается JSON с рекомендациями по входу в сделки
    Обработка всех других путей:
        Любые другие URL возвращают 404 ошибку
        Обрабатываются все возможные HTTP методы
    Документация:
        Автоматически генерируется Swagger UI с описанием эндпоинта (http://127.0.0.1:8000/docs)
        Указаны возможные коды ответов (200, 400, 404)
"""

model_folder = os.path.join("models", "model-1_5-3_9-multiplier")
app = FastAPI()
bybit_symbols = BybitSymbolsList()
predictor = TradingPredictor(model_folder=model_folder, threshold=0.6)
prognosis_cache = {}


# Модель для успешного ответа
class PrognosisResponse(BaseModel):
    timeframe: int
    candles_num: int
    prognosis: list # данные прогноза

# Модель для успешного ответа
class CandlesResponse(BaseModel):
    symbol: str
    timeframe: int
    candles_num: int
    candles_data: list  # данные свечей

# Модель для ошибки в бизнес-логике
class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

# эндпоинт
@app.get("/prognosis/", responses={
    200: {"model": PrognosisResponse},
    400: {"model": ErrorResponse},
    404: {"description": "Not found"},
})
async def get_prognosis(timeframe: int, candles_num: int, symbol:Optional[str] = None):
    """
    prognosis_cache = {
        symbol_name: {
            last_update: value,
            prognosis: json
        }
    }
    """

    symbols_list = bybit_symbols.get_bybit_symbols_list(limit=1000)
    current_time = int(time.time())

    for tmp_symbol in symbols_list:
        if symbol is not None and symbol != tmp_symbol:
            continue

        # Получаем данные из кеша, если они есть
        cached_data = prognosis_cache.get(tmp_symbol, {})
        last_update = cached_data.get('last_update', -1)

        # Если данные устарели или отсутствуют, обновляем
        if last_update == -1 or last_update < current_time - 60:
            try:
                df = bybit_candles_to_df(await get_candles(tmp_symbol, timeframe, candles_num))
                symbol_prognosis = predictor.predict(df)

                # Обновляем кеш только если получили прогноз
                if symbol_prognosis:  # Проверяем, что прогноз не пустой
                    prognosis_cache[tmp_symbol] = {
                        'last_update': current_time,
                        'symbol_prognosis': symbol_prognosis,
                    }
                elif tmp_symbol in prognosis_cache:
                    # Удаляем устаревшие пустые прогнозы
                    del prognosis_cache[tmp_symbol]

            except Exception as e:
                print(f"Error processing symbol {tmp_symbol}: {str(e)}")
                continue

    if symbol is not None:
        return prognosis_cache.get(symbol, {})

    return prognosis_cache

# эндпоинт
@app.get("/candles/", responses={
    200: {"model": CandlesResponse},
    400: {"model": ErrorResponse},
    404: {"description": "Not found"},
})
async def get_candles(symbol: str, timeframe: int, candles_num: int, save_to_csv: Optional[bool] = False):
    candles_data = get_bybit_candles(symbol, timeframe, candles_num)

    if save_to_csv:
        bybit_candles_to_csv(candles_data)

    return candles_data


# Обработка всех других путей с 404 ошибкой
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def handle_unsupported_paths(path: str):
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Некорректный запрос. Используй /candles/ или /prognosis/"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")



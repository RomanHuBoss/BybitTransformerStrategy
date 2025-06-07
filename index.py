import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import time
import os

from services.get_bybit_candles import get_bybit_candles
from services.bybit_candles_handlers import  bybit_candles_to_df
from services.bybit_symbols_list import BybitSymbolsList
from predictor import Predictor

app = FastAPI()

CACHE = {}  # symbol -> {last_update, result}

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", encoding="utf-8") as f:
        return f.read()

@app.get("/get_currency_pairs")
async def get_currency_pairs():
    symbolsList = BybitSymbolsList()
    return symbolsList.get_bybit_symbols_list(1000)

@app.get("/predict_info")
async def predict_info(symbol: str, timeframe: int, threshold: float = 0.65):
    now = time.time()
    cache_key = f"{symbol}_{timeframe}_{threshold}"
    cache_entry = CACHE.get(cache_key)

    if not cache_entry or now - cache_entry["last_update"] > 300:
        try:
            raw = get_bybit_candles(symbol, timeframe=timeframe, candles_num=200)
            df = bybit_candles_to_df(raw)
            dynamic_predictor = Predictor(model_folder="artifacts")

            result = dynamic_predictor.predict(df)
            CACHE[cache_key] = {
                "last_update": now,
                "result": result
            }
        except Exception as e:
            return JSONResponse({"error": f"Ошибка при получении данных: {str(e)}"}, status_code=500)

    result = CACHE[cache_key]["result"]

    # Фильтрация по threshold
    filtered = {
        "tp_sl_pairs": [],
        "classes": [],
        "confidences": [],
    }

    for tp_sl, cls, conf in zip(result["tp_sl_pairs"], result["classes"], result["confidences"]):
        if conf >= threshold and cls in [0, 1]:  # cls=2 — это "нет сигнала"
            filtered["tp_sl_pairs"].append(tp_sl)
            filtered["classes"].append(cls)
            filtered["confidences"].append(conf)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "threshold": threshold,
        "timestamp": CACHE[cache_key]["last_update"],
        "predictions": filtered
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("index:app", host="0.0.0.0", port=port, reload=True)

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import os

from services.bybit_symbols_list import BybitSymbolsList
from snapshot_inference import SnapshotInferenceEngine

app = FastAPI()

symbolsList = BybitSymbolsList()
symbols = symbolsList.get_bybit_symbols_list(1000)
snapshot_engine = SnapshotInferenceEngine(symbols, timeframe=30)

@app.get("/", response_class=HTMLResponse)
async def root():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_dir, "index_hybrid.html")
    with open(html_path, encoding="utf-8") as f:
        return f.read()

@app.get("/get_currency_pairs")
async def get_currency_pairs():
    return symbols

@app.get("/predict_hybrid")
async def predict_hybrid(symbol: str, timeframe: int, tp_coef: float = 0.7, sl_coef: float = 0.3):
    snapshot = snapshot_engine.get(symbol)
    if not snapshot:
        return JSONResponse({"error": "Нет данных для символа"}, status_code=404)

    return snapshot["result"]

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("index:app", host="0.0.0.0", port=port, reload=True)

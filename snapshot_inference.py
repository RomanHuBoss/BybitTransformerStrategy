import asyncio
import logging
from hybrid_predictor import HybridPredictor
from services.bybit_symbols_list import BybitSymbolsList
from services.get_bybit_candles import get_bybit_candles
from services.bybit_candles_handlers import bybit_candles_to_df, filter_closed_bars

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class SnapshotInference:
    def __init__(self):
        self.predictor = HybridPredictor()
        self.symbol_provider = BybitSymbolsList()
        self.symbols_all = self.symbol_provider.get_bybit_symbols_list(limit=1000)
        self.max_symbols = 100
        self.snapshot = {}

    async def run_loop(self):
        while True:
            await self.update_snapshot()
            await asyncio.sleep(60)

    async def update_snapshot(self):
        symbols = self.symbols_all[:self.max_symbols]
        logging.info(f"Обновление снимка: {len(symbols)} активов")
        snapshot_temp = {}

        for symbol in symbols:
            try:
                df = await self.get_recent_ohlcv(symbol)
                if df is None or len(df) < 100:
                    continue
                result = self.predictor.predict(df)
                snapshot_temp[symbol] = result
            except Exception as e:
                logging.error(f"Ошибка обработки {symbol}: {e}")

        self.snapshot = snapshot_temp

    async def get_recent_ohlcv(self, symbol, interval="30", limit=100):
        try:
            candles_json = await get_bybit_candles(symbol=symbol, timeframe=int(interval), candles_num=limit)
            df = bybit_candles_to_df(candles_json)
            df = filter_closed_bars(df, timeframe_minutes=int(interval))
            return df
        except Exception as e:
            logging.error(f"Ошибка получения OHLCV для {symbol}: {e}")
            return None

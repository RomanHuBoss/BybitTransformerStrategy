import asyncio
import logging
from asyncio import Semaphore
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
        self.max_symbols = 300
        self.snapshot = {}
        self.semaphore = Semaphore(10)  # Ограничиваем параллелизм до 10 запросов одновременно

    async def run_loop(self):
        while True:
            await self.update_snapshot()
            await asyncio.sleep(60)

    async def update_snapshot(self):
        symbols = self.symbols_all[:self.max_symbols]
        logging.info(f"Обновление снимка: {len(symbols)} активов")

        tasks = [self.process_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)

        snapshot_temp = {symbol: result for symbol, result in results if result is not None}
        self.snapshot = snapshot_temp
        logging.info(f"Собрано валидных сигналов: {len(self.snapshot)}")

    async def process_symbol(self, symbol):
        async with self.semaphore:
            try:
                df = await self.get_recent_ohlcv(symbol)

                if df is None or len(df) < 100:
                    logging.warning(f"Недостаточно свечей для {symbol}")
                    return (symbol, None)

                result = self.predictor.predict(df)
                return (symbol, result)

            except Exception as e:
                logging.error(f"Ошибка обработки {symbol}: {e}")
                return (symbol, None)

    async def get_recent_ohlcv(self, symbol, interval=30, limit=1000):
        try:
            candles_json = await get_bybit_candles(symbol=symbol, timeframe=int(interval), candles_num=limit)
            df = bybit_candles_to_df(candles_json)
            df = filter_closed_bars(df, timeframe_minutes=int(interval))
            return df
        except Exception as e:
            logging.error(f"Ошибка получения OHLCV для {symbol}: {e}")
            return None

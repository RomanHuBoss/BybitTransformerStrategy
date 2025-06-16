import asyncio
import logging
from asyncio import Semaphore
from hybrid_predictor import HybridPredictor
from feature_engineering import FeatureEngineer
from services.bybit_symbols_list import BybitSymbolsList
from services.get_bybit_candles import get_bybit_candles
from services.bybit_candles_handlers import bybit_candles_to_df, filter_closed_bars

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class SnapshotInference:
    def __init__(self):
        self.predictor = HybridPredictor()
        self.feature_engineer = FeatureEngineer()  # новый шаг — динамический генератор фичей
        self.symbol_provider = BybitSymbolsList()
        self.symbols_all = self.symbol_provider.get_bybit_symbols_list(limit=1000)
        self.max_symbols = 300
        self.snapshot = {}
        self.semaphore = Semaphore(15)
        self.symbol_pointer = 0  # указатель на следующую партию

    async def preload_snapshot(self):
        asyncio.create_task(self.update_snapshot_loop())

    async def update_snapshot_loop(self):
        while True:
            await self.incremental_update()
            await asyncio.sleep(1)

    async def incremental_update(self):
        batch_size = 15
        end_pointer = min(self.symbol_pointer + batch_size, len(self.symbols_all))
        batch_symbols = self.symbols_all[self.symbol_pointer:end_pointer]

        tasks = [self.process_symbol(symbol) for symbol in batch_symbols]
        for future in asyncio.as_completed(tasks):
            symbol, result = await future
            if result is not None:
                self.snapshot[symbol] = result
            elif symbol in self.snapshot:
                del self.snapshot[symbol]

        self.symbol_pointer = end_pointer if end_pointer < len(self.symbols_all) else 0
        logging.info(f"Инкрементальная обработка: {self.symbol_pointer}/{len(self.symbols_all)}")

    async def process_symbol(self, symbol):
        async with self.semaphore:
            try:
                df = await self.get_recent_ohlcv(symbol)
                if df is None or len(df) < 100:
                    logging.warning(f"Недостаточно свечей для {symbol}")
                    return (symbol, None)

                # Генерируем фичи на основе live OHLCV
                features_df = self.feature_engineer.generate_features(df, fit=False)

                # Предсказываем на последней строке
                result = self.predictor.predict(features_df.tail(1))
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

import logging
import asyncio
import random
from datetime import datetime, UTC

from hybrid_predictor import HybridPredictor
from services.get_bybit_candles import get_bybit_candles
from services.bybit_candles_handlers import bybit_candles_to_df
from services.bybit_symbols_list import BybitSymbolsList
from config import CFG


class SnapshotInference:
    def __init__(self):
        self.predictor = HybridPredictor()
        self.snapshot = {}

        # Загружаем актуальный список валют динамически
        symbol_provider = BybitSymbolsList()
        self.symbols = symbol_provider.get_bybit_symbols_list(limit=500)
        logging.info(f"✅ Загружено {len(self.symbols)} символов с Bybit.")

    async def update_snapshot(self):
        # Обновляем не все сразу, а по частям (например 5 валют за итерацию)
        symbols_to_update = random.sample(self.symbols, k=min(5, len(self.symbols)))
        tasks = [self.process_symbol(symbol) for symbol in symbols_to_update]

        await asyncio.gather(*tasks)

        self.snapshot["timestamp"] = datetime.now(UTC).isoformat()
        logging.info(f"Снимок обновлён. Активов: {len(self.snapshot) - 1}")

    async def process_symbol(self, symbol):
        try:
            df = bybit_candles_to_df(await get_bybit_candles(
                symbol,
                CFG.assets.timeframe,
                candles_num=CFG.assets.limit
            ))
            result = self.predictor.predict(df)
            self.snapshot[symbol] = result
        except Exception as e:
            logging.warning(f"Ошибка при обработке {symbol}: {e}")

    def get_snapshot(self):
        return self.snapshot


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    engine = SnapshotInference()

    asyncio.run(engine.update_snapshot())
    print(engine.get_snapshot())

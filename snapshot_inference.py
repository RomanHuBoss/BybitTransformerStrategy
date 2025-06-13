import logging
from datetime import datetime, UTC

from hybrid_predictor import HybridPredictor
from services.get_bybit_candles import get_bybit_candles
from config import CFG


class SnapshotInference:
    def __init__(self):
        self.predictor = HybridPredictor()
        self.snapshot = {}

    async def update_snapshot(self):
        tasks = []
        for symbol in CFG.assets.symbols:
            tasks.append(self.process_symbol(symbol))

        await asyncio.gather(*tasks)

        self.snapshot["timestamp"] = datetime.now(UTC).isoformat()
        logging.info(f"Снимок обновлён. Активов: {len(self.snapshot) - 1}")

    async def process_symbol(self, symbol):
        try:
            df = await get_bybit_candles(symbol, CFG.assets.timeframe, candles_num=CFG.assets.limit)
            result = self.predictor.predict(df)
            self.snapshot[symbol] = result
        except Exception as e:
            logging.warning(f"Ошибка при обработке {symbol}: {e}")

    def get_snapshot(self):
        return self.snapshot


if __name__ == '__main__':
    import asyncio

    logging.basicConfig(level=logging.INFO)
    engine = SnapshotInference()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(engine.update_snapshot())

    print(engine.get_snapshot())

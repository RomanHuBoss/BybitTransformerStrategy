import time
import threading
from services.get_bybit_candles import get_bybit_candles
from services.bybit_candles_handlers import bybit_candles_to_df, filter_closed_bars
from hybrid_predictor import AdaptiveHybridPredictor

class SnapshotInferenceEngine:
    def __init__(self, symbols, timeframe, update_interval=1800):
        self.symbols = symbols
        self.timeframe = timeframe
        self.update_interval = update_interval  # 30 минут
        self.cache = {}  # symbol -> {timestamp, result}

        self.hybrid_predictor = AdaptiveHybridPredictor(
            direction_model_folder="artifacts/direction_model_30m",
            amplitude_model_folder="artifacts/amplitude_model_30m"
        )

        self.start_initial_load()

    def start_initial_load(self):
        def load():
            self.update_all()
            self.start_auto_update()
        t = threading.Thread(target=load, daemon=True)
        t.start()

    def update_symbol(self, symbol):
        try:
            raw = get_bybit_candles(symbol, timeframe=self.timeframe, candles_num=1000)
            df = bybit_candles_to_df(raw)
            df = filter_closed_bars(df, timeframe_minutes=self.timeframe)

            if len(df) < 100:
                print(f"Not enough data for {symbol}, skipping.")
                return

            result = self.hybrid_predictor.predict(df)
            self.cache[symbol] = {
                "timestamp": int(time.time()),
                "result": result
            }
            print(f"Updated {symbol}: {result['signal_type']}")

        except Exception as e:
            print(f"Error updating {symbol}: {e}")

    def update_all(self):
        for symbol in self.symbols:
            self.update_symbol(symbol)

    def start_auto_update(self):
        def update_loop():
            while True:
                self.update_all()
                time.sleep(self.update_interval)
        t = threading.Thread(target=update_loop, daemon=True)
        t.start()

    def get(self, symbol):
        return self.cache.get(symbol)

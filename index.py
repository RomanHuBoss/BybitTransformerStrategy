import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from snapshot_inference import SnapshotInference
from config import CFG

# Продакшн логгирование на русском
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

snapshot_engine = SnapshotInference()

@asynccontextmanager
async def lifespan(_):
    logging.info("Запуск стримингового инференса...")
    task = asyncio.create_task(snapshot_loop())
    yield
    task.cancel()
    logging.info("Инференс остановлен.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def snapshot_loop():
    while True:
        try:
            await snapshot_engine.update_snapshot()
        except Exception as e:
            logging.error(f"Ошибка при обновлении snapshot: {e}")
        await asyncio.sleep(CFG.inference.update_interval)

@app.get("/snapshot")
async def get_snapshot():
    logging.info("Запрос на получение актуального snapshot.")
    return snapshot_engine.get_snapshot()

@app.get("/force_update")
async def force_update():
    logging.info("Принудительное обновление snapshot.")
    await snapshot_engine.update_snapshot()
    return {"status": "обновлено"}

if __name__ == "__main__":
    import uvicorn
    logging.info("Запуск сервера...")
    uvicorn.run(app, host="0.0.0.0", port=CFG.inference.api_port)

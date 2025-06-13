import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from snapshot_inference import SnapshotInference
from config import CFG

app = FastAPI()

# Разрешаем доступ с фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация snapshot движка
snapshot_engine = SnapshotInference()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(snapshot_loop())

async def snapshot_loop():
    while True:
        try:
            await snapshot_engine.update_snapshot()
        except Exception as e:
            logging.error(f"Ошибка при обновлении snapshot: {e}")
        await asyncio.sleep(CFG.inference.update_interval)

@app.get("/snapshot")
async def get_snapshot():
    return snapshot_engine.get_snapshot()

# Для ручного обновления (опционально)
@app.get("/force_update")
async def force_update():
    await snapshot_engine.update_snapshot()
    return {"status": "updated"}

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=CFG.inference.api_port)

import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from snapshot_inference import SnapshotInference
from config import CFG

snapshot_engine = SnapshotInference()

@asynccontextmanager
async def lifespan(_):
    task = asyncio.create_task(snapshot_loop())
    yield
    task.cancel()

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
    return snapshot_engine.get_snapshot()

@app.get("/force_update")
async def force_update():
    await snapshot_engine.update_snapshot()
    return {"status": "updated"}

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=CFG.inference.api_port)

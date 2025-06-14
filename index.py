import asyncio
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from snapshot_inference import SnapshotInference
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

snapshot_engine = SnapshotInference()

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(streaming_loop())
    logging.info("🚀 Запуск стримингового инференса...")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

@app.get("/snapshot")
async def get_snapshot(limit: int = 20):
    items = list(snapshot_engine.snapshot.items())
    filtered = dict(items[:limit])
    return filtered

async def streaming_loop():
    while True:
        await snapshot_engine.update_snapshot()
        await asyncio.sleep(1)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("index:app", host="0.0.0.0", port=8000, reload=True)

import asyncio
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from snapshot_inference import SnapshotInference
import os
from fastapi.responses import FileResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Инициализируем predictor заранее
snapshot_engine = SnapshotInference()

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(streaming_loop())
    logging.info("🚀 Запуск стримингового инференса...")
    yield

# Инициализация FastAPI с lifespan
app = FastAPI(lifespan=lifespan)

# CORS — разрешаем запросы для фронта
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Определяем путь до статики
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")

# Монтируем статику на /static
app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")

# Обработка корня: отдаём index.html при заходе на /
@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))

# Отдача snapshot с лимитом
@app.get("/snapshot")
async def get_snapshot(limit: int = 300):
    items = list(snapshot_engine.snapshot.items())
    filtered = dict(items[:limit])
    return filtered

# Стриминг обновления инференса в фоне
async def streaming_loop():
    while True:
        await snapshot_engine.update_snapshot()
        await asyncio.sleep(1)

# Healthcheck
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Запуск сервера из файла напрямую
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("index:app", host="0.0.0.0", port=8000, reload=True)

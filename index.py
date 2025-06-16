import asyncio
import json
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from snapshot_inference import SnapshotInference

app = FastAPI()

# Разрешаем CORS (можно ограничить при продакшн-деплое)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Монтируем директорию со статиками
app.mount("/static", StaticFiles(directory="static"), name="static")

# Отдаём HTML страницу
@app.get("/")
async def get():
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Инференс-объект
snapshot_inference = SnapshotInference()

# Запуск фоновой задачи обновления snapshot при старте сервера
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(snapshot_inference.preload_snapshot())

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.receive_text()
            snapshot = snapshot_inference.snapshot
            response = json.dumps(snapshot)
            await websocket.send_text(response)
    except Exception as e:
        print(f"WebSocket closed: {e}")

# Запуск сервера uvicorn
if __name__ == "__main__":
    uvicorn.run("index:app", host="0.0.0.0", port=8000, reload=False)

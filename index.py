import asyncio
import logging
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from starlette.websockets import WebSocketDisconnect

from snapshot_inference import SnapshotInference
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

snapshot_engine = SnapshotInference()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # preload snapshot перед запуском сервера:
    await snapshot_engine.update_snapshot()
    task = asyncio.create_task(snapshot_engine.run_loop())
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

app.mount("/static", StaticFiles(directory="static", html=True), name="static")

@app.get("/snapshot")
async def get_snapshot():
    return snapshot_engine.snapshot

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            snapshot = dict(list(snapshot_engine.snapshot.items())[:300])
            await websocket.send_json(snapshot)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        logging.info("WebSocket client disconnected.")



if __name__ == "__main__":
    uvicorn.run("index:app", host="0.0.0.0", port=8000, reload=False)

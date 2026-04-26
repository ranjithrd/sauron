import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db.connection import init_db
from app.db.writer import write_track
from app.kalman.tracker import KalmanTracker
from app.triangulation.pipeline import TriangulationPipeline
from app.video.stream_manager import StreamManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── DB ────────────────────────────────────────────────────────────
    await init_db()

    # ── Pipeline: StreamManager → TriangulationPipeline → KalmanTracker → DB
    kalman_tracker = KalmanTracker(on_update=write_track)
    pipeline = TriangulationPipeline(kalman_tracker=kalman_tracker)
    stream_manager = StreamManager()
    stream_manager.on_detection(pipeline.handle_detection)

    await pipeline.start()
    
    # ── Background Tasks ──────────────────────────────────────────────
    asyncio.create_task(stream_manager.run(), name="stream-manager-loop")

    app.state.stream_manager = stream_manager
    app.state.pipeline = pipeline
    app.state.kalman_tracker = kalman_tracker

    yield

    # ── Graceful shutdown ─────────────────────────────────────────────
    await pipeline.stop()
    await stream_manager.stop()


app = FastAPI(title="IOT EL Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.ingestion import router as ingestion_router  # noqa: E402
from app.api.dashboard import router as dashboard_router  # noqa: E402

app.include_router(ingestion_router)
app.include_router(dashboard_router)

# Serve the static dashboard
import os
os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

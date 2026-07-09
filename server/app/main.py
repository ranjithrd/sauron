import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings

_log_level = getattr(logging, os.environ.get("LOG_LEVEL", "DEBUG").upper(), logging.DEBUG)
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
# Keep noisy libraries quiet regardless of our level
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
from app.db.connection import init_db
from app.db.writer import write_rays, write_track
from app.ingestion.mqtt_client import MQTTClient
from app.kalman.tracker import KalmanTracker
from app.triangulation.pipeline import TriangulationPipeline
from app.vlm.scheduler import VLMScheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── DB ────────────────────────────────────────────────────────────
    await init_db()

    # ── Pipeline: MQTTClient → TriangulationPipeline → KalmanTracker → DB
    kalman_tracker = KalmanTracker(on_update=write_track)
    pipeline = TriangulationPipeline(kalman_tracker=kalman_tracker, on_rays=write_rays)
    mqtt_client = MQTTClient()
    mqtt_client.on_detection(pipeline.handle_detection)

    await pipeline.start()

    # ── VLM scheduler ────────────────────────────────────────────────
    vlm_scheduler = VLMScheduler()

    # ── Background Tasks ──────────────────────────────────────────────
    asyncio.create_task(mqtt_client.run(), name="mqtt-client-loop")
    asyncio.create_task(vlm_scheduler.run(), name="vlm-scheduler")

    app.state.mqtt_client = mqtt_client
    app.state.pipeline = pipeline
    app.state.kalman_tracker = kalman_tracker
    app.state.vlm_scheduler = vlm_scheduler

    yield

    # ── Graceful shutdown ─────────────────────────────────────────────
    await pipeline.stop()
    await mqtt_client.stop()


app = FastAPI(title="IOT EL Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.dashboard import router as dashboard_router  # noqa: E402

app.include_router(dashboard_router)

# Serve the static dashboard
import os
os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("app/static/index.html")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

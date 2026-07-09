"""
dashboard.py — API routes for the frontend visualizer.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.db.queries import get_all_cameras, get_live_rays, get_live_tracks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["dashboard"])


def _serialize(data: Any) -> Any:
    """Convert datetime/timedelta/UUID objects to JSON-safe types."""
    return jsonable_encoder(data)


# ─────────────────────────────────────────────────────────────────────────────
# Basic dashboard endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/cameras")
async def api_get_cameras() -> JSONResponse:
    """Return a list of all registered cameras."""
    data = await get_all_cameras()
    return JSONResponse(content=_serialize(data))


@router.get("/live_rays")
async def api_get_live_rays(within_seconds: int = 2) -> JSONResponse:
    """Return rays emitted within *within_seconds* for map overlay."""
    data = await get_live_rays(within_seconds=within_seconds)
    return JSONResponse(content=_serialize(data))


@router.get("/live_tracks")
async def api_get_live_tracks(within_seconds: int = 5) -> JSONResponse:
    """Return the most recent smoothed position of each active object."""
    data = await get_live_tracks(within_seconds=within_seconds)
    return JSONResponse(content=_serialize(data))


@router.get("/stats")
async def api_get_stats(within_seconds: int = 5) -> JSONResponse:
    """Return summary statistics for the dashboard header."""
    cameras = await get_all_cameras()
    tracks = await get_live_tracks(within_seconds=within_seconds)

    now = datetime.datetime.now(tz=datetime.timezone.utc)
    last_camera_seen: str | None = None
    for cam in cameras:
        seen = cam.get("last_seen")
        if seen is not None:
            if last_camera_seen is None or seen > last_camera_seen:
                last_camera_seen = seen

    stats: Dict[str, Any] = {
        "active_tracks": len(tracks),
        "camera_count": len(cameras),
        "server_time": now.isoformat(),
        "last_camera_seen": last_camera_seen,
    }
    return JSONResponse(content=_serialize(stats))


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot (S3 pre-signed URL)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/cameras/{device_id}/snapshot")
async def api_get_snapshot(device_id: str) -> JSONResponse:
    """Return a pre-signed S3 URL for the latest snapshot from *device_id*."""
    from app.config import settings
    from app.db.connection import get_pool

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT s3_snapshot_key, s3_snapshot_time FROM cameras WHERE device_id = $1",
            device_id,
        )

    if row is None or not row["s3_snapshot_key"]:
        raise HTTPException(status_code=404, detail="No snapshot available for this camera")

    if not settings.AWS_S3_BUCKET:
        raise HTTPException(status_code=503, detail="AWS_S3_BUCKET not configured")

    try:
        import boto3  # type: ignore[import]
        from botocore.config import Config  # type: ignore[import]

        kwargs: dict = {"config": Config(signature_version="s3v4")}
        if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
            kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
            kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
        if settings.AWS_REGION:
            kwargs["region_name"] = settings.AWS_REGION

        s3 = boto3.client("s3", **kwargs)
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.AWS_S3_BUCKET, "Key": row["s3_snapshot_key"]},
            ExpiresIn=300,
        )
    except Exception as exc:
        logger.error("Snapshot presign failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to generate presigned URL") from exc

    return JSONResponse(content={
        "url": url,
        "s3_key": row["s3_snapshot_key"],
        "time": _serialize(row["s3_snapshot_time"]),
    })


# ─────────────────────────────────────────────────────────────────────────────
# VLM
# ─────────────────────────────────────────────────────────────────────────────

class ToggleBody(BaseModel):
    enabled: bool


@router.get("/vlm/status")
async def api_vlm_status(request: Request) -> JSONResponse:
    scheduler = getattr(request.app.state, "vlm_scheduler", None)
    if scheduler is None:
        return JSONResponse(content={"enabled": False, "error": "scheduler not initialised"})
    return JSONResponse(content=_serialize(scheduler.status()))


@router.post("/vlm/toggle")
async def api_vlm_toggle(body: ToggleBody, request: Request) -> JSONResponse:
    scheduler = getattr(request.app.state, "vlm_scheduler", None)
    if scheduler is None:
        raise HTTPException(status_code=503, detail="VLM scheduler not initialised")
    scheduler.toggle(body.enabled)
    return JSONResponse(content={"enabled": body.enabled})


@router.post("/vlm/heartbeat")
async def api_vlm_heartbeat(request: Request) -> JSONResponse:
    """Extend the VLM deadman timer by 60 s."""
    scheduler = getattr(request.app.state, "vlm_scheduler", None)
    if scheduler is None:
        raise HTTPException(status_code=503, detail="VLM scheduler not initialised")
    scheduler.heartbeat()
    return JSONResponse(content={"enabled": scheduler.enabled})


@router.get("/vlm/latest")
async def api_vlm_latest() -> JSONResponse:
    """Return the most recent VLM analysis result."""
    from app.db.connection import get_pool

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM vlm_analyses ORDER BY created_at DESC LIMIT 1"
        )
    if row is None:
        return JSONResponse(content=None)
    return JSONResponse(content=_serialize(dict(row)))


@router.get("/messages")
async def api_messages(limit: int = 30) -> JSONResponse:
    """Return the most recent MQTT messages received by the server."""
    from app.ingestion.message_log import get_recent
    return JSONResponse(content=get_recent(min(limit, 60)))



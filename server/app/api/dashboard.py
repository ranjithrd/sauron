"""
dashboard.py — API routes for the frontend visualizer.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from app.db.queries import get_all_cameras, get_live_tracks

router = APIRouter(prefix="/api", tags=["dashboard"])


def _serialize(data: Any) -> Any:
    """Convert datetime/timedelta/UUID objects to JSON-safe types."""
    return jsonable_encoder(data)


@router.get("/cameras")
async def api_get_cameras() -> JSONResponse:
    """Return a list of all registered cameras."""
    data = await get_all_cameras()
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
    # Find the most recent camera heartbeat
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

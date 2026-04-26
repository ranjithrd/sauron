"""
writer.py — DB write operations (tracks + camera registry).
"""

from __future__ import annotations

import datetime
import logging

from app.db.connection import get_pool
from app.kalman.tracker import SmoothedPosition

logger = logging.getLogger(__name__)


async def write_track(position: SmoothedPosition) -> None:
    """Insert one smoothed position into object_tracks."""
    pool = await get_pool()
    time_dt = datetime.datetime.fromtimestamp(position.timestamp, tz=datetime.timezone.utc)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO object_tracks
                (time, object_id, lat, lon, vel_lat, vel_lon, source_cameras)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            time_dt,
            position.object_id,
            position.lat,
            position.lon,
            position.vel_lat,
            position.vel_lon,
            position.source_cameras,
        )
    logger.debug("DB: wrote track for %s", position.object_id)


async def upsert_camera(
    device_id: str,
    rtsp_url: str,
    lat: float,
    lon: float,
    bearing_deg: float,
    fov_deg: float,
) -> None:
    """Insert or update a camera registration row."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO cameras
                (device_id, rtsp_url, lat, lon, bearing_deg, fov_deg, last_seen)
            VALUES ($1, $2, $3, $4, $5, $6, NOW())
            ON CONFLICT (device_id) DO UPDATE SET
                rtsp_url    = EXCLUDED.rtsp_url,
                lat         = EXCLUDED.lat,
                lon         = EXCLUDED.lon,
                bearing_deg = EXCLUDED.bearing_deg,
                fov_deg     = EXCLUDED.fov_deg,
                last_seen   = NOW()
            """,
            device_id, rtsp_url, lat, lon, bearing_deg, fov_deg,
        )
    logger.debug("DB: upserted camera %s", device_id)

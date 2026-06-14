"""
queries.py — read operations exposed to the API layer.
All user inputs are parameterised — no string interpolation into SQL.
"""

from __future__ import annotations

import datetime
import logging
from typing import Optional

from app.db.connection import get_pool

logger = logging.getLogger(__name__)


async def get_live_tracks(within_seconds: int = 5) -> list[dict]:
    """Return the most recent position of each object active within *within_seconds*."""
    pool = await get_pool()
    interval = datetime.timedelta(seconds=within_seconds)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (object_id)
                object_id, lat, lon, altitude_m, vel_lat, vel_lon, time, source_cameras
            FROM object_tracks
            WHERE time > NOW() - $1::interval
            ORDER BY object_id, time DESC
            """,
            interval,
        )
    return [dict(r) for r in rows]


async def get_object_history(object_id: str, minutes: int = 5) -> list[dict]:
    """Return full position history for one object over the last *minutes* minutes."""
    pool = await get_pool()
    interval = datetime.timedelta(minutes=minutes)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT object_id, lat, lon, altitude_m, vel_lat, vel_lon, time, source_cameras
            FROM object_tracks
            WHERE object_id = $1
              AND time > NOW() - $2::interval
            ORDER BY time ASC
            """,
            object_id,
            interval,
        )
    return [dict(r) for r in rows]


async def get_playback(
    start: datetime.datetime, end: datetime.datetime
) -> list[dict]:
    """Return all tracks between *start* and *end* (for frontend scrubbing)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT object_id, lat, lon, altitude_m, vel_lat, vel_lon, time, source_cameras
            FROM object_tracks
            WHERE time >= $1 AND time <= $2
            ORDER BY time ASC
            """,
            start,
            end,
        )
    return [dict(r) for r in rows]


async def get_live_rays(within_seconds: int = 2) -> list[dict]:
    """Return all rays emitted within *within_seconds* (for map overlay)."""
    pool = await get_pool()
    interval = datetime.timedelta(seconds=within_seconds)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT device_id, camera_lat, camera_lon,
                   dx, dy, dz, xnorm, object_id,
                   tri_lat, tri_lon, tri_alt_m, time
            FROM detection_rays
            WHERE time > NOW() - $1::interval
            ORDER BY time DESC
            LIMIT 200
            """,
            interval,
        )
    return [dict(r) for r in rows]


async def get_all_cameras() -> list[dict]:
    """Return all registered cameras."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM cameras ORDER BY device_id")
    return [dict(r) for r in rows]


async def get_camera(device_id: str) -> Optional[dict]:
    """Return a single camera row, or None if not found."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM cameras WHERE device_id = $1", device_id
        )
    return dict(row) if row else None

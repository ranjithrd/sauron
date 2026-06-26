"""
writer.py — DB write operations (tracks + camera registry).
"""

from __future__ import annotations

import datetime
import logging
from typing import List, Optional, TYPE_CHECKING

from app.db.connection import get_pool
from app.kalman.tracker import SmoothedPosition

if TYPE_CHECKING:
    from app.triangulation.correlator import RayRecord

logger = logging.getLogger(__name__)


async def write_track(position: SmoothedPosition) -> None:
    """Insert one smoothed position into object_tracks."""
    pool = await get_pool()
    time_dt = datetime.datetime.fromtimestamp(position.timestamp, tz=datetime.timezone.utc)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO object_tracks
                (time, object_id, lat, lon, altitude_m, vel_lat, vel_lon, source_cameras)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            time_dt,
            position.object_id,
            position.lat,
            position.lon,
            position.altitude_m,
            position.vel_lat,
            position.vel_lon,
            position.source_cameras,
        )
    logger.debug("DB: wrote track for %s", position.object_id)


async def write_rays(rays: "List[RayRecord]") -> None:
    """Insert ray records produced by a single triangulation event."""
    if not rays:
        return
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO detection_rays
                (time, device_id, camera_lat, camera_lon,
                 dx, dy, dz, xnorm, object_id, tri_lat, tri_lon, tri_alt_m)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
            [
                (
                    datetime.datetime.fromtimestamp(r.timestamp, tz=datetime.timezone.utc),
                    r.device_id,
                    r.camera_lat, r.camera_lon,
                    r.dx, r.dy, r.dz,
                    r.xnorm,
                    r.object_id,
                    r.tri_lat, r.tri_lon, r.tri_alt_m,
                )
                for r in rays
            ],
        )


async def upsert_camera(
    device_id: str,
    lat: float,
    lon: float,
    heading: float,
    pitch: float,
    roll: float,
) -> None:
    """Insert or update latest camera telemetry metadata."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO cameras
                (device_id, lat, lon, last_heading, last_pitch, last_roll, last_seen)
            VALUES ($1, $2, $3, $4, $5, $6, NOW())
            ON CONFLICT (device_id) DO UPDATE SET
                lat         = EXCLUDED.lat,
                lon         = EXCLUDED.lon,
                last_heading = EXCLUDED.last_heading,
                last_pitch   = EXCLUDED.last_pitch,
                last_roll    = EXCLUDED.last_roll,
                last_seen   = NOW()
            """,
            device_id,
            lat,
            lon,
            heading,
            pitch,
            roll,
        )
    logger.debug("DB: upserted camera %s", device_id)


async def upsert_snapshot(device_id: str, s3_key: str, ts: float) -> None:
    """Store the latest S3 snapshot key for a camera."""
    pool = await get_pool()
    time_dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE cameras
            SET s3_snapshot_key = $2, s3_snapshot_time = $3
            WHERE device_id = $1
            """,
            device_id,
            s3_key,
            time_dt,
        )
    logger.debug("DB: upserted snapshot for %s → %s", device_id, s3_key)


async def write_vlm_result(
    device_id: Optional[str],
    s3_key: Optional[str],
    model: str,
    result: Optional[str],
    error: Optional[str],
    duration_ms: int,
) -> None:
    """Persist a VLM analysis result."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO vlm_analyses (device_id, s3_key, model, result, error, duration_ms)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            device_id,
            s3_key,
            model,
            result,
            error,
            duration_ms,
        )
    logger.debug("DB: wrote VLM result for device=%s key=%s", device_id, s3_key)

"""
connection.py — asyncpg connection pool and table/hypertable initialisation.
Idempotent: safe to call init_db() multiple times.
"""

from __future__ import annotations

import logging

import asyncpg

from app.config import settings

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None

_CREATE_OBJECT_TRACKS = """
CREATE TABLE IF NOT EXISTS object_tracks (
    time            TIMESTAMPTZ      NOT NULL,
    object_id       TEXT             NOT NULL,
    lat             DOUBLE PRECISION NOT NULL,
    lon             DOUBLE PRECISION NOT NULL,
    vel_lat         DOUBLE PRECISION,
    vel_lon         DOUBLE PRECISION,
    source_cameras  TEXT[]
);
"""

_CREATE_OBJECT_TRACKS_IDX = """
CREATE INDEX IF NOT EXISTS idx_object_tracks_object_id
ON object_tracks (object_id, time DESC);
"""

_CREATE_CAMERAS = """
CREATE TABLE IF NOT EXISTS cameras (
    device_id   TEXT PRIMARY KEY,
    rtsp_url    TEXT,
    lat         DOUBLE PRECISION,
    lon         DOUBLE PRECISION,
    bearing_deg DOUBLE PRECISION,
    fov_deg     DOUBLE PRECISION,
    last_seen   TIMESTAMPTZ DEFAULT NOW()
);
"""


async def init_db() -> None:
    """Create pool, tables, and TimescaleDB hypertable. Safe to call multiple times."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(settings.DATABASE_URL, min_size=2, max_size=10)
        logger.info("DB: connection pool created")

    async with _pool.acquire() as conn:
        await conn.execute(_CREATE_OBJECT_TRACKS)
        await conn.execute(_CREATE_OBJECT_TRACKS_IDX)
        await conn.execute(_CREATE_CAMERAS)
        logger.info("DB: tables ensured")

        # Create hypertable — ignore if already a hypertable
        try:
            await conn.execute(
                "SELECT create_hypertable('object_tracks', 'time', if_not_exists => TRUE)"
            )
            logger.info("DB: hypertable ensured for object_tracks")
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("DB: hypertable creation skipped: %s", exc)


async def get_pool() -> asyncpg.Pool:
    """Return the active connection pool. Raises if init_db() has not been called."""
    if _pool is None:
        raise RuntimeError("Database not initialised — call init_db() first.")
    return _pool

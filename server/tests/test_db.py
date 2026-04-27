"""
tests/test_db.py — integration tests for the database module.

Requirements:
    docker-compose up -d    # starts TimescaleDB on localhost:5432
    pytest tests/test_db.py -v

Tables are truncated between tests so they can run in any order.
"""

from __future__ import annotations

import datetime
import time

import pytest
import pytest_asyncio

from app.db.connection import get_pool, init_db
from app.db.queries import (
    get_all_cameras,
    get_camera,
    get_live_tracks,
    get_object_history,
    get_playback,
)
from app.db.writer import upsert_camera, write_track
from app.kalman.tracker import SmoothedPosition

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(autouse=True)
async def db():
    """Initialise DB once per test; truncate tables before each test."""
    from app.db import connection
    connection._pool = None
    
    await init_db()
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE object_tracks, cameras")
    yield pool
    
    await pool.close()
    connection._pool = None


def _position(
    object_id: str = "obj_001",
    lat: float = 1.0,
    lon: float = 2.0,
    ts: float | None = None,
) -> SmoothedPosition:
    return SmoothedPosition(
        object_id=object_id,
        lat=lat,
        lon=lon,
        vel_lat=0.001,
        vel_lon=0.002,
        timestamp=ts or time.time(),
        source_cameras=["cam_01", "cam_02"],
    )


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestInitDB:
    async def test_init_db_idempotent(self):
        """Calling init_db() twice must not raise."""
        await init_db()
        await init_db()

    async def test_tables_exist_after_init(self, db):
        """Both tables must exist and be queryable after init_db()."""
        rows = await db.fetch(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name IN ('object_tracks', 'cameras')"
        )
        names = {r["table_name"] for r in rows}
        assert "object_tracks" in names
        assert "cameras" in names


# ---------------------------------------------------------------------------
# write_track
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestWriteTrack:
    async def test_write_track_inserts_row(self, db):
        """write_track() must produce one row in object_tracks."""
        await write_track(_position(object_id="obj_write", lat=10.0, lon=20.0))
        row = await db.fetchrow(
            "SELECT * FROM object_tracks WHERE object_id = 'obj_write'"
        )
        assert row is not None
        assert float(row["lat"]) == pytest.approx(10.0)
        assert float(row["lon"]) == pytest.approx(20.0)
        assert row["source_cameras"] == ["cam_01", "cam_02"]


# ---------------------------------------------------------------------------
# upsert_camera
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestUpsertCamera:
    async def test_upsert_camera_new_device(self, db):
        """upsert_camera() for a new device must create a row."""
        await upsert_camera("cam_01", 1.0, 2.0, 90.0, -5.0, 1.2)
        row = await db.fetchrow("SELECT * FROM cameras WHERE device_id = 'cam_01'")
        assert row is not None
        assert float(row["last_heading"]) == pytest.approx(90.0)
        assert float(row["last_pitch"]) == pytest.approx(-5.0)
        assert float(row["last_roll"]) == pytest.approx(1.2)

    async def test_upsert_camera_updates_existing(self, db):
        """upsert_camera() for the same device must update, not duplicate."""
        await upsert_camera("cam_02", 1.0, 2.0, 90.0, -2.0, 0.5)
        await upsert_camera("cam_02", 3.0, 4.0, 180.0, -1.0, -0.3)

        count = await db.fetchval("SELECT COUNT(*) FROM cameras WHERE device_id = 'cam_02'")
        assert count == 1

        row = await db.fetchrow("SELECT * FROM cameras WHERE device_id = 'cam_02'")
        assert float(row["lat"]) == pytest.approx(3.0)
        assert float(row["last_heading"]) == pytest.approx(180.0)
        assert float(row["last_pitch"]) == pytest.approx(-1.0)
        assert float(row["last_roll"]) == pytest.approx(-0.3)


# ---------------------------------------------------------------------------
# get_live_tracks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGetLiveTracks:
    async def test_returns_latest_per_object(self, db):
        """get_live_tracks() must return one row per object (most recent)."""
        now = time.time()
        await write_track(_position("obj_A", lat=1.0, ts=now - 2))
        await write_track(_position("obj_A", lat=1.5, ts=now - 1))  # newest
        await write_track(_position("obj_B", lat=5.0, ts=now - 1))

        rows = await get_live_tracks(within_seconds=10)
        by_id = {r["object_id"]: r for r in rows}

        assert "obj_A" in by_id
        assert "obj_B" in by_id
        assert float(by_id["obj_A"]["lat"]) == pytest.approx(1.5)

    async def test_stale_data_returns_empty(self, db):
        """get_live_tracks() must exclude rows older than within_seconds."""
        old_ts = time.time() - 3600  # 1 hour ago
        old_dt = datetime.datetime.fromtimestamp(old_ts, tz=datetime.timezone.utc)
        async with (await get_pool()).acquire() as conn:
            await conn.execute(
                "INSERT INTO object_tracks (time, object_id, lat, lon, vel_lat, vel_lon, source_cameras) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7)",
                old_dt, "stale_obj", 1.0, 2.0, 0.0, 0.0, ["cam_01"],
            )

        rows = await get_live_tracks(within_seconds=5)
        ids = {r["object_id"] for r in rows}
        assert "stale_obj" not in ids


# ---------------------------------------------------------------------------
# get_object_history
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGetObjectHistory:
    async def test_returns_rows_in_asc_order(self, db):
        """get_object_history() must return rows in ascending time order."""
        now = time.time()
        for i in range(3):
            await write_track(_position("hist_obj", ts=now - (3 - i)))

        rows = await get_object_history("hist_obj", minutes=1)
        assert len(rows) == 3
        times = [r["time"] for r in rows]
        assert times == sorted(times)

    async def test_filters_by_object_id(self, db):
        """get_object_history() must not return rows from other objects."""
        await write_track(_position("wanted"))
        await write_track(_position("unwanted"))

        rows = await get_object_history("wanted", minutes=1)
        assert all(r["object_id"] == "wanted" for r in rows)


# ---------------------------------------------------------------------------
# get_playback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGetPlayback:
    async def test_returns_only_rows_within_range(self, db):
        """get_playback() must return only rows whose time is within [start, end]."""
        now = datetime.datetime.now(datetime.timezone.utc)
        inside_ts = now - datetime.timedelta(seconds=30)
        outside_ts = now - datetime.timedelta(hours=2)

        async with (await get_pool()).acquire() as conn:
            await conn.execute(
                "INSERT INTO object_tracks (time, object_id, lat, lon, vel_lat, vel_lon, source_cameras) "
                "VALUES ($1, 'inside', 1.0, 2.0, 0.0, 0.0, $2)",
                inside_ts, ["cam_01"],
            )
            await conn.execute(
                "INSERT INTO object_tracks (time, object_id, lat, lon, vel_lat, vel_lon, source_cameras) "
                "VALUES ($1, 'outside', 1.0, 2.0, 0.0, 0.0, $2)",
                outside_ts, ["cam_01"],
            )

        start = now - datetime.timedelta(minutes=5)
        end = now
        rows = await get_playback(start, end)
        ids = {r["object_id"] for r in rows}
        assert "inside" in ids
        assert "outside" not in ids


# ---------------------------------------------------------------------------
# get_all_cameras / get_camera
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCameraQueries:
    async def test_get_all_cameras(self, db):
        """get_all_cameras() must return all registered cameras."""
        await upsert_camera("cam_x", 0.0, 0.0, 0.0, -3.0, 0.1)
        await upsert_camera("cam_y", 1.0, 1.0, 90.0, -4.0, -0.2)

        cams = await get_all_cameras()
        ids = {c["device_id"] for c in cams}
        assert {"cam_x", "cam_y"}.issubset(ids)

    async def test_get_camera_known(self, db):
        """get_camera() must return the correct row for a known device_id."""
        await upsert_camera("cam_z", 5.0, 6.0, 45.0, -6.0, 0.0)
        cam = await get_camera("cam_z")
        assert cam is not None
        assert cam["device_id"] == "cam_z"
        assert float(cam["lat"]) == pytest.approx(5.0)

    async def test_get_camera_unknown_returns_none(self, db):
        """get_camera() must return None for an unregistered device_id."""
        result = await get_camera("nonexistent_device")
        assert result is None

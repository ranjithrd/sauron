"""
scheduler.py — background task that periodically runs VLM drone analysis.

When enabled via the frontend toggle, picks the most recent S3 snapshot,
fetches the latest track velocity for context, and calls analyze_snapshot().
Results are stored in vlm_analyses and served via /api/vlm/status.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class VLMScheduler:
    def __init__(self) -> None:
        self.enabled: bool = False
        self._last_run_ts: Optional[float] = None
        self._last_result: Optional[str] = None
        self._last_device_id: Optional[str] = None
        self._last_s3_key: Optional[str] = None
        self._last_error: Optional[str] = None

    def toggle(self, enabled: bool) -> None:
        self.enabled = enabled
        logger.info("VLMScheduler: %s", "enabled" if enabled else "disabled")

    def status(self) -> Dict[str, Any]:
        from app.config import settings
        return {
            "enabled": self.enabled,
            "interval_s": settings.VLM_INTERVAL_S,
            "model": settings.VLM_MODEL,
            "last_run": self._last_run_ts,
            "last_result": self._last_result,
            "last_device_id": self._last_device_id,
            "last_s3_key": self._last_s3_key,
            "last_error": self._last_error,
        }

    async def run(self) -> None:
        """Infinite loop — waits VLM_INTERVAL_S between analyses."""
        from app.config import settings
        from app.db.connection import get_pool
        from app.db.writer import write_vlm_result
        from app.vlm.analyzer import analyze_snapshot

        while True:
            interval = max(5, settings.VLM_INTERVAL_S)
            await asyncio.sleep(interval)

            if not self.enabled:
                continue

            if not settings.VLM_API_KEY:
                logger.warning(
                    "VLMScheduler: VLM_API_KEY not set — set OPENROUTER_API_KEY "
                    "(or VLM_API_KEY) in .env and restart the server"
                )
                continue

            if not settings.AWS_S3_BUCKET:
                logger.warning("VLMScheduler: AWS_S3_BUCKET not set — skipping run")
                continue

            # Pick the most recent snapshot across all cameras
            try:
                pool = await get_pool()
                async with pool.acquire() as conn:
                    row = await conn.fetchrow(
                        """
                        SELECT device_id, s3_snapshot_key, s3_snapshot_time
                        FROM cameras
                        WHERE s3_snapshot_key IS NOT NULL
                        ORDER BY s3_snapshot_time DESC NULLS LAST
                        LIMIT 1
                        """
                    )
            except Exception as exc:
                logger.error("VLMScheduler: DB query failed: %s", exc)
                continue

            if row is None or not row["s3_snapshot_key"]:
                logger.warning(
                    "VLMScheduler: no snapshots in DB yet — edge must have "
                    "image_upload.enabled: true in config.yaml and active detections"
                )
                continue

            device_id = row["device_id"]
            s3_key = row["s3_snapshot_key"]

            # Fetch latest track velocity for this run (any recent object)
            vel_lat = 0.0
            vel_lon = 0.0
            altitude_m = None
            try:
                async with pool.acquire() as conn:
                    track = await conn.fetchrow(
                        """
                        SELECT vel_lat, vel_lon, altitude_m
                        FROM object_tracks
                        WHERE time > NOW() - interval '30 seconds'
                        ORDER BY time DESC
                        LIMIT 1
                        """
                    )
                if track:
                    vel_lat = track["vel_lat"] or 0.0
                    vel_lon = track["vel_lon"] or 0.0
                    altitude_m = track["altitude_m"]
            except Exception as exc:
                logger.debug("VLMScheduler: could not fetch track velocity: %s", exc)

            try:
                result_str, duration_ms = await analyze_snapshot(
                    s3_key=s3_key,
                    device_id=device_id,
                    bucket=settings.AWS_S3_BUCKET,
                    region=settings.AWS_REGION,
                    model=settings.VLM_MODEL,
                    api_key=settings.VLM_API_KEY,
                    vel_lat=vel_lat,
                    vel_lon=vel_lon,
                    altitude_m=altitude_m,
                )
                self._last_run_ts = time.time()
                self._last_result = result_str
                self._last_device_id = device_id
                self._last_s3_key = s3_key
                self._last_error = None

                await write_vlm_result(
                    device_id=device_id,
                    s3_key=s3_key,
                    model=settings.VLM_MODEL,
                    result=result_str,
                    error=None,
                    duration_ms=duration_ms,
                )

            except Exception as exc:
                err = str(exc)
                logger.error("VLMScheduler: analysis failed: %s", err)
                self._last_run_ts = time.time()
                self._last_error = err
                self._last_result = None
                try:
                    await write_vlm_result(
                        device_id=device_id,
                        s3_key=s3_key,
                        model=settings.VLM_MODEL,
                        result=None,
                        error=err,
                        duration_ms=0,
                    )
                except Exception:
                    pass

"""
scheduler.py — background task that periodically runs VLM analysis.

The scheduler is toggled on/off via the frontend.  When enabled, it picks
the most recently uploaded snapshot from any camera and sends it to the VLM.
Results are persisted to vlm_analyses and exposed via /api/vlm/status.
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
        """Infinite loop — checks enabled flag each interval."""
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
                logger.warning("VLMScheduler: VLM_API_KEY not set — skipping run")
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
                logger.debug("VLMScheduler: no snapshots available yet")
                continue

            device_id = row["device_id"]
            s3_key = row["s3_snapshot_key"]

            try:
                result, duration_ms = await analyze_snapshot(
                    s3_key=s3_key,
                    device_id=device_id,
                    bucket=settings.AWS_S3_BUCKET,
                    region=settings.AWS_REGION,
                    model=settings.VLM_MODEL,
                    api_key=settings.VLM_API_KEY,
                )
                self._last_run_ts = time.time()
                self._last_result = result
                self._last_device_id = device_id
                self._last_s3_key = s3_key
                self._last_error = None

                await write_vlm_result(
                    device_id=device_id,
                    s3_key=s3_key,
                    model=settings.VLM_MODEL,
                    result=result,
                    error=None,
                    duration_ms=duration_ms,
                )

            except Exception as exc:  # pylint: disable=broad-except
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

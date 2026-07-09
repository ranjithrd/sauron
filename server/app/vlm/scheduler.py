"""
scheduler.py — background task that periodically runs VLM drone analysis.

Deadman: auto-disables 60 s after last heartbeat (frontend must POST
/api/vlm/heartbeat every 30 s to keep it alive).

Loop strategy: simple 1-second sleep poll — easy to reason about, no
event-based races. Runs analysis every VLM_INTERVAL_S seconds while enabled.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DEADMAN_SECONDS = 60.0
_HISTORY_SIZE    = 4
_POLL_INTERVAL_S = 1.0   # how often the loop wakes to check state


class VLMScheduler:
    def __init__(self) -> None:
        self._enabled_until: float = 0.0
        self._history: deque[Dict[str, Any]] = deque(maxlen=_HISTORY_SIZE)
        self._last_run_ts: float = 0.0       # epoch of last completed run
        self._run_immediately: bool = False   # fire next iteration instead of waiting

    # ── public interface ──────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return time.time() < self._enabled_until

    def toggle(self, enabled: bool) -> None:
        if enabled:
            self._enabled_until = time.time() + _DEADMAN_SECONDS
            self._run_immediately = True   # fire right away, don't wait interval
            self._last_run_ts = 0.0        # reset so the due-check passes immediately
        else:
            self._enabled_until = 0.0
            self._run_immediately = False
        logger.info("VLMScheduler: %s", "enabled" if enabled else "disabled")

    def heartbeat(self) -> None:
        if self.enabled:
            self._enabled_until = time.time() + _DEADMAN_SECONDS

    def status(self) -> Dict[str, Any]:
        from app.config import settings
        history = list(self._history)
        last = history[0] if history else None
        return {
            "enabled": self.enabled,
            "enabled_until": self._enabled_until if self.enabled else None,
            "interval_s": settings.VLM_INTERVAL_S,
            "model": settings.VLM_MODEL,
            "last_run": last["run_ts"] if last else None,
            "last_result": last["result"] if last else None,
            "last_device_id": last["device_id"] if last else None,
            "last_s3_key": last["s3_key"] if last else None,
            "last_error": last["error"] if last else None,
            "history": history,
        }

    # ── background loop ───────────────────────────────────────────────────────

    async def run(self) -> None:
        from app.config import settings

        while True:
            await asyncio.sleep(_POLL_INTERVAL_S)

            if not self.enabled:
                continue

            interval = max(20, settings.VLM_INTERVAL_S)
            due = self._run_immediately or (time.time() - self._last_run_ts >= interval)
            if not due:
                continue

            self._run_immediately = False
            await self._run_once(settings)

    async def _run_once(self, settings: Any) -> None:
        from app.db.connection import get_pool
        from app.db.writer import write_vlm_result
        from app.vlm.analyzer import analyze_snapshot

        run_ts = time.time()
        self._last_run_ts = run_ts

        # ── pre-flight checks (surface to UI, not silent skips) ──────────────
        if not settings.VLM_API_KEY:
            self._record_skip(run_ts, "VLM_API_KEY not set — configure it in .env")
            logger.warning("VLMScheduler: VLM_API_KEY not set")
            return

        if not settings.AWS_S3_BUCKET:
            self._record_skip(run_ts, "AWS_S3_BUCKET not set — configure it in .env")
            logger.warning("VLMScheduler: AWS_S3_BUCKET not set")
            return

        # ── fetch latest snapshot ─────────────────────────────────────────────
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
            self._record_error(run_ts, None, f"DB query failed: {exc}")
            logger.error("VLMScheduler: DB query failed: %s", exc)
            return

        if not row or not row["s3_snapshot_key"]:
            self._record_skip(run_ts, "No snapshot in DB — edge must have image_upload enabled")
            logger.warning("VLMScheduler: no snapshots in DB yet")
            return

        device_id = row["device_id"]
        s3_key    = row["s3_snapshot_key"]

        # ── fetch latest track context ────────────────────────────────────────
        vel_lat, vel_lon, altitude_m = 0.0, 0.0, None
        try:
            async with pool.acquire() as conn:
                track = await conn.fetchrow(
                    """
                    SELECT vel_lat, vel_lon, altitude_m
                    FROM object_tracks
                    WHERE time > NOW() - interval '30 seconds'
                    ORDER BY time DESC LIMIT 1
                    """
                )
            if track:
                vel_lat    = track["vel_lat"] or 0.0
                vel_lon    = track["vel_lon"] or 0.0
                altitude_m = track["altitude_m"]
        except Exception as exc:
            logger.debug("VLMScheduler: could not fetch track velocity: %s", exc)

        # ── run analysis ──────────────────────────────────────────────────────
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
            entry: Dict[str, Any] = {
                "run_ts":      run_ts,
                "result":      result_str,
                "device_id":   device_id,
                "s3_key":      s3_key,
                "model":       settings.VLM_MODEL,
                "duration_ms": duration_ms,
                "error":       None,
            }
            self._history.appendleft(entry)

            try:
                await write_vlm_result(
                    device_id=device_id, s3_key=s3_key,
                    model=settings.VLM_MODEL, result=result_str,
                    error=None, duration_ms=duration_ms,
                )
            except Exception as exc:
                logger.warning("VLMScheduler: DB write failed: %s", exc)

        except Exception as exc:
            err = str(exc)
            logger.error("VLMScheduler: analysis failed: %s", err)
            self._record_error(run_ts, device_id, err, s3_key=s3_key, model=settings.VLM_MODEL)
            try:
                await write_vlm_result(
                    device_id=device_id, s3_key=s3_key,
                    model=settings.VLM_MODEL, result=None,
                    error=err, duration_ms=0,
                )
            except Exception:
                pass

    def _record_skip(self, run_ts: float, reason: str) -> None:
        self._history.appendleft({
            "run_ts":      run_ts,
            "result":      None,
            "device_id":   None,
            "s3_key":      None,
            "model":       None,
            "duration_ms": 0,
            "error":       reason,
        })

    def _record_error(
        self,
        run_ts: float,
        device_id: Optional[str],
        error: str,
        s3_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self._history.appendleft({
            "run_ts":      run_ts,
            "result":      None,
            "device_id":   device_id,
            "s3_key":      s3_key,
            "model":       model,
            "duration_ms": 0,
            "error":       error,
        })

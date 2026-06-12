"""
correlator.py — stateful, async detection correlator.

Buffers incoming Detection objects, searches for cross-camera pairs within
the configured time window, triangulates, applies quality filters, and emits
CorrelatedDetection events via a callback.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable, Dict, List, Optional, Tuple

from app.config import settings
from app.triangulation.triangulator import (
    build_ray,
    flat_earth_distance_m,
    intersect_rays,
)

if TYPE_CHECKING:
    from app.ingestion.mqtt_client import Detection

logger = logging.getLogger(__name__)

_MIN_CONFIDENCE: float = 0.3


# ---------------------------------------------------------------------------
# Output dataclass consumed by the Kalman tracker
# ---------------------------------------------------------------------------

@dataclass
class CorrelatedDetection:
    object_id: str
    lat: float
    lon: float
    confidence: float           # [0, 1] quality of triangulation
    timestamp: float            # UTC unix
    source_cameras: List[str]   # device_ids of cameras used


# ---------------------------------------------------------------------------
# Correlator
# ---------------------------------------------------------------------------

class Correlator:
    """
    Buffers detections and emits triangulated positions when cross-camera
    detections are found within the configured time window.

    Parameters
    ----------
    on_correlated : async callable
        Called with each successfully triangulated :class:`CorrelatedDetection`.
    max_position_jump_m : float
        Maximum allowed distance (metres) between consecutive triangulated
        positions for the same ``object_id``.  Larger jumps are silently
        rejected as outliers.
    """

    def __init__(
        self,
        on_correlated: Callable[[CorrelatedDetection], Awaitable[None]],
        max_position_jump_m: float = 50.0,
    ) -> None:
        self._on_correlated = on_correlated
        self.max_position_jump_m = max_position_jump_m

        # Rolling buffer of recent detections
        self._buffer: List[Detection] = []

        # Last emitted lat/lon per camera-scoped object_id
        self._last_positions: Dict[str, Tuple[float, float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add(self, detection: Detection) -> None:
        """
        Ingest a new detection, search for cross-camera matches, and emit
        CorrelatedDetection events for each valid pair found.

        The new detection is added to the buffer *after* searching so that
        it cannot be matched against itself.
        """
        window_s = settings.DETECTION_CORRELATION_WINDOW_MS / 1_000.0
        prune_cutoff = detection.timestamp - (5 * window_s)

        # ── 1. Search buffer for cross-camera candidates ──────────────
        candidates_checked = 0
        pairs_attempted = 0
        for candidate in self._buffer:
            # Must be from a different camera
            if candidate.device_id == detection.device_id:
                continue

            candidates_checked += 1
            dt = abs(detection.timestamp - candidate.timestamp)

            # Must be within the time correlation window
            if dt > window_s:
                logger.debug(
                    "Correlator: skipping %s↔%s — dt=%.3fs > window=%.3fs",
                    detection.device_id, candidate.device_id, dt, window_s,
                )
                continue

            pairs_attempted += 1
            await self._try_correlate(detection, candidate)

        if candidates_checked == 0 and len(self._buffer) > 0:
            logger.debug(
                "Correlator: %s — buffer has %d entry(s) but all from same device",
                detection.device_id, len(self._buffer),
            )
        elif candidates_checked > 0 and pairs_attempted == 0:
            logger.debug(
                "Correlator: %s — %d cross-camera candidate(s) found but all outside %.0fms window",
                detection.device_id, candidates_checked, settings.DETECTION_CORRELATION_WINDOW_MS,
            )

        # ── 2. Append current detection ───────────────────────────────
        self._buffer.append(detection)

        # ── 3. Prune old detections ───────────────────────────────────
        self._buffer = [d for d in self._buffer if d.timestamp >= prune_cutoff]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    async def _try_correlate(
        self,
        d1: Detection,
        d2: Detection,
    ) -> None:
        """
        Attempt to triangulate a pair of cross-camera detections.
        Emits a callback on success; silently returns on rejection.
        """
        ray1 = build_ray(
            camera_lat=d1.camera_lat,
            camera_lon=d1.camera_lon,
            camera_heading=d1.camera_heading,
            camera_pitch=d1.camera_pitch,
            camera_roll=d1.camera_roll,
            camera_fov=d1.camera_fov,
            xnorm=d1.xnorm,
        )
        ray2 = build_ray(
            camera_lat=d2.camera_lat,
            camera_lon=d2.camera_lon,
            camera_heading=d2.camera_heading,
            camera_pitch=d2.camera_pitch,
            camera_roll=d2.camera_roll,
            camera_fov=d2.camera_fov,
            xnorm=d2.xnorm,
        )

        if ray1 is None:
            logger.debug(
                "Correlator: ray rejected for %s — heading=%.1f° pitch=%.1f° roll=%.1f° xnorm=%.3f (ray points upward or misses ground)",
                d1.device_id, d1.camera_heading, d1.camera_pitch, d1.camera_roll, d1.xnorm,
            )
        if ray2 is None:
            logger.debug(
                "Correlator: ray rejected for %s — heading=%.1f° pitch=%.1f° roll=%.1f° xnorm=%.3f (ray points upward or misses ground)",
                d2.device_id, d2.camera_heading, d2.camera_pitch, d2.camera_roll, d2.xnorm,
            )
        if ray1 is None or ray2 is None:
            return

        position = intersect_rays(ray1, ray2)

        # ── Reject: no intersection or behind cameras ─────────────────
        if position is None:
            return

        # ── Reject: low confidence (nearly parallel rays) ─────────────
        if position.confidence < _MIN_CONFIDENCE:
            logger.debug(
                "Correlator: rejected pair (%s, %s) — low confidence %.3f",
                d1.object_id,
                d2.object_id,
                position.confidence,
            )
            return

        # ── Reject: position jump too large ──────────────────────────
        if not self._position_jump_ok(d1.object_id, position.lat, position.lon):
            logger.debug(
                "Correlator: rejected pair (%s, %s) — position jump too large",
                d1.object_id,
                d2.object_id,
            )
            return

        # ── Accept: update last known position and emit ───────────────
        self._last_positions[d1.object_id] = (position.lat, position.lon)

        correlated = CorrelatedDetection(
            object_id=d1.object_id,
            lat=position.lat,
            lon=position.lon,
            confidence=position.confidence,
            timestamp=(d1.timestamp + d2.timestamp) / 2.0,
            source_cameras=[d1.device_id, d2.device_id],
        )

        try:
            await self._on_correlated(correlated)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "Correlator: on_correlated callback raised: %s", exc, exc_info=exc
            )

    def _position_jump_ok(
        self, object_id: str, new_lat: float, new_lon: float
    ) -> bool:
        """
        Return True if the new position is within ``max_position_jump_m`` of
        the last known position for this object_id, or if no prior position
        exists (first sighting).
        """
        if object_id not in self._last_positions:
            return True

        prev_lat, prev_lon = self._last_positions[object_id]
        dist = flat_earth_distance_m(prev_lat, prev_lon, new_lat, new_lon)
        return dist <= self.max_position_jump_m

"""
correlator.py — stateful, async detection correlator.

Buffers incoming Detection objects, searches for cross-camera pairs within
the configured time window, triangulates, applies quality filters, and emits
CorrelatedDetection events via a callback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Awaitable, Callable, Dict, List, Optional, Tuple

from app.config import settings
from app.triangulation.triangulator import (
    CameraRay,
    build_ray,
    flat_earth_distance_m,
    intersect_rays,
)

if TYPE_CHECKING:
    from app.ingestion.mqtt_client import Detection

logger = logging.getLogger(__name__)

_MIN_CONFIDENCE: float = 0.15


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RayRecord:
    """One camera ray from a successful triangulation pair."""
    timestamp: float
    device_id: str
    camera_lat: float
    camera_lon: float
    dx: float
    dy: float
    dz: float
    xnorm: float
    object_id: str       # pre-association hint; stable ID assigned later by tracker
    tri_lat: float
    tri_lon: float
    tri_alt_m: float


@dataclass
class CorrelatedDetection:
    object_id: str
    lat: float
    lon: float
    altitude_m: float            # estimated altitude of tracked object above ground
    confidence: float            # [0, 1] quality of triangulation
    timestamp: float             # UTC unix
    source_cameras: List[str]    # device_ids of cameras used
    rays: List[RayRecord] = field(default_factory=list)


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
    on_rays : async callable, optional
        Called with the pair of :class:`RayRecord` objects for each accepted
        triangulation.  Use this to persist rays to the database.
    max_position_jump_m : float
        Maximum allowed distance (metres) between consecutive triangulated
        positions for the same ``object_id``.  Larger jumps are silently
        rejected as outliers.
    """

    def __init__(
        self,
        on_correlated: Callable[[CorrelatedDetection], Awaitable[None]],
        on_rays: Optional[Callable[[List[RayRecord]], Awaitable[None]]] = None,
        max_position_jump_m: float = 500.0,
    ) -> None:
        self._on_correlated = on_correlated
        self._on_rays = on_rays
        self.max_position_jump_m = max_position_jump_m

        self._buffer: List[Detection] = []
        self._last_positions: Dict[str, Tuple[float, float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add(self, detection: Detection) -> None:
        """
        Ingest a new detection, search for cross-camera matches, and emit
        CorrelatedDetection events for each valid pair found.

        Only the most recent detection from each other device is considered —
        taking all buffered candidates would produce M×N correlation explosions
        when cameras publish multiple bounding boxes per frame.
        """
        window_s = settings.DETECTION_CORRELATION_WINDOW_MS / 1_000.0
        prune_cutoff = detection.timestamp - (5 * window_s)

        # Reduce buffer to the single most-recent entry per other device.
        latest_by_device: Dict[str, Detection] = {}
        for candidate in self._buffer:
            if candidate.device_id == detection.device_id:
                continue
            prev = latest_by_device.get(candidate.device_id)
            if prev is None or candidate.timestamp > prev.timestamp:
                latest_by_device[candidate.device_id] = candidate

        if not latest_by_device and len(self._buffer) > 0:
            logger.debug(
                "Correlator: %s — buffer has %d entry(s) but all from same device",
                detection.device_id, len(self._buffer),
            )

        for other_id, candidate in latest_by_device.items():
            dt = abs(detection.timestamp - candidate.timestamp)
            if dt > window_s:
                logger.debug(
                    "Correlator: skipping %s↔%s — dt=%.3fs > window=%.3fs",
                    detection.device_id, other_id, dt, window_s,
                )
                continue
            await self._try_correlate(detection, candidate)

        self._buffer.append(detection)
        self._buffer = [d for d in self._buffer if d.timestamp >= prune_cutoff]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    async def _try_correlate(self, d1: Detection, d2: Detection) -> None:
        ray1 = build_ray(
            camera_lat=d1.camera_lat, camera_lon=d1.camera_lon,
            camera_heading=d1.camera_heading, camera_pitch=d1.camera_pitch,
            camera_roll=d1.camera_roll, camera_fov=d1.camera_fov,
            xnorm=d1.xnorm,
        )
        ray2 = build_ray(
            camera_lat=d2.camera_lat, camera_lon=d2.camera_lon,
            camera_heading=d2.camera_heading, camera_pitch=d2.camera_pitch,
            camera_roll=d2.camera_roll, camera_fov=d2.camera_fov,
            xnorm=d2.xnorm,
        )

        if ray1 is None or ray2 is None:
            return

        position = intersect_rays(ray1, ray2)

        if position is None:
            return

        if position.confidence < _MIN_CONFIDENCE:
            logger.debug(
                "Correlator: rejected pair (%s, %s) — low confidence %.3f",
                d1.object_id, d2.object_id, position.confidence,
            )
            return

        pair_key = f"{d1.device_id}|{d2.device_id}"
        if not self._position_jump_ok(pair_key, position.lat, position.lon):
            logger.debug(
                "Correlator: rejected pair (%s, %s) — position jump too large",
                d1.device_id, d2.device_id,
            )
            return

        self._last_positions[pair_key] = (position.lat, position.lon)

        logger.debug(
            "Correlator: triangulated (%s, %s) → lat=%.6f lon=%.6f alt=%.1fm conf=%.3f",
            d1.device_id, d2.device_id,
            position.lat, position.lon, position.altitude_m, position.confidence,
        )

        ts = (d1.timestamp + d2.timestamp) / 2.0

        rays = [
            _make_ray_record(ts, d1, ray1, d1.object_id, position.lat, position.lon, position.altitude_m),
            _make_ray_record(ts, d2, ray2, d1.object_id, position.lat, position.lon, position.altitude_m),
        ]

        correlated = CorrelatedDetection(
            object_id=d1.object_id,
            lat=position.lat,
            lon=position.lon,
            altitude_m=position.altitude_m,
            confidence=position.confidence,
            timestamp=ts,
            source_cameras=[d1.device_id, d2.device_id],
            rays=rays,
        )

        if self._on_rays is not None:
            try:
                await self._on_rays(rays)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Correlator: on_rays callback raised: %s", exc, exc_info=exc)

        try:
            await self._on_correlated(correlated)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Correlator: on_correlated callback raised: %s", exc, exc_info=exc)

    def _position_jump_ok(self, object_id: str, new_lat: float, new_lon: float) -> bool:
        if object_id not in self._last_positions:
            return True
        prev_lat, prev_lon = self._last_positions[object_id]
        dist = flat_earth_distance_m(prev_lat, prev_lon, new_lat, new_lon)
        return dist <= self.max_position_jump_m


def _make_ray_record(
    timestamp: float,
    detection: Detection,
    ray: CameraRay,
    object_id: str,
    tri_lat: float,
    tri_lon: float,
    tri_alt_m: float,
) -> RayRecord:
    return RayRecord(
        timestamp=timestamp,
        device_id=detection.device_id,
        camera_lat=ray.lat,
        camera_lon=ray.lon,
        dx=ray.dx,
        dy=ray.dy,
        dz=ray.dz,
        xnorm=detection.xnorm,
        object_id=object_id,
        tri_lat=tri_lat,
        tri_lon=tri_lon,
        tri_alt_m=tri_alt_m,
    )

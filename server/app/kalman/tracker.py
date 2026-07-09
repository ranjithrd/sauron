"""
tracker.py — manages Kalman filter lifecycle for all active tracked objects.

One ObjectKalmanFilter is maintained per stable object ID.  Incoming
CorrelatedDetections are spatially associated to existing tracks before a
new filter is spawned, so the same physical object keeps the same ID across
frames rather than generating a new one every detection cycle.

Association is performed in 3-D (lat, lon, altitude) so that objects at
different altitudes are not incorrectly merged even if their ground
projections overlap.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from app.config import settings
from app.kalman.filter import ObjectKalmanFilter
from app.triangulation.correlator import CorrelatedDetection, RayRecord

logger = logging.getLogger(__name__)

_METERS_PER_DEG_LAT: float = 111_320.0


# ---------------------------------------------------------------------------
# Output dataclass — consumed by the DB writer
# ---------------------------------------------------------------------------

@dataclass
class SmoothedPosition:
    object_id: str
    lat: float
    lon: float
    altitude_m: float
    vel_lat: float
    vel_lon: float
    vel_alt: float          # vertical velocity in metres per second
    timestamp: float
    source_cameras: List[str]


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class KalmanTracker:
    """
    Manages one :class:`ObjectKalmanFilter` per active object.

    Incoming correlated detections are associated to the nearest active
    track before a new filter is created (3-D nearest-neighbour within
    ``settings.ASSOCIATION_MAX_DIST_M``).  This prevents a new object ID
    being minted for every frame of the same physical object.

    Parameters
    ----------
    on_update : async callable
        Receives a :class:`SmoothedPosition` after every successful update.
    """

    def __init__(
        self,
        on_update: Callable[[SmoothedPosition], Awaitable[None]],
    ) -> None:
        self._on_update = on_update
        # object_id → (filter, last_updated_wall_time)
        self._filters: Dict[str, Tuple[ObjectKalmanFilter, float]] = {}
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def update(self, correlated: CorrelatedDetection) -> None:
        """
        Ingest a :class:`CorrelatedDetection`.

        Associates the incoming position with the nearest active track, or
        spawns a new filter if nothing is close enough.  Emits a
        :class:`SmoothedPosition` via the callback in both cases.
        """
        wall_now = time.monotonic()

        matched = self._associate(correlated.lat, correlated.lon, correlated.altitude_m)
        if matched is not None:
            oid = matched
        else:
            self._next_id += 1
            oid = f"obj_{self._next_id:04d}"
            logger.debug(
                "KalmanTracker: spawned new track %s at (%.6f, %.6f, %.1fm)",
                oid, correlated.lat, correlated.lon, correlated.altitude_m,
            )

        if oid not in self._filters:
            kf = ObjectKalmanFilter(
                initial_lat=correlated.lat,
                initial_lon=correlated.lon,
                initial_alt=correlated.altitude_m,
            )
            self._filters[oid] = (kf, wall_now)
            state = kf.state
        else:
            kf, _ = self._filters[oid]
            self._filters[oid] = (kf, wall_now)
            kf.predict()
            state = kf.update(correlated.lat, correlated.lon, correlated.altitude_m)

        smoothed = SmoothedPosition(
            object_id=oid,
            lat=state.lat,
            lon=state.lon,
            altitude_m=state.altitude_m,
            vel_lat=state.vel_lat,
            vel_lon=state.vel_lon,
            vel_alt=state.vel_alt,
            timestamp=correlated.timestamp,
            source_cameras=correlated.source_cameras,
        )

        try:
            await self._on_update(smoothed)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("KalmanTracker: on_update callback raised: %s", exc, exc_info=exc)

    def prune_stale(self) -> None:
        """Remove filters not updated within ``settings.OBJECT_STALE_TIMEOUT_S``."""
        cutoff = time.monotonic() - settings.OBJECT_STALE_TIMEOUT_S
        stale = [oid for oid, (_, ts) in self._filters.items() if ts < cutoff]
        for oid in stale:
            del self._filters[oid]
            logger.debug("KalmanTracker: pruned stale object %s", oid)

    @property
    def active_ids(self) -> List[str]:
        return list(self._filters.keys())

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _associate(self, lat: float, lon: float, alt_m: float) -> Optional[str]:
        """
        Return the object_id of the nearest active track within the
        association threshold, or None if no match.

        Distance is computed in 3-D (horizontal + vertical) so that objects
        at different altitudes are correctly separated even when their ground
        footprints overlap.
        """
        best_id: Optional[str] = None
        best_dist = settings.ASSOCIATION_MAX_DIST_M

        for oid, (kf, _) in self._filters.items():
            s = kf.state
            dist = _dist_3d_m(lat, lon, alt_m, s.lat, s.lon, s.altitude_m)
            if dist < best_dist:
                best_dist = dist
                best_id = oid

        return best_id


def _dist_3d_m(
    lat1: float, lon1: float, alt1: float,
    lat2: float, lon2: float, alt2: float,
) -> float:
    """Approximate 3-D distance in metres between two (lat, lon, alt) points."""
    ref_lat_rad = math.radians((lat1 + lat2) / 2.0)
    meters_per_deg_lon = _METERS_PER_DEG_LAT * math.cos(ref_lat_rad)
    dlat = (lat2 - lat1) * _METERS_PER_DEG_LAT
    dlon = (lon2 - lon1) * meters_per_deg_lon
    dalt = alt2 - alt1
    return math.sqrt(dlat ** 2 + dlon ** 2 + dalt ** 2)

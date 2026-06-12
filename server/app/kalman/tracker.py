"""
tracker.py — manages Kalman filter lifecycle for all active tracked objects.

One ObjectKalmanFilter is maintained per active object_id.
Smoothed positions are emitted via an async callback after every update.
Stale objects (no recent measurement) are pruned by prune_stale().
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Tuple

from app.config import settings
from app.kalman.filter import ObjectKalmanFilter
from app.triangulation.correlator import CorrelatedDetection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass — consumed by the DB writer
# ---------------------------------------------------------------------------

@dataclass
class SmoothedPosition:
    object_id: str
    lat: float
    lon: float
    altitude_m: float          # estimated altitude above ground (metres)
    vel_lat: float             # velocity in lat direction (degrees/sec)
    vel_lon: float             # velocity in lon direction (degrees/sec)
    timestamp: float           # UTC unix
    source_cameras: List[str]


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class KalmanTracker:
    """
    Manages one :class:`ObjectKalmanFilter` per active object.

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def update(self, correlated: CorrelatedDetection) -> None:
        """
        Ingest a :class:`CorrelatedDetection`.

        * **New object** — spawn a filter initialised at the measured position.
        * **Existing object** — predict one step then correct with the measurement.

        Emits a :class:`SmoothedPosition` via the callback in both cases.
        """
        oid = correlated.object_id
        wall_now = time.monotonic()

        if oid not in self._filters:
            # First sighting — initialise filter at the measured position
            kf = ObjectKalmanFilter(
                initial_lat=correlated.lat,
                initial_lon=correlated.lon,
            )
            self._filters[oid] = (kf, wall_now)
            state = kf.state   # initial state: position known, velocity = 0

            logger.debug("KalmanTracker: spawned filter for new object %s", oid)
        else:
            kf, _ = self._filters[oid]
            self._filters[oid] = (kf, wall_now)   # refresh last-updated time

            kf.predict()
            state = kf.update(correlated.lat, correlated.lon)

        smoothed = SmoothedPosition(
            object_id=oid,
            lat=state.lat,
            lon=state.lon,
            altitude_m=correlated.altitude_m,
            vel_lat=state.vel_lat,
            vel_lon=state.vel_lon,
            timestamp=correlated.timestamp,
            source_cameras=correlated.source_cameras,
        )

        try:
            await self._on_update(smoothed)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "KalmanTracker: on_update callback raised: %s", exc, exc_info=exc
            )

    def prune_stale(self) -> None:
        """
        Remove any object whose filter has not been updated within
        ``settings.OBJECT_STALE_TIMEOUT_S`` seconds.
        """
        cutoff = time.monotonic() - settings.OBJECT_STALE_TIMEOUT_S
        stale = [oid for oid, (_, ts) in self._filters.items() if ts < cutoff]
        for oid in stale:
            del self._filters[oid]
            logger.debug("KalmanTracker: pruned stale object %s", oid)

    @property
    def active_ids(self) -> List[str]:
        """Return the IDs of all currently tracked objects."""
        return list(self._filters.keys())

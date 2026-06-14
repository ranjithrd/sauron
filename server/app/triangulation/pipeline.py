"""
pipeline.py — wires the triangulation correlator into the wider application.

The KalmanTracker is injected as a dependency and is not built here.
Expected interface:
    await kalman_tracker.update(correlated: CorrelatedDetection)
    kalman_tracker.prune_stale()
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from app.triangulation.correlator import CorrelatedDetection, Correlator

if TYPE_CHECKING:
    from app.ingestion.mqtt_client import Detection

logger = logging.getLogger(__name__)

_PRUNE_INTERVAL_S: float = 1.0


class TriangulationPipeline:
    """
    Connects telemetry detections → correlator → Kalman tracker.

    Parameters
    ----------
    kalman_tracker : Any
        An object implementing:
            ``await kalman_tracker.update(correlated: CorrelatedDetection)``
            ``kalman_tracker.prune_stale()``
    max_position_jump_m : float
        Forwarded to the :class:`Correlator`.
    """

    def __init__(
        self,
        kalman_tracker: Any,
        on_rays: Any = None,
        max_position_jump_m: float = 50.0,
    ) -> None:
        self._kalman = kalman_tracker
        self._correlator = Correlator(
            on_correlated=self._on_correlated,
            on_rays=on_rays,
            max_position_jump_m=max_position_jump_m,
        )
        self._prune_task: asyncio.Task | None = None  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background Kalman prune loop."""
        self._prune_task = asyncio.create_task(
            self._prune_loop(), name="kalman-prune-loop"
        )
        logger.info("TriangulationPipeline: started.")

    async def stop(self) -> None:
        """Cancel the background prune loop gracefully."""
        if self._prune_task and not self._prune_task.done():
            self._prune_task.cancel()
            try:
                await self._prune_task
            except asyncio.CancelledError:
                pass
        logger.info("TriangulationPipeline: stopped.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def handle_detection(self, detection: Detection) -> None:
        """
        Entry point — called by the MQTT client for every Detection event.
        Hands off to the Correlator which buffers and triangulates.
        """
        await self._correlator.add(detection)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    async def _on_correlated(self, correlated: CorrelatedDetection) -> None:
        """Callback: fired by the Correlator for each successful triangulation."""
        try:
            await self._kalman.update(correlated)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "TriangulationPipeline: kalman_tracker.update raised: %s",
                exc,
                exc_info=exc,
            )

    async def _prune_loop(self) -> None:
        """Background task: call kalman_tracker.prune_stale() every second."""
        while True:
            await asyncio.sleep(_PRUNE_INTERVAL_S)
            try:
                self._kalman.prune_stale()
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "TriangulationPipeline: kalman_tracker.prune_stale raised: %s",
                    exc,
                    exc_info=exc,
                )

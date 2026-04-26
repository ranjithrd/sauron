"""
centroid_tracker.py — stateful per-camera object tracker.

Uses greedy nearest-centroid matching in normalised [0, 1]² space to
maintain object identity across frames without any ML dependency.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import count
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class _TrackedObject:
    object_id: str
    centroid: Tuple[float, float]   # (x_norm, y_norm)
    disappeared_frames: int = 0


class CentroidTracker:
    """
    Tracks objects across frames for a single camera.

    Parameters
    ----------
    camera_id : str
        Used as the prefix for scoped IDs (e.g. ``"cam_01"``).
    max_disappeared : int
        Number of consecutive frames an object can be absent before being
        deregistered.
    max_distance : float
        Maximum Euclidean distance (in normalised [0,1] space) between an
        existing centroid and a new detection for them to be considered the
        same object.
    """

    def __init__(
        self,
        camera_id: str,
        max_disappeared: int = 10,
        max_distance: float = 0.15,
    ) -> None:
        self.camera_id = camera_id
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        self._objects: OrderedDict[str, _TrackedObject] = OrderedDict()
        self._id_counter = count(1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self, detections: List[Tuple[float, float]]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Update tracker state with the centroids detected in the current frame.

        Parameters
        ----------
        detections : list of (x_norm, y_norm)
            Normalised centroid coordinates produced by :class:`BlobDetector`.

        Returns
        -------
        dict
            Mapping of ``object_id → (x_norm, y_norm)`` for all currently
            tracked (and not yet stale) objects.
        """
        # ── No detections this frame ─────────────────────────────────
        if not detections:
            for obj in list(self._objects.values()):
                obj.disappeared_frames += 1
                if obj.disappeared_frames > self.max_disappeared:
                    self._deregister(obj.object_id)
            return self._active_centroids()

        # ── No existing objects → register everything as new ─────────
        if not self._objects:
            for det in detections:
                self._register(det)
            return self._active_centroids()

        # ── Match existing objects to new detections ──────────────────
        existing_ids = list(self._objects.keys())
        existing_centroids = np.array(
            [self._objects[oid].centroid for oid in existing_ids],
            dtype=np.float64,
        )
        new_centroids = np.array(detections, dtype=np.float64)

        # Pairwise Euclidean distance matrix: rows=existing, cols=new
        dist_matrix = np.linalg.norm(
            existing_centroids[:, np.newaxis] - new_centroids[np.newaxis, :],
            axis=2,
        )

        matched_existing: set[int] = set()
        matched_new: set[int] = set()

        # Greedy: sort all (existing, new) pairs by distance ascending
        num_existing, num_new = dist_matrix.shape
        pairs = sorted(
            (
                (dist_matrix[r, c], r, c)
                for r in range(num_existing)
                for c in range(num_new)
            ),
            key=lambda t: t[0],
        )

        for dist, row, col in pairs:
            if row in matched_existing or col in matched_new:
                continue
            if dist > self.max_distance:
                break  # remaining pairs are all farther

            oid = existing_ids[row]
            self._objects[oid].centroid = tuple(new_centroids[col])  # type: ignore[assignment]
            self._objects[oid].disappeared_frames = 0
            matched_existing.add(row)
            matched_new.add(col)

        # Unmatched existing → increment disappeared counter
        for row in range(num_existing):
            if row not in matched_existing:
                oid = existing_ids[row]
                self._objects[oid].disappeared_frames += 1
                if self._objects[oid].disappeared_frames > self.max_disappeared:
                    self._deregister(oid)

        # Unmatched new detections → new objects
        for col in range(num_new):
            if col not in matched_new:
                self._register(tuple(new_centroids[col]))  # type: ignore[arg-type]

        return self._active_centroids()

    def reset(self) -> None:
        """Clear all tracked objects (useful for testing or stream restarts)."""
        self._objects.clear()
        self._id_counter = count(1)

    @property
    def tracked_ids(self) -> List[str]:
        """Return the IDs of all currently tracked objects."""
        return list(self._objects.keys())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _register(self, centroid: Tuple[float, float]) -> str:
        n = next(self._id_counter)
        oid = f"{self.camera_id}_obj{n:04d}"
        self._objects[oid] = _TrackedObject(object_id=oid, centroid=centroid)
        logger.debug("CentroidTracker [%s]: registered %s at %s", self.camera_id, oid, centroid)
        return oid

    def _deregister(self, object_id: str) -> None:
        logger.debug("CentroidTracker [%s]: deregistered %s", self.camera_id, object_id)
        del self._objects[object_id]

    def _active_centroids(self) -> Dict[str, Tuple[float, float]]:
        return {oid: obj.centroid for oid, obj in self._objects.items()}

"""
Temporal stability filter for MOG2 detections.

Spurious detections (camera noise, lighting transients, MOG2 warm-up artefacts)
typically appear in only one or two frames before vanishing.  A genuine moving
object occupies the same region of the frame for many consecutive frames.

This filter divides the [0,1]×[0,1] normalised frame into a coarse grid.
Each cell tracks how many consecutive frames it has been occupied.  A detection
is only forwarded downstream once its grid cell has been continuously occupied
for at least *min_stable_frames* frames.

The returned index list is parallel to the caller's ncoords / bboxes lists, so
both can be filtered with a simple list comprehension.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from sauron_edge.schemas import NCoord

logger = logging.getLogger(__name__)

# Fixed grid resolution — 20×20 gives cells of ≈3% width/height.
# Coarse enough to absorb sub-pixel centroid jitter, fine enough to
# distinguish objects that are not immediately adjacent.
_GRID_W = 20
_GRID_H = 20


class DetectionStabilizer:
    """
    Grid-based temporal stability filter.

    Parameters
    ----------
    min_stable_frames : int
        Minimum number of consecutive frames a grid cell must be occupied
        before detections in that cell are forwarded.  1 disables filtering.
    """

    def __init__(self, min_stable_frames: int = 3) -> None:
        self._min_stable = max(1, min_stable_frames)
        # Per-cell consecutive-occupancy counter
        self._counts: np.ndarray = np.zeros((_GRID_H, _GRID_W), dtype=np.int32)

        logger.info(
            "DetectionStabilizer: grid=%dx%d min_stable_frames=%d",
            _GRID_W,
            _GRID_H,
            self._min_stable,
        )

    def apply_config(self, *, min_stable_frames: int) -> None:
        """Hot-reload min_stable_frames without discarding the existing counts."""
        new_val = max(1, min_stable_frames)
        if new_val != self._min_stable:
            logger.info(
                "DetectionStabilizer: min_stable_frames %d→%d | "
                "model will use updated threshold immediately",
                self._min_stable,
                new_val,
            )
            self._min_stable = new_val
        else:
            logger.debug(
                "DetectionStabilizer: apply_config — no change (min_stable_frames=%d)",
                self._min_stable,
            )

    # ------------------------------------------------------------------
    # Public API — call once per processed frame
    # ------------------------------------------------------------------

    def update(self, ncoords: List[NCoord]) -> List[int]:
        """
        Ingest this frame's detections, update stability counters, and return
        the *indices* (into `ncoords`) whose grid cells are considered stable.

        Cells not occupied this frame have their counter reset to zero.
        Cells occupied this frame have their counter incremented by one.
        Cells whose counter reaches ``min_stable_frames`` are declared stable.

        Multiple ncoords that fall into the same cell: only the first
        (lowest index) is returned to avoid duplicate reports.
        """
        # Map each ncoord to a (row, col) grid cell; keep the first index per cell.
        cell_to_first_idx: dict[tuple[int, int], int] = {}
        for i, nc in enumerate(ncoords):
            col = min(int(nc.xnorm * _GRID_W), _GRID_W - 1)
            row = min(int(nc.ynorm * _GRID_H), _GRID_H - 1)
            if (row, col) not in cell_to_first_idx:
                cell_to_first_idx[(row, col)] = i

        # Rebuild counter grid: occupied cells increment, all others reset to 0
        new_counts = np.zeros_like(self._counts)
        for row, col in cell_to_first_idx:
            new_counts[row, col] = self._counts[row, col] + 1
        self._counts = new_counts

        # Collect indices of stable cells
        stable_indices: List[int] = []
        for (row, col), idx in cell_to_first_idx.items():
            if self._counts[row, col] >= self._min_stable:
                stable_indices.append(idx)

        if ncoords:
            logger.debug(
                "DetectionStabilizer: %d raw → %d stable (min=%d)",
                len(ncoords),
                len(stable_indices),
                self._min_stable,
            )

        return sorted(stable_indices)

    def reset(self) -> None:
        """Clear all counters (e.g. after a scene cut or camera restart)."""
        self._counts[:] = 0

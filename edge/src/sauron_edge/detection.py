"""
MOG2 background subtraction pipeline.

Flow per frame:
  1. Apply MOG2 foreground/background model
  2. Morphological open  → remove salt-and-pepper noise
  3. Morphological close → fill small holes in blobs
  4. Find external contours
  5. Filter by minimum area
  6. Compute centroid of each surviving blob via image moments
  7. Normalise centroid to [0, 1] in both axes → NCoord

The MOG2 model is stateful — it builds a running background estimate across
frames, so the first several seconds of output may include many false positives
while the model converges.  Set learning_rate=-1 (auto) to let OpenCV pick a
sensible decay rate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from sauron_edge.schemas import NCoord

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Output of a single frame processed by the MOG2 detector."""

    ncoords: List[NCoord]
    """Normalised [0,1] centroids of all detected foreground blobs."""

    n_raw_contours: int = 0
    """Total contours found before area filtering (useful for tuning)."""

    n_rejected_area: int = 0
    """Contours rejected because their area was below min_contour_area."""

    foreground_pixel_count: int = 0
    """Number of non-zero foreground pixels in the mask (diagnostic)."""


class MOG2Detector:
    """
    Stateful MOG2 background subtractor that produces normalised detections.

    This is intentionally NOT thread-safe — call `process()` from a single
    thread.  The owning pipeline is responsible for serialising access.
    """

    def __init__(
        self,
        *,
        learning_rate: float = -1.0,
        var_threshold: float = 16.0,
        min_contour_area: float = 200.0,
        morph_kernel_size: int = 5,
    ) -> None:
        self._learning_rate = learning_rate
        self._min_contour_area = min_contour_area

        # Build the MOG2 subtractor
        self._subtractor: cv2.BackgroundSubtractorMOG2 = (
            cv2.createBackgroundSubtractorMOG2(
                varThreshold=var_threshold,
                detectShadows=False,  # shadows are noise for centroid detection
            )
        )

        # Morphological kernels — computed once
        self._open_kernel: np.ndarray = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        self._close_kernel: np.ndarray = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size + 2, morph_kernel_size + 2)
        )

        logger.info(
            "MOG2Detector: initialised — learning_rate=%.3f var_threshold=%.1f "
            "min_contour_area=%.1f morph_open=%dpx morph_close=%dpx",
            learning_rate,
            var_threshold,
            min_contour_area,
            morph_kernel_size,
            morph_kernel_size + 2,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> DetectionResult:
        """
        Process a single BGR frame.

        Args:
            frame: H×W×3 uint8 BGR frame from OpenCV.

        Returns:
            DetectionResult with normalised centroids and diagnostic counts.
        """
        if frame is None or frame.size == 0:
            logger.warning("MOG2Detector: received empty/None frame — skipping")
            return DetectionResult(ncoords=[])

        h, w = frame.shape[:2]
        logger.debug("MOG2Detector: processing frame %dx%d", w, h)

        # --- Step 1: apply MOG2 ---
        fg_mask: np.ndarray = self._subtractor.apply(
            frame, learningRate=self._learning_rate
        )

        fg_pixel_count = int(cv2.countNonZero(fg_mask))
        logger.debug(
            "MOG2Detector: foreground pixels before morphology = %d (%.1f%%)",
            fg_pixel_count, 100.0 * fg_pixel_count / (h * w),
        )

        # --- Step 2: morphological cleanup ---
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._open_kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self._close_kernel)

        fg_pixel_count_after = int(cv2.countNonZero(fg_mask))
        logger.debug(
            "MOG2Detector: foreground pixels after morphology = %d", fg_pixel_count_after
        )

        # --- Step 3: find contours ---
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        n_raw = len(contours)
        logger.debug("MOG2Detector: raw contour count = %d", n_raw)

        # --- Steps 4–6: filter, moments, normalise ---
        ncoords: List[NCoord] = []
        n_rejected_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self._min_contour_area:
                n_rejected_area += 1
                logger.debug(
                    "MOG2Detector: contour rejected — area=%.1f < min=%.1f",
                    area, self._min_contour_area,
                )
                continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                # Degenerate contour — skip
                logger.debug("MOG2Detector: contour skipped — zero moment (degenerate)")
                continue

            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            xnorm = cx / w
            ynorm = cy / h

            ncoords.append(NCoord(xnorm=round(xnorm, 4), ynorm=round(ynorm, 4)))
            logger.debug(
                "MOG2Detector: detection — centroid=(%.1f, %.1f) norm=(%.4f, %.4f) area=%.1f",
                cx, cy, xnorm, ynorm, area,
            )

        result = DetectionResult(
            ncoords=ncoords,
            n_raw_contours=n_raw,
            n_rejected_area=n_rejected_area,
            foreground_pixel_count=fg_pixel_count_after,
        )

        if ncoords:
            logger.info(
                "MOG2Detector: %d detection(s) — raw_contours=%d rejected_area=%d fg_px=%d",
                len(ncoords), n_raw, n_rejected_area, fg_pixel_count_after,
            )
        else:
            logger.debug(
                "MOG2Detector: no detections — raw_contours=%d rejected_area=%d fg_px=%d",
                n_raw, n_rejected_area, fg_pixel_count_after,
            )

        return result

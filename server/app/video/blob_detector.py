"""
blob_detector.py — pure, stateless OpenCV blob detection.

The edge device has already performed background subtraction; every frame
arriving over RTSP is a foreground mask (white blobs on black background).
This module just finds and filters contours in that mask.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Blob:
    """Detected blob with normalised centre and raw bounding box."""
    x_norm: float               # [0, 1] — 0 is left
    y_norm: float               # [0, 1] — 0 is top
    area: float                 # pixel area of the contour
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in pixel space


class BlobDetector:
    """
    Detects blobs in a single foreground-mask frame.

    Parameters
    ----------
    min_area : float
        Minimum contour area (pixels²) to keep.
    max_area : float
        Maximum contour area (pixels²) to keep.
    blur_kernel : int
        Side length of the Gaussian blur kernel (must be odd).
    threshold : int
        Binary threshold value (0–255).
    morph_kernel : int
        Size of the structuring element used for morphological ops.
    """

    def __init__(
        self,
        min_area: float = 50.0,
        max_area: float = 50_000.0,
        blur_kernel: int = 5,
        threshold: int = 30,
        morph_kernel: int = 5,
    ) -> None:
        if blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be an odd integer")
        self.min_area = min_area
        self.max_area = max_area
        self.blur_kernel = blur_kernel
        self.threshold = threshold
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Blob]:
        """
        Run blob detection on *frame* and return a list of :class:`Blob`.

        Parameters
        ----------
        frame : np.ndarray
            A single video frame (H×W or H×W×C, uint8).
            Expected to be a foreground mask from the edge device.

        Returns
        -------
        List[Blob]
            Detected blobs sorted by area (largest first).
        """
        if frame is None or frame.size == 0:
            logger.warning("BlobDetector received an empty frame; skipping.")
            return []

        h, w = frame.shape[:2]
        gray = self._to_gray(frame)
        blurred = cv2.GaussianBlur(
            gray, (self.blur_kernel, self.blur_kernel), 0
        )
        _, binary = cv2.threshold(
            blurred, self.threshold, 255, cv2.THRESH_BINARY
        )

        # Close small holes, then open small specks
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._morph_kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, self._morph_kernel)

        contours, _ = cv2.findContours(
            opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        blobs: List[Blob] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            cx = x + bw / 2.0
            cy = y + bh / 2.0

            blobs.append(
                Blob(
                    x_norm=cx / w,
                    y_norm=cy / h,
                    area=area,
                    bbox=(x, y, bw, bh),
                )
            )

        blobs.sort(key=lambda b: b.area, reverse=True)
        return blobs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        if frame.ndim == 3 and frame.shape[2] == 1:
            return frame[:, :, 0]
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

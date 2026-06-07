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
from typing import List, Optional, Tuple

import cv2
import numpy as np

from sauron_edge.schemas import NCoord

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Output of a single frame processed by the MOG2 detector."""

    ncoords: List[NCoord]
    """Normalised [0,1] centroids of all detected foreground blobs."""

    bboxes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    """
    Normalised bounding boxes of each detected blob, parallel to `ncoords`.
    Each entry is (xnorm, ynorm, wnorm, hnorm) where (xnorm, ynorm) is the
    top-left corner and (wnorm, hnorm) are width/height — all in [0, 1]
    relative to the *original* (unscaled) frame dimensions.
    """

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
        min_contour_area: float = 500.0,
        morph_kernel_size: int = 7,
        process_scale: float = 0.5,
        blur_kernel_size: int = 5,
    ) -> None:
        self._learning_rate = learning_rate
        # Scale min_contour_area to match the processing resolution
        self._process_scale = max(0.1, min(1.0, process_scale))
        self._min_contour_area = min_contour_area * (self._process_scale**2)

        # Gaussian pre-blur kernel size — must be odd and >= 1
        self._blur_kernel_size = (
            blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        )

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

        # Pre-allocated morphology output buffers.
        # Lazy-initialised on first frame (shape unknown until then).
        # Reusing these avoids repeated heap allocation that can cause
        # std::bad_alloc on memory-constrained devices like the Pi.
        self._morph_buf1: Optional[np.ndarray] = None
        self._morph_buf2: Optional[np.ndarray] = None
        self._buf_shape: tuple = ()

        # Frame counter — used to throttle periodic debug summaries.
        self._frame_count: int = 0

        logger.info(
            "MOG2Detector: initialised — learning_rate=%.3f var_threshold=%.1f "
            "min_contour_area=%.1f (scaled from %.1f at %.0f%%) "
            "morph_open=%dpx morph_close=%dpx process_scale=%.2f blur_kernel=%dpx",
            learning_rate,
            var_threshold,
            self._min_contour_area,
            min_contour_area,
            self._process_scale * 100,
            morph_kernel_size,
            morph_kernel_size + 2,
            self._process_scale,
            self._blur_kernel_size,
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
            Returns an empty result (no crash) on cv2.error (e.g. OOM).
        """
        if frame is None or frame.size == 0:
            logger.warning("MOG2Detector: received empty/None frame — skipping")
            return DetectionResult(ncoords=[])

        # Guard against degenerate frames (zero-dimension V4L2 glitch frames).
        # frame.size catches fully-empty arrays; this catches (H, 0, 3) etc.
        if frame.ndim < 2 or frame.shape[0] == 0 or frame.shape[1] == 0:
            logger.warning(
                "MOG2Detector: degenerate frame shape %s — skipping", frame.shape
            )
            return DetectionResult(ncoords=[])

        # --- Step 0: downscale for processing ---
        # Compute explicit target dimensions using round() (not C-style truncation)
        # and clamp to a minimum of 2 so that downstream morphology operations never
        # receive a 0- or 1-pixel dimension, which would trigger an OpenCV assertion.
        self._frame_count += 1
        if self._process_scale != 1.0:
            target_w = max(2, round(frame.shape[1] * self._process_scale))
            target_h = max(2, round(frame.shape[0] * self._process_scale))
            if self._frame_count % 30 == 1:
                logger.debug(
                    "MOG2Detector: frame #%d src=%dx%d → proc=%dx%d (scale=%.2f)",
                    self._frame_count,
                    frame.shape[1],
                    frame.shape[0],
                    target_w,
                    target_h,
                    self._process_scale,
                )
            try:
                proc_frame = cv2.resize(
                    frame,
                    (target_w, target_h),
                    interpolation=cv2.INTER_LINEAR,
                )
            except cv2.error as exc:
                logger.error(
                    "MOG2Detector: cv2.error during resize to %dx%d — skipping frame: %s",
                    target_w,
                    target_h,
                    exc,
                )
                return DetectionResult(ncoords=[])
        else:
            if self._frame_count % 30 == 1:
                logger.debug(
                    "MOG2Detector: frame #%d src=%dx%d (no scale)",
                    self._frame_count,
                    frame.shape[1],
                    frame.shape[0],
                )
            proc_frame = frame

        h, w = proc_frame.shape[:2]

        # Lazy-init / re-init pre-allocated morphology buffers if shape changed
        if proc_frame.shape[:2] != self._buf_shape:
            self._buf_shape = proc_frame.shape[:2]
            self._morph_buf1 = np.empty((h, w), dtype=np.uint8)
            self._morph_buf2 = np.empty((h, w), dtype=np.uint8)
            logger.info(
                "MOG2Detector: (re)allocated morphology buffers for %dx%d", w, h
            )

        # --- Step 1: apply MOG2 + morphological cleanup ---
        try:
            # Pre-blur to suppress per-pixel sensor noise before MOG2 sees the frame.
            # A 5×5 Gaussian kernel smooths out CMOS noise that would otherwise create
            # spurious foreground pixels, substantially reducing false-positive contours
            # in static scenes.
            blurred = cv2.GaussianBlur(
                proc_frame,
                (self._blur_kernel_size, self._blur_kernel_size),
                0,
            )
            fg_mask: np.ndarray = self._subtractor.apply(
                blurred, learningRate=self._learning_rate
            )

            fg_pixel_count = int(cv2.countNonZero(fg_mask))
            logger.debug(
                "MOG2Detector: foreground pixels before morphology = %d (%.1f%%)",
                fg_pixel_count,
                100.0 * fg_pixel_count / (h * w),
            )

            # Reuse pre-allocated buffers — avoids heap thrash on Pi
            fg_mask = cv2.morphologyEx(
                fg_mask, cv2.MORPH_OPEN, self._open_kernel, dst=self._morph_buf1
            )
            fg_mask = cv2.morphologyEx(
                fg_mask, cv2.MORPH_CLOSE, self._close_kernel, dst=self._morph_buf2
            )
        except cv2.error as exc:
            logger.error(
                "MOG2Detector: cv2.error during MOG2/morphology — skipping frame "
                "(likely OOM): %s",
                exc,
            )
            return DetectionResult(ncoords=[])

        fg_pixel_count_after = int(cv2.countNonZero(fg_mask))
        logger.debug(
            "MOG2Detector: foreground pixels after morphology = %d",
            fg_pixel_count_after,
        )

        # --- Step 2: find contours ---
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        n_raw = len(contours)
        logger.debug("MOG2Detector: raw contour count = %d", n_raw)

        # --- Steps 3–5: filter, moments, normalise ---
        ncoords: List[NCoord] = []
        bboxes: List[Tuple[float, float, float, float]] = []
        n_rejected_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self._min_contour_area:
                n_rejected_area += 1
                logger.debug(
                    "MOG2Detector: contour rejected — area=%.1f < min=%.1f",
                    area,
                    self._min_contour_area,
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

            # Bounding box in proc_frame pixels → normalise to [0,1].
            # Dividing by proc_frame dimensions gives coords relative to the
            # original frame because the normalised ratio is scale-invariant.
            bx, by, bw, bh = cv2.boundingRect(contour)
            bboxes.append(
                (
                    round(bx / w, 4),
                    round(by / h, 4),
                    round(bw / w, 4),
                    round(bh / h, 4),
                )
            )

            ncoords.append(NCoord(xnorm=round(xnorm, 4), ynorm=round(ynorm, 4)))
            logger.debug(
                "MOG2Detector: detection — centroid=(%.1f, %.1f) norm=(%.4f, %.4f) "
                "bbox=(%.3f, %.3f, %.3f, %.3f) area=%.1f",
                cx,
                cy,
                xnorm,
                ynorm,
                bx / w,
                by / h,
                bw / w,
                bh / h,
                area,
            )

        result = DetectionResult(
            ncoords=ncoords,
            bboxes=bboxes,
            n_raw_contours=n_raw,
            n_rejected_area=n_rejected_area,
            foreground_pixel_count=fg_pixel_count_after,
        )

        if ncoords:
            logger.info(
                "MOG2Detector: %d detection(s) — raw_contours=%d rejected_area=%d fg_px=%d",
                len(ncoords),
                n_raw,
                n_rejected_area,
                fg_pixel_count_after,
            )
        else:
            logger.debug(
                "MOG2Detector: no detections — raw_contours=%d rejected_area=%d fg_px=%d",
                n_raw,
                n_rejected_area,
                fg_pixel_count_after,
            )

        return result

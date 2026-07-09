"""
Thread-safe shared state store.

The main processing loop writes to this module after every frame and every
publish.  The dashboard Flask thread reads from it.  All access goes through
a single threading.Lock so neither side can observe a half-written update.

Design goals:
- Zero blocking: writers call update_*() which take the lock, copy in the
  new data, and release immediately.
- Zero copying overhead for readers: get_*() methods return lightweight
  copies or primitive values; numpy frames are copied once here so the
  dashboard never holds a reference into the main-loop frame buffer.
- Error capture: a custom logging.Handler is installed that funnels all
  ERROR+ records into a bounded deque so the dashboard can display them.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from copy import copy
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Maximum recent telemetry packets to retain
_MAX_TELEMETRY = 5
# Maximum error log entries to retain
_MAX_ERRORS = 30
# JPEG encoder hard limit; frames exceeding this in either dimension are dropped
_JPEG_MAX_DIM = 8192


# ---------------------------------------------------------------------------
# Error capture handler
# ---------------------------------------------------------------------------


@dataclass
class ErrorEntry:
    timestamp: float
    level: str
    logger_name: str
    message: str


class _ErrorCaptureHandler(logging.Handler):
    """Funnels WARNING+ log records into the shared state error deque."""

    def __init__(self, error_deque: Deque[ErrorEntry]) -> None:
        super().__init__(level=logging.WARNING)
        self._deque = error_deque

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._deque.append(
                ErrorEntry(
                    timestamp=record.created,
                    level=record.levelname,
                    logger_name=record.name,
                    message=self.format(record),
                )
            )
        except Exception:
            pass  # never raise inside a logging handler


# ---------------------------------------------------------------------------
# Health status
# ---------------------------------------------------------------------------


@dataclass
class ComponentHealth:
    camera_alive: bool = False
    """True if the camera capture thread is running and producing frames."""

    sensor_thread_alive: bool = False
    """True if the sensor reader thread is running."""

    gps_locked: bool = False
    """True if GPS has a valid satellite fix (effective value — see gps_override_active)."""

    gps_override_active: bool = False
    """True if this device publishes a fixed override position instead of the real GPS."""

    gps_actual_locked: bool = False
    """True if the real GPS module has a satellite fix, regardless of override."""

    gps_actual_lat: float = 0.0
    gps_actual_lon: float = 0.0
    gps_actual_sats: int = 0
    """Real GPS reading — always the ground truth, never masked by an override."""

    imu_calibrated: bool = False
    """True if BNO055 system calibration level >= 1 (effective value — see imu_override_active)."""

    imu_override_active: bool = False
    """True if this device publishes a fixed override orientation instead of the real IMU."""

    imu_actual_calibrated: bool = False
    imu_actual_heading: float = 0.0
    imu_actual_pitch: float = 0.0
    imu_actual_roll: float = 0.0
    imu_actual_calibration_status: str = "Sys:0 G:0 A:0 M:0"
    """Real IMU reading — always the ground truth, never masked by an override."""

    publisher_connected: bool = False
    """True if the Greengrass IPC connection is live."""

    uptime_s: float = 0.0
    """Seconds since the main loop started."""

    frames_processed: int = 0
    total_detections: int = 0
    total_publishes: int = 0
    total_publish_failures: int = 0

    last_publish_ts: Optional[float] = None
    """Unix timestamp of the last successful publish, or None."""

    camera_fps: float = 0.0
    """Actual FPS negotiated with the capture device."""

    camera_w: int = 0
    camera_h: int = 0
    """Actual frame dimensions negotiated with the capture device."""

    cpu_temp_c: float = 0.0
    """CPU/SoC temperature in Celsius (sampled every ~10 s)."""


# ---------------------------------------------------------------------------
# Singleton shared state
# ---------------------------------------------------------------------------


class SharedState:
    """
    Singleton shared state updated by the main processing loop and read by
    the dashboard Flask thread.

    Instantiate once in main.py and pass to both the main loop helpers and
    the dashboard.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_ts: float = time.time()

        # Latest annotated camera frame (BGR numpy array or None)
        self._latest_frame: Optional[np.ndarray] = None

        # Latest raw (unannotated) frame dimensions — for overlay calculations
        self._frame_w: int = 0
        self._frame_h: int = 0

        # Latest detection ncoords as list of dicts [{"xnorm":…, "ynorm":…}]
        self._latest_ncoords: List[Dict[str, float]] = []

        # Recent telemetry payloads (model_dump() dicts)
        self._recent_telemetry: Deque[Dict[str, Any]] = deque(maxlen=_MAX_TELEMETRY)

        # Component health snapshot
        self._health: ComponentHealth = ComponentHealth()

        # Error / warning log entries
        self._errors: Deque[ErrorEntry] = deque(maxlen=_MAX_ERRORS)

        # Install log capture handler on the root logger
        self._log_handler = _ErrorCaptureHandler(self._errors)
        self._log_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(self._log_handler)
        logger.debug("SharedState: error capture log handler installed")

    # ------------------------------------------------------------------
    # Write API (called from main loop thread)
    # ------------------------------------------------------------------

    def update_frame(
        self,
        frame: np.ndarray,
        ncoords: List[Dict[str, float]],
        bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> None:
        """
        Store the latest camera frame and its detections.
        `frame` must be the raw BGR frame from the camera — NOT the fg_mask.
        The frame is annotated with bounding boxes and a HUD, then stored.
        Frames with corrupt/garbage dimensions are dropped before JPEG encoding.
        """
        if frame.ndim < 2 or frame.shape[0] == 0 or frame.shape[1] == 0:
            logger.warning(
                "SharedState: dropping degenerate frame shape %s", frame.shape
            )
            return
        if frame.shape[0] > _JPEG_MAX_DIM or frame.shape[1] > _JPEG_MAX_DIM:
            logger.warning(
                "SharedState: dropping oversized frame %dx%d (limit %d) — "
                "likely a V4L2 driver glitch",
                frame.shape[1],
                frame.shape[0],
                _JPEG_MAX_DIM,
            )
            return
        annotated = _annotate_frame(frame, ncoords, bboxes or [])
        with self._lock:
            self._latest_frame = annotated
            self._frame_w = frame.shape[1]
            self._frame_h = frame.shape[0]
            self._latest_ncoords = list(ncoords)

    def record_telemetry(self, payload_dict: Dict[str, Any]) -> None:
        """Append the most recently published telemetry dict to the rolling window."""
        with self._lock:
            self._recent_telemetry.append(payload_dict)
            logger.debug(
                "SharedState: telemetry recorded — window size=%d",
                len(self._recent_telemetry),
            )

    def update_health(
        self,
        *,
        camera_alive: bool,
        sensor_thread_alive: bool,
        gps_locked: bool,
        imu_calibrated: bool,
        publisher_connected: bool,
        frames_processed: int,
        total_detections: int,
        total_publishes: int,
        total_publish_failures: int,
        last_publish_ts: Optional[float],
        camera_fps: float = 0.0,
        camera_w: int = 0,
        camera_h: int = 0,
        cpu_temp_c: float = 0.0,
        gps_override_active: bool = False,
        gps_actual_locked: bool = False,
        gps_actual_lat: float = 0.0,
        gps_actual_lon: float = 0.0,
        gps_actual_sats: int = 0,
        imu_override_active: bool = False,
        imu_actual_calibrated: bool = False,
        imu_actual_heading: float = 0.0,
        imu_actual_pitch: float = 0.0,
        imu_actual_roll: float = 0.0,
        imu_actual_calibration_status: str = "Sys:0 G:0 A:0 M:0",
    ) -> None:
        with self._lock:
            self._health = ComponentHealth(
                camera_alive=camera_alive,
                sensor_thread_alive=sensor_thread_alive,
                gps_locked=gps_locked,
                gps_override_active=gps_override_active,
                gps_actual_locked=gps_actual_locked,
                gps_actual_lat=gps_actual_lat,
                gps_actual_lon=gps_actual_lon,
                gps_actual_sats=gps_actual_sats,
                imu_calibrated=imu_calibrated,
                imu_override_active=imu_override_active,
                imu_actual_calibrated=imu_actual_calibrated,
                imu_actual_heading=imu_actual_heading,
                imu_actual_pitch=imu_actual_pitch,
                imu_actual_roll=imu_actual_roll,
                imu_actual_calibration_status=imu_actual_calibration_status,
                publisher_connected=publisher_connected,
                uptime_s=time.time() - self._start_ts,
                frames_processed=frames_processed,
                total_detections=total_detections,
                total_publishes=total_publishes,
                total_publish_failures=total_publish_failures,
                last_publish_ts=last_publish_ts,
                camera_fps=camera_fps,
                camera_w=camera_w,
                camera_h=camera_h,
                cpu_temp_c=cpu_temp_c,
            )

    # ------------------------------------------------------------------
    # Read API (called from dashboard thread)
    # ------------------------------------------------------------------

    def get_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the latest annotated frame, or None."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_recent_telemetry(self) -> List[Dict[str, Any]]:
        """Return list of up to 5 most recent telemetry payload dicts (newest last)."""
        with self._lock:
            return list(self._recent_telemetry)

    def get_health(self) -> ComponentHealth:
        with self._lock:
            return copy(self._health)

    def get_errors(self) -> List[ErrorEntry]:
        """Return list of recent error/warning entries (newest last)."""
        with self._lock:
            return list(self._errors)

    def get_latest_ncoords(self) -> List[Dict[str, float]]:
        with self._lock:
            return list(self._latest_ncoords)


# ---------------------------------------------------------------------------
# Frame annotation helper
# ---------------------------------------------------------------------------


def _annotate_frame(
    frame: np.ndarray,
    ncoords: List[Dict[str, float]],
    bboxes: List[Tuple[float, float, float, float]],
) -> np.ndarray:
    """
    Draw MOG2 detection overlays on a copy of the raw BGR `frame`.

    For each detection:
    - A green bounding rectangle (from the contour bounding box)
    - A small label with the detection index and normalised centroid

    Also draws a corner HUD with timestamp and detection count.
    The input frame is NEVER modified in place.
    """
    import cv2  # local import keeps this module importable without opencv on CI

    out = frame.copy()
    h, w = out.shape[:2]

    COLOUR_BOX = (0, 255, 0)  # bright green
    COLOUR_LABEL = (0, 255, 0)
    COLOUR_HUD = (0, 255, 0)
    COLOUR_HUD_ALERT = (0, 165, 255)  # orange when detections > 0

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SMALL = 0.4
    FONT_MED = 0.5
    THICK_1 = 1
    THICK_2 = 2

    # --- Draw bounding boxes on the raw frame ---
    for i, (xnorm, ynorm, wnorm, hnorm) in enumerate(bboxes):
        x1 = int(xnorm * w)
        y1 = int(ynorm * h)
        x2 = int((xnorm + wnorm) * w)
        y2 = int((ynorm + hnorm) * h)

        cv2.rectangle(out, (x1, y1), (x2, y2), COLOUR_BOX, THICK_2)

        # Label just above the top-left corner of the box
        if i < len(ncoords):
            nc = ncoords[i]
            label = f"#{i+1} ({nc['xnorm']:.2f},{nc['ynorm']:.2f})"
        else:
            label = f"#{i+1}"
        lx = min(x1 + 2, w - 100)
        ly = max(y1 - 4, 10)
        cv2.putText(out, label, (lx, ly), FONT, FONT_SMALL, COLOUR_LABEL, THICK_1)

    # --- HUD overlay (top-left corner) ---
    n = len(bboxes)
    hud_colour = COLOUR_HUD if n == 0 else COLOUR_HUD_ALERT
    ts_str = time.strftime("%H:%M:%S")
    cv2.putText(
        out, f"SAURON EDGE  {ts_str}", (8, 18), FONT, FONT_MED, COLOUR_HUD, THICK_2
    )
    cv2.putText(out, f"Detections: {n}", (8, 38), FONT, FONT_MED, hud_colour, THICK_1)

    return out

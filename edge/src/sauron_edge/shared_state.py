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
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Maximum recent telemetry packets to retain
_MAX_TELEMETRY = 5
# Maximum error log entries to retain
_MAX_ERRORS = 30


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
    """True if GPS has a valid satellite fix."""

    imu_calibrated: bool = False
    """True if BNO055 system calibration level >= 1."""

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
    ) -> None:
        """
        Store the latest camera frame and its detections.
        The frame is copied once here so the main loop can reuse its buffer.
        """
        annotated = _annotate_frame(frame, ncoords)
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
                "SharedState: telemetry recorded — window size=%d", len(self._recent_telemetry)
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
    ) -> None:
        with self._lock:
            self._health = ComponentHealth(
                camera_alive=camera_alive,
                sensor_thread_alive=sensor_thread_alive,
                gps_locked=gps_locked,
                imu_calibrated=imu_calibrated,
                publisher_connected=publisher_connected,
                uptime_s=time.time() - self._start_ts,
                frames_processed=frames_processed,
                total_detections=total_detections,
                total_publishes=total_publishes,
                total_publish_failures=total_publish_failures,
                last_publish_ts=last_publish_ts,
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


def _annotate_frame(frame: np.ndarray, ncoords: List[Dict[str, float]]) -> np.ndarray:
    """
    Draw detection markers on a copy of `frame`.

    Each ncoord gets:
    - A filled circle at the centroid
    - A crosshair (horizontal + vertical line through the centroid)
    - A small label with normalised coordinates

    Also draws a corner timestamp and detection count HUD.
    """
    import cv2  # local import keeps this module importable without opencv on CI

    out = frame.copy()
    h, w = out.shape[:2]

    COLOUR_DETECTION = (0, 255, 0)      # bright green
    COLOUR_CROSSHAIR = (0, 200, 0)
    COLOUR_HUD       = (0, 255, 0)
    COLOUR_HUD_WARN  = (0, 165, 255)    # orange

    FONT      = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SMALL = 0.4
    FONT_MED   = 0.5
    THICK_1    = 1
    THICK_2    = 2

    # --- Draw each detection ---
    for i, nc in enumerate(ncoords):
        cx = int(nc["xnorm"] * w)
        cy = int(nc["ynorm"] * h)

        # Crosshair arms
        arm = 15
        cv2.line(out, (cx - arm, cy), (cx + arm, cy), COLOUR_CROSSHAIR, THICK_1)
        cv2.line(out, (cx, cy - arm), (cx, cy + arm), COLOUR_CROSSHAIR, THICK_1)

        # Centre dot
        cv2.circle(out, (cx, cy), 4, COLOUR_DETECTION, -1)

        # Label
        label = f"#{i+1} ({nc['xnorm']:.2f},{nc['ynorm']:.2f})"
        lx = min(cx + 8, w - 120)
        ly = max(cy - 8, 12)
        cv2.putText(out, label, (lx, ly), FONT, FONT_SMALL, COLOUR_DETECTION, THICK_1)

    # --- HUD overlay (top-left) ---
    n = len(ncoords)
    hud_colour = COLOUR_HUD if n == 0 else COLOUR_HUD_WARN
    ts_str = time.strftime("%H:%M:%S")
    cv2.putText(out, f"SAURON EDGE  {ts_str}", (8, 18), FONT, FONT_MED, COLOUR_HUD, THICK_2)
    cv2.putText(out, f"Detections: {n}", (8, 38), FONT, FONT_MED, hud_colour, THICK_1)

    return out

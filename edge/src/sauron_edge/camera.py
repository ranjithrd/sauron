"""
Camera source abstraction.

Supports two backends, both configured via config.yaml:
  - 'raspi': Raspberry Pi embedded camera via V4L2 (/dev/video0)
  - 'rtsp':  Any RTSP network stream (proxied via FFMPEG/GStreamer)

Each backend runs a dedicated background thread that does nothing except
continuously drain the capture buffer and store the latest frame. This
eliminates accumulated-buffer lag — the main pipeline always processes the
most recent frame, not one that is N-frames stale.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# How many consecutive failed reads before the capture thread gives up
_MAX_CONSECUTIVE_FAILURES = 30

# Seconds to sleep between failed read retries
_RETRY_SLEEP_S = 0.05


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class CameraSource(ABC):
    """Abstract camera backend interface."""

    @abstractmethod
    def start(self) -> None:
        """Open the capture device and launch the background reader thread."""

    @abstractmethod
    def stop(self) -> None:
        """Signal the reader thread to stop, release the capture device."""

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """
        Return a copy of the latest captured frame, or None if no frame is
        available yet (e.g. camera is still initialising).
        """

    @abstractmethod
    def is_alive(self) -> bool:
        """True if the background reader thread is still running."""


# ---------------------------------------------------------------------------
# Shared background reader thread
# ---------------------------------------------------------------------------


class _CaptureThread(threading.Thread):
    """
    Continuously reads from a cv2.VideoCapture, always keeping only the
    latest frame in memory. Designed to be started once and run until
    stop() is called or too many consecutive failures occur.
    """

    def __init__(
        self, cap: cv2.VideoCapture, thread_name: str, width: int, height: int
    ) -> None:
        super().__init__(name=thread_name, daemon=True)
        self._cap = cap
        self._width = width
        self._height = height
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._healthy = False

    # --- public API ---

    def get_frame(self) -> Optional[np.ndarray]:
        """Return a thread-safe copy of the latest frame, or None."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self) -> None:
        """Signal the thread to exit on its next iteration."""
        self._stop_event.set()

    def is_healthy(self) -> bool:
        return self._healthy

    # --- thread body ---

    def run(self) -> None:
        consecutive_failures = 0
        logger.info("[%s] Capture thread started", self.name)

        # Calculate the exact number of elements a standard BGR frame should have
        expected_size = self._width * self._height * 3

        while not self._stop_event.is_set():
            grabbed, frame = self._cap.read()

            if grabbed and frame is not None:
                # Intercept the raw libcamera flat buffer bug and reshape it
                if (
                    frame.ndim == 2
                    and frame.shape[0] == 1
                    and frame.size == expected_size
                ):
                    frame = frame.reshape((self._height, self._width, 3))

                # Now run the standard validation
                if frame.ndim >= 2 and frame.shape[0] > 1 and frame.shape[1] > 1:
                    with self._lock:
                        self._frame = frame
                    self._healthy = True
                    consecutive_failures = 0
                    continue

                # If we get here, the frame was grabbed but still has a degenerate shape
                logger.debug(
                    "[%s] Dropping malformed frame shape=%s — skipping",
                    self.name,
                    frame.shape,
                )

            else:
                consecutive_failures += 1
                logger.warning(
                    "[%s] Frame read failed (consecutive=%d/%d)",
                    self.name,
                    consecutive_failures,
                    _MAX_CONSECUTIVE_FAILURES,
                )
                self._healthy = False

                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    logger.error(
                        "[%s] Camera unresponsive — capture thread exiting after %d failures",
                        self.name,
                        consecutive_failures,
                    )
                    break

                time.sleep(_RETRY_SLEEP_S)

        logger.info(
            "[%s] Capture thread stopped (healthy=%s)", self.name, self._healthy
        )


# ---------------------------------------------------------------------------
# Raspi backend (V4L2)
# ---------------------------------------------------------------------------


class RaspiCamera(CameraSource):
    """
    Raspberry Pi camera via V4L2 (/dev/video0).

    Mirrors the pattern from reference.txt — buffer size forced to 1 frame,
    resolution and FPS set explicitly before the background reader is started.
    """

    def __init__(self, width: int, height: int, fps: int) -> None:
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[_CaptureThread] = None
        # Populated after start() — reflect the V4L2-negotiated values
        self._actual_w: int = width
        self._actual_h: int = height
        self._actual_fps: float = float(fps)

    @property
    def actual_width(self) -> int:
        return self._actual_w

    @property
    def actual_height(self) -> int:
        return self._actual_h

    @property
    def actual_fps(self) -> float:
        return self._actual_fps

    def start(self) -> None:
        logger.info(
            "RaspiCamera: opening /dev/video0 — %dx%d @ %d fps",
            self._width,
            self._height,
            self._fps,
        )
        self._cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

        # Request MJPEG pixel format first — before setting resolution/FPS.
        # This is critical on Pi cameras: V4L2 defaults to raw YUYV at low
        # resolutions, but at some resolutions it can't decode the stream and
        # hands back the raw MJPEG bytes as a 1×N array instead of a proper
        # BGR frame.  Explicitly requesting MJPEG ensures OpenCV always gets
        # a decodable stream regardless of the configured resolution.
        mjpeg_fourcc = cv2.VideoWriter.fourcc("M", "J", "P", "G")
        self._cap.set(cv2.CAP_PROP_FOURCC, mjpeg_fourcc)

        # Force a single-frame buffer to kill latency (same as reference.txt)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        if not self._cap.isOpened():
            raise RuntimeError(
                "RaspiCamera: cv2.VideoCapture(0, CAP_V4L2) failed to open — "
                "is the camera connected and /dev/video0 accessible?"
            )

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        raw_fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc = "".join(chr((raw_fourcc >> (8 * i)) & 0xFF) for i in range(4))
        logger.info(
            "RaspiCamera: opened — actual resolution=%dx%d fps=%.1f fourcc=%s",
            actual_w,
            actual_h,
            actual_fps,
            actual_fourcc,
        )
        self._actual_w = actual_w
        self._actual_h = actual_h
        self._actual_fps = actual_fps

        self._thread = _CaptureThread(
            self._cap, thread_name="raspi-capture", width=actual_w, height=actual_h
        )
        self._thread.start()
        logger.info("RaspiCamera: background capture thread launched")

    def stop(self) -> None:
        logger.info("RaspiCamera: stopping")
        if self._thread is not None:
            self._thread.stop()
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("RaspiCamera: capture thread did not exit cleanly")
        if self._cap is not None:
            self._cap.release()
            logger.info("RaspiCamera: capture device released")

    def read(self) -> Optional[np.ndarray]:
        if self._thread is None:
            return None
        frame = self._thread.get_frame()
        if frame is not None:
            # Physical mount is upside-down — apply 180° software flip
            frame = cv2.flip(frame, -1)
        return frame

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# ---------------------------------------------------------------------------
# RTSP backend
# ---------------------------------------------------------------------------


class RTSPCamera(CameraSource):
    """
    RTSP stream camera, decoded via OpenCV's FFMPEG/GStreamer backend.
    Used for test streams and remote cameras.
    """

    def __init__(self, url: str, width: int, height: int) -> None:
        self._url = url
        self._width = width
        self._height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[_CaptureThread] = None

    def start(self) -> None:
        logger.info("RTSPCamera: opening stream — %s", self._url)
        self._cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"RTSPCamera: failed to open RTSP stream {self._url!r} — "
                "check the URL and network connectivity"
            )

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            "RTSPCamera: stream opened — actual resolution=%dx%d", actual_w, actual_h
        )

        self._thread = _CaptureThread(
            self._cap, thread_name="rtsp-capture", width=actual_w, height=actual_h
        )
        self._thread.start()
        logger.info("RTSPCamera: background capture thread launched")

    def stop(self) -> None:
        logger.info("RTSPCamera: stopping")
        if self._thread is not None:
            self._thread.stop()
            self._thread.join(timeout=8.0)  # RTSP teardown can be slow
            if self._thread.is_alive():
                logger.warning("RTSPCamera: capture thread did not exit cleanly")
        if self._cap is not None:
            self._cap.release()
            logger.info("RTSPCamera: stream released")

    def read(self) -> Optional[np.ndarray]:
        if self._thread is None:
            return None
        return self._thread.get_frame()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_camera(cfg) -> CameraSource:
    """
    Construct the correct CameraSource from the loaded Configuration.
    Raises ValueError for an invalid/incomplete config before any device
    is opened.
    """
    if cfg.camera_source == "raspi":
        logger.info(
            "make_camera: selecting RaspiCamera (%dx%d @ %dfps)",
            cfg.camera_width(),
            cfg.camera_height(),
            cfg.camera_fps,
        )
        return RaspiCamera(
            width=cfg.camera_width(),
            height=cfg.camera_height(),
            fps=cfg.camera_fps,
        )

    if cfg.camera_source == "rtsp":
        if not cfg.camera_rtsp_url:
            raise ValueError(
                "camera_source is 'rtsp' but camera_rtsp_url is empty — "
                "set it in config.yaml"
            )
        logger.info(
            "make_camera: selecting RTSPCamera (%s, %dx%d)",
            cfg.camera_rtsp_url,
            cfg.camera_width(),
            cfg.camera_height(),
        )
        return RTSPCamera(
            url=cfg.camera_rtsp_url,
            width=cfg.camera_width(),
            height=cfg.camera_height(),
        )

    # Should never reach here — validator in Configuration catches this
    raise ValueError(f"make_camera: unknown camera_source {cfg.camera_source!r}")

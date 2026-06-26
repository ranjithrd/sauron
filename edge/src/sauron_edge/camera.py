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

# Seconds to sleep between failed read retries (normal)
_RETRY_SLEEP_S = 0.05

# Seconds to sleep after an extended run of failures (avoids log spam)
_RETRY_BACKOFF_S = 5.0

# Number of consecutive failures before switching to backoff sleep
_BACKOFF_THRESHOLD = 30


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
                self._healthy = False

                if consecutive_failures <= _BACKOFF_THRESHOLD:
                    logger.warning(
                        "[%s] Frame read failed (consecutive=%d)",
                        self.name,
                        consecutive_failures,
                    )
                    time.sleep(_RETRY_SLEEP_S)
                else:
                    # Log once when we first hit backoff, then stay quiet
                    if consecutive_failures == _BACKOFF_THRESHOLD + 1:
                        logger.warning(
                            "[%s] Stream unresponsive after %d failures — retrying every %.0fs",
                            self.name,
                            consecutive_failures,
                            _RETRY_BACKOFF_S,
                        )
                    time.sleep(_RETRY_BACKOFF_S)

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


class _RTSPCaptureThread(threading.Thread):
    """
    RTSP-specific capture thread that owns the connect/reconnect loop.
    Never exits until stop() is called — if the stream is down it retries
    every _RETRY_BACKOFF_S seconds, silently after the first failure log.
    """

    def __init__(self, url: str, width: int, height: int) -> None:
        super().__init__(name="rtsp-capture", daemon=True)
        self._url = url
        self._width = width
        self._height = height
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._healthy = False

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self) -> None:
        self._stop_event.set()

    def is_healthy(self) -> bool:
        return self._healthy

    def _open_cap(self) -> Optional[cv2.VideoCapture]:
        """Try once to open the RTSP stream. Returns cap on success, None on failure."""
        import os as _os
        _os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp"
            "|fflags;nobuffer"
            "|flags;low_delay"
            "|framedrop;1"
            "|buffer_size;65536"
        )
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def run(self) -> None:
        logger.info("[rtsp-capture] Capture thread started — connecting to %s", self._url)
        first_attempt = True

        while not self._stop_event.is_set():
            # --- connect ---
            cap = self._open_cap()
            if cap is None:
                if first_attempt:
                    logger.warning(
                        "[rtsp-capture] Could not open stream %s — will retry every %.0fs",
                        self._url,
                        _RETRY_BACKOFF_S,
                    )
                    first_attempt = False
                self._stop_event.wait(timeout=_RETRY_BACKOFF_S)
                continue

            first_attempt = True
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(
                "[rtsp-capture] Stream connected — %dx%d", actual_w, actual_h
            )

            # --- read loop ---
            consecutive_failures = 0
            while not self._stop_event.is_set():
                grabbed, frame = cap.read()
                if grabbed and frame is not None and frame.ndim >= 2 and frame.shape[0] > 1:
                    with self._lock:
                        self._frame = frame
                    self._healthy = True
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    self._healthy = False
                    if consecutive_failures <= _BACKOFF_THRESHOLD:
                        logger.warning(
                            "[rtsp-capture] Frame read failed (consecutive=%d)",
                            consecutive_failures,
                        )
                        time.sleep(_RETRY_SLEEP_S)
                    else:
                        if consecutive_failures == _BACKOFF_THRESHOLD + 1:
                            logger.warning(
                                "[rtsp-capture] Stream lost after %d failures — reconnecting",
                                consecutive_failures,
                            )
                        cap.release()
                        break  # back to outer connect loop

        logger.info("[rtsp-capture] Capture thread stopped")


class RTSPCamera(CameraSource):
    """
    RTSP stream camera, decoded via OpenCV's FFMPEG backend.
    start() returns immediately — connection and reconnection happen in the
    background thread, which retries forever until stop() is called.
    """

    def __init__(self, url: str, width: int, height: int) -> None:
        self._url = url
        self._width = width
        self._height = height
        self._thread: Optional[_RTSPCaptureThread] = None

    def start(self) -> None:
        logger.info("RTSPCamera: launching capture thread for %s", self._url)
        self._thread = _RTSPCaptureThread(self._url, self._width, self._height)
        self._thread.start()

    def stop(self) -> None:
        logger.info("RTSPCamera: stopping")
        if self._thread is not None:
            self._thread.stop()
            self._thread.join(timeout=8.0)
            if self._thread.is_alive():
                logger.warning("RTSPCamera: capture thread did not exit cleanly")

    def read(self) -> Optional[np.ndarray]:
        if self._thread is None:
            return None
        return self._thread.get_frame()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# ---------------------------------------------------------------------------
# NTP-synced video loop backend
# ---------------------------------------------------------------------------


class NTPLoopCamera(CameraSource):
    """
    Plays a video file on a loop that is synchronised to wall-clock (NTP) time.

    Frame selection: frame_idx = floor((unix_time % duration_s) * fps) % total_frames

    Because all NTP-synced clocks share the same unix_time, two edge agents
    playing the same file will always be on the same frame at the same instant
    — making multi-camera test footage fully coordinated.

    Constraint: best results when 60 % duration_s == 0 so that the loop
    realigns at every minute boundary (durations: 1 2 3 4 5 6 10 12 15 20 30 60s).

    All frames are preloaded into RAM at start() to avoid seek latency.
    """

    def __init__(self, filepath: str, width: int, height: int, reverse: bool = False) -> None:
        self._filepath = filepath
        self._width = width
        self._height = height
        self._reverse = reverse
        self._frames: list = []
        self._fps: float = 25.0
        self._duration_s: float = 0.0
        self._total: int = 0
        self._alive: bool = False

    @property
    def actual_width(self) -> int:
        return self._width

    @property
    def actual_height(self) -> int:
        return self._height

    @property
    def actual_fps(self) -> float:
        return self._fps

    def start(self) -> None:
        import cv2 as _cv2

        cap = _cv2.VideoCapture(self._filepath)
        if not cap.isOpened():
            raise RuntimeError(
                f"NTPLoopCamera: could not open video file {self._filepath!r}"
            )

        total_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(_cv2.CAP_PROP_FPS) or 25.0
        duration_s = total_frames / fps if fps > 0 else 0.0

        if duration_s > 0:
            remainder = 60.0 % duration_s
            if remainder > 0.1:
                logger.warning(
                    "NTPLoopCamera: video duration %.2fs does not divide 60s "
                    "(remainder %.2fs) — loops will not realign at minute boundaries",
                    duration_s,
                    remainder,
                )

        frames: list = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(_cv2.resize(frame, (self._width, self._height)))
        cap.release()

        if not frames:
            raise RuntimeError(
                f"NTPLoopCamera: no frames could be read from {self._filepath!r}"
            )

        if self._reverse:
            frames = frames[::-1]

        self._frames = frames
        self._fps = fps
        self._duration_s = duration_s if duration_s > 0 else len(frames) / fps
        self._total = len(frames)
        self._alive = True

        mem_mb = (self._width * self._height * 3 * self._total) / (1024 * 1024)
        logger.info(
            "NTPLoopCamera: loaded %d frames (%.2fs @ %.1f fps) from %s — %.0f MB RAM%s",
            self._total,
            self._duration_s,
            self._fps,
            self._filepath,
            mem_mb,
            " [REVERSED]" if self._reverse else "",
        )

    def stop(self) -> None:
        self._alive = False
        self._frames.clear()
        logger.info("NTPLoopCamera: stopped, frames released")

    def read(self) -> Optional[np.ndarray]:
        if not self._frames:
            return None
        pos_s = time.time() % self._duration_s
        idx = int(pos_s * self._fps) % self._total
        return self._frames[idx].copy()

    def is_alive(self) -> bool:
        return self._alive


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

    if cfg.camera_source == "ntp_loop":
        if not cfg.camera_ntp_loop_file:
            raise ValueError(
                "camera_source is 'ntp_loop' but camera_ntp_loop_file is empty — "
                "set it in config.yaml"
            )
        logger.info(
            "make_camera: selecting NTPLoopCamera (%s, %dx%d)",
            cfg.camera_ntp_loop_file,
            cfg.camera_width(),
            cfg.camera_height(),
        )
        return NTPLoopCamera(
            filepath=cfg.camera_ntp_loop_file,
            width=cfg.camera_width(),
            height=cfg.camera_height(),
            reverse=cfg.camera_ntp_loop_reverse,
        )

    # Should never reach here — validator in Configuration catches this
    raise ValueError(f"make_camera: unknown camera_source {cfg.camera_source!r}")

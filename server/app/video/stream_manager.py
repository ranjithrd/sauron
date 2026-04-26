"""
stream_manager.py — async orchestrator for per-camera RTSP streams.

Design decisions
----------------
* Each camera runs its capture loop inside ``asyncio.to_thread`` so that
  OpenCV's blocking calls never stall the event loop.
* ``cv2.VideoCapture`` is **created and released within every iteration** —
  OpenCV captures are not thread-safe when shared across iterations.
* Frame skipping is applied before any heavy processing.
* All errors are logged and swallowed; the loop never crashes.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import os
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Callable, Coroutine, Dict, List, Optional

# MUST be set before cv2 is imported to force TCP for all RTSP streams
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2

from app.config import settings
from app.ingestion.schemas import DevicePayload
from app.video.blob_detector import BlobDetector
from app.video.centroid_tracker import CentroidTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detection dataclass — consumed by the triangulation pipeline
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    device_id: str
    object_id: str        # camera-scoped, e.g. "cam_01_obj0001"
    x_norm: float         # [0, 1] horizontal position (0=left, 1=right)
    y_norm: float         # [0, 1] vertical position  (0=top,  1=bottom)
    timestamp: float      # UTC unix timestamp
    camera_lat: float
    camera_lon: float
    camera_bearing: float
    camera_fov: float


# ---------------------------------------------------------------------------
# Internal per-camera state
# ---------------------------------------------------------------------------

@dataclass
class _CameraState:
    payload: DevicePayload
    detector: BlobDetector
    tracker: CentroidTracker
    frame_queue: Queue[Any]
    frame_counter: int = 0
    running: bool = True
    capture_thread: threading.Thread | None = None


# ---------------------------------------------------------------------------
# StreamManager
# ---------------------------------------------------------------------------

AsyncCallback = Callable[[Detection], Coroutine[Any, Any, None]]

_TARGET_FPS: float = 30.0
_FRAME_SLEEP: float = 1.0 / _TARGET_FPS


class StreamManager:
    """
    Manages one async task per registered camera.

    Usage::

        manager = StreamManager()
        manager.on_detection(my_async_handler)
        await manager.register_camera(payload)
        await manager.run()
    """

    def __init__(self) -> None:
        self._cameras: Dict[str, _CameraState] = {}
        self._tasks: Dict[str, asyncio.Task] = {}  # type: ignore[type-arg]
        self._callbacks: List[AsyncCallback] = []
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_detection(self, callback: AsyncCallback) -> None:
        """Register an async callback that receives each :class:`Detection`."""
        self._callbacks.append(callback)

    async def register_camera(self, payload: DevicePayload) -> None:
        """
        Add or update a camera stream.

        If a stream for ``payload.device_id`` is already active it is
        gracefully replaced (old task cancelled, new one started).
        """
        device_id = payload.device_id

        if device_id in self._cameras:
            logger.info("StreamManager: updating camera %s", device_id)
            await self._cancel_camera(device_id)

        state = _CameraState(
            payload=payload,
            detector=BlobDetector(
                min_area=50.0,
                max_area=50_000.0,
                blur_kernel=5,
            ),
            tracker=CentroidTracker(
                camera_id=device_id,
                max_disappeared=settings.FRAME_SKIP * 5,
                max_distance=0.15,
            ),
            frame_queue=Queue(maxsize=2),
        )
        self._cameras[device_id] = state

        if self._running:
            self._spawn_task(device_id)

    async def run(self) -> None:
        """
        Start the stream manager.  Spawns one async task per registered camera
        and then loops until :meth:`stop` is called.
        """
        self._running = True
        for device_id in self._cameras:
            self._spawn_task(device_id)

        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self._shutdown_all()

    async def stop(self) -> None:
        """Signal all camera loops to stop and await their completion."""
        self._running = False
        await self._shutdown_all()

    # ------------------------------------------------------------------
    # Private – task management
    # ------------------------------------------------------------------

    def _spawn_task(self, device_id: str) -> None:
        task = asyncio.create_task(
            self._camera_loop(device_id), name=f"stream-{device_id}"
        )
        self._tasks[device_id] = task
        task.add_done_callback(
            lambda t: self._on_task_done(device_id, t)
        )

    def _on_task_done(self, device_id: str, task: asyncio.Task) -> None:  # type: ignore[type-arg]
        exc = task.exception() if not task.cancelled() else None
        if exc is not None:
            logger.error(
                "StreamManager: task for %s raised an unhandled exception: %s",
                device_id, exc, exc_info=exc,
            )

    async def _cancel_camera(self, device_id: str) -> None:
        if device_id in self._cameras:
            self._cameras[device_id].running = False
        if device_id in self._tasks:
            self._tasks[device_id].cancel()
            try:
                await self._tasks[device_id]
            except (asyncio.CancelledError, Exception):
                pass
            del self._tasks[device_id]
        self._cameras.pop(device_id, None)

    async def _shutdown_all(self) -> None:
        for device_id in list(self._cameras.keys()):
            self._cameras[device_id].running = False
        for task in self._tasks.values():
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()

    # ------------------------------------------------------------------
    # Private – per-camera capture loop
    # ------------------------------------------------------------------

    async def _camera_loop(self, device_id: str) -> None:
        """Async task: pulls frames from the thread's queue and processes them."""
        state = self._cameras.get(device_id)
        if state is None:
            return

        rtsp_url = state.payload.rtsp_url
        logger.info("StreamManager: starting loop for %s → %s", device_id, rtsp_url)

        # Spawn the persistent capture thread
        state.capture_thread = threading.Thread(
            target=self._capture_thread_func,
            args=(state, rtsp_url),
            daemon=True,
            name=f"capture-{device_id}"
        )
        state.capture_thread.start()

        while state.running:
            try:
                frame = state.frame_queue.get_nowait()
            except Empty:
                await asyncio.sleep(0.01)
                continue

            await self._process_frame(state, frame)
            await asyncio.sleep(0)  # yield control

        if state.capture_thread.is_alive():
            state.capture_thread.join(timeout=1.0)
            
        logger.info("StreamManager: loop stopped for %s", device_id)

    # ------------------------------------------------------------------
    # Private – frame capture (runs in a persistent thread)
    # ------------------------------------------------------------------

    @staticmethod
    def _capture_thread_func(state: _CameraState, rtsp_url: str) -> None:
        """
        Thread target: keeps VideoCapture open, reads frames, handles frame skipping,
        and pushes valid frames to the queue.
        """
        cap = cv2.VideoCapture(rtsp_url)

        while state.running:
            if not cap.isOpened():
                logger.warning("StreamManager: attempting to reconnect to %s", rtsp_url)
                cap.release()
                time.sleep(1.0)
                cap = cv2.VideoCapture(rtsp_url)
                continue

            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                logger.warning("StreamManager: bad frame from %s, reconnecting", rtsp_url)
                cap.release()
                time.sleep(0.5)
                cap = cv2.VideoCapture(rtsp_url)
                continue

            # Frame skip: only process every FRAME_SKIP-th frame
            state.frame_counter += 1
            if state.frame_counter % settings.FRAME_SKIP != 0:
                continue

            try:
                # Push to queue, dropping the frame if queue is full
                state.frame_queue.put(frame, block=False)
            except Exception:  # queue.Full
                pass

        cap.release()

    # ------------------------------------------------------------------
    # Private – detection pipeline
    # ------------------------------------------------------------------

    async def _process_frame(
        self, state: _CameraState, frame: Any
    ) -> None:
        """Blob-detect → track → emit Detection callbacks."""
        blobs = state.detector.detect(frame)
        centroids = [(b.x_norm, b.y_norm) for b in blobs]
        active = state.tracker.update(centroids)

        now = time.time()
        camera = state.payload.camera

        detections = [
            Detection(
                device_id=state.payload.device_id,
                object_id=oid,
                x_norm=x,
                y_norm=y,
                timestamp=now,
                camera_lat=camera.lat,
                camera_lon=camera.lon,
                camera_bearing=camera.bearing_deg,
                camera_fov=camera.fov_deg,
            )
            for oid, (x, y) in active.items()
        ]

        for det in detections:
            await self._fire_callbacks(det)

    async def _fire_callbacks(self, detection: Detection) -> None:
        for cb in self._callbacks:
            try:
                await cb(detection)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "StreamManager: callback %s raised: %s", cb, exc, exc_info=exc
                )

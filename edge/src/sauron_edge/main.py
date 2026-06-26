"""
Sauron Edge — main entry point.

Component lifecycle:
  1. Load configuration (CONFIGURATION_FILE_PATH or default path)
  2. Start camera background capture thread
  3. Start sensor reader thread (GPS + IMU)
  4. Connect to Greengrass IPC
  5. Start monitoring dashboard on port 5000
  6. Main loop:
       a. Read latest frame from camera
       b. Run MOG2 background subtraction
       c. Update shared state (for dashboard)
       d. At the configured publish rate, build a TelemetryPayload and publish
  7. On SIGINT/SIGTERM: drain threads, close connections, log final stats, exit 0

Threading model:
  - raspi-capture / rtsp-capture thread: drains V4L2/RTSP buffer, stores latest frame
  - sensor-reader thread: polls GPS UART + IMU I2C, stores latest state
  - dashboard thread (daemon): Flask server on port 5000, reads SharedState
  - Main thread: owns detection (MOG2 is stateful and not thread-safe) and publishing
"""

from __future__ import annotations

import logging
import signal
import subprocess
import sys
import time
from typing import Optional

import cv2

from sauron_edge.camera import make_camera
from sauron_edge.config import get_config_file_path, get_configuration
from sauron_edge.dashboard import start_dashboard
from sauron_edge.detection import DetectionResult, MOG2Detector
from sauron_edge.image_uploader import ImageUploader
from sauron_edge.publisher import GreengrassPublisher
from sauron_edge.schemas import TelemetryPayload
from sauron_edge.sensors import SensorReader
from sauron_edge.shared_state import SharedState
from sauron_edge.stability import DetectionStabilizer

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    """
    Configure root logger.  All module loggers inherit this config.
    Level is DEBUG so that every subsystem's verbose output is captured —
    operators can tail the component log and see exactly what's happening.
    """
    fmt = "%(asctime)s.%(msecs)03d [%(levelname)-8s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"

    logging.basicConfig(
        level=logging.DEBUG,
        format=fmt,
        datefmt=datefmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Third-party libraries that are excessively chatty at DEBUG
    for noisy in ("awscrt", "awsiot", "botocore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Per-frame/per-message subsystems — suppress DEBUG spam; summaries in main loop
    for chatty in (
        "sauron_edge.sensors",
        "sauron_edge.detection",
        "sauron_edge.shared_state",
        "sauron_edge.publisher",
        "sauron_edge.dashboard",
    ):
        logging.getLogger(chatty).setLevel(logging.INFO)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901
    _configure_logging()

    logger.info("=" * 60)
    logger.info("Sauron Edge — starting up")
    logger.info("=" * 60)

    # ------------------------------------------------------------------ #
    # 0. Shared state (must be created before dashboard and main loop)
    # ------------------------------------------------------------------ #
    state = SharedState()
    logger.info("SharedState: initialised")

    # ------------------------------------------------------------------ #
    # 1. Configuration
    # ------------------------------------------------------------------ #
    cfg = get_configuration()

    try:
        from pathlib import Path as _Path

        _raw_yaml = _Path(get_config_file_path()).read_text(encoding="utf-8")
        logger.info("config.yaml contents:\n%s", _raw_yaml)
    except Exception as _e:
        logger.warning("Could not read config.yaml for display: %s", _e)

    logger.info(
        "Configuration summary:\n"
        "  camera_source    : %s\n"
        "  camera_rtsp_url  : %s\n"
        "  camera_dimensions: %s\n"
        "  camera_fov       : %.1f°\n"
        "  camera_fps       : %d\n"
        "  publish_rate_hz  : %.1f\n"
        "  detection        : learning_rate=%.3f var_threshold=%.1f "
        "min_area=%.1f morph_kernel=%d\n"
        "  gps_port         : %s @ %d baud\n"
        "  imu_enabled      : %s",
        cfg.camera_source,
        cfg.camera_rtsp_url or "(not set)",
        cfg.camera_dimensions,
        cfg.camera_fov,
        cfg.camera_fps,
        cfg.publish_rate_hz,
        cfg.detection.learning_rate,
        cfg.detection.var_threshold,
        cfg.detection.min_contour_area,
        cfg.detection.morph_kernel_size,
        cfg.sensors.gps_port,
        cfg.sensors.gps_baud,
        cfg.sensors.imu_enabled,
    )

    # ------------------------------------------------------------------ #
    # 2. Signal handling
    # ------------------------------------------------------------------ #
    shutdown_requested = False

    def _handle_signal(signum: int, _frame: object) -> None:
        nonlocal shutdown_requested
        sig_name = signal.Signals(signum).name
        logger.info("Signal %s received — requesting graceful shutdown", sig_name)
        shutdown_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    logger.info("Signal handlers installed for SIGINT and SIGTERM")

    # ------------------------------------------------------------------ #
    # 3. Build subsystems
    # ------------------------------------------------------------------ #
    logger.info("Building subsystems...")

    logger.info(
        "Creating camera: source=%s%s",
        cfg.camera_source,
        f" url={cfg.camera_rtsp_url}" if cfg.camera_source == "rtsp" else "",
    )
    camera = make_camera(cfg)

    detector = MOG2Detector(
        learning_rate=cfg.detection.learning_rate,
        var_threshold=cfg.detection.var_threshold,
        min_contour_area=cfg.detection.min_contour_area,
        morph_kernel_size=cfg.detection.morph_kernel_size,
        process_scale=cfg.detection.process_scale,
        blur_kernel_size=cfg.detection.blur_kernel_size,
    )

    sensor_reader = SensorReader(
        gps_port=cfg.sensors.gps_port,
        gps_baud=cfg.sensors.gps_baud,
        imu_enabled=cfg.sensors.imu_enabled,
        gps_default_lat=cfg.sensors.gps_default_lat,
        gps_default_lon=cfg.sensors.gps_default_lon,
        imu_default_heading=cfg.sensors.imu_default_heading,
        imu_default_pitch=cfg.sensors.imu_default_pitch,
        imu_default_roll=cfg.sensors.imu_default_roll,
        gps_override=cfg.sensors.gps_override,
        gps_override_lat=cfg.sensors.gps_override_lat,
        gps_override_lon=cfg.sensors.gps_override_lon,
        imu_override=cfg.sensors.imu_override,
        imu_override_heading=cfg.sensors.imu_override_heading,
        imu_override_pitch=cfg.sensors.imu_override_pitch,
        imu_override_roll=cfg.sensors.imu_override_roll,
    )

    stabilizer = DetectionStabilizer(
        min_stable_frames=cfg.detection.min_stable_frames,
    )

    publisher = GreengrassPublisher()

    # Image uploader — optional, only active when image_upload.enabled is true
    import os as _os
    _img_bucket = cfg.image_upload.s3_bucket or _os.environ.get("AWS_S3_BUCKET", "")
    image_uploader: Optional[ImageUploader] = None
    if cfg.image_upload.enabled:
        _img_region = cfg.image_upload.s3_region or _os.environ.get("AWS_DEFAULT_REGION", "") or _os.environ.get("AWS_REGION", "")
        image_uploader = ImageUploader(
            bucket=_img_bucket,
            prefix=cfg.image_upload.s3_prefix,
            min_interval_s=cfg.image_upload.min_interval_s,
            jpeg_quality=cfg.image_upload.jpeg_quality,
            region=_img_region,
            always_upload=cfg.image_upload.always_upload,
        )
    else:
        logger.info("ImageUploader: disabled (image_upload.enabled=false in config.yaml)")

    # ------------------------------------------------------------------ #
    # 4. Start subsystems
    # ------------------------------------------------------------------ #

    # Camera — will raise immediately on device open failure
    logger.info("Starting camera (%s)...", cfg.camera_source)
    camera.start()
    logger.info("Camera started — connection happens in background thread")

    # Sensor reader — always starts; individual sensors are optional
    logger.info("Starting sensor reader thread...")
    sensor_reader.start()
    logger.info("Sensor reader thread launched")

    # Greengrass IPC — attempt connection; failures are non-fatal (will retry on publish)
    logger.info("Connecting to Greengrass IPC...")
    if publisher.connect():
        logger.info(
            "Greengrass IPC connected — publishing to topic: %s", publisher.topic
        )
    else:
        logger.warning(
            "Greengrass IPC connection failed at startup — will retry on first publish. "
            "If running outside Greengrass, this is expected."
        )

    # Start command listener for telemetry toggle (non-fatal if Greengrass unavailable)
    publisher.start_command_listener()

    # ------------------------------------------------------------------ #
    # 4b. Start monitoring dashboard
    # ------------------------------------------------------------------ #
    start_dashboard(state, port=cfg.dashboard_port)
    logger.info("Dashboard: available at http://0.0.0.0:%d", cfg.dashboard_port)

    # ------------------------------------------------------------------ #
    # 5. Main processing loop
    # ------------------------------------------------------------------ #

    publish_interval_s = 1.0 / cfg.publish_rate_hz
    last_publish_monotonic: float = 0.0
    last_publish_ts: float | None = None

    # Counters for the final stats log
    total_frames_processed: int = 0
    total_detections: int = 0
    total_publishes: int = 0
    total_publish_failures: int = 0

    # CPU temperature — polled every 10 s via vcgencmd
    _cpu_temp_c: float = 0.0
    _last_temp_poll: float = 0.0
    _TEMP_POLL_INTERVAL_S: float = 10.0

    # Per-10s frame summary
    _last_summary_mono: float = time.monotonic()
    _summary_frames: int = 0
    _summary_detections: int = 0
    _SUMMARY_INTERVAL_S: float = 10.0

    # Config auto-refresh every 1 s — fast enough to catch typo fixes while camera is down
    _last_config_refresh_mono: float = time.monotonic()
    _CONFIG_REFRESH_INTERVAL_S: float = 15.0

    # Camera hardware info (read once after start)
    _cam_fps: float = float(cfg.camera_fps)
    _cam_w: int = cfg.camera_width()
    _cam_h: int = cfg.camera_height()
    if hasattr(camera, "actual_fps"):
        _cam_fps = camera.actual_fps  # type: ignore[attr-defined]
        _cam_w = camera.actual_width  # type: ignore[attr-defined]
        _cam_h = camera.actual_height  # type: ignore[attr-defined]

    logger.info(
        "=" * 60 + "\n"
        "Main loop running — publish interval %.3fs (%.1f Hz)\n" + "=" * 60,
        publish_interval_s,
        cfg.publish_rate_hz,
    )

    frame_skip_log_counter = 0  # for throttling "no frame yet" log

    try:
        while not shutdown_requested:
            loop_start = time.monotonic()

            # -- Camera health check --
            if not camera.is_alive():
                logger.error("Camera capture thread is no longer alive — shutting down")
                break

            # -- Config refresh (runs even when no frame is available) --
            _now_mono = time.monotonic()
            if _now_mono - _last_config_refresh_mono >= _CONFIG_REFRESH_INTERVAL_S:
                _last_config_refresh_mono = _now_mono
                try:
                    new_cfg = get_configuration(refetch=True)
                    publish_interval_s = 1.0 / new_cfg.publish_rate_hz

                    camera_changed = (
                        new_cfg.camera_source != cfg.camera_source
                        or new_cfg.camera_rtsp_url != cfg.camera_rtsp_url
                        or new_cfg.camera_dimensions != cfg.camera_dimensions
                    )

                    cfg = new_cfg

                    # Apply detection params immediately — no restart needed
                    detector.apply_config(
                        learning_rate=cfg.detection.learning_rate,
                        var_threshold=cfg.detection.var_threshold,
                        min_contour_area=cfg.detection.min_contour_area,
                        morph_kernel_size=cfg.detection.morph_kernel_size,
                        process_scale=cfg.detection.process_scale,
                        blur_kernel_size=cfg.detection.blur_kernel_size,
                    )
                    stabilizer.apply_config(
                        min_stable_frames=cfg.detection.min_stable_frames,
                    )

                    if camera_changed:
                        logger.info(
                            "Config refresh: camera config changed → restarting camera "
                            "(source=%s url=%s)",
                            cfg.camera_source,
                            cfg.camera_rtsp_url or "(none)",
                        )
                        camera.stop()
                        camera = make_camera(cfg)
                        try:
                            camera.start()
                            if hasattr(camera, "actual_fps"):
                                _cam_fps = camera.actual_fps  # type: ignore[attr-defined]
                                _cam_w = camera.actual_width  # type: ignore[attr-defined]
                                _cam_h = camera.actual_height  # type: ignore[attr-defined]
                            logger.info("Config refresh: camera restarted successfully")
                        except Exception as _cam_err:
                            logger.error(
                                "Config refresh: camera restart failed — %s",
                                _cam_err,
                            )

                    logger.info(
                        "Config refreshed — camera_source=%s publish_rate=%.1fHz",
                        cfg.camera_source,
                        cfg.publish_rate_hz,
                    )
                except Exception as _cfg_err:
                    logger.warning(
                        "Config refresh failed — keeping previous config: %s", _cfg_err
                    )

            # -- Read latest frame --
            frame = camera.read()
            if frame is None:
                frame_skip_log_counter += 1
                if frame_skip_log_counter % 50 == 1:
                    logger.info(
                        "Waiting for camera frame (attempt %d)...",
                        frame_skip_log_counter,
                    )
                time.sleep(0.01)
                continue

            if frame_skip_log_counter > 0:
                logger.info(
                    "Camera frame available after %d wait iterations",
                    frame_skip_log_counter,
                )
                frame_skip_log_counter = 0

            total_frames_processed += 1

            # -- Periodic checks (temp, summary, config refresh) --
            _now_mono = time.monotonic()
            if _now_mono - _last_temp_poll >= _TEMP_POLL_INTERVAL_S:
                _last_temp_poll = _now_mono
                try:
                    _tp = subprocess.run(
                        ["vcgencmd", "measure_temp"],
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                    # output format: "temp=73.1'C"
                    _raw = _tp.stdout.strip()
                    _cpu_temp_c = float(_raw.split("=")[1].rstrip("'C"))
                    logger.debug("CPU temperature: %.1f°C", _cpu_temp_c)
                except Exception as _te:
                    logger.debug("Could not read CPU temperature: %s", _te)

            # -- Run detection --
            try:
                result = detector.process(frame)
            except cv2.error as _cv_err:
                logger.error(
                    "Unhandled cv2.error in detector.process — skipping frame: %s",
                    _cv_err,
                )
                result = DetectionResult(ncoords=[])

            # -- Apply temporal stability filter --
            stable_indices = stabilizer.update(result.ncoords)
            stable_ncoords = [result.ncoords[i] for i in stable_indices]
            stable_bboxes = [
                result.bboxes[i] for i in stable_indices if i < len(result.bboxes)
            ]

            total_detections += len(stable_ncoords)
            _summary_frames += 1
            _summary_detections += len(stable_ncoords)

            # -- 10-second frame summary --
            if _now_mono - _last_summary_mono >= _SUMMARY_INTERVAL_S:
                current_sensors_snap = sensor_reader.get_state()
                logger.info(
                    "Frame summary (last %.0fs): frames=%d detections=%d | "
                    "GPS locked=%s lat=%.6f lon=%.6f | "
                    "IMU heading=%.1f° pitch=%.1f° roll=%.1f° cal=%s | "
                    "cpu_temp=%.1f°C",
                    _now_mono - _last_summary_mono,
                    _summary_frames,
                    _summary_detections,
                    current_sensors_snap.gps_locked,
                    current_sensors_snap.lat,
                    current_sensors_snap.lon,
                    current_sensors_snap.heading,
                    current_sensors_snap.pitch,
                    current_sensors_snap.roll,
                    current_sensors_snap.imu_calibration_status,
                    _cpu_temp_c,
                )
                _last_summary_mono = _now_mono
                _summary_frames = 0
                _summary_detections = 0

            # -- Update shared state for dashboard (every frame) --
            ncoords_as_dicts = [nc.model_dump() for nc in stable_ncoords]
            state.update_frame(frame, ncoords_as_dicts, stable_bboxes)

            # Update health every frame (cheap lock + copy)
            current_sensors = sensor_reader.get_state()
            state.update_health(
                camera_alive=camera.is_alive(),
                sensor_thread_alive=sensor_reader.is_alive(),
                gps_locked=current_sensors.gps_locked,
                imu_calibrated=current_sensors.imu_calibrated,
                publisher_connected=publisher._connected,
                frames_processed=total_frames_processed,
                total_detections=total_detections,
                total_publishes=total_publishes,
                total_publish_failures=total_publish_failures,
                last_publish_ts=last_publish_ts,
                camera_fps=_cam_fps,
                camera_w=_cam_w,
                camera_h=_cam_h,
                cpu_temp_c=_cpu_temp_c,
            )

            # -- Upload snapshot if enabled --
            snapshot_s3_key: Optional[str] = None
            if image_uploader is not None:
                snapshot_s3_key = image_uploader.maybe_upload(
                    frame=frame,
                    device_id=publisher.device_id,
                    has_detections=bool(stable_ncoords),
                )

            # -- Publish at configured rate --
            now = time.monotonic()
            time_since_last_publish = now - last_publish_monotonic

            if time_since_last_publish >= publish_interval_s:
                # Check global telemetry toggle (set via server command)
                if not publisher.telemetry_enabled:
                    logger.debug("Publish suppressed — telemetry disabled via command")
                    last_publish_monotonic = now
                # Optionally suppress MQTT message when nothing stable is detected
                elif cfg.detection.suppress_empty_publishes and not stable_ncoords:
                    logger.debug(
                        "Publish suppressed — no stable detections (suppress_empty_publishes=True)"
                    )
                    last_publish_monotonic = now
                else:
                    sensors = sensor_reader.get_state()

                    payload = TelemetryPayload.build(
                        lat=sensors.lat,
                        lon=sensors.lon,
                        heading=sensors.heading,
                        pitch=sensors.pitch,
                        roll=sensors.roll,
                        fov=cfg.camera_fov,
                        ncoords=stable_ncoords,
                        fps=round(_cam_fps, 1),
                        resolution_w=_cam_w,
                        resolution_h=_cam_h,
                        cpu_temp_c=round(_cpu_temp_c, 1) if _cpu_temp_c else None,
                        snapshot_s3_key=snapshot_s3_key,
                    )

                    logger.debug(
                        "Telemetry publish #%d — "
                        "ncoords=%d (stable) raw=%d | "
                        "GPS locked=%s lat=%.6f lon=%.6f sats=%d | "
                        "IMU heading=%.1f° pitch=%.1f° roll=%.1f° cal=%s | "
                        "frames_since_last=%d detections_total=%d",
                        total_publishes + 1,
                        len(payload.ncoords),
                        len(result.ncoords),
                        sensors.gps_locked,
                        sensors.lat,
                        sensors.lon,
                        sensors.gps_sats,
                        sensors.heading,
                        sensors.pitch,
                        sensors.roll,
                        sensors.imu_calibration_status,
                        total_frames_processed,
                        total_detections,
                    )

                    success = publisher.publish(payload.model_dump())

                    if success:
                        total_publishes += 1
                        last_publish_ts = time.time()
                        state.record_telemetry(payload.model_dump())
                    else:
                        total_publish_failures += 1
                        logger.warning(
                            "Publish failed — total_failures=%d total_successes=%d",
                            total_publish_failures,
                            total_publishes,
                        )

                    last_publish_monotonic = now

            # -- Frame-rate pacing --
            # Sleep just enough to match the configured camera FPS.
            # The camera thread runs independently, so this only throttles
            # how often the main loop reads+processes frames.
            elapsed = time.monotonic() - loop_start
            target_frame_time = 1.0 / cfg.camera_fps
            sleep_s = max(0.0, target_frame_time - elapsed)
            if sleep_s > 0:
                time.sleep(sleep_s)

    except Exception as exc:
        logger.critical(
            "Unhandled exception in main loop — initiating emergency shutdown: %s",
            exc,
            exc_info=True,
        )

    # ------------------------------------------------------------------ #
    # 6. Graceful shutdown
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("Sauron Edge — shutting down")
    logger.info("=" * 60)

    logger.info(
        "Final stats:\n"
        "  frames processed : %d\n"
        "  total detections : %d\n"
        "  publishes OK     : %d\n"
        "  publish failures : %d\n"
        "  publisher stats  : published=%d failures=%d",
        total_frames_processed,
        total_detections,
        total_publishes,
        total_publish_failures,
        publisher.publish_count,
        publisher.failure_count,
    )

    logger.info("Stopping camera...")
    camera.stop()

    logger.info("Stopping sensor reader...")
    sensor_reader.stop()
    sensor_reader.join(timeout=5.0)
    if sensor_reader.is_alive():
        logger.warning(
            "Sensor reader thread did not exit within timeout — continuing shutdown"
        )

    logger.info("Disconnecting from Greengrass IPC...")
    publisher.disconnect()

    logger.info("=" * 60)
    logger.info("Sauron Edge — shut down cleanly. Goodbye.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

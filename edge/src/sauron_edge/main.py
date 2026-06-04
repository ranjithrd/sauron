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
import sys
import time

from sauron_edge.camera import make_camera
from sauron_edge.config import get_configuration
from sauron_edge.dashboard import start_dashboard
from sauron_edge.detection import MOG2Detector
from sauron_edge.publisher import GreengrassPublisher
from sauron_edge.schemas import TelemetryPayload
from sauron_edge.sensors import SensorReader
from sauron_edge.shared_state import SharedState

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

    camera = make_camera(cfg)

    detector = MOG2Detector(
        learning_rate=cfg.detection.learning_rate,
        var_threshold=cfg.detection.var_threshold,
        min_contour_area=cfg.detection.min_contour_area,
        morph_kernel_size=cfg.detection.morph_kernel_size,
    )

    sensor_reader = SensorReader(
        gps_port=cfg.sensors.gps_port,
        gps_baud=cfg.sensors.gps_baud,
        imu_enabled=cfg.sensors.imu_enabled,
    )

    publisher = GreengrassPublisher()

    # ------------------------------------------------------------------ #
    # 4. Start subsystems
    # ------------------------------------------------------------------ #

    # Camera — will raise immediately on device open failure
    logger.info("Starting camera (%s)...", cfg.camera_source)
    try:
        camera.start()
        logger.info("Camera started successfully")
    except Exception as exc:
        logger.critical("Camera startup failed — cannot continue: %s", exc, exc_info=True)
        sys.exit(1)

    # Sensor reader — always starts; individual sensors are optional
    logger.info("Starting sensor reader thread...")
    sensor_reader.start()
    logger.info("Sensor reader thread launched")

    # Greengrass IPC — attempt connection; failures are non-fatal (will retry on publish)
    logger.info("Connecting to Greengrass IPC...")
    if publisher.connect():
        logger.info("Greengrass IPC connected — publishing to topic: %s", publisher.topic)
    else:
        logger.warning(
            "Greengrass IPC connection failed at startup — will retry on first publish. "
            "If running outside Greengrass, this is expected."
        )

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

    logger.info(
        "=" * 60 + "\n"
        "Main loop running — publish interval %.3fs (%.1f Hz)\n" + "=" * 60,
        publish_interval_s, cfg.publish_rate_hz,
    )

    frame_skip_log_counter = 0  # for throttling "no frame yet" log

    try:
        while not shutdown_requested:
            loop_start = time.monotonic()

            # -- Camera health check --
            if not camera.is_alive():
                logger.error(
                    "Camera capture thread is no longer alive — shutting down"
                )
                break

            # -- Read latest frame --
            frame = camera.read()
            if frame is None:
                frame_skip_log_counter += 1
                if frame_skip_log_counter % 50 == 1:
                    logger.info(
                        "Waiting for first camera frame (attempt %d)...",
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

            # -- Run detection --
            result = detector.process(frame)
            total_detections += len(result.ncoords)

            logger.debug(
                "Frame #%d: detections=%d raw_contours=%d rejected_area=%d fg_pixels=%d",
                total_frames_processed,
                len(result.ncoords),
                result.n_raw_contours,
                result.n_rejected_area,
                result.foreground_pixel_count,
            )

            # -- Update shared state for dashboard (every frame) --
            ncoords_as_dicts = [nc.model_dump() for nc in result.ncoords]
            state.update_frame(frame, ncoords_as_dicts)

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
            )

            # -- Publish at configured rate --
            now = time.monotonic()
            time_since_last_publish = now - last_publish_monotonic

            if time_since_last_publish >= publish_interval_s:
                sensors = sensor_reader.get_state()

                payload = TelemetryPayload.build(
                    lat=sensors.lat,
                    lon=sensors.lon,
                    heading=sensors.heading,
                    pitch=sensors.pitch,
                    roll=sensors.roll,
                    fov=cfg.camera_fov,
                    ncoords=result.ncoords,
                )

                logger.info(
                    "Telemetry publish #%d — "
                    "ncoords=%d | "
                    "GPS locked=%s lat=%.6f lon=%.6f sats=%d | "
                    "IMU heading=%.1f° pitch=%.1f° roll=%.1f° cal=%s | "
                    "frames_since_last=%d detections_total=%d",
                    total_publishes + 1,
                    len(payload.ncoords),
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
                        total_publish_failures, total_publishes,
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
        logger.warning("Sensor reader thread did not exit within timeout — continuing shutdown")

    logger.info("Disconnecting from Greengrass IPC...")
    publisher.disconnect()

    logger.info("=" * 60)
    logger.info("Sauron Edge — shut down cleanly. Goodbye.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

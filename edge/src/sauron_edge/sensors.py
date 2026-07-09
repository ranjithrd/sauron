"""
Hardware sensor reader — GPS (UART/NMEA) and IMU (I2C/BNO055).

Both sensors are optional. If a sensor fails to initialise, the thread logs
a warning and continues running with zero/default values for that sensor.
This allows the pipeline to function (and publish) even on hardware that
only has a camera attached.

Thread safety: all reads go through a threading.Lock.  get_state() returns
a cheap shallow copy — callers never hold the lock.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from copy import copy
from dataclasses import dataclass
from pathlib import Path

from sauron_edge.config import get_config_file_path

logger = logging.getLogger(__name__)

# Seconds between sensor poll iterations
_DEFAULT_POLL_INTERVAL_S = 0.05  # 20 Hz

# Raw NMEA sentences are mirrored to a small rolling file next to config.yaml
# (not into the Greengrass component log — that would be way too much volume).
_GPS_LOG_MAX_LINES = 100


# ---------------------------------------------------------------------------
# Shared state dataclass
# ---------------------------------------------------------------------------


@dataclass
class SensorState:
    """
    Snapshot of the latest sensor readings.
    All fields default to safe zeros so callers always get a valid object
    even before the first successful sensor read.
    """

    # GPS
    lat: float = 0.0
    """Latitude in decimal degrees. 0.0 if no GPS fix."""

    lon: float = 0.0
    """Longitude in decimal degrees. 0.0 if no GPS fix."""

    gps_sats: int = 0
    """Number of tracked satellites. 0 if no fix."""

    gps_locked: bool = False
    """True when the GPS module reports a valid fix."""

    # GPS — actual sensor reading, never touched by gps_override.
    # `lat`/`lon`/`gps_locked` above are the *effective* values (what gets
    # published); these `gps_actual_*` fields are the ground truth so a
    # dashboard/operator can always see real fix status even when an
    # override is masking it for the published telemetry.
    gps_actual_lat: float = 0.0
    gps_actual_lon: float = 0.0
    gps_actual_sats: int = 0
    gps_actual_locked: bool = False

    gps_override_active: bool = False
    """True if this device is configured to override GPS with a fixed position."""

    # IMU
    heading: float = 0.0
    """Compass bearing in degrees (0 = North). 0.0 if IMU unavailable."""

    pitch: float = 0.0
    """Pitch (nose-up/down). 0.0 if IMU unavailable."""

    roll: float = 0.0
    """Roll (sideways tilt). 0.0 if IMU unavailable."""

    imu_calibrated: bool = False
    """True when system calibration level >= 1."""

    imu_calibration_status: str = "Sys:0 G:0 A:0 M:0"
    """Human-readable calibration string, e.g. 'Sys:3 G:3 A:3 M:3'."""

    # IMU — actual sensor reading, never touched by imu_override. Same split
    # as gps_actual_*: heading/pitch/roll/imu_calibrated above are the
    # *effective* values (what gets published); these mirror the real BNO055
    # reading regardless of override.
    imu_actual_heading: float = 0.0
    imu_actual_pitch: float = 0.0
    imu_actual_roll: float = 0.0
    imu_actual_calibrated: bool = False
    imu_actual_calibration_status: str = "Sys:0 G:0 A:0 M:0"

    imu_override_active: bool = False
    """True if this device is configured to override IMU orientation with fixed values."""


# ---------------------------------------------------------------------------
# Reader thread
# ---------------------------------------------------------------------------


class SensorReader(threading.Thread):
    """
    Long-lived daemon thread that polls GPS and IMU continuously.

    Usage::

        reader = SensorReader(gps_port="/dev/serial0", gps_baud=9600, imu_enabled=True)
        reader.start()
        ...
        state = reader.get_state()   # thread-safe snapshot
        ...
        reader.stop()
        reader.join(timeout=3.0)
    """

    def __init__(
        self,
        gps_port: str,
        gps_baud: int,
        imu_enabled: bool,
        *,
        gps_default_lat: float = 0.0,
        gps_default_lon: float = 0.0,
        imu_default_heading: float = 0.0,
        imu_default_pitch: float = 0.0,
        imu_default_roll: float = 0.0,
        gps_override: bool = False,
        gps_override_lat: float = 0.0,
        gps_override_lon: float = 0.0,
        imu_override: bool = False,
        imu_override_heading: float = 0.0,
        imu_override_pitch: float = 0.0,
        imu_override_roll: float = 0.0,
        poll_interval: float = _DEFAULT_POLL_INTERVAL_S,
    ) -> None:
        super().__init__(name="sensor-reader", daemon=True)
        self._gps_port = gps_port
        self._gps_baud = gps_baud
        self._imu_enabled = imu_enabled
        self._poll_interval = poll_interval

        # Override flags
        self._gps_override = gps_override
        self._gps_override_lat = gps_override_lat
        self._gps_override_lon = gps_override_lon
        self._imu_override = imu_override
        self._imu_override_heading = imu_override_heading
        self._imu_override_pitch = imu_override_pitch
        self._imu_override_roll = imu_override_roll

        # Raw NMEA line mirror — last _GPS_LOG_MAX_LINES lines, written next to config.yaml
        self._gps_log_path = Path(get_config_file_path()).parent / "gps.log"
        self._gps_log_lines: deque[str] = deque(maxlen=_GPS_LOG_MAX_LINES)

        # Initialise state with config-provided defaults (used when sensor is absent/unfixed).
        # gps_actual_lat/lon start at the same fallback — they're only ever
        # overwritten by a real NMEA fix, never by gps_override.
        self._state = SensorState(
            lat=gps_default_lat,
            lon=gps_default_lon,
            gps_actual_lat=gps_default_lat,
            gps_actual_lon=gps_default_lon,
            heading=imu_default_heading,
            pitch=imu_default_pitch,
            roll=imu_default_roll,
            imu_actual_heading=imu_default_heading,
            imu_actual_pitch=imu_default_pitch,
            imu_actual_roll=imu_default_roll,
            gps_override_active=gps_override,
            imu_override_active=imu_override,
        )
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API (called from the main thread)
    # ------------------------------------------------------------------

    def get_state(self) -> SensorState:
        """Return a copy of the current sensor state. Never blocks for long."""
        with self._lock:
            return copy(self._state)

    def _log_raw_gps_line(self, line: str) -> None:
        """Mirror a raw NMEA sentence to gps.log, keeping only the last N lines."""
        self._gps_log_lines.append(line)
        try:
            self._gps_log_path.write_text(
                "\n".join(self._gps_log_lines) + "\n", encoding="utf-8"
            )
        except Exception as exc:
            logger.debug("SensorReader: failed to write %s — %s", self._gps_log_path, exc)

    def stop(self) -> None:
        """Signal the thread to exit cleanly."""
        logger.info("SensorReader: stop requested")
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------

    def run(self) -> None:  # noqa: C901  (complexity is unavoidable here)
        logger.info(
            "SensorReader: starting — gps_port=%s gps_baud=%d imu_enabled=%s",
            self._gps_port,
            self._gps_baud,
            self._imu_enabled,
        )

        # ---- Attempt GPS init ----
        ser = None
        gps_active = False
        try:
            import serial  # pyserial

            ser = serial.Serial(self._gps_port, baudrate=self._gps_baud, timeout=1)
            gps_active = True
            logger.info(
                "SensorReader: GPS serial port %s opened at %d baud",
                self._gps_port,
                self._gps_baud,
            )
        except Exception as exc:
            logger.warning(
                "SensorReader: GPS unavailable — will publish zero coordinates. Reason: %s",
                exc,
            )

        # pynmea2 is required for GPS parsing
        pynmea2 = None
        if gps_active:
            try:
                import pynmea2 as _pynmea2

                pynmea2 = _pynmea2
                logger.info("SensorReader: pynmea2 loaded for NMEA parsing")
            except ImportError:
                logger.warning(
                    "SensorReader: pynmea2 not importable — GPS parsing disabled"
                )
                gps_active = False

        # ---- Attempt IMU init ----
        imu_sensor = None
        imu_active = False
        if self._imu_enabled:
            try:
                import adafruit_bno055
                import board  # adafruit-blinka
                import busio

                i2c = busio.I2C(board.SCL, board.SDA)
                imu_sensor = adafruit_bno055.BNO055_I2C(i2c)

                # --- Load persisted calibration data ---
                _CAL_FILE = "/root/.config/sauron/bno055_cal.json"
                try:
                    import json as _json

                    with open(_CAL_FILE, "r") as _f:
                        _cal_data = _json.load(_f)
                    # JSON deserialises as list[int]; the BNO055 setter expects bytes
                    imu_sensor.calibration = bytes(_cal_data)
                    logger.info(
                        "SensorReader: BNO055 calibration loaded from %s", _CAL_FILE
                    )
                except FileNotFoundError:
                    logger.info(
                        "SensorReader: no saved BNO055 calibration — starting fresh"
                    )
                except Exception as _cal_err:
                    logger.warning(
                        "SensorReader: failed to load BNO055 calibration from %s — %s",
                        _CAL_FILE,
                        _cal_err,
                    )

                imu_active = True
                logger.info("SensorReader: BNO055 IMU initialised over I2C")
            except Exception as exc:
                logger.warning(
                    "SensorReader: IMU unavailable — will publish zero orientation. Reason: %s",
                    exc,
                )
        else:
            logger.info("SensorReader: IMU disabled in config — skipping init")

        logger.info(
            "SensorReader: entering poll loop — gps_active=%s imu_active=%s poll_interval=%.3fs",
            gps_active,
            imu_active,
            self._poll_interval,
        )

        imu_read_count = 0
        gps_read_count = 0
        loop_count = 0
        _imu_cal_saved = (
            False  # save calibration once per session when fully calibrated
        )

        while not self._stop_event.is_set():
            loop_count += 1

            # ---- Read IMU ----
            if imu_active and imu_sensor is not None:
                try:
                    euler = imu_sensor.euler
                    sys_cal, gyro_cal, accel_cal, mag_cal = (
                        imu_sensor.calibration_status
                    )
                    cal_str = f"Sys:{sys_cal} G:{gyro_cal} A:{accel_cal} M:{mag_cal}"

                    with self._lock:
                        if euler is not None and euler.count(None) == 0:
                            self._state.imu_actual_heading = round(float(euler[0]) % 360.0, 2)
                            self._state.imu_actual_roll = round(float(euler[1]), 2)
                            self._state.imu_actual_pitch = round(float(euler[2]), 2)
                        self._state.imu_actual_calibrated = sys_cal >= 1
                        self._state.imu_actual_calibration_status = cal_str

                    imu_read_count += 1
                    logger.debug(
                        "SensorReader [loop=%d]: IMU heading=%.2f roll=%.2f pitch=%.2f cal=%s",
                        loop_count,
                        self._state.imu_actual_heading,
                        self._state.imu_actual_roll,
                        self._state.imu_actual_pitch,
                        cal_str,
                    )

                    # --- Persist calibration when fully calibrated (once per session) ---
                    if (
                        not _imu_cal_saved
                        and sys_cal == 3
                        and gyro_cal == 3
                        and accel_cal == 3
                        and mag_cal == 3
                    ):
                        try:
                            import json as _json
                            import os as _os

                            _CAL_FILE = "/root/.config/sauron/bno055_cal.json"
                            _os.makedirs(_os.path.dirname(_CAL_FILE), exist_ok=True)
                            with open(_CAL_FILE, "w") as _f:
                                _json.dump(list(imu_sensor.calibration), _f)
                            _imu_cal_saved = True
                            logger.info(
                                "SensorReader: BNO055 fully calibrated (3,3,3,3) — "
                                "calibration saved to %s",
                                _CAL_FILE,
                            )
                        except Exception as _save_err:
                            logger.warning(
                                "SensorReader: failed to save BNO055 calibration — %s",
                                _save_err,
                            )

                except Exception as exc:
                    logger.debug(
                        "SensorReader [loop=%d]: IMU read error (transient) — %s",
                        loop_count,
                        exc,
                    )

            # ---- Read GPS ----
            if gps_active and ser is not None and pynmea2 is not None:
                try:
                    bytes_waiting = ser.in_waiting
                    if bytes_waiting > 0:
                        raw_line = (
                            ser.readline().decode("utf-8", errors="ignore").strip()
                        )

                        if raw_line:
                            self._log_raw_gps_line(raw_line)

                        logger.debug(
                            "SensorReader [loop=%d]: NMEA line — %s",
                            loop_count,
                            raw_line,
                        )

                        if raw_line.startswith("$GPGGA"):
                            try:
                                msg = pynmea2.parse(raw_line)
                                if msg.is_valid:
                                    with self._lock:
                                        self._state.gps_actual_lat = round(float(msg.latitude), 6)
                                        self._state.gps_actual_lon = round(float(msg.longitude), 6)
                                        self._state.gps_actual_sats = int(msg.num_sats)
                                        self._state.gps_actual_locked = True
                                    gps_read_count += 1
                                    logger.debug(
                                        "SensorReader [loop=%d]: GPS fix — lat=%.6f lon=%.6f sats=%d",
                                        loop_count,
                                        self._state.gps_actual_lat,
                                        self._state.gps_actual_lon,
                                        self._state.gps_actual_sats,
                                    )
                                else:
                                    with self._lock:
                                        self._state.gps_actual_locked = False
                                    logger.debug(
                                        "SensorReader [loop=%d]: $GPGGA received but fix invalid",
                                        loop_count,
                                    )
                            except pynmea2.ParseError as exc:
                                logger.debug(
                                    "SensorReader [loop=%d]: NMEA parse error — %s",
                                    loop_count,
                                    exc,
                                )
                    else:
                        logger.debug(
                            "SensorReader [loop=%d]: GPS serial — no bytes waiting",
                            loop_count,
                        )

                except Exception as exc:
                    logger.debug(
                        "SensorReader [loop=%d]: GPS read error (transient) — %s",
                        loop_count,
                        exc,
                    )

            # ---- Compute effective GPS/IMU (override wins) vs. actual (always real) ----
            # gps_actual_*/imu_actual_* above are ground truth from the sensors and are
            # never touched here. The non-"actual" fields are the *effective* values
            # used for telemetry: override wins when configured, otherwise they simply
            # mirror the actual reading.
            with self._lock:
                if self._gps_override:
                    self._state.lat = self._gps_override_lat
                    self._state.lon = self._gps_override_lon
                    self._state.gps_locked = True
                else:
                    self._state.lat = self._state.gps_actual_lat
                    self._state.lon = self._state.gps_actual_lon
                    self._state.gps_locked = self._state.gps_actual_locked
                self._state.gps_sats = self._state.gps_actual_sats

                if self._imu_override:
                    self._state.heading = self._imu_override_heading
                    self._state.pitch = self._imu_override_pitch
                    self._state.roll = self._imu_override_roll
                    self._state.imu_calibrated = True
                else:
                    self._state.heading = self._state.imu_actual_heading
                    self._state.pitch = self._state.imu_actual_pitch
                    self._state.roll = self._state.imu_actual_roll
                    self._state.imu_calibrated = self._state.imu_actual_calibrated
                self._state.imu_calibration_status = self._state.imu_actual_calibration_status

            # Periodic info log so we know the thread is alive
            if loop_count % 200 == 0:
                state_snap = self.get_state()
                logger.info(
                    "SensorReader: heartbeat [loop=%d] — GPS effective locked=%s (%.6f, %.6f) sats=%d "
                    "(override=%s, actual locked=%s (%.6f, %.6f)) | "
                    "IMU effective heading=%.1f pitch=%.1f roll=%.1f cal=%s "
                    "(override=%s, actual heading=%.1f pitch=%.1f roll=%.1f cal=%s) | "
                    "imu_reads=%d gps_reads=%d",
                    loop_count,
                    state_snap.gps_locked,
                    state_snap.lat,
                    state_snap.lon,
                    state_snap.gps_sats,
                    state_snap.gps_override_active,
                    state_snap.gps_actual_locked,
                    state_snap.gps_actual_lat,
                    state_snap.gps_actual_lon,
                    state_snap.heading,
                    state_snap.pitch,
                    state_snap.roll,
                    state_snap.imu_calibration_status,
                    state_snap.imu_override_active,
                    state_snap.imu_actual_heading,
                    state_snap.imu_actual_pitch,
                    state_snap.imu_actual_roll,
                    state_snap.imu_actual_calibration_status,
                    imu_read_count,
                    gps_read_count,
                )

            time.sleep(self._poll_interval)

        # ---- Cleanup ----
        if ser is not None:
            try:
                ser.close()
                logger.info("SensorReader: GPS serial port closed")
            except Exception as exc:
                logger.debug("SensorReader: error closing serial port — %s", exc)

        logger.info(
            "SensorReader: thread exited — total loops=%d imu_reads=%d gps_reads=%d",
            loop_count,
            imu_read_count,
            gps_read_count,
        )

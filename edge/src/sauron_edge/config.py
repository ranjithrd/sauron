"""
Configuration management for Sauron Edge.

Mirrors the Go agent's approach:
- Path resolved via CONFIGURATION_FILE_PATH env var, defaulting to ~/.config/sauron/config.yaml
- File is created with defaults if it doesn't exist
- In-process cache with optional refetch
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

# Load .env from the working directory at import time
load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class DetectionConfig(BaseModel):
    """MOG2 background subtraction parameters."""

    learning_rate: float = -1.0
    """MOG2 learning rate. -1 = automatic (recommended)."""

    min_contour_area: float = 500.0
    """Minimum foreground blob area in pixels². Smaller blobs are ignored."""

    var_threshold: float = 16.0
    """MOG2 varThreshold. Lower values = more sensitive to change."""

    morph_kernel_size: int = 7
    """Size of the elliptical morphological kernel used for noise removal."""

    blur_kernel_size: int = 5
    """
    Size of the Gaussian blur kernel applied to each frame before MOG2.
    Must be odd. Reduces CMOS sensor noise and prevents spurious detections
    in static scenes. Set to 1 to disable.
    """

    process_scale: float = 0.5
    """
    Fraction of the capture resolution used internally for MOG2 processing.
    0.5 means 320x240 when capturing at 640x480 — reduces memory by 4× with
    negligible accuracy loss since NCoords are normalised anyway.
    Set to 1.0 to process at full resolution (higher memory pressure).
    """

    min_stable_frames: int = 3
    """
    Minimum number of consecutive frames a detection must appear in before it is
    forwarded to the publisher. Increases this to suppress flicker/noise; set to 1
    to disable stability filtering entirely.
    """

    suppress_empty_publishes: bool = False
    """
    When True, skip the MQTT publish entirely on frames where the stability filter
    produces no confirmed detections. Reduces IoT Core message volume when the
    scene is quiet. Heartbeat at publish_rate_hz is lost when enabled.
    """


class SensorsConfig(BaseModel):
    """Hardware sensor settings."""

    gps_port: str = "/dev/serial0"
    """UART device path for the GPS module."""

    gps_baud: int = 9600
    """GPS serial baud rate."""

    imu_enabled: bool = True
    """Whether to attempt to initialise the BNO055 IMU over I2C."""

    # ---- Default fallback values (used when sensor is absent / unlocked) ----

    gps_default_lat: float = 0.0
    """Fallback latitude when GPS has no fix. Used as initial state on startup."""

    gps_default_lon: float = 0.0
    """Fallback longitude when GPS has no fix. Used as initial state on startup."""

    imu_default_heading: float = 0.0
    """Fallback compass bearing when IMU is absent or uncalibrated."""

    imu_default_pitch: float = 0.0
    """Fallback pitch when IMU is absent or uncalibrated."""

    imu_default_roll: float = 0.0
    """Fallback roll when IMU is absent or uncalibrated."""

    # ---- Override flags (force fixed values even if sensor is healthy) ----

    gps_override: bool = False
    """
    When True, always publish gps_override_lat / gps_override_lon regardless of
    what the GPS module reports. Use when the GPS is fried but you know the unit's
    fixed position.
    """

    gps_override_lat: float = 0.0
    """Override latitude. Effective only when gps_override is True."""

    gps_override_lon: float = 0.0
    """Override longitude. Effective only when gps_override is True."""

    imu_override: bool = False
    """
    When True, always publish imu_override_heading / pitch / roll regardless of
    what the IMU reports. Use when the IMU is fried but you know the fixed orientation.
    """

    imu_override_heading: float = 0.0
    """Override compass bearing. Effective only when imu_override is True."""

    imu_override_pitch: float = 0.0
    """Override pitch. Effective only when imu_override is True."""

    imu_override_roll: float = 0.0
    """Override roll. Effective only when imu_override is True."""


# ---------------------------------------------------------------------------
# Root configuration model
# ---------------------------------------------------------------------------


class Configuration(BaseModel):
    """Full runtime configuration for the edge component."""

    camera_source: str = "raspi"
    """Video source. One of 'raspi' (V4L2 /dev/video0) or 'rtsp'."""

    camera_rtsp_url: str = ""
    """RTSP stream URL. Required when camera_source == 'rtsp'."""

    camera_dimensions: str = "640x480"
    """Frame dimensions as 'WxH'. Applied to both capture and output."""

    camera_fov: float = 90.0
    """Horizontal field of view in degrees. Published in every telemetry message."""

    camera_fps: int = 24
    """Desired camera capture frame rate."""

    publish_rate_hz: float = 5.0
    """How many telemetry messages per second to publish to IoT Core."""

    dashboard_port: int = 5000
    """Port for the local monitoring dashboard (Flask). Set to 0 to disable."""

    detection: DetectionConfig = DetectionConfig()
    sensors: SensorsConfig = SensorsConfig()

    @field_validator("camera_source")
    @classmethod
    def _validate_camera_source(cls, v: str) -> str:
        if v not in ("raspi", "rtsp"):
            raise ValueError(f"camera_source must be 'raspi' or 'rtsp', got {v!r}")
        return v

    @field_validator("camera_rtsp_url")
    @classmethod
    def _validate_rtsp_url(cls, v: str, info) -> str:
        # Only validate presence — format check happens at camera start time
        return v

    def camera_width(self) -> int:
        """Parsed frame width in pixels."""
        return int(self.camera_dimensions.split("x")[0])

    def camera_height(self) -> int:
        """Parsed frame height in pixels."""
        return int(self.camera_dimensions.split("x")[1])


# ---------------------------------------------------------------------------
# File helpers (mirrors Go's FetchOrCreateConfigurationFile pattern)
# ---------------------------------------------------------------------------

_config_lock = threading.Lock()
_cached_config: Optional[Configuration] = None


def get_config_file_path() -> str:
    """
    Return the resolved config file path.
    Checks CONFIGURATION_FILE_PATH env var first, then uses the default
    ~/.config/sauron/config.yaml — identical logic to the Go agent.
    """
    env_path = os.environ.get("CONFIGURATION_FILE_PATH", "").strip()
    if env_path:
        return os.path.expanduser(env_path)
    return os.path.expanduser("~/.config/sauron/config.yaml")


def _ensure_config_file(path: str) -> None:
    """Create the config file with safe defaults if it does not exist."""
    p = Path(path)
    if p.exists():
        return

    logger.warning("Config file not found at %s — creating with default values", path)
    p.parent.mkdir(parents=True, exist_ok=True)

    defaults = Configuration()
    content = yaml.dump(
        defaults.model_dump(), default_flow_style=False, sort_keys=False
    )
    p.write_text(content, encoding="utf-8")
    logger.info("Default configuration written to %s", path)


def get_configuration(refetch: bool = False) -> Configuration:
    """
    Return the parsed Configuration, reading from disk the first time
    (or whenever refetch=True).  Thread-safe.
    """
    global _cached_config

    with _config_lock:
        if _cached_config is not None and not refetch:
            logger.debug("Config: returning cached configuration")
            return _cached_config

        path = get_config_file_path()
        _ensure_config_file(path)

        logger.info("Config: loading from %s", path)
        raw_text = Path(path).read_text(encoding="utf-8")
        data: dict = yaml.safe_load(raw_text) or {}

        _cached_config = Configuration.model_validate(data)

        logger.info(
            "Config: loaded — source=%s dimensions=%s fov=%.1f° fps=%d publish=%.1fHz",
            _cached_config.camera_source,
            _cached_config.camera_dimensions,
            _cached_config.camera_fov,
            _cached_config.camera_fps,
            _cached_config.publish_rate_hz,
        )
        logger.debug("Config: full dump — %s", _cached_config.model_dump())
        return _cached_config

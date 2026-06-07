"""
Telemetry payload schemas.

These are intentionally kept in sync with the server's ingestion/schemas.py.
The server subscribes to devices/+/telemetry and validates against these types,
so field names and types must not diverge.
"""

from __future__ import annotations

import time
from typing import List, Optional

from pydantic import BaseModel


class GPS(BaseModel):
    lat: float
    """Latitude in decimal degrees."""

    lon: float
    """Longitude in decimal degrees."""


class IMU(BaseModel):
    heading: float
    """Compass bearing in degrees (0 = North, 90 = East)."""

    pitch: float
    """Nose-up / nose-down tilt in degrees. Negative = tilted down."""

    roll: float
    """Sideways tilt in degrees."""


class CameraInfo(BaseModel):
    gps: GPS
    imu: IMU
    fov: float
    """Horizontal field of view in degrees."""


class NCoord(BaseModel):
    xnorm: float
    """Normalised horizontal position in frame — [0.0, 1.0], left to right."""

    ynorm: float
    """Normalised vertical position in frame — [0.0, 1.0], top to bottom."""


class TelemetryPayload(BaseModel):
    cameras: CameraInfo
    ncoords: List[NCoord]
    timestamp: float
    """Unix epoch seconds (float) at the moment of assembly."""

    fps: Optional[float] = None
    """Actual camera FPS negotiated with V4L2."""

    resolution_w: Optional[int] = None
    resolution_h: Optional[int] = None
    """Actual camera resolution negotiated with V4L2."""

    cpu_temp_c: Optional[float] = None
    """CPU/SoC temperature in Celsius, sampled every ~10 seconds."""

    @classmethod
    def build(
        cls,
        *,
        lat: float,
        lon: float,
        heading: float,
        pitch: float,
        roll: float,
        fov: float,
        ncoords: List[NCoord],
        fps: Optional[float] = None,
        resolution_w: Optional[int] = None,
        resolution_h: Optional[int] = None,
        cpu_temp_c: Optional[float] = None,
    ) -> "TelemetryPayload":
        """Convenience factory — assembles the full payload from flat sensor readings."""
        return cls(
            cameras=CameraInfo(
                gps=GPS(lat=lat, lon=lon),
                imu=IMU(heading=heading, pitch=pitch, roll=roll),
                fov=fov,
            ),
            ncoords=ncoords,
            timestamp=time.time(),
            fps=fps,
            resolution_w=resolution_w,
            resolution_h=resolution_h,
            cpu_temp_c=cpu_temp_c,
        )

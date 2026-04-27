from typing import List

from pydantic import BaseModel


class GPS(BaseModel):
    lat: float
    lon: float


class IMU(BaseModel):
    heading: float   # compass bearing (0=North, 90=East)
    pitch: float     # tilt up/down in degrees (negative = tilted down)
    roll: float      # sideways tilt in degrees


class CameraInfo(BaseModel):
    gps: GPS
    imu: IMU
    fov: float       # horizontal field of view in degrees


class NCoord(BaseModel):
    xnorm: float     # [0, 1] horizontal position in frame
    ynorm: float     # [0, 1] vertical position in frame


class TelemetryPayload(BaseModel):
    cameras: CameraInfo
    ncoords: List[NCoord]
    timestamp: float

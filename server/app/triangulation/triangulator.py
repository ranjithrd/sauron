"""
triangulator.py — pure math, no I/O, no state.

Build a 3D line-of-sight ray from camera heading/pitch/roll, project it onto
the ground plane, then triangulate as the midpoint between two cameras' ground
projections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from app.config import settings

_METERS_PER_DEG_LAT: float = 111_320.0
_UPWARD_EPS: float = 1e-9
_HORIZON_FALLBACK_DZ: float = -1e-3
_CONFIDENCE_DISTANCE_SCALE_M: float = 25.0


@dataclass(frozen=True)
class CameraRay:
    lat: float
    lon: float
    bearing_deg: float
    ground_offset_m: tuple[float, float]   # (east_m, north_m)


@dataclass(frozen=True)
class TriangulatedPosition:
    lat: float
    lon: float
    confidence: float    # [0, 1] — higher when two projected points agree


def compute_object_bearing(
    camera_heading_deg: float,
    fov_deg: float,
    xnorm: float,
) -> float:
    """Return absolute compass bearing from camera heading + horizontal offset."""
    x_offset_deg = (xnorm - 0.5) * fov_deg
    return (camera_heading_deg + x_offset_deg) % 360.0


def build_ray(
    camera_lat: float,
    camera_lon: float,
    camera_heading: float,
    camera_pitch: float,
    camera_roll: float,
    camera_fov: float,
    xnorm: float,
) -> Optional[CameraRay]:
    """Construct a pitch/roll-corrected camera ray projected to the ground plane."""
    bearing = compute_object_bearing(camera_heading, camera_fov, xnorm)
    bearing_rad = math.radians(bearing)

    # ENU direction from bearing (flat horizon reference).
    dx = math.sin(bearing_rad)  # East
    dy = math.cos(bearing_rad)  # North

    # Apply pitch around East-West axis.
    pitch_rad = math.radians(camera_pitch)
    dy_pitched = dy * math.cos(pitch_rad)
    dz_pitched = dy * math.sin(pitch_rad)

    # Apply roll around North-South axis.
    roll_rad = math.radians(camera_roll)
    dx_rolled = dx * math.cos(roll_rad) - dz_pitched * math.sin(roll_rad)
    dz_final = dx * math.sin(roll_rad) + dz_pitched * math.cos(roll_rad)
    dy_final = dy_pitched

    # Keep zero-pitch rays triangulatable by nudging exact-horizon rays slightly down.
    if abs(dz_final) <= _UPWARD_EPS:
        dz_final = _HORIZON_FALLBACK_DZ

    # Upward rays do not hit the ground in front of camera.
    if dz_final > 0.0:
        return None

    t = -settings.CAMERA_HEIGHT_M / dz_final
    if t <= 0.0:
        return None

    east_m = t * dx_rolled
    north_m = t * dy_final

    return CameraRay(
        lat=camera_lat,
        lon=camera_lon,
        bearing_deg=bearing,
        ground_offset_m=(east_m, north_m),
    )


def intersect_rays(
    ray1: CameraRay,
    ray2: CameraRay,
) -> Optional[TriangulatedPosition]:
    """Triangulate by midpoint between the two projected ground points."""
    p1_lat, p1_lon = _ground_point_lat_lon(ray1)
    p2_lat, p2_lon = _ground_point_lat_lon(ray2)

    ref_lat_rad = math.radians((p1_lat + p2_lat) / 2.0)
    meters_per_deg_lon = _METERS_PER_DEG_LAT * math.cos(ref_lat_rad)
    if abs(meters_per_deg_lon) < 1e-12:
        return None

    x1 = p1_lon * meters_per_deg_lon
    y1 = p1_lat * _METERS_PER_DEG_LAT
    x2 = p2_lon * meters_per_deg_lon
    y2 = p2_lat * _METERS_PER_DEG_LAT

    midpoint_x = (x1 + x2) / 2.0
    midpoint_y = (y1 + y2) / 2.0

    separation_m = math.hypot(x2 - x1, y2 - y1)
    confidence = 1.0 / (1.0 + (separation_m / _CONFIDENCE_DISTANCE_SCALE_M))
    confidence = max(0.0, min(1.0, confidence))

    return TriangulatedPosition(
        lat=midpoint_y / _METERS_PER_DEG_LAT,
        lon=midpoint_x / meters_per_deg_lon,
        confidence=confidence,
    )


def _ground_point_lat_lon(ray: CameraRay) -> tuple[float, float]:
    east_m, north_m = ray.ground_offset_m

    lat = ray.lat + (north_m / _METERS_PER_DEG_LAT)

    meters_per_deg_lon = _METERS_PER_DEG_LAT * math.cos(math.radians(ray.lat))
    if abs(meters_per_deg_lon) < 1e-12:
        return lat, ray.lon

    lon = ray.lon + (east_m / meters_per_deg_lon)
    return lat, lon


def flat_earth_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return approximate distance in metres between two lat/lon points."""
    ref_lat_rad = math.radians((lat1 + lat2) / 2.0)
    meters_per_deg_lon = _METERS_PER_DEG_LAT * math.cos(ref_lat_rad)
    dlat = (lat2 - lat1) * _METERS_PER_DEG_LAT
    dlon = (lon2 - lon1) * meters_per_deg_lon
    return math.sqrt(dlat ** 2 + dlon ** 2)

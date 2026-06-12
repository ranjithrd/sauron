"""
triangulator.py — pure math, no I/O, no state.

Build a 3D line-of-sight ray from camera heading/pitch/roll and find the
closest approach between two rays in 3D space (ENU coordinates).  Works for
cameras that face upward into shared airspace as well as cameras that face
downward toward the ground.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from app.config import settings

_METERS_PER_DEG_LAT: float = 111_320.0
_CONFIDENCE_DISTANCE_SCALE_M: float = 25.0


@dataclass(frozen=True)
class CameraRay:
    lat: float
    lon: float
    height_m: float              # camera height above ground (metres)
    dx: float                    # East component of unit direction vector
    dy: float                    # North component
    dz: float                    # Up component (positive = skyward)


@dataclass(frozen=True)
class TriangulatedPosition:
    lat: float
    lon: float
    altitude_m: float            # estimated altitude of tracked object above ground
    confidence: float            # [0, 1] — higher when rays converge tightly


def compute_object_bearing(
    camera_heading_deg: float,
    fov_deg: float,
    xnorm: float,
) -> float:
    """Return absolute compass bearing from camera heading + horizontal pixel offset."""
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
    """
    Construct a pitch/roll-corrected 3D ray from the camera into the scene.

    The ray direction is expressed as a unit vector in ENU (East-North-Up)
    space.  Upward-pointing rays are valid — this is an airspace tracker.
    Returns None only if the direction vector degenerates (camera pointing
    exactly at its own origin, which cannot happen physically).
    """
    bearing = compute_object_bearing(camera_heading, camera_fov, xnorm)
    bearing_rad = math.radians(bearing)

    # Horizontal ENU direction from bearing.
    dx = math.sin(bearing_rad)   # East
    dy = math.cos(bearing_rad)   # North

    # Pitch: rotate (dy, 0) toward Up around the lateral (East-West) axis.
    pitch_rad = math.radians(camera_pitch)
    dy_pitched = dy * math.cos(pitch_rad)
    dz_pitched = dy * math.sin(pitch_rad)

    # Roll: rotate (dx, dz_pitched) around the forward (North-South) axis.
    roll_rad = math.radians(camera_roll)
    dx_final = dx * math.cos(roll_rad) - dz_pitched * math.sin(roll_rad)
    dz_final = dx * math.sin(roll_rad) + dz_pitched * math.cos(roll_rad)
    dy_final = dy_pitched

    mag = math.sqrt(dx_final ** 2 + dy_final ** 2 + dz_final ** 2)
    if mag < 1e-9:
        return None

    return CameraRay(
        lat=camera_lat,
        lon=camera_lon,
        height_m=settings.CAMERA_HEIGHT_M,
        dx=dx_final / mag,
        dy=dy_final / mag,
        dz=dz_final / mag,
    )


def intersect_rays(
    ray1: CameraRay,
    ray2: CameraRay,
) -> Optional[TriangulatedPosition]:
    """
    Find the closest approach between two 3D rays and return the midpoint.

    Uses the standard skew-lines formula.  Both rays must point forward
    (t > 0) from their respective cameras.  Confidence is derived from the
    separation distance at closest approach — tightly converging rays score
    near 1.0; nearly parallel rays score near 0.
    """
    ref_lat_rad = math.radians(ray1.lat)
    meters_per_deg_lon = _METERS_PER_DEG_LAT * math.cos(ref_lat_rad)
    if abs(meters_per_deg_lon) < 1e-12:
        return None

    # Express ray2 origin in ENU metres relative to ray1 origin.
    east2 = (ray2.lon - ray1.lon) * meters_per_deg_lon
    north2 = (ray2.lat - ray1.lat) * _METERS_PER_DEG_LAT

    # Ray 1 origin (0, 0, h1); Ray 2 origin (east2, north2, h2).
    p1 = (0.0, 0.0, ray1.height_m)
    p2 = (east2, north2, ray2.height_m)
    d1 = (ray1.dx, ray1.dy, ray1.dz)
    d2 = (ray2.dx, ray2.dy, ray2.dz)

    w = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])

    a = _dot(d1, d1)
    b = _dot(d1, d2)
    c = _dot(d2, d2)
    d = _dot(d1, w)
    e = _dot(d2, w)

    denom = a * c - b * b
    if abs(denom) < 1e-9:
        # Rays are parallel — no unique intersection.
        return None

    t1 = (b * e - c * d) / denom
    t2 = (a * e - b * d) / denom

    # Object must be in front of both cameras.
    if t1 < 0.0 or t2 < 0.0:
        return None

    # Closest point on each ray.
    q1 = (p1[0] + t1 * d1[0], p1[1] + t1 * d1[1], p1[2] + t1 * d1[2])
    q2 = (p2[0] + t2 * d2[0], p2[1] + t2 * d2[1], p2[2] + t2 * d2[2])

    mid_east  = (q1[0] + q2[0]) / 2.0
    mid_north = (q1[1] + q2[1]) / 2.0
    mid_up    = (q1[2] + q2[2]) / 2.0

    separation_m = math.sqrt(
        (q1[0] - q2[0]) ** 2 + (q1[1] - q2[1]) ** 2 + (q1[2] - q2[2]) ** 2
    )
    confidence = 1.0 / (1.0 + separation_m / _CONFIDENCE_DISTANCE_SCALE_M)
    confidence = max(0.0, min(1.0, confidence))

    lat = ray1.lat + mid_north / _METERS_PER_DEG_LAT
    lon = ray1.lon + mid_east / meters_per_deg_lon

    return TriangulatedPosition(
        lat=lat,
        lon=lon,
        altitude_m=mid_up,
        confidence=confidence,
    )


def flat_earth_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate distance in metres between two lat/lon points."""
    ref_lat_rad = math.radians((lat1 + lat2) / 2.0)
    meters_per_deg_lon = _METERS_PER_DEG_LAT * math.cos(ref_lat_rad)
    dlat = (lat2 - lat1) * _METERS_PER_DEG_LAT
    dlon = (lon2 - lon1) * meters_per_deg_lon
    return math.sqrt(dlat ** 2 + dlon ** 2)


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

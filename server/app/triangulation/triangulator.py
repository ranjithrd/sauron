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
from typing import Optional, Tuple

from app.config import settings

_METERS_PER_DEG_LAT: float = 111_320.0


@dataclass(frozen=True)
class CameraRay:
    lat: float
    lon: float
    height_m: float              # camera height above ground (metres)
    dx: float                    # East component of unit direction vector
    dy: float                    # North component
    dz: float                    # Up component (positive = skyward, negative = toward ground)

    @property
    def ground_offset_m(self) -> Tuple[float, float]:
        """
        Return (east_m, north_m) — the ENU ground-plane offset from the camera
        origin where this ray intersects the ground (z=0 plane).

        Only valid when dz < 0 (ray points toward the ground).  Horizontal
        (dz == 0) and skyward (dz > 0) rays project to infinity and return
        (inf, inf).
        """
        if self.dz >= 0.0:
            return (math.inf, math.inf)
        # Ray: p(t) = (0, 0, height_m) + t*(dx, dy, dz)
        # Ground at z=0 → height_m + t*dz = 0 → t = -height_m / dz
        t_ground = -self.height_m / self.dz
        east_m  = t_ground * self.dx
        north_m = t_ground * self.dy
        return (east_m, north_m)


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
    space.  Upward-pointing rays are valid — this is an airspace tracker, so
    a camera tilted up at an object above camera height must still produce
    a usable ray for triangulation.

    Returns None only if the direction vector degenerates (camera pointing
    exactly at its own origin, which cannot happen physically).
    """
    bearing = compute_object_bearing(camera_heading, camera_fov, xnorm)
    bearing_rad = math.radians(bearing)

    # Horizontal ENU direction from bearing.
    dx = math.sin(bearing_rad)   # East
    dy = math.cos(bearing_rad)   # North

    # Pitch: rotate the horizontal (dx, dy) ray toward/away from vertical.
    # Positive pitch = tilted up, negative = tilted down toward ground.
    pitch_rad = math.radians(camera_pitch)
    # The forward component (along the bearing direction) is scaled by cos(pitch).
    # The vertical component (dz) is introduced by sin(pitch).
    dy_pitched = dy * math.cos(pitch_rad)
    dz_pitched = math.sin(pitch_rad)   # shared vertical from pitch alone
    dx_pitched = dx * math.cos(pitch_rad)

    # Roll: rotate (dx_pitched, dz_pitched) around the forward (bearing) axis.
    # Standard right-hand rule: positive roll tilts the right side of the camera
    # down.  This rotates the East-Up plane around the North (bearing) axis.
    roll_rad = math.radians(camera_roll)
    cos_r = math.cos(roll_rad)
    sin_r = math.sin(roll_rad)

    # Rotation around the forward (dy_pitched / bearing) axis:
    #   dx' =  dx·cos(r) + dz·sin(r)
    #   dz' = -dx·sin(r) + dz·cos(r)
    dx_final = dx_pitched * cos_r + dz_pitched * sin_r
    dz_final = -dx_pitched * sin_r + dz_pitched * cos_r
    dy_final = dy_pitched

    mag = math.sqrt(dx_final ** 2 + dy_final ** 2 + dz_final ** 2)
    if mag < 1e-9:
        return None

    dx_n = dx_final / mag
    dy_n = dy_final / mag
    dz_n = dz_final / mag

    return CameraRay(
        lat=camera_lat,
        lon=camera_lon,
        height_m=settings.CAMERA_HEIGHT_M,
        dx=dx_n,
        dy=dy_n,
        dz=dz_n,
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
    confidence = 1.0 / (1.0 + separation_m / settings.TRIANGULATION_CONFIDENCE_SCALE_M)
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

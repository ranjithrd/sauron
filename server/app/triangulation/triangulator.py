"""
triangulator.py — pure math, no I/O, no state.

Flat-earth approximation throughout (valid for areas < ~5 km).
All angles are compass bearings: 0=North, 90=East, 180=South, 270=West.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_METERS_PER_DEG_LAT: float = 111_320.0   # ~constant globally
_PARALLEL_THRESHOLD: float = 1e-10        # denominator below this → parallel rays


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraRay:
    lat: float
    lon: float
    bearing_deg: float   # absolute compass bearing of this ray


@dataclass(frozen=True)
class TriangulatedPosition:
    lat: float
    lon: float
    confidence: float    # [0, 1] — sine of angle between rays


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_object_bearing(
    camera_bearing_deg: float,
    fov_deg: float,
    x_norm: float,
) -> float:
    """
    Return the absolute compass bearing from a camera to a detected object.

    Parameters
    ----------
    camera_bearing_deg : float
        The direction the camera centre points (0=North, 90=East …).
    fov_deg : float
        Horizontal field of view of the camera in degrees.
    x_norm : float
        Normalised horizontal position of the object in the frame [0, 1].
        0 = left edge, 0.5 = centre, 1 = right edge.

    Returns
    -------
    float
        Absolute compass bearing in [0, 360).
    """
    offset = (x_norm - 0.5) * fov_deg
    return (camera_bearing_deg + offset) % 360.0


def build_ray(
    camera_lat: float,
    camera_lon: float,
    camera_bearing: float,
    camera_fov: float,
    x_norm: float,
) -> CameraRay:
    """
    Construct a :class:`CameraRay` from camera metadata and a detection position.
    """
    bearing = compute_object_bearing(camera_bearing, camera_fov, x_norm)
    return CameraRay(lat=camera_lat, lon=camera_lon, bearing_deg=bearing)


def intersect_rays(
    ray1: CameraRay,
    ray2: CameraRay,
) -> Optional[TriangulatedPosition]:
    """
    Find the real-world intersection of two camera rays using the flat-earth
    approximation.

    Returns ``None`` if:
    - The rays are parallel (denominator < ``_PARALLEL_THRESHOLD``).
    - The intersection lies *behind* either camera (``t < 0`` or ``s < 0``).

    Parameters
    ----------
    ray1, ray2 : CameraRay
        The two rays to intersect.

    Returns
    -------
    TriangulatedPosition | None
    """
    # ------------------------------------------------------------------
    # 1. Convert lat/lon → flat-earth metres
    #    Use the average latitude for longitude scaling.
    # ------------------------------------------------------------------
    ref_lat_rad = math.radians((ray1.lat + ray2.lat) / 2.0)
    meters_per_deg_lon = _METERS_PER_DEG_LAT * math.cos(ref_lat_rad)

    x1 = ray1.lon * meters_per_deg_lon
    y1 = ray1.lat * _METERS_PER_DEG_LAT
    x2 = ray2.lon * meters_per_deg_lon
    y2 = ray2.lat * _METERS_PER_DEG_LAT

    # ------------------------------------------------------------------
    # 2. Direction unit vectors (compass bearing → Cartesian)
    #    dx = sin(bearing), dy = cos(bearing)
    # ------------------------------------------------------------------
    b1_rad = math.radians(ray1.bearing_deg)
    b2_rad = math.radians(ray2.bearing_deg)

    dx1 = math.sin(b1_rad)
    dy1 = math.cos(b1_rad)
    dx2 = math.sin(b2_rad)
    dy2 = math.cos(b2_rad)

    # ------------------------------------------------------------------
    # 3. Solve:  [x1,y1] + t*[dx1,dy1] = [x2,y2] + s*[dx2,dy2]
    #
    #    Rearranged:
    #        t*dx1 - s*dx2 = x2 - x1   … (A)
    #        t*dy1 - s*dy2 = y2 - y1   … (B)
    #
    #    Matrix form:  [[dx1, -dx2], [dy1, -dy2]] * [t, s]^T = [Δx, Δy]
    #    det = dx1*(−dy2) − (−dx2)*dy1 = dx2*dy1 − dx1*dy2
    # ------------------------------------------------------------------
    denom = dx2 * dy1 - dx1 * dy2   # same magnitude as cross-product

    # ------------------------------------------------------------------
    # 4. Confidence — sine of angle between rays = |cross product|
    # ------------------------------------------------------------------
    confidence = abs(denom)   # unit vectors, so this is already in [0, 1]

    if abs(denom) < _PARALLEL_THRESHOLD:
        return None   # rays are parallel

    dx = x2 - x1
    dy = y2 - y1

    # Cramer's rule
    t = (-dx * dy2 + dx2 * dy) / denom
    s = (-dx * dy1 + dx1 * dy) / denom

    if t < 0.0 or s < 0.0:
        return None   # intersection is behind at least one camera

    # ------------------------------------------------------------------
    # 5. Intersection point → back to lat/lon
    # ------------------------------------------------------------------
    ix = x1 + t * dx1
    iy = y1 + t * dy1

    result_lat = iy / _METERS_PER_DEG_LAT
    result_lon = ix / meters_per_deg_lon

    return TriangulatedPosition(lat=result_lat, lon=result_lon, confidence=confidence)


# ---------------------------------------------------------------------------
# Internal helper (also used by correlator for distance checks)
# ---------------------------------------------------------------------------

def flat_earth_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Return the approximate distance in metres between two lat/lon points
    using the flat-earth approximation.
    """
    ref_lat_rad = math.radians((lat1 + lat2) / 2.0)
    meters_per_deg_lon = _METERS_PER_DEG_LAT * math.cos(ref_lat_rad)
    dlat = (lat2 - lat1) * _METERS_PER_DEG_LAT
    dlon = (lon2 - lon1) * meters_per_deg_lon
    return math.sqrt(dlat ** 2 + dlon ** 2)

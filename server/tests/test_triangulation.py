"""
tests/test_triangulation.py — unit tests for triangulation and correlator logic.

Run with:
    pytest tests/test_triangulation.py -v
"""

from __future__ import annotations

import math
import time
from typing import List

import pytest

from app.ingestion.mqtt_client import Detection
from app.triangulation.correlator import CorrelatedDetection, Correlator
from app.triangulation.triangulator import (
    CameraRay,
    build_ray,
    compute_object_bearing,
    flat_earth_distance_m,
    intersect_rays,
)

_METERS_PER_DEG_LAT = 111_320.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detection(
    device_id: str = "cam_01",
    object_id: str = "cam_01_obj_001",
    xnorm: float = 0.5,
    ynorm: float = 0.5,
    timestamp: float = 0.0,
    camera_lat: float = 0.0,
    camera_lon: float = 0.0,
    camera_heading: float = 0.0,
    camera_pitch: float = -10.0,
    camera_roll: float = 0.0,
    camera_fov: float = 60.0,
) -> Detection:
    return Detection(
        device_id=device_id,
        object_id=object_id,
        xnorm=xnorm,
        ynorm=ynorm,
        timestamp=timestamp,
        camera_lat=camera_lat,
        camera_lon=camera_lon,
        camera_heading=camera_heading,
        camera_pitch=camera_pitch,
        camera_roll=camera_roll,
        camera_fov=camera_fov,
    )


def _ground_point_from_ray(ray: CameraRay) -> tuple[float, float]:
    """Convert a ray's ground_offset_m to lat/lon."""
    east_m, north_m = ray.ground_offset_m
    ref_lat_rad = math.radians(ray.lat)
    meters_per_deg_lon = _METERS_PER_DEG_LAT * math.cos(ref_lat_rad)
    lat = ray.lat + (north_m / _METERS_PER_DEG_LAT)
    lon = ray.lon + (east_m / meters_per_deg_lon)
    return lat, lon


# ---------------------------------------------------------------------------
# compute_object_bearing
# ---------------------------------------------------------------------------

class TestComputeObjectBearing:

    def test_centre_equals_camera_heading(self):
        bearing = compute_object_bearing(
            camera_heading_deg=120.0,
            fov_deg=60.0,
            xnorm=0.5,
        )
        assert math.isclose(bearing, 120.0, abs_tol=1e-9)

    def test_left_edge_minus_half_fov(self):
        bearing = compute_object_bearing(
            camera_heading_deg=90.0,
            fov_deg=60.0,
            xnorm=0.0,
        )
        assert math.isclose(bearing, 60.0, abs_tol=1e-9)

    def test_right_edge_plus_half_fov(self):
        bearing = compute_object_bearing(
            camera_heading_deg=90.0,
            fov_deg=60.0,
            xnorm=1.0,
        )
        assert math.isclose(bearing, 120.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# build_ray / intersect_rays
# ---------------------------------------------------------------------------

class TestRayProjection:

    def test_zero_pitch_roll_preserves_heading_direction(self):
        """Zero pitch/roll should produce a ray along the heading — dz=0, valid for airspace."""
        ray = build_ray(
            camera_lat=0.0,
            camera_lon=0.0,
            camera_heading=90.0,
            camera_pitch=0.0,
            camera_roll=0.0,
            camera_fov=60.0,
            xnorm=0.5,
        )

        assert ray is not None
        # With pitch=0 the ray is horizontal: dz should be 0.0, dx > 0 (East)
        assert ray.dx > 0.0
        assert abs(ray.dy) < 1e-6
        assert abs(ray.dz) < 1e-6

    def test_negative_pitch_hits_ground_closer_than_zero_pitch(self):
        ray_zero = build_ray(
            camera_lat=0.0,
            camera_lon=0.0,
            camera_heading=0.0,
            camera_pitch=0.0,
            camera_roll=0.0,
            camera_fov=60.0,
            xnorm=0.5,
        )
        ray_down = build_ray(
            camera_lat=0.0,
            camera_lon=0.0,
            camera_heading=0.0,
            camera_pitch=-10.0,
            camera_roll=0.0,
            camera_fov=60.0,
            xnorm=0.5,
        )

        assert ray_zero is not None
        assert ray_down is not None

        # pitch=0 → horizontal → infinite ground distance
        # pitch=-10 → tilted down → finite ground distance
        e0, n0 = ray_zero.ground_offset_m
        e_down, n_down = ray_down.ground_offset_m
        assert math.isinf(math.hypot(e0, n0)), "Horizontal ray should project to infinity"
        assert math.isfinite(math.hypot(e_down, n_down)), "Downward ray should have finite ground hit"

    def test_upward_ray_returns_none(self):
        ray = build_ray(
            camera_lat=0.0,
            camera_lon=0.0,
            camera_heading=0.0,
            camera_pitch=8.0,
            camera_roll=0.0,
            camera_fov=60.0,
            xnorm=0.5,
        )
        assert ray is None

    def test_two_cameras_with_different_pitch_still_converge(self):
        """Different camera pitch values should still triangulate to a finite midpoint."""
        cam2_lon = 10.0 / _METERS_PER_DEG_LAT

        ray1 = build_ray(
            camera_lat=0.0,
            camera_lon=0.0,
            camera_heading=45.0,
            camera_pitch=-45.0,
            camera_roll=0.0,
            camera_fov=60.0,
            xnorm=0.5,
        )
        ray2 = build_ray(
            camera_lat=0.0,
            camera_lon=cam2_lon,
            camera_heading=315.0,
            camera_pitch=-35.0,
            camera_roll=0.0,
            camera_fov=60.0,
            xnorm=0.5,
        )

        assert ray1 is not None
        assert ray2 is not None

        pos = intersect_rays(ray1, ray2)
        assert pos is not None
        assert 0.0 <= pos.confidence <= 1.0

        # Triangulated point should be in the vicinity of both rays' ground projections.
        # Use a generous tolerance (10 m) — exact midpoint differs by method.
        p1_lat, p1_lon = _ground_point_from_ray(ray1)
        p2_lat, p2_lon = _ground_point_from_ray(ray2)
        midpoint_lat = (p1_lat + p2_lat) / 2.0
        midpoint_lon = (p1_lon + p2_lon) / 2.0

        assert flat_earth_distance_m(pos.lat, pos.lon, midpoint_lat, midpoint_lon) < 10.0


# ---------------------------------------------------------------------------
# Correlator
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCorrelator:

    async def test_same_camera_detections_not_correlated(self):
        received: List[CorrelatedDetection] = []

        async def cb(c: CorrelatedDetection) -> None:
            received.append(c)

        correlator = Correlator(on_correlated=cb)
        now = time.time()
        await correlator.add(_make_detection(device_id="cam_01", timestamp=now))
        await correlator.add(_make_detection(device_id="cam_01", timestamp=now))

        assert received == []

    async def test_cross_camera_within_window_is_correlated(self):
        received: List[CorrelatedDetection] = []

        async def cb(c: CorrelatedDetection) -> None:
            received.append(c)

        correlator = Correlator(on_correlated=cb)
        now = time.time()
        cam2_lon = 10.0 / _METERS_PER_DEG_LAT

        d1 = _make_detection(
            device_id="cam_01",
            object_id="cam_01_obj_001",
            timestamp=now,
            camera_lat=0.0,
            camera_lon=0.0,
            camera_heading=45.0,
            camera_pitch=-45.0,
        )
        d2 = _make_detection(
            device_id="cam_02",
            object_id="cam_02_obj_001",
            timestamp=now,
            camera_lat=0.0,
            camera_lon=cam2_lon,
            camera_heading=315.0,
            camera_pitch=-35.0,
        )

        await correlator.add(d1)
        await correlator.add(d2)

        assert len(received) == 1
        assert set(received[0].source_cameras) == {"cam_01", "cam_02"}
        assert received[0].confidence >= 0.3

    async def test_upward_ray_pair_is_rejected(self):
        received: List[CorrelatedDetection] = []

        async def cb(c: CorrelatedDetection) -> None:
            received.append(c)

        correlator = Correlator(on_correlated=cb)
        now = time.time()

        d1 = _make_detection(
            device_id="cam_01",
            object_id="cam_01_obj_001",
            timestamp=now,
            camera_heading=45.0,
            camera_pitch=10.0,   # strictly upward → build_ray returns None
        )
        d2 = _make_detection(
            device_id="cam_02",
            object_id="cam_02_obj_001",
            timestamp=now,
            camera_heading=315.0,
            camera_pitch=-10.0,
            camera_lon=0.001,
        )

        await correlator.add(d1)
        await correlator.add(d2)

        assert received == []

    async def test_low_confidence_pair_rejected(self):
        """Nearly-parallel rays (cameras aimed the same direction) give low confidence."""
        received: List[CorrelatedDetection] = []

        async def cb(c: CorrelatedDetection) -> None:
            received.append(c)

        correlator = Correlator(on_correlated=cb)
        now = time.time()

        # Both cameras face exactly North with identical pitch — rays are parallel.
        # parallel rays → denom ≈ 0 → intersect_rays returns None → no correlation.
        d1 = _make_detection(
            device_id="cam_01",
            object_id="cam_01_obj_001",
            timestamp=now,
            camera_lat=0.0,
            camera_lon=0.0,
            camera_heading=0.0,
            camera_pitch=-20.0,
        )
        d2 = _make_detection(
            device_id="cam_02",
            object_id="cam_02_obj_001",
            timestamp=now,
            camera_lat=0.0,
            camera_lon=0.001,
            camera_heading=0.0,   # same direction as cam_01 → parallel rays
            camera_pitch=-20.0,
        )

        await correlator.add(d1)
        await correlator.add(d2)

        assert received == []

    async def test_position_jump_rejected(self):
        received: List[CorrelatedDetection] = []

        async def cb(c: CorrelatedDetection) -> None:
            received.append(c)

        correlator = Correlator(on_correlated=cb, max_position_jump_m=1.0)

        correlator._last_positions["cam_01_obj_001"] = (0.0, 0.0)
        correlator._last_positions["cam_02_obj_001"] = (0.0, 0.0)

        now = time.time()
        cam2_lon = 10.0 / _METERS_PER_DEG_LAT

        d1 = _make_detection(
            device_id="cam_01",
            object_id="cam_01_obj_001",
            timestamp=now,
            camera_lat=0.0,
            camera_lon=0.0,
            camera_heading=45.0,
            camera_pitch=-45.0,
        )
        d2 = _make_detection(
            device_id="cam_02",
            object_id="cam_02_obj_001",
            timestamp=now,
            camera_lat=0.0,
            camera_lon=cam2_lon,
            camera_heading=315.0,
            camera_pitch=-35.0,
        )

        await correlator.add(d1)
        await correlator.add(d2)

        assert received == []

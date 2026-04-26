"""
tests/test_triangulation.py — unit tests for the triangulation module.

Run with:
    pytest tests/test_triangulation.py -v
"""

from __future__ import annotations

import math
import time
from typing import List
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from app.triangulation.correlator import CorrelatedDetection, Correlator
from app.triangulation.triangulator import (
    CameraRay,
    build_ray,
    compute_object_bearing,
    flat_earth_distance_m,
    intersect_rays,
)
from app.video.stream_manager import Detection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detection(
    device_id: str = "cam_01",
    object_id: str = "cam_01_obj0001",
    x_norm: float = 0.5,
    timestamp: float = 0.0,
    camera_lat: float = 0.0,
    camera_lon: float = 0.0,
    camera_bearing: float = 0.0,
    camera_fov: float = 60.0,
) -> Detection:
    return Detection(
        device_id=device_id,
        object_id=object_id,
        x_norm=x_norm,
        y_norm=0.5,
        timestamp=timestamp,
        camera_lat=camera_lat,
        camera_lon=camera_lon,
        camera_bearing=camera_bearing,
        camera_fov=camera_fov,
    )


# ---------------------------------------------------------------------------
# compute_object_bearing
# ---------------------------------------------------------------------------

class TestComputeObjectBearing:

    def test_centre_equals_camera_bearing(self):
        """Object at x_norm=0.5 → bearing equals camera bearing exactly."""
        bearing = compute_object_bearing(
            camera_bearing_deg=120.0, fov_deg=60.0, x_norm=0.5
        )
        assert math.isclose(bearing, 120.0, abs_tol=1e-9)

    def test_left_edge_minus_fov_half(self):
        """Object at x_norm=0.0 → bearing offset by -fov/2."""
        bearing = compute_object_bearing(
            camera_bearing_deg=90.0, fov_deg=60.0, x_norm=0.0
        )
        assert math.isclose(bearing, 60.0, abs_tol=1e-9)

    def test_right_edge_plus_fov_half(self):
        """Object at x_norm=1.0 → bearing offset by +fov/2."""
        bearing = compute_object_bearing(
            camera_bearing_deg=90.0, fov_deg=60.0, x_norm=1.0
        )
        assert math.isclose(bearing, 120.0, abs_tol=1e-9)

    def test_wraps_past_360(self):
        """Bearing wraps correctly past 360°."""
        bearing = compute_object_bearing(
            camera_bearing_deg=350.0, fov_deg=60.0, x_norm=1.0
        )
        assert math.isclose(bearing, 20.0, abs_tol=1e-9)

    def test_wraps_below_zero(self):
        """Bearing wraps correctly below 0° (stays non-negative)."""
        bearing = compute_object_bearing(
            camera_bearing_deg=10.0, fov_deg=60.0, x_norm=0.0
        )
        assert math.isclose(bearing, 340.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# intersect_rays
# ---------------------------------------------------------------------------

class TestIntersectRays:

    def test_perpendicular_rays_intersect_correctly(self):
        """
        Camera 1 at (0, 0) facing East (90°),
        Camera 2 at (0.001°N, 0.001°E) facing South (180°).

        East ray:  x↑ along lat=0.
        South ray: y↓ along lon=0.001°.
        Intersection expected near (lat=0, lon=0.001°).
        Confidence should be > 0.9 (rays are perpendicular).
        """
        ray1 = CameraRay(lat=0.0, lon=0.0, bearing_deg=90.0)      # East
        ray2 = CameraRay(lat=0.001, lon=0.001, bearing_deg=180.0)  # South

        result = intersect_rays(ray1, ray2)

        assert result is not None
        assert math.isclose(result.lat, 0.0, abs_tol=1e-5), (
            f"Expected lat≈0, got {result.lat}"
        )
        assert math.isclose(result.lon, 0.001, abs_tol=1e-5), (
            f"Expected lon≈0.001, got {result.lon}"
        )
        assert result.confidence > 0.9, (
            f"Expected confidence > 0.9 for perpendicular rays, got {result.confidence:.4f}"
        )

    def test_parallel_rays_returns_none(self):
        """Two rays pointing in the same direction never intersect."""
        ray1 = CameraRay(lat=0.0, lon=0.0, bearing_deg=45.0)
        ray2 = CameraRay(lat=0.001, lon=0.0, bearing_deg=45.0)
        assert intersect_rays(ray1, ray2) is None

    def test_anti_parallel_rays_returns_none(self):
        """Two rays pointing in exactly opposite directions are also parallel."""
        ray1 = CameraRay(lat=0.0, lon=0.0, bearing_deg=90.0)
        ray2 = CameraRay(lat=0.0, lon=0.001, bearing_deg=270.0)
        assert intersect_rays(ray1, ray2) is None

    def test_intersection_behind_camera_returns_none(self):
        """
        Camera 1 at (0, 0) facing South (180°).
        Camera 2 at (0.001°N, 0°) facing West (270°).

        The only geometric intersection is to the South of cam1, but cam1
        faces South (t>0 would put us south); actually let's verify:
        Camera 1 facing South (180°): dx=0, dy=-1 → goes south
        Camera 2 at (0.001, 0) facing West (270°): dx=-1, dy=0 → goes west

        Solve: 0 + t*0 = 0 + s*(-1) → s=0  … borderline.
               0 + t*(-1) = 111.32 + s*0  → t = -111.32

        t < 0 → behind camera 1 → None.
        """
        ray1 = CameraRay(lat=0.0, lon=0.0, bearing_deg=180.0)      # South
        ray2 = CameraRay(lat=0.001, lon=0.0, bearing_deg=270.0)    # West

        result = intersect_rays(ray1, ray2)
        assert result is None, (
            f"Expected None for behind-camera intersection, got {result}"
        )

    def test_confidence_near_zero_for_nearly_parallel(self):
        """Nearly parallel rays produce near-zero confidence."""
        ray1 = CameraRay(lat=0.0, lon=0.0, bearing_deg=89.9)
        ray2 = CameraRay(lat=0.001, lon=0.0, bearing_deg=90.1)
        result = intersect_rays(ray1, ray2)
        # May or may not intersect, but confidence must be very low
        if result is not None:
            assert result.confidence < 0.05


# ---------------------------------------------------------------------------
# Correlator — async tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCorrelator:

    async def test_same_camera_detections_not_correlated(self):
        """Two detections from the same camera must never produce a correlation."""
        received: List[CorrelatedDetection] = []

        async def cb(c: CorrelatedDetection) -> None:
            received.append(c)

        correlator = Correlator(on_correlated=cb)
        now = time.time()
        await correlator.add(_make_detection(device_id="cam_01", timestamp=now))
        await correlator.add(_make_detection(device_id="cam_01", timestamp=now))

        assert received == [], f"Expected no correlations, got {received}"

    async def test_outside_time_window_not_correlated(self):
        """Detections outside the time window must not be correlated."""
        received: List[CorrelatedDetection] = []

        async def cb(c: CorrelatedDetection) -> None:
            received.append(c)

        correlator = Correlator(on_correlated=cb)
        # DETECTION_CORRELATION_WINDOW_MS default = 200 ms
        await correlator.add(_make_detection(device_id="cam_01", timestamp=0.0))
        await correlator.add(_make_detection(device_id="cam_02", timestamp=1.0))  # 1 s later

        assert received == [], f"Expected no correlations outside window, got {received}"

    async def test_cross_camera_within_window_is_correlated(self):
        """Two detections from different cameras within the window → one CorrelatedDetection."""
        received: List[CorrelatedDetection] = []

        async def cb(c: CorrelatedDetection) -> None:
            received.append(c)

        correlator = Correlator(on_correlated=cb)
        now = time.time()

        # cam_01 facing East, cam_02 at a slightly different position facing South.
        # These produce perpendicular rays → high confidence intersection.
        d1 = _make_detection(
            device_id="cam_01",
            object_id="cam_01_obj0001",
            x_norm=0.5,
            timestamp=now,
            camera_lat=0.0,
            camera_lon=0.0,
            camera_bearing=90.0,   # East
            camera_fov=60.0,
        )
        d2 = _make_detection(
            device_id="cam_02",
            object_id="cam_02_obj0001",
            x_norm=0.5,
            timestamp=now,          # same moment
            camera_lat=0.001,
            camera_lon=0.001,
            camera_bearing=180.0,   # South
            camera_fov=60.0,
        )

        await correlator.add(d1)
        await correlator.add(d2)

        assert len(received) == 1, f"Expected 1 correlation, got {len(received)}"
        c = received[0]
        assert isinstance(c, CorrelatedDetection)
        assert set(c.source_cameras) == {"cam_01", "cam_02"}
        assert 0.0 <= c.confidence <= 1.0

    async def test_low_confidence_pair_rejected(self):
        """Nearly parallel rays (confidence < 0.3) must not trigger the callback."""
        received: List[CorrelatedDetection] = []

        async def cb(c: CorrelatedDetection) -> None:
            received.append(c)

        correlator = Correlator(on_correlated=cb)
        now = time.time()

        # Both cameras facing almost the same direction (1° apart) → near-zero confidence
        d1 = _make_detection(
            device_id="cam_01",
            object_id="cam_01_obj0001",
            x_norm=0.5,
            timestamp=now,
            camera_lat=0.0,
            camera_lon=0.0,
            camera_bearing=89.5,
            camera_fov=1.0,
        )
        d2 = _make_detection(
            device_id="cam_02",
            object_id="cam_02_obj0001",
            x_norm=0.5,
            timestamp=now,
            camera_lat=0.001,
            camera_lon=0.0,
            camera_bearing=90.5,
            camera_fov=1.0,
        )

        await correlator.add(d1)
        await correlator.add(d2)

        assert received == [], (
            f"Expected no correlation for nearly-parallel rays, got {received}"
        )

    async def test_position_jump_rejected(self):
        """
        A triangulated position that jumps more than max_position_jump_m from
        the last known position for that object_id must be rejected.

        Strategy: seed the correlator's internal position cache directly with a
        known anchor, then feed a pair whose intersection is guaranteed to be
        > 1 m away from that anchor.
        """
        received: List[CorrelatedDetection] = []

        async def cb(c: CorrelatedDetection) -> None:
            received.append(c)

        correlator = Correlator(on_correlated=cb, max_position_jump_m=1.0)

        # Seed known positions for both sides — the correlator uses the
        # incoming detection's object_id as the key, and either camera's
        # detection can be the "primary" depending on add() order.
        correlator._last_positions["cam_01_obj0001"] = (0.0, 0.0)
        correlator._last_positions["cam_02_obj0001"] = (0.0, 0.0)

        now = time.time()

        # Submit a pair whose intersection is at (lat=0, lon≈0.001°) ≈ 111 m east —
        # well beyond the 1 m jump limit.
        d1 = _make_detection(
            device_id="cam_01", object_id="cam_01_obj0001",
            x_norm=0.5, timestamp=now,
            camera_lat=0.0, camera_lon=0.0,
            camera_bearing=90.0, camera_fov=60.0,   # East
        )
        d2 = _make_detection(
            device_id="cam_02", object_id="cam_02_obj0001",
            x_norm=0.5, timestamp=now,
            camera_lat=0.001, camera_lon=0.001,
            camera_bearing=180.0, camera_fov=60.0,  # South
        )

        await correlator.add(d1)
        await correlator.add(d2)

        # Intersection ≈ 111 m away from (0, 0) → rejected
        assert received == [], (
            f"Expected callback NOT called (position jump > 1 m), got {len(received)} events"
        )

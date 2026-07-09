"""
tests/test_kalman.py — unit tests for ObjectKalmanFilter and KalmanTracker.

Run with:
    pytest tests/test_kalman.py -v
"""

from __future__ import annotations

import time
from typing import List
from unittest.mock import patch

import numpy as np
import pytest

from app.kalman.filter import KalmanState, ObjectKalmanFilter
from app.kalman.tracker import KalmanTracker, SmoothedPosition
from app.triangulation.correlator import CorrelatedDetection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_correlated(
    object_id: str = "cam_01_obj0001",
    lat: float = 1.0,
    lon: float = 1.0,
    altitude_m: float = 10.0,
    confidence: float = 0.9,
    timestamp: float = 0.0,
    source_cameras: list[str] | None = None,
) -> CorrelatedDetection:
    return CorrelatedDetection(
        object_id=object_id,
        lat=lat,
        lon=lon,
        altitude_m=altitude_m,
        confidence=confidence,
        timestamp=timestamp,
        source_cameras=source_cameras or ["cam_01", "cam_02"],
    )


# ---------------------------------------------------------------------------
# ObjectKalmanFilter tests
# ---------------------------------------------------------------------------

class TestObjectKalmanFilter:

    def test_initial_state_equals_given_position(self):
        """Filter's initial state must reflect the given lat/lon/alt with zero velocity."""
        kf = ObjectKalmanFilter(initial_lat=10.0, initial_lon=20.0, initial_alt=50.0)
        s = kf.state
        assert s.lat == pytest.approx(10.0, abs=1e-9)
        assert s.lon == pytest.approx(20.0, abs=1e-9)
        assert s.altitude_m == pytest.approx(50.0, abs=1e-9)
        assert s.vel_lat == pytest.approx(0.0, abs=1e-9)
        assert s.vel_lon == pytest.approx(0.0, abs=1e-9)
        assert s.vel_alt == pytest.approx(0.0, abs=1e-9)

    def test_single_update_output_close_to_measurement(self):
        """After one update, the filtered position should be close to the measurement."""
        kf = ObjectKalmanFilter(initial_lat=0.0, initial_lon=0.0, initial_alt=0.0)
        kf.predict()
        state = kf.update(lat=1.0, lon=2.0, alt=30.0)

        # With high initial uncertainty (P=I) the filter trusts the measurement heavily
        assert abs(state.lat - 1.0) < 0.1, f"lat too far from measurement: {state.lat}"
        assert abs(state.lon - 2.0) < 0.1, f"lon too far from measurement: {state.lon}"
        assert abs(state.altitude_m - 30.0) < 1.0, f"alt too far from measurement: {state.altitude_m}"

    def test_repeated_same_position_converges(self):
        """
        Feeding the same position repeatedly should make the estimate converge
        toward that position — each successive output should be closer.
        """
        true_lat, true_lon, true_alt = 5.0, 10.0, 25.0
        kf = ObjectKalmanFilter(initial_lat=0.0, initial_lon=0.0, initial_alt=0.0)

        errors = []
        for _ in range(10):
            kf.predict()
            state = kf.update(lat=true_lat, lon=true_lon, alt=true_alt)
            errors.append(
                abs(state.lat - true_lat) + abs(state.lon - true_lon) + abs(state.altitude_m - true_alt)
            )

        # The filter should converge overall
        first_half_avg = sum(errors[:5]) / 5
        second_half_avg = sum(errors[5:]) / 5
        assert second_half_avg < first_half_avg, (
            f"Expected convergence: first-half avg {first_half_avg:.6f} > "
            f"second-half avg {second_half_avg:.6f}"
        )

        assert errors[-1] < 0.1

    def test_constant_velocity_recovers_velocity(self):
        """
        Feeding measurements along a straight line at constant velocity should
        cause vel_lat / vel_lon to converge toward the true velocity.
        """
        true_vel_lat = 0.001   # degrees/step
        true_vel_lon = 0.002
        lat, lon, alt = 0.0, 0.0, 0.0
        kf = ObjectKalmanFilter(initial_lat=lat, initial_lon=lon, initial_alt=alt, dt=1.0)

        for _ in range(30):
            lat += true_vel_lat
            lon += true_vel_lon
            kf.predict()
            state = kf.update(lat=lat, lon=lon, alt=alt)

        # After many steps the velocity estimate should be in the right ballpark
        assert abs(state.vel_lat - true_vel_lat) < true_vel_lat * 0.5, (
            f"vel_lat {state.vel_lat:.6f} too far from true {true_vel_lat}"
        )
        assert abs(state.vel_lon - true_vel_lon) < true_vel_lon * 0.5, (
            f"vel_lon {state.vel_lon:.6f} too far from true {true_vel_lon}"
        )

    def test_altitude_velocity_recovered(self):
        """Vertical velocity (vel_alt) should be estimated from climbing altitude measurements."""
        true_vel_alt = 1.0   # metres/step
        alt = 0.0
        kf = ObjectKalmanFilter(initial_lat=0.0, initial_lon=0.0, initial_alt=alt, dt=1.0)

        for _ in range(30):
            alt += true_vel_alt
            kf.predict()
            state = kf.update(lat=0.0, lon=0.0, alt=alt)

        assert abs(state.vel_alt - true_vel_alt) < true_vel_alt * 0.5, (
            f"vel_alt {state.vel_alt:.4f} too far from true {true_vel_alt}"
        )

    def test_predict_only_advances_position(self):
        """
        If the filter has a non-zero velocity, a predict() step should move the
        position estimate in that direction without any measurement.
        """
        kf = ObjectKalmanFilter(initial_lat=0.0, initial_lon=0.0, initial_alt=0.0, dt=1.0)

        # Manually inject velocities into the state vector
        kf.x[3, 0] = 0.01    # vel_lat = +0.01 deg/s
        kf.x[4, 0] = -0.005  # vel_lon = -0.005 deg/s
        kf.x[5, 0] = 2.0     # vel_alt = +2.0 m/s

        before_lat = float(kf.x[0, 0])
        before_lon = float(kf.x[1, 0])
        before_alt = float(kf.x[2, 0])

        state = kf.predict()

        assert state.lat > before_lat, "Positive vel_lat should increase lat after predict"
        assert state.lon < before_lon, "Negative vel_lon should decrease lon after predict"
        assert state.altitude_m > before_alt, "Positive vel_alt should increase altitude after predict"

    def test_returns_kalman_state_instance(self):
        """predict() and update() must both return KalmanState dataclass instances."""
        kf = ObjectKalmanFilter(initial_lat=1.0, initial_lon=2.0, initial_alt=5.0)
        pred = kf.predict()
        assert isinstance(pred, KalmanState)
        upd = kf.update(1.0, 2.0, 5.0)
        assert isinstance(upd, KalmanState)


# ---------------------------------------------------------------------------
# KalmanTracker tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestKalmanTracker:

    async def test_new_object_spawns_filter_and_emits(self):
        """First update for a new object_id → filter spawned, callback fired once."""
        received: List[SmoothedPosition] = []

        async def cb(s: SmoothedPosition) -> None:
            received.append(s)

        tracker = KalmanTracker(on_update=cb)
        await tracker.update(_make_correlated(object_id="obj_A", lat=1.0, lon=2.0, altitude_m=15.0))

        assert len(received) == 1
        # Tracker mints its own sequential IDs (obj_0001, obj_0002, ...)
        assert len(tracker.active_ids) == 1
        track_id = tracker.active_ids[0]
        assert received[0].object_id == track_id
        assert received[0].lat == pytest.approx(1.0, abs=1e-6)
        assert received[0].lon == pytest.approx(2.0, abs=1e-6)
        assert received[0].altitude_m == pytest.approx(15.0, abs=1e-3)

    async def test_existing_object_reuses_filter(self):
        """
        Second update for the same object_id must reuse the existing filter,
        not spawn a new one.  The active_ids list should still have length 1.
        """
        received: List[SmoothedPosition] = []

        async def cb(s: SmoothedPosition) -> None:
            received.append(s)

        tracker = KalmanTracker(on_update=cb)
        await tracker.update(_make_correlated(object_id="obj_B", lat=0.0, lon=0.0, altitude_m=10.0))
        await tracker.update(_make_correlated(object_id="obj_B", lat=0.001, lon=0.001, altitude_m=12.0))

        # Both updates should be associated to the same internal track
        assert len(tracker.active_ids) == 1
        assert len(received) == 2, "Callback must fire on both updates"

    async def test_multiple_objects_tracked_independently(self):
        """Two different object_ids should each get their own filter."""
        received: List[SmoothedPosition] = []

        async def cb(s: SmoothedPosition) -> None:
            received.append(s)

        tracker = KalmanTracker(on_update=cb)
        # Two detections far apart in space → each gets its own track
        await tracker.update(_make_correlated(object_id="obj_X", lat=1.0, lon=1.0, altitude_m=5.0))
        await tracker.update(_make_correlated(object_id="obj_Y", lat=5.0, lon=5.0, altitude_m=50.0))

        assert len(tracker.active_ids) == 2
        assert len(received) == 2

    async def test_stale_object_removed_by_prune(self):
        """An object not updated for OBJECT_STALE_TIMEOUT_S seconds is pruned."""
        received: List[SmoothedPosition] = []

        async def cb(s: SmoothedPosition) -> None:
            received.append(s)

        tracker = KalmanTracker(on_update=cb)
        await tracker.update(_make_correlated(object_id="stale_obj"))

        # Tracker generates its own ID — get the actual internal ID
        assert len(tracker.active_ids) == 1
        internal_id = tracker.active_ids[0]

        # Wind the filter's last-updated time into the past
        kf, _ = tracker._filters[internal_id]
        tracker._filters[internal_id] = (kf, 0.0)   # epoch — definitely stale

        tracker.prune_stale()

        assert internal_id not in tracker.active_ids, (
            "Stale object should have been removed by prune_stale()"
        )

    async def test_recent_object_not_pruned(self):
        """An object updated very recently must survive prune_stale()."""
        received: List[SmoothedPosition] = []

        async def cb(s: SmoothedPosition) -> None:
            received.append(s)

        tracker = KalmanTracker(on_update=cb)
        await tracker.update(_make_correlated(object_id="fresh_obj"))

        tracker.prune_stale()

        assert len(tracker.active_ids) == 1, (
            "Recently updated object should NOT be pruned"
        )

    async def test_smoothed_position_has_required_fields(self):
        """SmoothedPosition emitted must have all required fields with correct types."""
        received: List[SmoothedPosition] = []

        async def cb(s: SmoothedPosition) -> None:
            received.append(s)

        tracker = KalmanTracker(on_update=cb)
        c = _make_correlated(
            object_id="obj_C",
            lat=37.77,
            lon=-122.41,
            altitude_m=20.0,
            timestamp=12345.0,
            source_cameras=["cam_01", "cam_02"],
        )
        await tracker.update(c)

        s = received[0]
        assert isinstance(s.object_id, str)
        assert isinstance(s.lat, float)
        assert isinstance(s.lon, float)
        assert isinstance(s.altitude_m, float)
        assert isinstance(s.vel_lat, float)
        assert isinstance(s.vel_lon, float)
        assert isinstance(s.vel_alt, float)
        assert isinstance(s.timestamp, float)
        assert isinstance(s.source_cameras, list)
        assert s.timestamp == pytest.approx(12345.0)
        assert s.source_cameras == ["cam_01", "cam_02"]

    async def test_3d_association_separates_objects_at_different_altitudes(self):
        """
        Two objects with the same lat/lon but very different altitudes should
        be tracked as separate objects when 3-D association is used.
        """
        received: List[SmoothedPosition] = []

        async def cb(s: SmoothedPosition) -> None:
            received.append(s)

        tracker = KalmanTracker(on_update=cb)
        # Object on the ground
        await tracker.update(_make_correlated(
            object_id="low_obj", lat=0.0, lon=0.0, altitude_m=2.0
        ))
        # Object at high altitude — same lat/lon but 500 m up
        await tracker.update(_make_correlated(
            object_id="high_obj", lat=0.0, lon=0.0, altitude_m=500.0
        ))

        # Should have spawned two separate tracks
        assert len(tracker.active_ids) == 2, (
            "Objects at very different altitudes should not be merged by association"
        )

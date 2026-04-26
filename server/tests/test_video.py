"""
tests/test_video.py — unit tests for BlobDetector and CentroidTracker.

Run with:
    pytest tests/test_video.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from app.video.blob_detector import Blob, BlobDetector
from app.video.centroid_tracker import CentroidTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _black_frame(h: int = 240, w: int = 320) -> np.ndarray:
    """Return a solid-black uint8 frame."""
    return np.zeros((h, w), dtype=np.uint8)


def _frame_with_rect(
    h: int = 240,
    w: int = 320,
    rect_x: int = 100,
    rect_y: int = 80,
    rect_w: int = 60,
    rect_h: int = 40,
) -> np.ndarray:
    """Return a black frame with one white rectangle drawn on it."""
    frame = _black_frame(h, w)
    frame[rect_y : rect_y + rect_h, rect_x : rect_x + rect_w] = 255
    return frame


# ---------------------------------------------------------------------------
# BlobDetector tests
# ---------------------------------------------------------------------------


class TestBlobDetector:

    def test_blank_frame_returns_empty(self):
        """A pure-black frame contains no foreground blobs."""
        detector = BlobDetector(min_area=10)
        blobs = detector.detect(_black_frame())
        assert blobs == []

    def test_detects_white_rectangle(self):
        """A white rectangle should produce exactly one blob at the correct position."""
        h, w = 240, 320
        rect_x, rect_y, rect_w, rect_h = 100, 80, 60, 40
        frame = _frame_with_rect(h, w, rect_x, rect_y, rect_w, rect_h)

        detector = BlobDetector(min_area=10, max_area=100_000)
        blobs = detector.detect(frame)

        assert len(blobs) >= 1, "Expected at least one blob"

        # The primary blob should be centred on the rectangle (±5 % tolerance)
        expected_x_norm = (rect_x + rect_w / 2) / w
        expected_y_norm = (rect_y + rect_h / 2) / h
        blob = blobs[0]

        assert abs(blob.x_norm - expected_x_norm) < 0.05, (
            f"x_norm mismatch: got {blob.x_norm:.3f}, expected ~{expected_x_norm:.3f}"
        )
        assert abs(blob.y_norm - expected_y_norm) < 0.05, (
            f"y_norm mismatch: got {blob.y_norm:.3f}, expected ~{expected_y_norm:.3f}"
        )

    def test_filters_blob_below_min_area(self):
        """A blob whose area is below min_area must be filtered out."""
        h, w = 240, 320
        # Small 5×5 white square → area ≈ 25 px²
        frame = _frame_with_rect(h, w, rect_x=10, rect_y=10, rect_w=5, rect_h=5)

        # min_area set higher than the rectangle's area
        detector = BlobDetector(min_area=200, max_area=100_000)
        blobs = detector.detect(frame)
        assert blobs == [], f"Expected no blobs, got {blobs}"

    def test_blob_normalised_coordinates_in_range(self):
        """x_norm and y_norm must always be in [0, 1]."""
        frame = _frame_with_rect()
        detector = BlobDetector(min_area=10)
        for blob in detector.detect(frame):
            assert 0.0 <= blob.x_norm <= 1.0
            assert 0.0 <= blob.y_norm <= 1.0

    def test_returns_blob_dataclass(self):
        """Returned items must be Blob instances with expected fields."""
        frame = _frame_with_rect()
        detector = BlobDetector(min_area=10)
        blobs = detector.detect(frame)
        assert len(blobs) >= 1
        b = blobs[0]
        assert isinstance(b, Blob)
        assert hasattr(b, "x_norm")
        assert hasattr(b, "y_norm")
        assert hasattr(b, "area")
        assert hasattr(b, "bbox")
        assert b.area > 0


# ---------------------------------------------------------------------------
# CentroidTracker tests
# ---------------------------------------------------------------------------


class TestCentroidTracker:

    def test_first_frame_two_detections_assigns_two_ids(self):
        """Two detections on the first frame → two unique object IDs registered."""
        tracker = CentroidTracker(camera_id="cam_01", max_disappeared=5, max_distance=0.3)
        result = tracker.update([(0.2, 0.3), (0.7, 0.8)])

        assert len(result) == 2
        ids = list(result.keys())
        assert ids[0] != ids[1], "IDs must be unique"
        # Both IDs should be scoped to the camera
        for oid in ids:
            assert oid.startswith("cam_01_"), f"ID {oid!r} not scoped to cam_01"

    def test_slight_movement_preserves_id(self):
        """An object that moves slightly between frames keeps the same ID."""
        tracker = CentroidTracker(camera_id="cam_02", max_disappeared=5, max_distance=0.3)

        # Frame 1: object at (0.5, 0.5)
        result1 = tracker.update([(0.5, 0.5)])
        assert len(result1) == 1
        original_id = list(result1.keys())[0]

        # Frame 2: same object has moved a tiny bit
        result2 = tracker.update([(0.52, 0.51)])
        assert len(result2) == 1
        new_id = list(result2.keys())[0]

        assert original_id == new_id, (
            f"Expected same ID after small movement; got {original_id!r} → {new_id!r}"
        )

    def test_absent_object_removed_after_max_disappeared(self):
        """An object absent for max_disappeared+1 frames is deregistered."""
        max_d = 3
        tracker = CentroidTracker(camera_id="cam_03", max_disappeared=max_d, max_distance=0.3)

        # Register the object
        tracker.update([(0.5, 0.5)])
        assert len(tracker.tracked_ids) == 1

        # Feed empty frames until the object should be evicted
        for _ in range(max_d + 1):
            tracker.update([])

        assert len(tracker.tracked_ids) == 0, (
            "Object should have been deregistered after max_disappeared empty frames"
        )

    def test_far_detection_registered_as_new_object(self):
        """A detection far from all existing objects is registered as a new object."""
        tracker = CentroidTracker(camera_id="cam_04", max_disappeared=5, max_distance=0.1)

        # Register object near top-left
        r1 = tracker.update([(0.05, 0.05)])
        assert len(r1) == 1

        # New detection far away (bottom-right) → must create a second object
        r2 = tracker.update([(0.05, 0.05), (0.95, 0.95)])
        assert len(r2) == 2, f"Expected 2 objects, got {len(r2)}: {list(r2.keys())}"

        ids = list(r2.keys())
        assert ids[0] != ids[1]

    def test_empty_first_frame_returns_empty(self):
        """No detections on the first frame → no tracked objects."""
        tracker = CentroidTracker(camera_id="cam_05")
        result = tracker.update([])
        assert result == {}

    def test_object_ids_are_incrementing(self):
        """IDs within a camera should be sequentially numbered."""
        tracker = CentroidTracker(camera_id="cam_06", max_disappeared=5, max_distance=0.05)
        tracker.update([(0.1, 0.1)])
        tracker.update([(0.1, 0.1), (0.9, 0.9)])  # second detection is new (far away)
        ids = sorted(tracker.tracked_ids)
        # Expect cam_06_obj0001 and cam_06_obj0002
        assert ids[0] == "cam_06_obj0001"
        assert ids[1] == "cam_06_obj0002"

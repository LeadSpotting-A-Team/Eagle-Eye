# -*- coding: utf-8 -*-
"""
Headless tests for deterministic dual-path change detection.
"""

import os
import sys
import traceback

import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.Lock import ALIGNMENT_OK, GEOMETRY_FAILURE, register_images
from backend.change_detection import (
    ChangeDetection,
    ChangeDetectorConfig,
    ChangeTracker,
    ChangeType,
    DualPathChangeDetector,
)
from backend.no_ai_guard import assert_no_ai_policy


PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def run_test(name, fn):
    try:
        fn()
        results.append((PASS, name, ""))
        print(f"  {PASS}  {name}")
    except AssertionError as e:
        results.append((FAIL, name, str(e)))
        print(f"  {FAIL}  {name}")
        print(f"         AssertionError: {e}")
    except Exception:
        msg = traceback.format_exc()
        results.append((FAIL, name, msg))
        print(f"  {FAIL}  {name}")
        print(msg)


def _make_reference(h=260, w=360):
    img = np.full((h, w, 3), 186, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (95, 95), (45, 45, 45), -1)
    cv2.rectangle(img, (250, 30), (335, 105), (70, 70, 70), -1)
    cv2.circle(img, (70, 205), 28, (30, 90, 170), -1)
    cv2.circle(img, (290, 205), 30, (20, 145, 80), -1)
    cv2.line(img, (0, 135), (359, 135), (110, 110, 110), 3)
    cv2.putText(img, "EE-42", (125, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (25, 25, 25), 2)
    for x in range(20, w, 45):
        cv2.line(img, (x, 150), (x + 15, 165), (80, 80, 80), 1)
    return img


def _make_warped_pair():
    ref = _make_reference()
    h, w = ref.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    dst = np.float32([[7, 8], [w - 12, 5], [10, h - 11], [w - 7, h - 6]])
    ref_to_new = cv2.getPerspectiveTransform(src, dst)
    new = cv2.warpPerspective(ref, ref_to_new, (w, h))
    return ref, new


def _detector(config=None):
    return DualPathChangeDetector(config or ChangeDetectorConfig())


def test_alignment_known_homography_passes():
    ref, new = _make_warped_pair()
    result = register_images(
        cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(new, cv2.COLOR_BGR2GRAY),
    )
    assert result.status == ALIGNMENT_OK, result.failure_reason
    assert result.homography is not None, "Expected homography"
    assert result.reprojection_error <= 0.5, "Strict RMS threshold should pass"
    assert result.inlier_count >= 20, "Expected enough inliers"


def test_alignment_low_feature_fails():
    blank = np.full((160, 220), 128, dtype=np.uint8)
    result = register_images(blank, blank)
    assert result.status == GEOMETRY_FAILURE, "Blank images must fail closed"


def test_geometry_failure_returns_no_detections():
    ref, new = _make_warped_pair()
    config = ChangeDetectorConfig(alignment_max_reprojection_error=0.01)
    result = _detector(config).detect(ref, new)
    assert result.status == GEOMETRY_FAILURE, "Expected strict geometry failure"
    assert result.detections == [], "Geometry failure must emit no detections"


def test_uniform_brightness_shift_no_change():
    ref = _make_reference()
    shifted = np.clip(ref.astype(np.int16) + 35, 0, 255).astype(np.uint8)
    result = _detector().detect(ref, shifted)
    assert result.status == ALIGNMENT_OK, result.alignment.failure_reason
    assert result.detections == [], "ZNCC/entropy path should suppress uniform light shifts"


def test_ground_texture_detected():
    ref = _make_reference()
    changed = ref.copy()
    rng = np.random.default_rng(7)
    roi = changed[88:152, 170:234].copy()
    speckles = rng.random((64, 64)) < 0.10
    values = rng.integers(50, 240, size=(64, 64, 1), dtype=np.uint8)
    roi[speckles] = np.repeat(values, 3, axis=2)[speckles]
    changed[88:152, 170:234] = roi
    result = _detector().detect(ref, changed)
    assert result.status == ALIGNMENT_OK, result.alignment.failure_reason
    assert any(d.change_type == ChangeType.GROUND for d in result.detections), "Expected Ground Change"


def test_closed_loop_rejected_from_ground():
    ref = _make_reference()
    changed = ref.copy()
    cv2.rectangle(changed, (155, 82), (220, 130), (20, 20, 20), 3)
    result = _detector().detect(ref, changed)
    assert result.status == ALIGNMENT_OK, result.alignment.failure_reason
    assert all(d.change_type != ChangeType.GROUND for d in result.detections), "Closed loops are not ground texture"


def test_textured_compact_object_detected():
    ref = _make_reference()
    changed = ref.copy()
    cv2.rectangle(changed, (145, 150), (205, 210), (42, 42, 42), -1)
    for offset in range(0, 60, 8):
        cv2.line(changed, (145, 150 + offset), (205, 150 + offset), (210, 210, 210), 2)
    result = _detector().detect(ref, changed)
    assert result.status == ALIGNMENT_OK, result.alignment.failure_reason
    objects = [d for d in result.detections if d.change_type == ChangeType.OBJECT]
    assert objects, "Expected Object detection"
    assert all(d.compactness is not None and d.compactness < 25.0 for d in objects)


def test_thin_elongated_region_fails_object_compactness():
    ref = _make_reference()
    changed = ref.copy()
    cv2.line(changed, (115, 180), (245, 195), (20, 20, 20), 4)
    result = _detector().detect(ref, changed)
    assert result.status == ALIGNMENT_OK, result.alignment.failure_reason
    assert all(d.change_type != ChangeType.OBJECT for d in result.detections), "Thin regions are not solid objects"


def test_flat_blob_fails_gradient_density():
    ref = _make_reference()
    changed = ref.copy()
    cv2.rectangle(changed, (145, 150), (205, 210), (35, 35, 35), -1)
    result = _detector().detect(ref, changed)
    assert result.status == ALIGNMENT_OK, result.alignment.failure_reason
    assert all(d.change_type != ChangeType.OBJECT for d in result.detections), "Flat blobs lack internal gradient density"


def test_shadow_like_region_suppressed():
    h, w = 220, 340
    gradient = np.tile(np.linspace(95, 220, w, dtype=np.uint8), (h, 1))
    ref = cv2.merge([gradient, gradient, gradient])
    cv2.putText(ref, "SHADOW", (70, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    cv2.rectangle(ref, (25, 150), (90, 205), (50, 50, 50), -1)
    changed = ref.copy()
    cv2.rectangle(changed, (95, 105), (260, 135), (45, 45, 45), -1)
    for x in range(100, 260, 14):
        cv2.line(changed, (x, 105), (x + 10, 135), (85, 85, 85), 1)
    config = ChangeDetectorConfig(
        object_compactness_max=200.0,
        shadow_elongation_min=2.0,
        alignment_min_matches=20,
    )
    result = _detector(config).detect(ref, changed)
    assert result.status == ALIGNMENT_OK, result.alignment.failure_reason
    assert all(d.change_type != ChangeType.OBJECT for d in result.detections), "Aligned dark elongated regions should be shadow-suppressed"


def test_fusion_prioritizes_compact_object():
    detector = _detector()
    ground = [
        ChangeDetection((10, 10, 50, 50), ChangeType.GROUND, 0.8, 0.2),
    ]
    objects = [
        ChangeDetection(
            (12, 12, 45, 45),
            ChangeType.OBJECT,
            0.9,
            0.1,
            compactness=16.0,
            compactness_passed=True,
        ),
    ]
    fused = detector._fuse(ground, objects)
    assert len(fused) == 1, "Object should suppress overlapping ground detection"
    assert fused[0].change_type == ChangeType.OBJECT, "Object path has priority"


def test_temporal_three_frame_consensus():
    tracker = ChangeTracker(required_persistence=3, iou_threshold=0.35)
    det = ChangeDetection((20, 20, 40, 40), ChangeType.GROUND, 0.9, 0.1)
    assert tracker.update([det], frame_id=1) == [], "Frame 1 is not real yet"
    assert tracker.update([det], frame_id=2) == [], "Frame 2 is not real yet"
    real = tracker.update([det], frame_id=3)
    assert len(real) == 1, "Frame 3 should reach consensus"
    assert real[0].persistence_count == 3, "Persistence count should be 3"
    tracker.update([], frame_id=4)
    real_after_gap = tracker.update([det], frame_id=5)
    assert real_after_gap == [], "A missed frame must break consecutiveness"


def test_no_ai_policy_guard():
    assert_no_ai_policy(ROOT_DIR)


print()
print("=" * 60)
print("  Eagle-Eye Dual-Path Detector -- Test Suite")
print("=" * 60)

run_test("T1 - known homography alignment", test_alignment_known_homography_passes)
run_test("T2 - low-feature geometry failure", test_alignment_low_feature_fails)
run_test("T3 - strict geometry failure emits no detections", test_geometry_failure_returns_no_detections)
run_test("T4 - uniform brightness shift suppressed", test_uniform_brightness_shift_no_change)
run_test("T5 - ground texture detected", test_ground_texture_detected)
run_test("T6 - closed loop rejected from ground", test_closed_loop_rejected_from_ground)
run_test("T7 - compact textured object detected", test_textured_compact_object_detected)
run_test("T8 - thin elongated object rejected", test_thin_elongated_region_fails_object_compactness)
run_test("T9 - flat blob rejected", test_flat_blob_fails_gradient_density)
run_test("T10 - shadow-like region suppressed", test_shadow_like_region_suppressed)
run_test("T11 - fusion object priority", test_fusion_prioritizes_compact_object)
run_test("T12 - temporal consensus", test_temporal_three_frame_consensus)
run_test("T13 - no-AI policy guard", test_no_ai_policy_guard)

print()
print("=" * 60)
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
total = len(results)
print(f"  Results: {passed}/{total} passed  |  {failed} failed")
print("=" * 60)

if failed:
    print("\nFailed tests:")
    for status, name, msg in results:
        if status == FAIL:
            print(f"  - {name}: {msg[:300]}")

sys.exit(0 if failed == 0 else 1)

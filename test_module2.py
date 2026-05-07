# -*- coding: utf-8 -*-
"""
Headless tests for the current Module 2 implementation.
"""

import os
import sys
import traceback

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from backend.Filter import Module2Processor, ProcessedImage


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


def _make_synthetic_pair(H=128, W=256):
    base = np.tile(np.linspace(80, 220, W, dtype=np.uint8), (H, 1))
    img_a = cv2.merge([base, base, base])
    img_b = img_a.copy()

    img_a[30:90, 60:180] = (img_a[30:90, 60:180] * 0.3).astype(np.uint8)
    img_b[35:95, 70:190] = (img_b[35:95, 70:190] * 0.45).astype(np.uint8)
    img_b[45:60, 150:190] = 255
    return img_a, img_b


IMG_A, IMG_B = _make_synthetic_pair()
PROC = Module2Processor(
    shadow_threshold_ratio=0.6,
    entropy_threshold=5.5,
    enable_weberface=True,
)
H, W = IMG_A.shape[:2]


def test_process_single():
    out = PROC.process_single(IMG_A)
    assert isinstance(out, ProcessedImage), "Expected ProcessedImage"
    assert out.original.shape == IMG_A.shape, "Original shape mismatch"
    assert out.final.shape == IMG_A.shape, "Final shape mismatch"
    assert out.final.dtype == np.uint8, "Final must be uint8"
    assert out.gradient.shape == (H, W), "Gradient shape mismatch"
    assert out.lbp.shape == (H, W), "LBP shape mismatch"
    assert out.structure_map.shape == (H, W), "Structure map shape mismatch"
    assert out.shadow_mask.shape == (H, W), "Shadow mask shape mismatch"
    assert out.shadow_mask.dtype == np.uint8, "Shadow mask must be uint8"


def test_normalized_maps_range():
    out = PROC.process_single(IMG_A)
    for arr, name in [
        (out.L_raw, "L_raw"),
        (out.L_normalized, "L_normalized"),
        (out.L_blurred, "L_blurred"),
        (out.gradient, "gradient"),
        (out.lbp, "lbp"),
        (out.structure_map, "structure_map"),
    ]:
        assert arr.dtype == np.float32, f"{name} must be float32"
        assert arr.min() >= 0.0 and arr.max() <= 1.0, f"{name} out of [0,1]"


def test_process_pair():
    a, b = PROC.process_pair(IMG_A, IMG_B)
    assert isinstance(a, ProcessedImage), "A must be ProcessedImage"
    assert isinstance(b, ProcessedImage), "B must be ProcessedImage"
    assert a.final.shape == b.final.shape == IMG_A.shape, "Pair shape mismatch"


def test_module3_payload():
    payload = PROC.process_pair_for_module3(IMG_A, IMG_B)
    assert set(payload.keys()) == {"image_a", "image_b", "debug"}, "Unexpected payload keys"
    for key in ("image_a", "image_b"):
        item = payload[key]
        for map_name in ("structure_map", "gradient", "lbp", "luminance"):
            arr = item[map_name]
            assert arr.dtype == np.float32, f"{key}.{map_name} must be float32"
            assert arr.shape == (H, W), f"{key}.{map_name} shape mismatch"
        assert item["shadow_mask"].dtype == np.uint8, f"{key}.shadow_mask must be uint8"
    assert payload["debug"]["a_final_bgr"].shape == IMG_A.shape, "Debug A final shape mismatch"
    assert payload["debug"]["b_final_bgr"].shape == IMG_B.shape, "Debug B final shape mismatch"


def test_shape_mismatch_rejected():
    bad = IMG_B[:100, :, :]
    try:
        PROC.process_pair_for_module3(IMG_A, bad)
    except ValueError:
        return
    raise AssertionError("Expected ValueError on mismatched image shapes")


def test_shadow_compensation_effect():
    img = IMG_A.astype(np.float32) / 255.0
    out = PROC.process_single(IMG_A)
    mask = np.zeros((H, W), dtype=bool)
    mask[30:90, 60:180] = True
    compensated = PROC._compensate(
        img,
        out.L_raw,
        out.lbp_raw,
        out.gradient,
        mask,
        out.weberface,
    )
    orig_mean = img[mask].mean()
    comp_mean = compensated[mask].mean()
    assert comp_mean >= orig_mean, "Compensation should not darken the masked region"


print()
print("=" * 60)
print("  Eagle-Eye Module 2 -- Test Suite")
print("=" * 60)

run_test("T1 - process_single()", test_process_single)
run_test("T2 - normalized maps range", test_normalized_maps_range)
run_test("T3 - process_pair()", test_process_pair)
run_test("T4 - Module 3 payload", test_module3_payload)
run_test("T5 - shape mismatch guard", test_shape_mismatch_rejected)
run_test("T6 - shadow compensation effect", test_shadow_compensation_effect)

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
            print(f"  - {name}: {msg[:200]}")

sys.exit(0 if failed == 0 else 1)

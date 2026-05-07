# -*- coding: utf-8 -*-
"""
Integration tests for the Module 1 -> Module 2 pipeline.
"""

import os
import sys
import traceback

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from backend.Filter import Module2Processor
from backend.pipelines import run_module1_module2


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


def _make_pipeline_pair(H=240, W=320):
    ref = np.full((H, W, 3), 235, dtype=np.uint8)

    cv2.rectangle(ref, (25, 30), (110, 140), (30, 30, 30), -1)
    cv2.rectangle(ref, (180, 55), (285, 150), (70, 70, 70), -1)
    cv2.circle(ref, (75, 185), 24, (0, 0, 255), -1)
    cv2.circle(ref, (230, 190), 28, (0, 255, 0), -1)
    cv2.line(ref, (0, 210), (319, 210), (255, 0, 0), 4)
    cv2.putText(ref, "E1", (130, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)

    src = np.float32([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]])
    dst = np.float32([[8, 10], [W - 18, 6], [12, H - 15], [W - 8, H - 8]])
    warp = cv2.getPerspectiveTransform(src, dst)
    new = cv2.warpPerspective(ref, warp, (W, H))

    new[70:150, 180:285] = np.clip(new[70:150, 180:285].astype(np.int16) - 65, 0, 255).astype(np.uint8)
    return ref, new


REF_IMG, NEW_IMG = _make_pipeline_pair()


def test_pipeline_output_contract():
    out = run_module1_module2(REF_IMG, NEW_IMG, Module2Processor())
    assert set(out.keys()) == {
        "reference_gray",
        "new_gray",
        "aligned_gray",
        "homography",
        "aligned_bgr",
        "module2_payload",
    }, "Unexpected top-level output keys"

    assert out["reference_gray"].shape == REF_IMG.shape[:2], "reference_gray shape mismatch"
    assert out["new_gray"].shape == NEW_IMG.shape[:2], "new_gray shape mismatch"
    assert out["aligned_gray"].shape == REF_IMG.shape[:2], "aligned_gray shape mismatch"
    assert out["aligned_bgr"].shape == REF_IMG.shape, "aligned_bgr shape mismatch"
    assert out["homography"].shape == (3, 3), "homography must be 3x3"


def test_module2_payload_contract():
    out = run_module1_module2(REF_IMG, NEW_IMG, Module2Processor())
    payload = out["module2_payload"]
    assert set(payload.keys()) == {"image_a", "image_b", "debug"}, "Unexpected payload keys"

    for image_key in ("image_a", "image_b"):
        item = payload[image_key]
        for map_name in ("structure_map", "gradient", "lbp", "luminance"):
            arr = item[map_name]
            assert arr.dtype == np.float32, f"{image_key}.{map_name} must be float32"
            assert arr.shape == REF_IMG.shape[:2], f"{image_key}.{map_name} shape mismatch"
        assert item["shadow_mask"].dtype == np.uint8, f"{image_key}.shadow_mask must be uint8"

    assert payload["debug"]["a_final_bgr"].shape == REF_IMG.shape, "debug A image shape mismatch"
    assert payload["debug"]["b_final_bgr"].shape == REF_IMG.shape, "debug B image shape mismatch"


def test_pipeline_rejects_non_bgr_input():
    gray = cv2.cvtColor(REF_IMG, cv2.COLOR_BGR2GRAY)
    try:
        run_module1_module2(gray, NEW_IMG)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for non-BGR reference image")


print()
print("=" * 60)
print("  Eagle-Eye Pipeline -- Test Suite")
print("=" * 60)

run_test("T1 - pipeline output contract", test_pipeline_output_contract)
run_test("T2 - Module 2 payload contract", test_module2_payload_contract)
run_test("T3 - non-BGR input guard", test_pipeline_rejects_non_bgr_input)

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

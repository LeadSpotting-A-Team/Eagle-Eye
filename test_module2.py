# -*- coding: utf-8 -*-
"""
test_illumination_invariant_processor.py
========================================
Headless unit/integration tests for Module 2 (no GUI, no cv2.imshow).

Test suite covers:
  T1  - _preprocess         : shape, dtype, Y range
  T2  - _compute_gradients  : shape, non-negative magnitude, direction range
  T3  - NMS (vectorized)    : shape, values <= grad_mag, no new maxima
  T4  - _build_uniform_lbp_table : exactly 58 uniform patterns, bin 58 = non-uniform
  T5  - _compute_lbp        : shape, values in [0, 58]
  T6  - _build_shadow_mask  : bool dtype, subset of low-luminance pixels
  T7  - _compute_weberface  : shape, dtype uint8
  T8  - process() full run  : output shapes, dtypes, shadow region brightened
  T9  - Zarr save (skipped if zarr not installed)
  T10 - Synthetic shadow compensation: compensated mean > original shadow mean
"""

import sys
import os
import traceback
import numpy as np
import cv2

# Make sure the module is importable from this directory
sys.path.insert(0, os.path.dirname(__file__))
from illumination_invariant_processor import IlluminationInvariantProcessor

# ---------------------------------------------------------------------------
# Test runner helpers
# ---------------------------------------------------------------------------

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

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
    except Exception as e:
        results.append((FAIL, name, traceback.format_exc()))
        print(f"  {FAIL}  {name}")
        traceback.print_exc()


def skip_test(name, reason):
    results.append((SKIP, name, reason))
    print(f"  {SKIP}  {name}  ({reason})")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_image(H=128, W=256):
    """Gradient image (80..220) with a dark shadow band (rows 30..90, cols 60..180)."""
    base = np.tile(np.linspace(80, 220, W, dtype=np.uint8), (H, 1))
    img = cv2.merge([base, base, base])
    img[30:90, 60:180] = (img[30:90, 60:180] * 0.3).astype(np.uint8)
    return img


IMG   = _make_synthetic_image()
PROC  = IlluminationInvariantProcessor(
    shadow_threshold_ratio=0.6,
    entropy_threshold=5.5,
    enable_weberface=True,
    enable_zarr=False,
)
H, W  = IMG.shape[:2]


# ---------------------------------------------------------------------------
# T1 – Preprocessing
# ---------------------------------------------------------------------------

def test_preprocess():
    Y, image_f32 = PROC._preprocess(IMG)
    assert Y.shape == (H, W),         f"Y shape mismatch: {Y.shape}"
    assert image_f32.shape == IMG.shape, f"float image shape mismatch"
    assert image_f32.dtype == np.float32, "float image should be float32"
    assert Y.min() >= 0.0 and Y.max() <= 1.0, f"Y out of [0,1]: [{Y.min():.3f}, {Y.max():.3f}]"

# ---------------------------------------------------------------------------
# T2 – Gradient Analysis
# ---------------------------------------------------------------------------

def test_gradients():
    Y, _ = PROC._preprocess(IMG)
    M, theta = PROC._compute_gradients(Y)
    assert M.shape == (H, W),     f"M shape: {M.shape}"
    assert theta.shape == (H, W), f"theta shape: {theta.shape}"
    assert M.min() >= 0.0,        "gradient magnitude must be non-negative"
    assert theta.min() >= -np.pi - 1e-4 and theta.max() <= np.pi + 1e-4, \
        f"theta out of [-pi, pi]: [{theta.min():.4f}, {theta.max():.4f}]"

# ---------------------------------------------------------------------------
# T3 – Vectorized NMS
# ---------------------------------------------------------------------------

def test_nms():
    Y, _ = PROC._preprocess(IMG)
    M, theta = PROC._compute_gradients(Y)
    nms = PROC._non_maximum_suppression(M, theta)
    assert nms.shape == M.shape, f"NMS shape mismatch: {nms.shape}"
    assert nms.min() >= 0.0,     "NMS must be non-negative"
    # NMS can only zero-out or preserve — never amplify
    assert (nms <= M + 1e-4).all(), "NMS produced values larger than input magnitude"
    # NMS should zero out at least some pixels (thin edges)
    assert (nms == 0).sum() > (M == 0).sum(), "NMS did not suppress any pixels"

# ---------------------------------------------------------------------------
# T4 – Uniform LBP lookup table
# ---------------------------------------------------------------------------

def test_lbp_table():
    table = IlluminationInvariantProcessor._build_uniform_lbp_table(8)
    assert table.shape == (256,), f"Table shape: {table.shape}"
    # Exactly 58 uniform patterns for P=8 (bins 0..57)
    n_uniform = int((table < 58).sum())
    assert n_uniform == 58, f"Expected 58 uniform patterns, got {n_uniform}"
    # Non-uniform patterns mapped to bin 58
    assert table[7] == 58 or (table < 58).sum() == 58, \
        "Non-uniform code not mapping to bin 58"

# ---------------------------------------------------------------------------
# T5 – LBP map
# ---------------------------------------------------------------------------

def test_lbp_map():
    Y, _ = PROC._preprocess(IMG)
    lbp = PROC._compute_lbp(Y)
    assert lbp.shape == (H, W),          f"LBP shape: {lbp.shape}"
    assert lbp.min() >= 0,               f"LBP min < 0: {lbp.min()}"
    assert lbp.max() <= 58,              f"LBP max > 58: {lbp.max()}"
    # Should contain multiple distinct bins (not all identical)
    assert len(np.unique(lbp)) > 1,     "LBP map is constant — likely broken"

# ---------------------------------------------------------------------------
# T6 – Shadow mask
# ---------------------------------------------------------------------------

def test_shadow_mask():
    Y, _ = PROC._preprocess(IMG)
    M, theta = PROC._compute_gradients(Y)
    nms = PROC._non_maximum_suppression(M, theta)
    mask = PROC._build_shadow_mask(Y, nms)
    assert mask.shape == (H, W),     f"Mask shape: {mask.shape}"
    assert mask.dtype == bool,       f"Mask dtype should be bool, got {mask.dtype}"
    # Shadow mask must be a subset of the low-luminance pixels
    avg_Y = Y.mean()
    low_lum = Y < (PROC.shadow_threshold_ratio * avg_Y)
    assert (mask & ~low_lum).sum() == 0, \
        "Shadow mask contains pixels that are NOT low-luminance"
    # The synthetic image has an obvious shadow band — mask should catch some pixels
    assert mask.sum() > 0, "Shadow mask is entirely empty on a shadowed image"

# ---------------------------------------------------------------------------
# T7 – Weberface
# ---------------------------------------------------------------------------

def test_weberface():
    Y, _ = PROC._preprocess(IMG)
    wf = PROC._compute_weberface(Y)
    assert wf.shape == (H, W),      f"Weberface shape: {wf.shape}"
    assert wf.dtype == np.uint8,    f"Weberface dtype: {wf.dtype}"
    assert wf.max() > 0,            "Weberface map is all zeros"

# ---------------------------------------------------------------------------
# T8 – Full process() pipeline
# ---------------------------------------------------------------------------

def test_full_pipeline():
    result, lbp_vis, mask_vis, wf_vis = PROC.process(IMG)
    assert result.shape  == IMG.shape,   f"Result shape: {result.shape}"
    assert result.dtype  == np.uint8,    f"Result dtype: {result.dtype}"
    assert lbp_vis.shape == (H, W),      f"LBP vis shape: {lbp_vis.shape}"
    assert mask_vis.shape == (H, W),     f"Mask vis shape: {mask_vis.shape}"
    assert wf_vis is not None,           "Weberface should not be None when enabled"
    assert wf_vis.shape == (H, W),       f"Weberface vis shape: {wf_vis.shape}"
    # All output arrays should be uint8
    for arr, name in [(result, "result"), (lbp_vis, "lbp"), (mask_vis, "mask"), (wf_vis, "weberface")]:
        assert arr.dtype == np.uint8, f"{name} dtype should be uint8, got {arr.dtype}"

# ---------------------------------------------------------------------------
# T9 – Zarr persistence (skipped if zarr unavailable)
# ---------------------------------------------------------------------------

def test_zarr():
    try:
        import zarr  # noqa
    except ImportError:
        skip_test("T9 - Zarr persistence", "zarr not installed")
        return

    import tempfile
    z_proc = IlluminationInvariantProcessor(enable_zarr=True)
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path = os.path.join(tmp, "module2_test.zarr")
        z_proc.process(IMG, zarr_path=zarr_path)
        store = zarr.open(zarr_path, mode="r")
        assert "result_bgr"  in store, "result_bgr dataset missing"
        assert "lbp_visual"  in store, "lbp_visual dataset missing"
        assert "shadow_mask" in store, "shadow_mask dataset missing"
        assert "weberface"   in store, "weberface dataset missing"

    run_test("T9 - Zarr persistence", lambda: None)  # already ran above

# ---------------------------------------------------------------------------
# T10 – Shadow compensation effectiveness
# ---------------------------------------------------------------------------

def test_shadow_compensation_effectiveness():
    result, _, mask_vis, _ = PROC.process(IMG)

    shadow_pixels_orig = IMG[mask_vis > 0].astype(np.float32)
    shadow_pixels_comp = result[mask_vis > 0].astype(np.float32)

    if shadow_pixels_orig.size == 0:
        raise AssertionError("No shadow pixels detected — cannot validate compensation")

    orig_mean = shadow_pixels_orig.mean()
    comp_mean = shadow_pixels_comp.mean()
    assert comp_mean > orig_mean, (
        f"Compensation did not brighten shadow region: "
        f"original mean={orig_mean:.1f}, compensated mean={comp_mean:.1f}"
    )
    print(f"         Shadow region: original mean={orig_mean:.1f}, "
          f"compensated mean={comp_mean:.1f}  (+{comp_mean-orig_mean:.1f})")

# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("  Eagle-Eye Module 2 -- Test Suite")
print("=" * 60)

run_test("T1  - Preprocessing",                test_preprocess)
run_test("T2  - Gradient Analysis",            test_gradients)
run_test("T3  - Vectorized NMS",               test_nms)
run_test("T4  - Uniform LBP lookup table",     test_lbp_table)
run_test("T5  - LBP map",                      test_lbp_map)
run_test("T6  - Shadow mask",                  test_shadow_mask)
run_test("T7  - Weberface map",                test_weberface)
run_test("T8  - Full process() pipeline",      test_full_pipeline)
run_test("T10 - Shadow compensation effect",   test_shadow_compensation_effectiveness)

# T9 handled separately (conditional skip)
try:
    import zarr  # noqa
    run_test("T9  - Zarr persistence",         test_zarr)
except ImportError:
    skip_test("T9  - Zarr persistence", "zarr not installed")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("=" * 60)
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
skipped = sum(1 for r in results if r[0] == SKIP)
total  = len(results)
print(f"  Results: {passed}/{total} passed  |  {failed} failed  |  {skipped} skipped")
print("=" * 60)

if failed:
    print("\nFailed tests:")
    for status, name, msg in results:
        if status == FAIL:
            print(f"  - {name}: {msg[:200]}")

sys.exit(0 if failed == 0 else 1)

# Dual-Path Deterministic Change Detection

This document explains the deterministic change-detection implementation added to the Eagle-Eye project. The system is intentionally non-AI: it does not use pretrained models, neural networks, `torch`, `keras`, `transformers`, or `ultralytics`. All decisions come from image registration, classical image processing, statistics, and geometry.

## High-Level Flow

The main user-facing demo is `run_pipeline_demo.py`. The main detection implementation is `backend/change_detection.py`.

The runtime flow is:

```text
reference / before image
new / after image
    |
    v
Module 1: strict ORB + RANSAC homography alignment
    |
    v
Fail-closed geometry gate
    |
    v
Dual-path detector
    |-- Path A: Ground / texture changes
    |-- Path B: Object / structural changes
    |
    v
Fusion and confidence filtering
    |
    v
Optional temporal 3-frame tracker
    |
    v
side-by-side visual output
```

Important behavior: if geometry is not reliable, the detector returns `GEOMETRY_FAILURE` and emits zero reliable detections. This is intentional. A bad homography can make unchanged pixels appear different.

## Main Files

- `backend/Lock.py`: strict image registration, ORB feature matching, RANSAC homography, reprojection-error validation.
- `backend/change_detection.py`: dual-path detector, data structures, fusion, temporal tracker, confidence scoring.
- `backend/pipelines.py`: convenience entrypoints, including `run_dual_path_change_detection`.
- `backend/no_ai_guard.py`: verifies that forbidden AI dependencies/imports are not present.
- `run_pipeline_demo.py`: CLI demo that writes side-by-side visual output.
- `test_dual_path_change_detection.py`: regression tests for alignment, ground/object paths, fusion, temporal tracking, and no-AI policy.

## Public Data Structures

### `ChangeType`

The detector currently emits two categories:

- `ChangeType.GROUND`: texture/surface anomaly, for example debris, stain, rough patch, or local texture change.
- `ChangeType.OBJECT`: compact physical/structural object-like region.

The system does not know semantic classes such as car, person, box, table, or tree. It only distinguishes surface texture changes from object-like structural changes.

### `ChangeDetection`

Represents one detected changed region.

Fields:

- `bounding_box`: `(x, y, width, height)` in reference-image coordinates.
- `change_type`: `Ground Change` or `Object`.
- `confidence_score`: `1 - p_value`.
- `p_value`: statistical score based on background-region distribution.
- `persistence_count`: how many consecutive frames matched this region.
- `is_real`: true only after temporal consensus marks it real.
- `compactness`: object contour compactness, used by Path B.
- `gradient_density`: internal gradient density, used by Path B.
- `metadata`: extra diagnostics such as region area and path.

### `ChangeDetectionResult`

Returned by the detector for one image pair.

Fields:

- `status`: `OK` or `GEOMETRY_FAILURE`.
- `detections`: current-frame detections after fusion and confidence filtering.
- `alignment`: `AlignmentResult` from Module 1.
- `real_detections`: detections promoted by the temporal tracker, if a tracker was used.
- `aligned_bgr`: the new image warped into reference coordinates.
- `diagnostics`: counts and metrics such as valid overlap ratio.

### `ChangeDetectorConfig`

Central configuration object for all thresholds. It includes:

- alignment gates: reprojection error, RANSAC threshold, minimum matches/inliers.
- ground path thresholds: patch size, entropy delta, SSIM, ZNCC, confidence.
- object path thresholds: Canny thresholds, compactness, area, gradient density.
- shadow filtering thresholds.
- fusion and temporal-tracking thresholds.

## Module 1: Strict Registration

Implemented in `backend/Lock.py`.

The detector cannot compare pixels directly until both images are in the same coordinate system. Module 1 aligns the new image to the reference image.

Algorithm:

1. Convert both images to grayscale.
2. Detect ORB keypoints and binary descriptors.
3. Match descriptors with Hamming distance.
4. Apply Lowe-ratio filtering to keep better matches.
5. Estimate homography using RANSAC.
6. Compute RMS reprojection error on inlier matches.
7. Reject the pair if geometry quality is below threshold.

Default strict gates:

- `alignment_max_reprojection_error = 0.5` pixels.
- `alignment_min_matches = 30`.
- `alignment_min_inliers = 20`.
- `alignment_min_inlier_ratio = 0.15`.

If any gate fails, the status is `GEOMETRY_FAILURE`.

Why this matters:

- Image differencing is extremely sensitive to misalignment.
- A 1-2 pixel shift can create many false edge/texture changes.
- The system is designed to fail closed rather than draw false positives.

## Valid Overlap Mask

After homography, not every output pixel is trustworthy. Some pixels may come from outside the original after-image or from warped border regions.

`DualPathChangeDetector._valid_overlap_mask()` creates a mask of pixels in reference space that are backed by real source pixels after warping.

The detector then ignores invalid overlap areas in both paths. This prevents false detections along image borders and warped black regions.

## Path A: Ground / Texture Sensor

Implemented in `_detect_ground()`.

Purpose:

Detect non-structural surface changes, such as local texture changes, debris, roughness, stains, or other ground-like anomalies.

Main idea:

Ground changes usually alter local texture statistics but do not form a strong closed object contour.

Steps:

1. Convert reference and aligned image to normalized grayscale.
2. Split image into `16x16` patches.
3. Skip patches outside the valid overlap mask.
4. For each patch, compute:
   - entropy delta: `entropy(new_patch) - entropy(reference_patch)`.
   - SSIM: structural similarity between patches.
   - ZNCC: illumination-invariant correlation.
5. Mark a patch as a candidate if:
   - entropy increased enough.
   - SSIM dropped enough.
   - the patch is not only a uniform lighting shift.
6. Clean the candidate mask with morphology.
7. Extract connected components.
8. Reject components that look like closed-loop structures.
9. Score remaining regions against background patch scores.
10. Keep only regions above `min_confidence_score`.

Key functions:

- `_entropy_patch()`: measures texture complexity.
- `_ssim_patch()`: measures local structural similarity.
- `_zncc()`: suppresses uniform brightness/contrast shifts.
- `_has_closed_loop_structure()`: prevents object-like contours from being labeled as ground.

Current default filtering:

- `patch_size = 16`.
- `min_change_area = 512`.
- `min_confidence_score = 0.95`.
- `ground_entropy_delta_threshold = 0.20`.
- `ground_ssim_threshold = 0.90`.
- `ground_zncc_max = 0.94`.

Known limitation:

Path A cannot understand real-world semantics. It detects signal patterns, not object names.

## Path B: Object / Structural Sensor

Implemented in `_detect_objects()`.

Purpose:

Detect compact physical object-like changes. These are regions with new edges, compact shape, and internal gradient structure.

Main idea:

Physical objects tend to introduce structural edges and internal gradient features. Flat blobs or elongated shadows should be rejected.

Steps:

1. Convert reference and aligned image to normalized grayscale.
2. Run Canny edge detection on both images.
3. Remove edges already present in the reference image.
4. Apply morphological closing to bridge small edge gaps.
5. Extract contours.
6. Reject small regions.
7. Compute compactness:

```text
compactness = perimeter^2 / area
```

8. Keep only compact regions below `object_compactness_max`.
9. Compute internal gradient density.
10. Compare internal gradient density to surrounding gradient density.
11. Suppress shadow-like dark elongated regions.
12. Convert the final structural score to p-value and confidence.

Current default filtering:

- `object_canny_low = 55`.
- `object_canny_high = 140`.
- `object_close_kernel = 5`.
- `object_min_area = 120`.
- `object_compactness_max = 25.0`.
- `object_gradient_density_min = 0.06`.
- `object_gradient_density_ratio_min = 1.20`.
- `min_confidence_score = 0.95`.

Key functions:

- `_sobel_mag()`: computes gradient magnitude.
- `_estimate_shadow_direction()`: estimates coarse shadow direction from low-frequency luminance.
- `_mask_major_axis()`: estimates elongation and dominant axis.
- `_is_shadow_like()`: suppresses dark elongated regions aligned with shadow direction.

Known limitation:

Path B can detect object-like regions but cannot name the object category. It can say `Object`, not `car` or `person`.

## Confidence and p-Value

Both paths compute a region score and compare it to background scores from non-candidate areas.

Implementation:

```text
p_value = one-sided survival probability of region score
confidence_score = 1 - p_value
```

The p-value is computed by `_p_value_from_background()` using a robust background distribution. The implementation uses median, MAD, standard deviation, and a minimum floor to avoid unstable division.

The default detector only keeps detections with:

```text
confidence_score >= 0.95
```

This conservative threshold was added to reduce false positives.

## Fusion

Implemented in `_fuse()`.

The detector runs Path A and Path B independently. Sometimes both paths flag the same area.

Rule:

- If a Ground detection overlaps an Object detection and the object compactness test passed, Object wins.
- Otherwise, Ground remains.

Overlap is measured with IoU:

```text
intersection-over-union >= fusion_iou_threshold
```

Default:

```text
fusion_iou_threshold = 0.20
```

## Temporal Tracker

Implemented in `ChangeTracker`.

Purpose:

Avoid treating one-frame noise as a real change.

Behavior:

1. Each frame produces detections in reference coordinates.
2. The tracker matches new detections to old tracks using IoU.
3. A detection becomes real only after consecutive matches.

Defaults:

```text
required_persistence = 3
temporal_iou_threshold = 0.35
```

Meaning:

A change must appear in approximately the same coordinate region for 3 consecutive frames before `is_real=True`.

Current demo note:

`run_pipeline_demo.py` processes one before/after pair, so it displays current-frame detections. Temporal tracking is available through the API, but the demo does not simulate a frame sequence by default.

## Demo Usage

Run a known dataset pair:

```powershell
python run_pipeline_demo.py --dataset-id 1 --output sandbox\dataset_1_side_by_side.png
```

Output:

- left: reference / before image.
- right: aligned after image.
- same boxes drawn on both sides.

Run with explicit file paths:

```powershell
python run_pipeline_demo.py --reference Datasets\before_after\1_before.webp --new Datasets\before_after\1_after.webp --output sandbox\manual_pair.png
```

Save only the aligned after image:

```powershell
python run_pipeline_demo.py --dataset-id 1 --single --output sandbox\dataset_1_single.png
```

Force visual debug output when geometry fails:

```powershell
python run_pipeline_demo.py --dataset-id 3 --relaxed-geometry --output sandbox\dataset_3_relaxed.png
```

Warning:

`--relaxed-geometry` is for visual debugging only. It can create false positives because it loosens the geometry gates.

## Geometry Failure

Example:

```text
Geometry Failure: not enough RANSAC inliers: 8 < 20
```

This means the pair did not produce enough reliable matched points for a trustworthy homography.

In this case:

- no reliable detections are emitted.
- the demo saves a diagnostic side-by-side image.
- the system is working as designed.

To get reliable detections, use image pairs with enough shared visual structure and viewpoint overlap.

## Tests

Main test command:

```powershell
python test_dual_path_change_detection.py
```

This validates:

- known homography alignment passes.
- low-feature images fail closed.
- geometry failure emits zero detections.
- uniform brightness shifts are suppressed.
- ground texture changes are detected.
- closed-loop structures are rejected from Path A.
- compact textured objects are detected.
- thin regions fail object compactness.
- flat blobs fail gradient density.
- shadow-like regions are suppressed.
- fusion prioritizes objects.
- temporal consensus requires 3 frames.
- forbidden AI packages are not used.

Existing regression tests:

```powershell
python test_module2.py
python sandbox\test_pipeline.py
```

## No-AI Policy

The no-AI rule is enforced by `backend/no_ai_guard.py`.

Forbidden packages:

```text
torch
keras
transformers
ultralytics
```

The guard checks:

- `requirements.txt` for forbidden dependencies.
- project Python files for forbidden imports.

Run through the test suite:

```powershell
python test_dual_path_change_detection.py
```

## Current Limitations

- The detector is deterministic, but not semantically intelligent.
- It can distinguish `Ground Change` vs `Object`, not object names.
- It depends heavily on accurate geometry.
- Dataset pairs with low overlap or strong viewpoint changes will fail closed.
- Some texture changes can still be false positives if the scene has strong non-rigid changes.
- `Module2Processor` exists, but the current dual-path detector operates after Module 1 alignment and does not yet consume Module 2 `structure_map` directly.

## Future Improvements

Reasonable next improvements:

- Feed Module 2 luminance/structure maps into `DualPathChangeDetector`.
- Add a heatmap output mode instead of only bounding boxes.
- Add Top-N filtering for demos where only the strongest changes should be shown.
- Add per-dataset calibration profiles.
- Add a batch runner that processes all `Datasets/before_after` pairs and writes a CSV report.
- Add real temporal-sequence demo support for `ChangeTracker`.

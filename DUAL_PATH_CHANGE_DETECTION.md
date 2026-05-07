# Dual-Path Change Detection

This document describes the current Module 3 backend implementation:
Dual-Path Change Detection, also called "The Sensor".

The module receives a reference image and a new image, aligns the new image into
reference coordinates, generates candidate change regions, fuses duplicate
signals, and optionally applies temporal persistence tracking.

## Design Goal

The classical image-processing logic is not treated as the final truth source.
It is a candidate generator. Its job is to find suspicious regions while
remaining deterministic, testable, and usable in classical-only mode.

The current architecture supports two modes:

- Classical-only: no AI packages are imported statically; object detection uses
  a no-op adapter plus deterministic structural fallback.
- Hybrid-ready: a real detector such as YOLO can be plugged in behind the
  `ObjectDetector` adapter when explicitly enabled.

## Runtime Flow

```text
reference_bgr
new_bgr
    |
    v
Module 1 alignment: ORB + RANSAC homography
    |
    v
Geometry gate
    |-- GEOMETRY_FAILURE -> structured empty result
    |
    v
aligned_new_bgr + valid overlap mask
    |
    v
Path A: ground candidate generator
Path B: object before/after comparator
    |
    v
Fusion
    |
    v
ChangeDetectionResult
    |
    v
Optional ChangeTracker
```

Important behavior: if alignment is invalid, the detector returns
`GEOMETRY_FAILURE` with no candidates and no final detections.

## Main Files

- `backend/change_types.py`: shared public dataclasses and enums.
- `backend/change_detection.py`: detector config, Path A, Path B, fusion, and
  `run_dual_path_change_detection`.
- `backend/object_detector.py`: object detector adapter, `NoOpDetector`, and
  optional guarded `YoloDetector`.
- `backend/change_tracker.py`: temporal persistence tracker.
- `backend/dependency_policy.py`: runtime dependency mode.
- `backend/no_ai_guard.py`: static scanner for forbidden AI imports.
- `test_dual_path_change_detection.py`: Module 3 regression tests.

## Public Types

### `ChangeType`

Current categories:

- `GROUND_CHANGE`
- `OBJECT_ADDED`
- `OBJECT_REMOVED`
- `UNKNOWN_CHANGE`

Legacy aliases are kept:

- `GROUND` -> `GROUND_CHANGE`
- `OBJECT` -> `OBJECT_ADDED`

### `ChangeDetection`

One candidate or final detection in reference-image coordinates.

Important fields:

- `bbox`: `(x, y, width, height)`
- `change_type`
- `confidence_score`: normalized to `[0, 1]`
- `source`: e.g. `path_a`, `path_b`, `path_b_classical`, `fusion`
- `label`: optional object label from a detector
- `mask`: optional boolean mask
- `frame_id`: optional caller-supplied frame identifier
- `reason`: human-readable detection reason
- `metadata`: debug payload

Backward-compatible fields are preserved:

- `bounding_box`
- `p_value`
- `persistence_count`
- `is_real`
- `compactness`
- `gradient_density`
- `compactness_passed`

### `ChangeDetectionResult`

Returned by `DualPathChangeDetector.detect(...)` and
`run_dual_path_change_detection(...)`.

Fields:

- `status`
- `alignment_result`
- `ground_candidates`
- `object_candidates`
- `final_detections`
- `rejected_candidates`
- `debug_info`
- `real_detections`
- `aligned_bgr`

Backward-compatible aliases:

- `detections` -> `final_detections`
- `alignment` -> `alignment_result`
- `diagnostics` -> `debug_info`

## Configuration

`ChangeDetectorConfig` stores thresholds and feature flags. Important groups:

- alignment gates: reprojection error, RANSAC threshold, minimum matches,
  minimum inliers, inlier ratio, ORB features.
- Path A thresholds: patch size, minimum area, entropy delta, SSIM, ZNCC,
  confidence.
- Path B detector thresholds: detector confidence, object match IoU, same-class
  matching.
- Path B deterministic fallback thresholds: Canny, compactness, area, gradient
  density, shadow suppression.
- fusion thresholds: overlap IoU and minimum object confidence.
- tracker defaults: required persistence and temporal IoU.

## Path A: Ground Candidate Generator

Implemented by `_detect_ground(...)`.

Path A compares normalized grayscale patches between the reference image and
the aligned new image.

For each patch it computes:

- local entropy difference
- local SSIM drop
- ZNCC for illumination-invariant suppression

The candidate mask is cleaned with morphology, converted into connected
regions, and filtered by:

- minimum area
- valid overlap mask
- closed-loop/object-like structure rejection
- statistical confidence

Output candidates are `GROUND_CHANGE`.

Path A does not make the final truth decision. It only marks suspicious ground
or texture regions.

## Path B: Object Change Detection

Implemented by `_detect_objects(...)`.

Path B now follows the before/after object comparison rule:

1. Run `ObjectDetector.detect(reference_bgr)`.
2. Run `ObjectDetector.detect(aligned_new_bgr)`.
3. Match detections by IoU and class label/class id.
4. Objects only in the new image become `OBJECT_ADDED`.
5. Objects only in the reference image become `OBJECT_REMOVED`.
6. Matched objects are treated as unchanged and are not reported.

The default adapter is `NoOpDetector`, so no AI package is required. When no
detector is active, the module can use the deterministic structural fallback
that existed previously. That fallback finds compact edge/gradient regions and
emits object-like candidates as `OBJECT_ADDED`.

## Object Detector Adapter

The adapter interface is:

```python
class ObjectDetector:
    def detect(self, image_bgr) -> list[ObjectDetection]:
        ...
```

Available implementations:

- `NoOpDetector`: returns an empty list.
- `YoloDetector`: optional YOLOv8-backed detector, guarded by environment and
  dependency policy.

YOLO is intentionally not imported with `from ultralytics import YOLO`. The
code uses dynamic import only inside the optional guarded path, so the
classical-only scanner remains clean.

## Fusion

Implemented by `_fuse(...)`.

Fusion resolves duplicate regions between Path A and Path B.

Rule:

- If a ground candidate overlaps an object candidate by
  `fusion_iou_threshold`, and the object confidence is at least
  `fusion_object_confidence_min`, prefer the object candidate.
- Otherwise keep the ground candidate.

Suppressed ground candidates are returned in `rejected_candidates` with a
reason and debug fusion decision.

## Confidence

All confidence scores are normalized to `[0, 1]`.

Ground candidates combine:

- entropy/SSIM-derived statistical score
- p-value background comparison
- alignment confidence

Object candidates combine:

- detector confidence or deterministic structural score
- alignment confidence

Raw unnormalized values are stored only in metadata/debug fields.

## Temporal Tracking

Implemented in `backend/change_tracker.py`.

Default behavior:

```text
required_persistence = 3
iou_threshold = 0.35
same change_type required
```

A detection is confirmed only after it appears in matching reference-image
coordinates for 3 consecutive valid frames.

`run_dual_path_change_detection(...)` does not update the tracker when the
result status is `GEOMETRY_FAILURE`. Geometry failure frames are ignored for
persistence purposes.

## Debug Output

`debug_info` can include:

- alignment confidence
- valid overlap ratio
- object detections before
- object detections after
- rejected object detections
- object match decisions
- fusion decisions
- final detection summaries

Debug images are not saved automatically. `save_debug_images` exists in config
as a future-facing flag, but persisted debug image export is still future work.

## Demo Usage

Run a bundled pair:

```powershell
python run_pipeline_demo.py --dataset-id 1
```

Choose output path:

```powershell
python run_pipeline_demo.py --dataset-id 1 --output Output\module3_result.png
```

Run explicit files:

```powershell
python run_pipeline_demo.py --reference Datasets\before_after\1_before.webp --new Datasets\before_after\1_after.webp --output sandbox\manual_pair.png
```

The no-argument form expects `sandbox\1.jpeg` and `sandbox\2.jpeg`.

## Tests

Run:

```powershell
python test_dual_path_change_detection.py
python test_module2.py
```

Current expected status:

```text
test_dual_path_change_detection.py: 18/18 passed
test_module2.py: 6/6 passed
```

Module 3 tests cover:

- geometry failure produces empty detections
- brightness shifts are suppressed
- ground texture changes are detected
- small/flat/noisy changes are rejected
- object added is reported
- object removed is reported
- matched before/after objects are not reported
- high-confidence object suppresses overlapping ground
- low-confidence object does not erase ground
- temporal persistence requires 3 frames
- geometry failure does not advance tracker
- forbidden AI imports are not present in the core path

## Current Limitations

- No verifier model is active yet.
- YOLO is optional but not enabled by default.
- The deterministic fallback can find object-like structure but cannot provide
  semantic labels.
- Module 2 maps are not yet fully consumed by Module 3.
- Precision/recall has not been measured against a labeled benchmark.

## Next Work

- Add verifier model support over before/after/diff candidate crops.
- Add explicit hybrid-AI enablement and optional dependency setup docs.
- Add batch dataset reporting.
- Add persisted debug image export behind config.
- Add semantic reporting and optional segmentation after candidate verification.

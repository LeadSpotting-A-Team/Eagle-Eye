# Eagle-Eye Change Detection

Eagle-Eye is a computer-vision backend for comparing a reference image with a
new image and reporting candidate physical changes in reference-image
coordinates.

The current backend focuses on Module 3: Dual-Path Change Detection, also
called "The Sensor".

## Current Pipeline

```text
reference / before image
new / after image
    |
    v
Module 1: ORB + RANSAC homography alignment
    |
    v
Fail-closed geometry gate
    |
    v
Module 2: Illumination correction and adaptive structure mapping
    |
    v
Module 3: Dual-path candidate generation
    |-- Path A: ground / texture changes
    |-- Path B: object added / object removed
    |
    v
Deterministic fusion
    |
    v
Optional temporal tracker
```

If geometry is unreliable, the system returns `GEOMETRY_FAILURE` and emits no
detections. This is intentional: a bad homography can create false changes.

## Main Files

- `backend/Lock.py`: Module 1 registration and geometry validation.
- `backend/Filter.py`: Module 2 illumination and structure preprocessing.
- `backend/change_types.py`: public Module 3 result types.
- `backend/change_detection.py`: dual-path detector, fusion, and main API.
- `backend/object_detector.py`: optional object-detector adapter, including
  `NoOpDetector` and guarded `YoloDetector`.
- `backend/change_tracker.py`: temporal persistence tracker.
- `backend/dependency_policy.py`: classical-only vs hybrid-AI dependency mode.
- `backend/no_ai_guard.py`: static no-AI policy scanner.
- `run_pipeline_demo.py`: CLI demo for before/after pairs.

## Public API

Use:

```python
from backend.change_detection import run_dual_path_change_detection

result = run_dual_path_change_detection(reference_bgr, new_bgr)
```

The returned `ChangeDetectionResult` includes:

- `status`
- `alignment_result`
- `ground_candidates`
- `object_candidates`
- `final_detections`
- `rejected_candidates`
- `debug_info`

Backward-compatible aliases are still available:

- `result.detections` -> `result.final_detections`
- `result.alignment` -> `result.alignment_result`
- `result.diagnostics` -> `result.debug_info`

## Change Types

The current public categories are:

- `GROUND_CHANGE`
- `OBJECT_ADDED`
- `OBJECT_REMOVED`
- `UNKNOWN_CHANGE`

Legacy aliases `GROUND` and `OBJECT` are kept for older tests and callers.

## Running Tests

From the project root:

```powershell
python test_dual_path_change_detection.py
python test_module2.py
```

Current expected status:

```text
test_dual_path_change_detection.py: 18/18 passed
test_module2.py: 6/6 passed
```

## Running The Demo

The demo needs input images. Running it with no arguments expects
`sandbox\1.jpeg` and `sandbox\2.jpeg`, which may not exist.

Use one of the bundled dataset pairs:

```powershell
python run_pipeline_demo.py --dataset-id 1
```

Save to a specific output path:

```powershell
python run_pipeline_demo.py --dataset-id 1 --output sandbox\module3_result.png
```

Run on explicit files:

```powershell
python run_pipeline_demo.py --reference "path\to\before.jpg" --new "path\to\after.jpg" --output sandbox\my_result.png
```

For visual debugging only, loosen geometry gates:

```powershell
python run_pipeline_demo.py --dataset-id 1 --relaxed-geometry
```

`--relaxed-geometry` can create false positives and should not be used for
reliable reporting.

## Dependency Policy

Default mode is classical-only. The core path must not import forbidden AI
packages statically:

```text
torch
keras
transformers
ultralytics
```

YOLO support is isolated behind `backend/object_detector.py` and a runtime
adapter. In normal no-AI mode, the system uses `NoOpDetector` plus the
deterministic structural fallback.

## Current Limitations

- YOLO is not enabled by default.
- The AI verifier over candidate crops has not been implemented yet (Module 3.5).
- Module 2 payload is fully integrated, extracting invariant structure maps alongside the candidates.
- Real precision/recall has not been calibrated on a labeled dataset.
- The demo processes one before/after pair; temporal tracking is available via
  API but not shown as a full video sequence demo.

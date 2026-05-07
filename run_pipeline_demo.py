from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from backend.Lock import GEOMETRY_FAILURE
from backend.change_detection import ChangeDetectorConfig, ChangeType
from backend.pipelines import run_dual_path_change_detection


def _safe_read(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _safe_write(path: str, image) -> None:
    ext = os.path.splitext(path)[1]
    ok, buffer = cv2.imencode(ext, image)
    if not ok:
        raise IOError(f"Could not encode output image for: {path}")
    buffer.tofile(path)


def _draw_detections(image_bgr: np.ndarray, detections: list) -> np.ndarray:
    canvas = image_bgr.copy()
    for detection in detections:
        x, y, w, h = detection.bounding_box
        color = (255, 0, 0) if detection.change_type == ChangeType.GROUND else (0, 0, 255)
        label = f"{detection.change_type.value} {detection.confidence_score:.2f}"
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            canvas,
            label,
            (x, max(15, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def _draw_comparison(
    reference_bgr: np.ndarray,
    aligned_bgr: np.ndarray,
    detections: list,
) -> np.ndarray:
    left = _draw_detections(reference_bgr, detections)
    right = _draw_detections(aligned_bgr, detections)
    height = max(left.shape[0], right.shape[0])
    if left.shape[0] != height:
        left = _resize_to_height(left, height)
    if right.shape[0] != height:
        right = _resize_to_height(right, height)
    divider = np.full((height, 10, 3), 35, dtype=np.uint8)
    body = np.hstack([left, divider, right])
    banner = np.full((70, body.shape[1], 3), 25, dtype=np.uint8)
    cv2.putText(
        banner,
        "BEFORE / REFERENCE",
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        banner,
        "AFTER / ALIGNED",
        (left.shape[1] + divider.shape[1] + 20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([banner, body])


def _resize_to_height(image_bgr: np.ndarray, height: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if h == height:
        return image_bgr
    new_w = max(1, int(w * (height / h)))
    return cv2.resize(image_bgr, (new_w, height), interpolation=cv2.INTER_AREA)


def _failure_canvas(reference_bgr: np.ndarray, new_bgr: np.ndarray, reason: str) -> np.ndarray:
    height = min(reference_bgr.shape[0], new_bgr.shape[0], 720)
    reference_view = _resize_to_height(reference_bgr, height)
    new_view = _resize_to_height(new_bgr, height)
    divider = np.full((height, 8, 3), 40, dtype=np.uint8)
    canvas = np.hstack([reference_view, divider, new_view])
    banner = np.full((90, canvas.shape[1], 3), 25, dtype=np.uint8)
    cv2.putText(
        banner,
        "GEOMETRY FAILURE - no reliable detections emitted",
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        banner,
        f"Reason: {reason}",
        (15, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([banner, canvas])


def _find_dataset_pair(item_id: str) -> tuple[str, str]:
    root = os.path.join("Datasets", "before_after")
    names = os.listdir(root)
    before = next((n for n in names if n.startswith(f"{item_id}_before")), None)
    after = next((n for n in names if n.startswith(f"{item_id}_after")), None)
    if before is None or after is None:
        raise FileNotFoundError(f"Could not find before/after pair for dataset id {item_id}")
    return os.path.join(root, before), os.path.join(root, after)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic Eagle-Eye change detection.")
    parser.add_argument("--reference", default=None, help="Reference/before image path.")
    parser.add_argument("--new", default=None, help="New/after image path.")
    parser.add_argument("--dataset-id", default=None, help="Use Datasets/before_after/<id>_before and <id>_after.")
    parser.add_argument("--output", default=os.path.join("sandbox", "pipeline_result.png"))
    parser.add_argument(
        "--single",
        action="store_true",
        help="Save only the aligned after image with boxes instead of side-by-side comparison.",
    )
    parser.add_argument(
        "--relaxed-geometry",
        action="store_true",
        help="Debug-only: loosen geometry support gates to force visual inspection output.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.dataset_id is not None:
        reference_path, new_path = _find_dataset_pair(args.dataset_id)
    else:
        reference_path = args.reference or os.path.join("sandbox", "1.jpeg")
        new_path = args.new or os.path.join("sandbox", "2.jpeg")
    output_path = args.output

    reference_bgr = _safe_read(reference_path)
    new_bgr = _safe_read(new_path)

    if reference_bgr is None:
        raise FileNotFoundError(f"Could not read reference image: {reference_path}")
    if new_bgr is None:
        raise FileNotFoundError(f"Could not read new image: {new_path}")

    config = None
    if args.relaxed_geometry:
        config = ChangeDetectorConfig(
            alignment_min_matches=8,
            alignment_min_inliers=6,
            alignment_min_inlier_ratio=0.05,
        )

    result = run_dual_path_change_detection(reference_bgr, new_bgr, config=config)
    if result.status == GEOMETRY_FAILURE:
        output = _failure_canvas(reference_bgr, new_bgr, result.alignment.failure_reason or "unknown")
        _safe_write(output_path, output)
        print("Geometry Failure")
        print(f"Reason: {result.alignment.failure_reason}")
        print("No reliable detections emitted.")
        print(f"Saved diagnostic output: {output_path}")
        return

    if args.single:
        output = _draw_detections(result.aligned_bgr, result.detections)
    else:
        output = _draw_comparison(reference_bgr, result.aligned_bgr, result.detections)
    _safe_write(output_path, output)

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Output file was not created: {output_path}")

    print("Pipeline finished")
    print(f"Saved output: {output_path}")
    print(f"Detections: {len(result.detections)}")
    if args.relaxed_geometry:
        print("WARNING: relaxed geometry mode is for visual debugging only.")


if __name__ == "__main__":
    main()

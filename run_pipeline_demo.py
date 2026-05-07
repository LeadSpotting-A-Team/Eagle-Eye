from __future__ import annotations

import os

import cv2
import numpy as np

from backend.pipelines import run_module1_module2


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


def main() -> None:
    reference_path = os.path.join("sandbox", "1.jpeg")
    new_path = os.path.join("sandbox", "2.jpeg")
    output_path = os.path.join("sandbox", "pipeline_result.png")

    reference_bgr = _safe_read(reference_path)
    new_bgr = _safe_read(new_path)

    if reference_bgr is None:
        raise FileNotFoundError(f"Could not read reference image: {reference_path}")
    if new_bgr is None:
        raise FileNotFoundError(f"Could not read new image: {new_path}")

    result = run_module1_module2(reference_bgr, new_bgr)
    _safe_write(output_path, result["module2_payload"]["debug"]["b_final_bgr"])

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Output file was not created: {output_path}")

    print("Pipeline finished")
    print(f"Saved output: {output_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import os

import cv2

from backend.pipelines import run_module1_module2


def main() -> None:
    reference_path = os.path.join("sandbox", "1.jpeg")
    new_path = os.path.join("sandbox", "2.jpeg")
    output_dir = os.path.join("sandbox", "Output")
    output_path = os.path.join(output_dir, "pipeline_result.png")

    reference_bgr = cv2.imread(reference_path)
    new_bgr = cv2.imread(new_path)

    if reference_bgr is None:
        raise FileNotFoundError(f"Could not read reference image: {reference_path}")
    if new_bgr is None:
        raise FileNotFoundError(f"Could not read new image: {new_path}")

    result = run_module1_module2(reference_bgr, new_bgr)

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, result["module2_payload"]["debug"]["b_final_bgr"])

    print("Pipeline finished")
    print(f"Saved output: {output_path}")


if __name__ == "__main__":
    main()

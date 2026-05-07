from __future__ import annotations

import cv2
import numpy as np

from backend.Filter import Module2Processor
from backend.Lock import The_Lock


def run_module1_module2(
    reference_bgr: np.ndarray,
    new_bgr: np.ndarray,
    module2_processor: Module2Processor | None = None,
) -> dict:
    """
    Run the Module 1 -> Module 2 pipeline on a pair of BGR images.

    Expected flow:
    1. Validate inputs.
    2. Convert both images to grayscale for feature matching.
    3. Align the new image to the reference image with Module 1.
    4. Run Module 2 on the aligned BGR pair.
    5. Return the alignment data and Module 2 payload.
    """
    if reference_bgr is None:
        raise ValueError("reference_bgr is None")
    if new_bgr is None:
        raise ValueError("new_bgr is None")
    if reference_bgr.ndim != 3 or reference_bgr.shape[2] != 3:
        raise ValueError(
            f"reference_bgr must be a BGR image with 3 channels, got shape {reference_bgr.shape}"
        )
    if new_bgr.ndim != 3 or new_bgr.shape[2] != 3:
        raise ValueError(
            f"new_bgr must be a BGR image with 3 channels, got shape {new_bgr.shape}"
        )

    if module2_processor is None:
        module2_processor = Module2Processor()

    # Step 1: prepare grayscale views for Module 1.
    reference_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor(new_bgr, cv2.COLOR_BGR2GRAY)

    # Step 2: compute alignment with Module 1.
    aligned_gray, homography = The_Lock(reference_gray, new_gray)

    # Step 3: warp the original BGR image using the homography.
    if homography is None:
        raise ValueError("Module 1 failed to compute a valid homography")

    height, width = reference_bgr.shape[:2]
    aligned_bgr = cv2.warpPerspective(new_bgr, homography, (width, height))

    # Step 4: run Module 2 on the reference image and aligned image.
    module2_payload = module2_processor.process_pair_for_module3(
        reference_bgr,
        aligned_bgr,
    )

    return {
        "reference_gray": reference_gray,
        "new_gray": new_gray,
        "aligned_gray": aligned_gray,
        "homography": homography,
        "aligned_bgr": aligned_bgr,
        "module2_payload": module2_payload,
    }

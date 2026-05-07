from __future__ import annotations

from dataclasses import dataclass
import logging

import cv2 as cv
import numpy as np


LOGGER = logging.getLogger(__name__)
GEOMETRY_FAILURE = "GEOMETRY_FAILURE"
ALIGNMENT_OK = "OK"


@dataclass(frozen=True)
class AlignmentResult:
    """Strict image registration result in reference-image coordinates."""

    status: str
    aligned_image: np.ndarray | None
    homography: np.ndarray | None
    reprojection_error: float
    inlier_count: int
    match_count: int
    keypoints_reference: int
    keypoints_new: int
    inlier_ratio: float
    failure_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == ALIGNMENT_OK


def _geometry_failure(
    reason: str,
    *,
    keypoints_reference: int = 0,
    keypoints_new: int = 0,
    match_count: int = 0,
    inlier_count: int = 0,
    inlier_ratio: float = 0.0,
    reprojection_error: float = float("inf"),
) -> AlignmentResult:
    LOGGER.warning("Geometry Failure: %s", reason)
    return AlignmentResult(
        status=GEOMETRY_FAILURE,
        aligned_image=None,
        homography=None,
        reprojection_error=float(reprojection_error),
        inlier_count=int(inlier_count),
        match_count=int(match_count),
        keypoints_reference=int(keypoints_reference),
        keypoints_new=int(keypoints_new),
        inlier_ratio=float(inlier_ratio),
        failure_reason=reason,
    )


def register_images(
    reference_image: np.ndarray,
    new_image: np.ndarray,
    *,
    max_reprojection_error: float = 0.5,
    ransac_reprojection_threshold: float = 0.75,
    min_matches: int = 30,
    min_inliers: int = 20,
    min_inlier_ratio: float = 0.15,
    orb_features: int = 10000,
    rng_seed: int = 1337,
) -> AlignmentResult:
    """
    Align ``new_image`` to ``reference_image`` using ORB + RANSAC homography.

    The function is fail-closed: if any geometry quality gate fails, no aligned
    image is returned. All output coordinates are in the reference image space.
    """
    if reference_image is None:
        raise ValueError("reference_image is None")
    if new_image is None:
        raise ValueError("new_image is None")
    if reference_image.ndim != 2 or new_image.ndim != 2:
        raise ValueError("register_images expects grayscale 2D images")

    cv.setRNGSeed(int(rng_seed))
    detector = cv.ORB_create(
        nfeatures=int(orb_features),
        scaleFactor=1.1,
        nlevels=12,
        edgeThreshold=5,
        patchSize=31,
        fastThreshold=5,
    )

    kp_ref, des_ref = detector.detectAndCompute(reference_image, None)
    kp_new, des_new = detector.detectAndCompute(new_image, None)
    kp_ref_count = len(kp_ref)
    kp_new_count = len(kp_new)

    if des_ref is None or des_new is None:
        return _geometry_failure(
            "insufficient descriptors",
            keypoints_reference=kp_ref_count,
            keypoints_new=kp_new_count,
        )

    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(des_new, des_ref, k=2)
    matches = []
    for pair in knn:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < 0.75 * second.distance:
            matches.append(first)
    matches = sorted(matches, key=lambda m: (m.distance, m.queryIdx, m.trainIdx))

    if len(matches) < min_matches:
        return _geometry_failure(
            f"not enough ORB matches: {len(matches)} < {min_matches}",
            keypoints_reference=kp_ref_count,
            keypoints_new=kp_new_count,
            match_count=len(matches),
        )

    new_pts = np.float32([kp_new[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    ref_pts = np.float32([kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    homography, mask = cv.findHomography(
        new_pts,
        ref_pts,
        cv.RANSAC,
        ransac_reprojection_threshold,
        maxIters=10000,
        confidence=0.999,
    )
    if homography is None or mask is None:
        return _geometry_failure(
            "homography estimation failed",
            keypoints_reference=kp_ref_count,
            keypoints_new=kp_new_count,
            match_count=len(matches),
        )

    inliers = mask.ravel().astype(bool)
    inlier_count = int(inliers.sum())
    inlier_ratio = float(inlier_count / max(len(matches), 1))
    if inlier_count < min_inliers:
        return _geometry_failure(
            f"not enough RANSAC inliers: {inlier_count} < {min_inliers}",
            keypoints_reference=kp_ref_count,
            keypoints_new=kp_new_count,
            match_count=len(matches),
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
        )
    if inlier_ratio < min_inlier_ratio:
        return _geometry_failure(
            f"low RANSAC inlier ratio: {inlier_ratio:.3f} < {min_inlier_ratio:.3f}",
            keypoints_reference=kp_ref_count,
            keypoints_new=kp_new_count,
            match_count=len(matches),
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
        )

    projected = cv.perspectiveTransform(new_pts[inliers], homography)
    errors = np.linalg.norm(projected - ref_pts[inliers], axis=2).ravel()
    rms_error = float(np.sqrt(np.mean(errors**2))) if errors.size else float("inf")
    if rms_error > max_reprojection_error:
        return _geometry_failure(
            f"reprojection error {rms_error:.4f}px exceeds {max_reprojection_error:.4f}px",
            keypoints_reference=kp_ref_count,
            keypoints_new=kp_new_count,
            match_count=len(matches),
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            reprojection_error=rms_error,
        )

    height, width = reference_image.shape
    aligned = cv.warpPerspective(new_image, homography, (width, height))
    return AlignmentResult(
        status=ALIGNMENT_OK,
        aligned_image=aligned,
        homography=homography.astype(np.float64),
        reprojection_error=rms_error,
        inlier_count=inlier_count,
        match_count=len(matches),
        keypoints_reference=kp_ref_count,
        keypoints_new=kp_new_count,
        inlier_ratio=inlier_ratio,
    )


def The_Lock(
    reference_image: np.ndarray,
    new_image: np.ndarray,
    keypoints_detector=None,
):
    """
    Backward-compatible Module 1 entrypoint.

    ``keypoints_detector`` is accepted for the historical signature but strict
    registration always uses the deterministic ORB configuration above.
    """
    _ = keypoints_detector
    result = register_images(reference_image, new_image)
    if not result.ok:
        raise ValueError(f"Geometry Failure: {result.failure_reason}")
    return result.aligned_image, result.homography

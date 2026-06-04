from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import Any

import cv2
import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.stats import norm

from backend.Lock import ALIGNMENT_OK, GEOMETRY_FAILURE, AlignmentResult, register_images
from backend.change_tracker import ChangeTracker
from backend.change_types import ChangeDetection, ChangeDetectionResult, ChangeType
from backend.object_detector import (
    NoOpDetector,
    ObjectDetection,
    ObjectDetector,
    build_object_detector,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChangeDetectorConfig:
    """All deterministic thresholds used by alignment, detection, fusion, and tracking."""

    alignment_max_reprojection_error: float = 0.5
    alignment_ransac_threshold: float = 0.75
    alignment_min_matches: int = 30
    alignment_min_inliers: int = 20
    alignment_min_inlier_ratio: float = 0.15
    alignment_orb_features: int = 10000
    alignment_rng_seed: int = 1337

    patch_size: int = 16
    min_change_area: int = 512
    min_confidence_score: float = 0.95
    ground_entropy_delta_threshold: float = 0.20
    ground_high_entropy_override: float = 0.45
    ground_ssim_threshold: float = 0.90
    ground_zncc_max: float = 0.94
    ground_closed_loop_area_ratio: float = 0.15

    object_canny_low: int = 55
    object_canny_high: int = 140
    object_close_kernel: int = 5
    object_min_area: int = 120
    object_compactness_max: float = 25.0
    object_gradient_density_min: float = 0.06
    object_gradient_density_ratio_min: float = 1.20
    object_detector_confidence_threshold: float = 0.35
    object_match_iou_threshold: float = 0.50
    object_same_class_required: bool = True
    object_classical_fallback_enabled: bool = True

    shadow_elongation_min: float = 2.5
    shadow_alignment_cosine: float = 0.82
    shadow_dark_delta: float = -0.04

    fusion_iou_threshold: float = 0.20
    fusion_object_confidence_min: float = 0.50
    temporal_iou_threshold: float = 0.35
    required_persistence: int = 3
    pvalue_floor_std: float = 1e-3
    save_debug_images: bool = False


def _validate_bgr(image: np.ndarray, name: str) -> None:
    """Reject missing or non-BGR images before running geometry or detection."""

    if image is None:
        raise ValueError(f"{name} is None")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must be a BGR image with 3 channels, got {image.shape}")


def _to_gray_u8(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR image to uint8 grayscale for OpenCV feature/edge operations."""

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def _to_gray_f32(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR image to normalized float32 grayscale in the range [0, 1]."""

    return _to_gray_u8(image_bgr).astype(np.float32) / 255.0


def _norm01(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize an array into float32 [0, 1], returning zeros if flat."""

    arr = arr.astype(np.float32, copy=False)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _sobel_mag(gray_f32: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude on a normalized grayscale image."""

    gx = cv2.Sobel(gray_f32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f32, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def _entropy_patch(patch: np.ndarray) -> float:
    """Estimate local texture entropy for one normalized image patch."""

    counts, _ = np.histogram(patch, bins=16, range=(0.0, 1.0))
    total = int(counts.sum())
    if total == 0:
        return 0.0
    probs = counts.astype(np.float64) / float(total)
    return float(scipy_entropy(probs, base=2))


def _ssim_patch(a: np.ndarray, b: np.ndarray) -> float:
    """Compute a lightweight SSIM score between two same-sized grayscale patches."""

    a64 = a.astype(np.float64, copy=False)
    b64 = b.astype(np.float64, copy=False)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_a = float(a64.mean())
    mu_b = float(b64.mean())
    var_a = float(((a64 - mu_a) ** 2).mean())
    var_b = float(((b64 - mu_b) ** 2).mean())
    cov = float(((a64 - mu_a) * (b64 - mu_b)).mean())
    denom = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    if denom <= 1e-12:
        return 1.0
    return float(((2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2)) / denom)


def _zncc(a: np.ndarray, b: np.ndarray) -> float:
    """Compute zero-mean normalized cross-correlation for illumination invariance."""

    a64 = a.astype(np.float64, copy=False)
    b64 = b.astype(np.float64, copy=False)
    a0 = a64 - float(a64.mean())
    b0 = b64 - float(b64.mean())
    a_std = float(np.sqrt((a0 * a0).mean()))
    b_std = float(np.sqrt((b0 * b0).mean()))
    if a_std < 1e-9 and b_std < 1e-9:
        return 1.0
    if a_std < 1e-9 or b_std < 1e-9:
        return 0.0
    return float(np.clip((a0 * b0).mean() / (a_std * b_std), -1.0, 1.0))


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return an inclusive mask bounding box encoded as (x, y, width, height)."""

    ys, xs = np.where(mask)
    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    """Compute area for a bounding box encoded as (x, y, width, height)."""

    return max(0, int(box[2])) * max(0, int(box[3]))


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Compute intersection-over-union for two (x, y, width, height) boxes."""

    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = _bbox_area(a) + _bbox_area(b) - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _is_object_change(change_type: ChangeType) -> bool:
    """Return True for semantic object add/remove candidates."""

    return change_type in {ChangeType.OBJECT_ADDED, ChangeType.OBJECT_REMOVED, ChangeType.OBJECT}


def _object_classes_match(
    before: ObjectDetection,
    after: ObjectDetection,
    same_class_required: bool,
) -> bool:
    """Decide whether two detector outputs can represent the same object."""

    if not same_class_required:
        return True
    if before.class_id >= 0 and after.class_id >= 0:
        return before.class_id == after.class_id
    return before.class_name == after.class_name


def _alignment_confidence(
    alignment: AlignmentResult,
    config: ChangeDetectorConfig,
) -> float:
    """Convert strict alignment diagnostics into a conservative [0, 1] score."""

    if not alignment.ok:
        return 0.0
    max_error = max(float(config.alignment_max_reprojection_error), 1e-6)
    error_fraction = float(np.clip(alignment.reprojection_error / max_error, 0.0, 1.0))
    inlier_floor = float(config.alignment_min_inlier_ratio)
    inlier_score = (alignment.inlier_ratio - inlier_floor) / max(1.0 - inlier_floor, 1e-6)
    inlier_score = float(np.clip(inlier_score, 0.0, 1.0))

    # Reaching the geometry gate already means the pair is usable. Keep the
    # score high enough that valid borderline alignments do not zero out strong
    # visual evidence, while still reflecting alignment quality.
    return float(np.clip(0.90 + 0.07 * (1.0 - error_fraction) + 0.03 * inlier_score, 0.0, 1.0))


def _rect_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    """Create a boolean rectangle mask clipped to image bounds."""

    height, width = shape
    x, y, w, h = box
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(width, int(x + w))
    y1 = min(height, int(y + h))
    mask = np.zeros((height, width), dtype=bool)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def _bbox_valid_ratio(valid_mask: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """Return how much of a bbox is inside the valid warped-overlap mask."""

    mask = _rect_mask(valid_mask.shape, box)
    if not mask.any():
        return 0.0
    return float(valid_mask[mask].mean())


def _serialize_object_detection(det: ObjectDetection) -> dict[str, Any]:
    """Small debug payload for detector outputs."""

    return {
        "bbox": det.bbox,
        "class_id": det.class_id,
        "class_name": det.class_name,
        "confidence": float(det.confidence),
    }


def _p_value_from_background(
    score: float,
    background_scores: list[float],
    floor_std: float,
) -> float:
    """Convert a region score into a one-sided p-value using background scores."""

    bg = np.asarray([s for s in background_scores if np.isfinite(s)], dtype=np.float64)
    if bg.size < 5:
        mu = 0.0
        sigma = max(floor_std, 0.05)
    else:
        mu = float(np.median(bg))
        mad = float(np.median(np.abs(bg - mu)))
        sigma = max(1.4826 * mad, float(bg.std()), floor_std)
    z = (float(score) - mu) / sigma
    return float(np.clip(norm.sf(z), 1e-12, 1.0))


def _has_closed_loop_structure(
    ref_roi: np.ndarray,
    new_roi: np.ndarray,
    region_area: int,
    area_ratio_threshold: float,
) -> bool:
    """Detect object-like closed edge loops so Path A does not claim them as ground."""

    if ref_roi.size == 0 or new_roi.size == 0:
        return False
    ref_edges = cv2.Canny((ref_roi * 255.0).astype(np.uint8), 55, 140)
    new_edges = cv2.Canny((new_roi * 255.0).astype(np.uint8), 55, 140)
    delta_edges = cv2.bitwise_and(
        new_edges,
        cv2.bitwise_not(cv2.dilate(ref_edges, np.ones((3, 3), np.uint8))),
    )
    delta_edges = cv2.morphologyEx(
        delta_edges,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
    )
    contours, _ = cv2.findContours(delta_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(20.0, float(region_area) * area_ratio_threshold)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1e-6:
            continue
        compactness = (perimeter * perimeter) / max(area, 1.0)
        if compactness <= 25.0:
            return True
    return False


def _estimate_shadow_direction(gray_f32: np.ndarray) -> np.ndarray | None:
    """Infer a coarse scene shadow direction from the low-frequency luminance field."""

    blurred = cv2.GaussianBlur(gray_f32, (0, 0), sigmaX=25.0, sigmaY=25.0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    vec = np.array([-float(gx.mean()), -float(gy.mean())], dtype=np.float64)
    norm_value = float(np.linalg.norm(vec))
    if norm_value < 1e-6:
        return None
    return vec / norm_value


def _mask_major_axis(mask: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Estimate the dominant axis and elongation ratio of a binary region mask."""

    ys, xs = np.where(mask)
    if xs.size < 5:
        return None, 1.0
    coords = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    coords -= coords.mean(axis=0, keepdims=True)
    cov = np.cov(coords, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vec = vecs[:, order[0]]
    elongation = float(np.sqrt(max(vals[0], 1e-9) / max(vals[-1], 1e-9)))
    return vec.astype(np.float64), elongation


def _is_shadow_like(
    mask: np.ndarray,
    ref_gray: np.ndarray,
    new_gray: np.ndarray,
    shadow_direction: np.ndarray | None,
    config: ChangeDetectorConfig,
) -> bool:
    """Suppress dark elongated regions aligned with the estimated shadow direction."""

    if shadow_direction is None:
        return False
    mean_delta = float((new_gray[mask] - ref_gray[mask]).mean())
    if mean_delta > config.shadow_dark_delta:
        return False
    major_axis, elongation = _mask_major_axis(mask)
    if major_axis is None or elongation < config.shadow_elongation_min:
        return False
    cosine = abs(float(np.dot(major_axis, shadow_direction)))
    return cosine >= config.shadow_alignment_cosine


class DualPathChangeDetector:
    """Deterministic two-path change detector with strict geometry gating."""

    def __init__(
        self,
        config: ChangeDetectorConfig | None = None,
        object_detector: ObjectDetector | None = None,
    ) -> None:
        """Create a detector using explicit thresholds or the default conservative config."""

        self.config = config or ChangeDetectorConfig()
        self.object_detector = object_detector or build_object_detector(
            confidence_threshold=self.config.object_detector_confidence_threshold,
        )
        self.module2_processor = None

    def detect(
        self,
        reference_bgr: np.ndarray,
        new_bgr: np.ndarray,
        frame_id: Any = None,
    ) -> ChangeDetectionResult:
        """Align a pair, run both detection paths, fuse results, and return diagnostics."""

        _validate_bgr(reference_bgr, "reference_bgr")
        _validate_bgr(new_bgr, "new_bgr")

        reference_gray = _to_gray_u8(reference_bgr)
        new_gray = _to_gray_u8(new_bgr)
        alignment = register_images(
            reference_gray,
            new_gray,
            max_reprojection_error=self.config.alignment_max_reprojection_error,
            ransac_reprojection_threshold=self.config.alignment_ransac_threshold,
            min_matches=self.config.alignment_min_matches,
            min_inliers=self.config.alignment_min_inliers,
            min_inlier_ratio=self.config.alignment_min_inlier_ratio,
            orb_features=self.config.alignment_orb_features,
            rng_seed=self.config.alignment_rng_seed,
        )

        if not alignment.ok or alignment.homography is None:
            LOGGER.warning("Geometry Failure: no detections emitted")
            return ChangeDetectionResult(
                status=GEOMETRY_FAILURE,
                alignment_result=alignment,
                ground_candidates=[],
                object_candidates=[],
                final_detections=[],
                rejected_candidates=[],
                debug_info={"failure_reason": alignment.failure_reason},
            )

        height, width = reference_bgr.shape[:2]
        aligned_bgr = cv2.warpPerspective(new_bgr, alignment.homography, (width, height))
        valid_mask = self._valid_overlap_mask(new_bgr.shape[:2], (height, width), alignment.homography)
        alignment_score = _alignment_confidence(alignment, self.config)
        debug_info: dict[str, Any] = {
            "alignment_confidence": alignment_score,
            "valid_overlap_ratio": float(valid_mask.mean()),
            "save_debug_images": self.config.save_debug_images,
        }
        
        module2_payload = None
        # Module 3 must operate on the "normalized image pair": both the
        # reference and the aligned-new image after shadow removal. When a
        # Module 2 processor is wired in, both detection paths (A ground and
        # B objects) run on the shadow-removed BGR pair. Without Module 2 we
        # fall back to the raw aligned pair so the deterministic, AI-free path
        # (and its tests) keep working unchanged.
        ref_for_detection = reference_bgr
        new_for_detection = aligned_bgr
        if self.module2_processor is not None:
            try:
                module2_payload = self.module2_processor.process_pair_for_module3(reference_bgr, aligned_bgr)
                ref_for_detection = module2_payload["debug"]["a_final_bgr"]
                new_for_detection = module2_payload["debug"]["b_final_bgr"]
                debug_info["module2_used"] = True
                debug_info["detection_input"] = "module2_shadow_removed"
            except Exception as e:
                LOGGER.warning(f"Module 2 failed: {e}")
                debug_info["module2_error"] = str(e)
                debug_info["detection_input"] = "raw_aligned_module2_failed"
        else:
            debug_info["detection_input"] = "raw_aligned"

        ground = self._detect_ground(
            ref_for_detection,
            new_for_detection,
            valid_mask,
            alignment_confidence=alignment_score,
            module2_payload=module2_payload,
        )
        objects = self._detect_objects(
            ref_for_detection,
            new_for_detection,
            valid_mask,
            alignment_confidence=alignment_score,
            debug_info=debug_info,
        )
        fused, rejected, fusion_decisions = self._fuse(ground, objects, return_debug=True)
        debug_info.update(
            {
                "ground_candidate_count": len(ground),
                "object_candidate_count": len(objects),
                "final_detection_count": len(fused),
                "rejected_candidate_count": len(rejected),
                "fusion_decisions": fusion_decisions,
                "final_detections": [
                    {
                        "bbox": det.bbox,
                        "change_type": det.change_type.value,
                        "confidence_score": det.confidence_score,
                        "source": det.source,
                        "label": det.label,
                        "reason": det.reason,
                    }
                    for det in fused
                ],
            }
        )

        return ChangeDetectionResult(
            status=ALIGNMENT_OK,
            alignment_result=alignment,
            ground_candidates=ground,
            object_candidates=objects,
            final_detections=fused,
            rejected_candidates=rejected,
            aligned_bgr=aligned_bgr,
            debug_info=debug_info,
        )

    @staticmethod
    def _valid_overlap_mask(
        source_shape: tuple[int, int],
        target_shape: tuple[int, int],
        homography: np.ndarray,
    ) -> np.ndarray:
        """Return pixels in reference space that are backed by valid warped source pixels."""

        source_h, source_w = source_shape
        target_h, target_w = target_shape
        source_valid = np.full((source_h, source_w), 255, dtype=np.uint8)
        warped = cv2.warpPerspective(source_valid, homography, (target_w, target_h))
        mask = warped > 0
        mask = cv2.erode(mask.astype(np.uint8), np.ones((17, 17), np.uint8), iterations=1) > 0
        return mask

    def _detect_ground(
        self,
        reference_bgr: np.ndarray,
        aligned_bgr: np.ndarray,
        valid_mask: np.ndarray | None = None,
        alignment_confidence: float = 1.0,
        module2_payload: dict | None = None,
    ) -> list[ChangeDetection]:
        """Path A: find non-structural texture changes on valid overlapping pixels."""

        ref = _to_gray_f32(reference_bgr)
        new = _to_gray_f32(aligned_bgr)
        height, width = ref.shape
        if valid_mask is None:
            valid_mask = np.ones((height, width), dtype=bool)
        patch = int(self.config.patch_size)
        candidate_mask = np.zeros((height, width), dtype=np.uint8)
        score_map = np.zeros((height, width), dtype=np.float32)
        background_scores: list[float] = []

        for y in range(0, height, patch):
            for x in range(0, width, patch):
                y2 = min(y + patch, height)
                x2 = min(x + patch, width)
                if y2 - y < patch // 2 or x2 - x < patch // 2:
                    continue
                patch_valid = valid_mask[y:y2, x:x2]
                if float(patch_valid.mean()) < 0.95:
                    continue
                ref_patch = ref[y:y2, x:x2]
                new_patch = new[y:y2, x:x2]

                entropy_delta = _entropy_patch(new_patch) - _entropy_patch(ref_patch)
                ssim = _ssim_patch(ref_patch, new_patch)
                zncc = _zncc(ref_patch, new_patch)
                ssim_drop = max(0.0, 1.0 - ssim)
                score = max(0.0, entropy_delta) + ssim_drop
                score_map[y:y2, x:x2] = score

                light_invariant_match = zncc > self.config.ground_zncc_max
                candidate = (
                    entropy_delta >= self.config.ground_entropy_delta_threshold
                    and ssim <= self.config.ground_ssim_threshold
                    and (
                        not light_invariant_match
                        or entropy_delta >= self.config.ground_high_entropy_override
                    )
                )
                
                if candidate:
                    candidate_mask[y:y2, x:x2] = 255
                else:
                    background_scores.append(score)

        kernel = np.ones((3, 3), np.uint8)
        candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel)
        candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel)
        candidate_mask = np.where(valid_mask, candidate_mask, 0).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, 8)
        detections: list[ChangeDetection] = []
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.config.min_change_area:
                continue
            mask = labels == label
            if float(valid_mask[mask].mean()) < 0.98:
                continue
            x, y, w, h = (
                int(stats[label, cv2.CC_STAT_LEFT]),
                int(stats[label, cv2.CC_STAT_TOP]),
                int(stats[label, cv2.CC_STAT_WIDTH]),
                int(stats[label, cv2.CC_STAT_HEIGHT]),
            )
            if _has_closed_loop_structure(
                ref[y : y + h, x : x + w],
                new[y : y + h, x : x + w],
                area,
                self.config.ground_closed_loop_area_ratio,
            ):
                continue

            region_score = float(score_map[mask].mean())
            p_value = _p_value_from_background(
                region_score,
                background_scores,
                self.config.pvalue_floor_std,
            )
            confidence = float((1.0 - p_value) * (0.95 + 0.05 * alignment_confidence))
            if confidence < self.config.min_confidence_score:
                continue
            detections.append(
                ChangeDetection(
                    bbox=(x, y, w, h),
                    change_type=ChangeType.GROUND_CHANGE,
                    confidence_score=confidence,
                    source="path_a",
                    mask=mask,
                    reason="Patch texture/structure statistics changed after illumination and geometry gates",
                    p_value=p_value,
                    metadata={
                        "score": region_score,
                        "area": area,
                        "path": "A",
                        "alignment_confidence": alignment_confidence,
                    },
                )
            )
        return detections

    def _detect_objects(
        self,
        reference_bgr: np.ndarray,
        aligned_bgr: np.ndarray,
        valid_mask: np.ndarray | None = None,
        alignment_confidence: float = 1.0,
        debug_info: dict[str, Any] | None = None,
    ) -> list[ChangeDetection]:
        """Path B: compare detected objects before/after in reference coordinates."""

        if valid_mask is None:
            valid_mask = np.ones(reference_bgr.shape[:2], dtype=bool)
        if debug_info is None:
            debug_info = {}

        before_objects = self.object_detector.detect(reference_bgr)
        after_objects = self.object_detector.detect(aligned_bgr)
        debug_info["object_detections_before"] = [
            _serialize_object_detection(det) for det in before_objects
        ]
        debug_info["object_detections_after"] = [
            _serialize_object_detection(det) for det in after_objects
        ]

        object_changes = self._compare_object_detections(
            before_objects,
            after_objects,
            valid_mask,
            alignment_confidence,
            debug_info,
        )
        if object_changes or not isinstance(self.object_detector, NoOpDetector):
            return object_changes

        if not self.config.object_classical_fallback_enabled:
            return []

        fallback = self._detect_classical_structural_objects(
            reference_bgr,
            aligned_bgr,
            valid_mask,
            alignment_confidence=alignment_confidence,
        )
        debug_info["classical_object_fallback_count"] = len(fallback)
        return fallback

    def _compare_object_detections(
        self,
        before_objects: list[ObjectDetection],
        after_objects: list[ObjectDetection],
        valid_mask: np.ndarray,
        alignment_confidence: float,
        debug_info: dict[str, Any],
    ) -> list[ChangeDetection]:
        """Convert detector before/after differences into object change candidates."""

        filtered_before = self._filter_detector_outputs(before_objects, valid_mask, "before", debug_info)
        filtered_after = self._filter_detector_outputs(after_objects, valid_mask, "after", debug_info)
        candidate_pairs: list[tuple[float, int, int]] = []
        for before_idx, before in enumerate(filtered_before):
            for after_idx, after in enumerate(filtered_after):
                if not _object_classes_match(before, after, self.config.object_same_class_required):
                    continue
                overlap = _iou(before.bbox, after.bbox)
                if overlap >= self.config.object_match_iou_threshold:
                    candidate_pairs.append((overlap, before_idx, after_idx))

        matched_before: set[int] = set()
        matched_after: set[int] = set()
        for overlap, before_idx, after_idx in sorted(candidate_pairs, reverse=True):
            if before_idx in matched_before or after_idx in matched_after:
                continue
            matched_before.add(before_idx)
            matched_after.add(after_idx)
            debug_info.setdefault("object_match_decisions", []).append(
                {
                    "before_bbox": filtered_before[before_idx].bbox,
                    "after_bbox": filtered_after[after_idx].bbox,
                    "iou": overlap,
                    "decision": "same_object_no_change",
                }
            )

        detections: list[ChangeDetection] = []
        for after_idx, after in enumerate(filtered_after):
            if after_idx not in matched_after:
                detections.append(
                    self._object_change_from_detection(
                        after,
                        ChangeType.OBJECT_ADDED,
                        alignment_confidence,
                        "Object detected only in aligned new image",
                    )
                )
        for before_idx, before in enumerate(filtered_before):
            if before_idx not in matched_before:
                detections.append(
                    self._object_change_from_detection(
                        before,
                        ChangeType.OBJECT_REMOVED,
                        alignment_confidence,
                        "Object detected only in reference image",
                    )
                )
        return sorted(detections, key=lambda d: (d.bbox[1], d.bbox[0], d.change_type.value))

    def _filter_detector_outputs(
        self,
        detections: list[ObjectDetection],
        valid_mask: np.ndarray,
        side: str,
        debug_info: dict[str, Any],
    ) -> list[ObjectDetection]:
        """Apply confidence and valid-overlap gates to raw object detections."""

        filtered: list[ObjectDetection] = []
        for det in detections:
            if det.confidence < self.config.object_detector_confidence_threshold:
                debug_info.setdefault("rejected_object_detections", []).append(
                    {
                        "side": side,
                        "bbox": det.bbox,
                        "reason": "detector confidence below threshold",
                        "confidence": float(det.confidence),
                    }
                )
                continue
            valid_ratio = _bbox_valid_ratio(valid_mask, det.bbox)
            if valid_ratio < 0.98:
                debug_info.setdefault("rejected_object_detections", []).append(
                    {
                        "side": side,
                        "bbox": det.bbox,
                        "reason": "bbox outside valid aligned overlap",
                        "valid_ratio": valid_ratio,
                    }
                )
                continue
            filtered.append(det)
        return filtered

    def _object_change_from_detection(
        self,
        detection: ObjectDetection,
        change_type: ChangeType,
        alignment_confidence: float,
        reason: str,
    ) -> ChangeDetection:
        """Convert an ObjectDetection into a public ChangeDetection."""

        confidence = float(
            np.clip(
                detection.confidence * (0.95 + 0.05 * alignment_confidence),
                0.0,
                1.0,
            )
        )
        label = detection.class_name if detection.class_name != "unknown" else None
        return ChangeDetection(
            bbox=detection.bbox,
            change_type=change_type,
            confidence_score=confidence,
            source="path_b",
            label=label,
            mask=None,
            reason=reason,
            p_value=1.0 - confidence,
            compactness_passed=True,
            metadata={
                "class_id": detection.class_id,
                "class_name": detection.class_name,
                "detector_confidence": float(detection.confidence),
                "alignment_confidence": alignment_confidence,
                "path": "B",
            },
        )

    def _detect_classical_structural_objects(
        self,
        reference_bgr: np.ndarray,
        aligned_bgr: np.ndarray,
        valid_mask: np.ndarray | None = None,
        alignment_confidence: float = 1.0,
    ) -> list[ChangeDetection]:
        """Path B: find compact structural objects using edge and gradient heuristics."""

        ref = _to_gray_f32(reference_bgr)
        new = _to_gray_f32(aligned_bgr)
        if valid_mask is None:
            valid_mask = np.ones(ref.shape, dtype=bool)
        ref_u8 = (ref * 255.0).astype(np.uint8)
        new_u8 = (new * 255.0).astype(np.uint8)
        ref_edges = cv2.Canny(ref_u8, self.config.object_canny_low, self.config.object_canny_high)
        new_edges = cv2.Canny(new_u8, self.config.object_canny_low, self.config.object_canny_high)
        ref_edge_guard = cv2.dilate(ref_edges, np.ones((3, 3), np.uint8), iterations=1)
        edge_delta = cv2.bitwise_and(new_edges, cv2.bitwise_not(ref_edge_guard))

        close_size = max(3, int(self.config.object_close_kernel) | 1)
        close_kernel = np.ones((close_size, close_size), np.uint8)
        closed = cv2.morphologyEx(edge_delta, cv2.MORPH_CLOSE, close_kernel)
        closed = cv2.dilate(closed, np.ones((3, 3), np.uint8), iterations=1)
        closed = np.where(valid_mask, closed, 0).astype(np.uint8)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        grad_new = _norm01(_sobel_mag(new))
        grad_delta = _norm01(np.abs(_sobel_mag(new) - _sobel_mag(ref)))
        grad_threshold = max(0.12, float(np.percentile(grad_new, 75.0)))
        background_scores = grad_delta[closed == 0].astype(np.float32).ravel().tolist()
        shadow_direction = _estimate_shadow_direction(new)
        detections: list[ChangeDetection] = []

        for contour in contours:
            if len(contour) < 3:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            region_mask = np.zeros(ref.shape, dtype=np.uint8)
            cv2.drawContours(region_mask, [contour], -1, 255, thickness=-1)
            mask = region_mask > 0
            if float(valid_mask[mask].mean()) < 0.98:
                continue
            area = int(mask.sum())
            if area < self.config.object_min_area:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            compactness = float((perimeter * perimeter) / max(float(area), 1.0))
            compactness_passed = compactness <= self.config.object_compactness_max
            if not compactness_passed:
                continue

            inner = cv2.erode(region_mask, np.ones((5, 5), np.uint8), iterations=1) > 0
            if not inner.any():
                inner = mask
            dilated = cv2.dilate(region_mask, np.ones((15, 15), np.uint8), iterations=1) > 0
            surround = dilated & (~mask)
            internal_density = float((grad_new[inner] > grad_threshold).mean())
            if surround.any():
                surround_density = float((grad_new[surround] > grad_threshold).mean())
            else:
                surround_density = 0.0
            density_ratio = internal_density / max(surround_density, 1e-3)
            if internal_density < self.config.object_gradient_density_min:
                continue
            if density_ratio < self.config.object_gradient_density_ratio_min:
                continue
            if _is_shadow_like(mask, ref, new, shadow_direction, self.config):
                continue

            score = float(grad_delta[mask].mean())
            p_value = _p_value_from_background(
                score,
                background_scores,
                self.config.pvalue_floor_std,
            )
            confidence = float((1.0 - p_value) * (0.95 + 0.05 * alignment_confidence))
            if confidence < self.config.min_confidence_score:
                continue
            detections.append(
                ChangeDetection(
                    bbox=(int(x), int(y), int(w), int(h)),
                    change_type=ChangeType.OBJECT_ADDED,
                    confidence_score=confidence,
                    source="path_b_classical",
                    mask=mask,
                    reason="Classical structural fallback found new compact edge/gradient region",
                    p_value=p_value,
                    compactness=compactness,
                    gradient_density=internal_density,
                    compactness_passed=True,
                    metadata={
                        "area": area,
                        "density_ratio": density_ratio,
                        "path": "B",
                        "alignment_confidence": alignment_confidence,
                    },
                )
            )
        return detections

    def _fuse(
        self,
        ground: list[ChangeDetection],
        objects: list[ChangeDetection],
        return_debug: bool = False,
    ) -> list[ChangeDetection] | tuple[list[ChangeDetection], list[ChangeDetection], list[dict[str, Any]]]:
        """Resolve overlapping Path A/Path B detections with object priority."""

        kept_ground: list[ChangeDetection] = []
        rejected: list[ChangeDetection] = []
        decisions: list[dict[str, Any]] = []
        for ground_detection in ground:
            suppressing_object: ChangeDetection | None = None
            suppressing_iou = 0.0
            for obj in objects:
                overlap = _iou(ground_detection.bbox, obj.bbox)
                high_confidence_object = obj.confidence_score >= self.config.fusion_object_confidence_min
                if (
                    _is_object_change(obj.change_type)
                    and high_confidence_object
                    and overlap >= self.config.fusion_iou_threshold
                    and overlap > suppressing_iou
                ):
                    suppressing_object = obj
                    suppressing_iou = overlap

            if suppressing_object is None:
                kept_ground.append(ground_detection)
                continue

            rejected_candidate = replace(
                ground_detection,
                source="fusion",
                reason="Suppressed by overlapping high-confidence object candidate",
            )
            rejected.append(rejected_candidate)
            decisions.append(
                {
                    "ground_bbox": ground_detection.bbox,
                    "object_bbox": suppressing_object.bbox,
                    "iou": suppressing_iou,
                    "decision": "prefer_object",
                }
            )

        fused = [*objects, *kept_ground]
        sorted_fused = sorted(
            fused,
            key=lambda d: (
                d.bbox[1],
                d.bbox[0],
                0 if _is_object_change(d.change_type) else 1,
            ),
        )
        if return_debug:
            return sorted_fused, rejected, decisions
        return sorted_fused


def run_dual_path_change_detection(
    reference_bgr: np.ndarray,
    new_bgr: np.ndarray,
    tracker: ChangeTracker | None = None,
    config: ChangeDetectorConfig | None = None,
    frame_id: Any = None,
    object_detector: ObjectDetector | None = None,
) -> ChangeDetectionResult:
    """Convenience API for one detection call, optionally applying temporal tracking."""

    detector = DualPathChangeDetector(config, object_detector=object_detector)
    result = detector.detect(reference_bgr, new_bgr, frame_id=frame_id)
    if tracker is not None and result.status != GEOMETRY_FAILURE:
        result.real_detections = tracker.update(result.detections, frame_id=frame_id)
    return result

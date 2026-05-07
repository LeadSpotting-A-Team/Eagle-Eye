from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import logging
from typing import Any

import cv2
import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.stats import norm

from backend.Lock import ALIGNMENT_OK, GEOMETRY_FAILURE, AlignmentResult, register_images


LOGGER = logging.getLogger(__name__)


class ChangeType(str, Enum):
    """Supported deterministic change categories emitted by the detector."""

    GROUND = "Ground Change"
    OBJECT = "Object"


@dataclass
class ChangeDetection:
    """Single detected changed region in reference-image coordinates."""

    bounding_box: tuple[int, int, int, int]
    change_type: ChangeType
    confidence_score: float
    p_value: float
    persistence_count: int = 1
    is_real: bool = False
    compactness: float | None = None
    gradient_density: float | None = None
    compactness_passed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChangeDetectionResult:
    """Full result for one before/after pair, including alignment diagnostics."""

    status: str
    detections: list[ChangeDetection]
    alignment: AlignmentResult
    real_detections: list[ChangeDetection] = field(default_factory=list)
    aligned_bgr: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


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

    shadow_elongation_min: float = 2.5
    shadow_alignment_cosine: float = 0.82
    shadow_dark_delta: float = -0.04

    fusion_iou_threshold: float = 0.20
    temporal_iou_threshold: float = 0.35
    required_persistence: int = 3
    pvalue_floor_std: float = 1e-3


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


class ChangeTracker:
    """Stateful 3-frame consensus tracker in reference-image coordinates."""

    def __init__(
        self,
        required_persistence: int = 3,
        iou_threshold: float = 0.35,
    ) -> None:
        """Create a tracker that promotes detections after consecutive IoU matches."""

        self.required_persistence = int(required_persistence)
        self.iou_threshold = float(iou_threshold)
        self._tracks: list[dict[str, Any]] = []
        self._ordinal = 0

    def update(
        self,
        detections: list[ChangeDetection],
        frame_id: Any = None,
    ) -> list[ChangeDetection]:
        """Update tracks with one frame of detections and return only real changes."""

        ordinal = self._ordinal
        self._ordinal += 1
        matched_tracks: set[int] = set()
        output: list[ChangeDetection] = []

        for detection in sorted(detections, key=lambda d: (d.bounding_box[1], d.bounding_box[0], d.change_type.value)):
            best_idx = None
            best_iou = 0.0
            for idx, track in enumerate(self._tracks):
                if idx in matched_tracks or track["change_type"] != detection.change_type:
                    continue
                overlap = _iou(track["bounding_box"], detection.bounding_box)
                if overlap >= self.iou_threshold and overlap > best_iou:
                    best_idx = idx
                    best_iou = overlap

            if best_idx is None:
                count = 1
                self._tracks.append(
                    {
                        "change_type": detection.change_type,
                        "bounding_box": detection.bounding_box,
                        "persistence_count": count,
                        "last_seen": ordinal,
                    }
                )
                best_idx = len(self._tracks) - 1
            else:
                track = self._tracks[best_idx]
                count = (
                    int(track["persistence_count"]) + 1
                    if int(track["last_seen"]) == ordinal - 1
                    else 1
                )
                track["bounding_box"] = detection.bounding_box
                track["persistence_count"] = count
                track["last_seen"] = ordinal

            matched_tracks.add(best_idx)
            tracked = replace(
                detection,
                persistence_count=count,
                is_real=count >= self.required_persistence,
            )
            tracked.metadata = dict(tracked.metadata)
            tracked.metadata["frame_id"] = frame_id
            output.append(tracked)

        max_gap = max(1, self.required_persistence)
        self._tracks = [
            track for track in self._tracks if ordinal - int(track["last_seen"]) <= max_gap
        ]
        return [detection for detection in output if detection.is_real]


class DualPathChangeDetector:
    """Deterministic two-path change detector with strict geometry gating."""

    def __init__(self, config: ChangeDetectorConfig | None = None) -> None:
        """Create a detector using explicit thresholds or the default conservative config."""

        self.config = config or ChangeDetectorConfig()

    def detect(
        self,
        reference_bgr: np.ndarray,
        new_bgr: np.ndarray,
        frame_id: Any = None,
    ) -> ChangeDetectionResult:
        """Align a pair, run both detection paths, fuse results, and return diagnostics."""

        _ = frame_id
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
                detections=[],
                alignment=alignment,
                diagnostics={"failure_reason": alignment.failure_reason},
            )

        height, width = reference_bgr.shape[:2]
        aligned_bgr = cv2.warpPerspective(new_bgr, alignment.homography, (width, height))
        valid_mask = self._valid_overlap_mask(new_bgr.shape[:2], (height, width), alignment.homography)
        ground = self._detect_ground(reference_bgr, aligned_bgr, valid_mask)
        objects = self._detect_objects(reference_bgr, aligned_bgr, valid_mask)
        fused = self._fuse(ground, objects)

        return ChangeDetectionResult(
            status=ALIGNMENT_OK,
            detections=fused,
            alignment=alignment,
            aligned_bgr=aligned_bgr,
            diagnostics={
                "ground_candidates": len(ground),
                "object_candidates": len(objects),
                "fused_detections": len(fused),
                "valid_overlap_ratio": float(valid_mask.mean()),
            },
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
            confidence = float(1.0 - p_value)
            if confidence < self.config.min_confidence_score:
                continue
            detections.append(
                ChangeDetection(
                    bounding_box=(x, y, w, h),
                    change_type=ChangeType.GROUND,
                    confidence_score=confidence,
                    p_value=p_value,
                    metadata={
                        "score": region_score,
                        "area": area,
                        "path": "A",
                    },
                )
            )
        return detections

    def _detect_objects(
        self,
        reference_bgr: np.ndarray,
        aligned_bgr: np.ndarray,
        valid_mask: np.ndarray | None = None,
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
            confidence = float(1.0 - p_value)
            if confidence < self.config.min_confidence_score:
                continue
            detections.append(
                ChangeDetection(
                    bounding_box=(int(x), int(y), int(w), int(h)),
                    change_type=ChangeType.OBJECT,
                    confidence_score=confidence,
                    p_value=p_value,
                    compactness=compactness,
                    gradient_density=internal_density,
                    compactness_passed=True,
                    metadata={
                        "area": area,
                        "density_ratio": density_ratio,
                        "path": "B",
                    },
                )
            )
        return detections

    def _fuse(
        self,
        ground: list[ChangeDetection],
        objects: list[ChangeDetection],
    ) -> list[ChangeDetection]:
        """Resolve overlapping Path A/Path B detections with object priority."""

        kept_ground: list[ChangeDetection] = []
        for ground_detection in ground:
            suppressed = any(
                obj.compactness_passed
                and _iou(ground_detection.bounding_box, obj.bounding_box)
                >= self.config.fusion_iou_threshold
                for obj in objects
            )
            if not suppressed:
                kept_ground.append(ground_detection)

        fused = [*objects, *kept_ground]
        return sorted(
            fused,
            key=lambda d: (
                d.bounding_box[1],
                d.bounding_box[0],
                0 if d.change_type == ChangeType.OBJECT else 1,
            ),
        )


def run_dual_path_change_detection(
    reference_bgr: np.ndarray,
    new_bgr: np.ndarray,
    tracker: ChangeTracker | None = None,
    config: ChangeDetectorConfig | None = None,
    frame_id: Any = None,
) -> ChangeDetectionResult:
    """Convenience API for one detection call, optionally applying temporal tracking."""

    detector = DualPathChangeDetector(config)
    result = detector.detect(reference_bgr, new_bgr, frame_id=frame_id)
    if tracker is not None:
        result.real_detections = tracker.update(result.detections, frame_id=frame_id)
    return result

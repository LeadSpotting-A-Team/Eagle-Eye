"""
backend/change_types.py
-----------------------
Public types shared across the dual-path change-detection pipeline.

This module contains *only* pure dataclasses and enums so it can be imported
from any context without pulling in CV or ML dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Change category taxonomy
# ---------------------------------------------------------------------------

class ChangeType(str, Enum):
    """Semantic category of a detected change candidate."""

    GROUND_CHANGE = "Ground Change"
    OBJECT_ADDED = "Object Added"
    OBJECT_REMOVED = "Object Removed"
    UNKNOWN_CHANGE = "Unknown Change"

    # Legacy aliases kept for backward compatibility with old code
    # that refers to ChangeType.GROUND / ChangeType.OBJECT.
    GROUND = "Ground Change"
    OBJECT = "Object Added"


def _coerce_change_type(value: ChangeType | str) -> ChangeType:
    """Accept current enum values plus the older ``Object`` string."""

    if isinstance(value, ChangeType):
        return value
    if value == "Object":
        return ChangeType.OBJECT_ADDED
    return ChangeType(value)


# ---------------------------------------------------------------------------
# Single change candidate
# ---------------------------------------------------------------------------

@dataclass(init=False)
class ChangeDetection:
    """
    One detected or inferred change region in reference-image coordinates.

    Fields
    ------
    bbox : (x, y, width, height) in reference-image pixels.
    change_type : semantic category.
    confidence_score : normalised [0, 1].
    source : which pipeline stage produced this candidate
        (e.g. ``"path_a"``, ``"path_b"``, ``"fusion"``).
    label : optional semantic label such as ``"car"`` or ``"person"``.
    mask : optional boolean ndarray with the same h×w as the reference image.
    frame_id : opaque frame identifier passed through from the caller.
    reason : human-readable explanation of why this region was flagged.

    Legacy fields (kept for backward compatibility with existing tests)
    ------------------------------------------------------------------
    bounding_box : same as bbox (older name).
    p_value : statistical p-value from background model (Path A / B).
    persistence_count : tracker consecutive-frame count.
    is_real : True once persistence_count >= required_persistence.
    compactness : perimeter²/area for object edge compactness gate.
    gradient_density : internal gradient density (Path B).
    compactness_passed : True when compactness gate passed.
    metadata : free-form diagnostic dict.
    """

    bbox: tuple[int, int, int, int]
    change_type: ChangeType
    confidence_score: float
    source: str = "unknown"
    label: str | None = None
    mask: np.ndarray | None = None
    frame_id: Any = None
    reason: str = ""

    # ------------------------------------------------------------------
    # Legacy / extended fields used by existing code and tests
    # ------------------------------------------------------------------
    p_value: float = 0.0
    persistence_count: int = 1
    is_real: bool = False
    compactness: float | None = None
    gradient_density: float | None = None
    compactness_passed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Backward-compatibility shim: ``bounding_box`` → ``bbox``
    # ------------------------------------------------------------------

    def __init__(
        self,
        bbox: tuple[int, int, int, int] | None = None,
        change_type: ChangeType | str | None = None,
        confidence_score: float = 0.0,
        source: str | float | None = "unknown",
        label: str | None = None,
        mask: np.ndarray | None = None,
        frame_id: Any = None,
        reason: str = "",
        p_value: float | None = None,
        persistence_count: int = 1,
        is_real: bool = False,
        compactness: float | None = None,
        gradient_density: float | None = None,
        compactness_passed: bool = False,
        metadata: dict[str, Any] | None = None,
        *,
        bounding_box: tuple[int, int, int, int] | None = None,
    ) -> None:
        if bbox is None:
            bbox = bounding_box
        elif bounding_box is not None and bbox != bounding_box:
            raise ValueError("bbox and bounding_box disagree")
        if bbox is None:
            raise TypeError("ChangeDetection requires bbox or bounding_box")
        if change_type is None:
            raise TypeError("ChangeDetection requires change_type")

        # Legacy constructor support:
        # ChangeDetection(box, type, confidence, p_value, ...)
        if source is not None and not isinstance(source, str) and p_value is None:
            p_value = float(source)
            source = "unknown"

        self.bbox = tuple(int(v) for v in bbox)
        self.change_type = _coerce_change_type(change_type)
        self.confidence_score = float(np.clip(confidence_score, 0.0, 1.0))
        self.source = source or "unknown"
        self.label = label
        self.mask = mask
        self.frame_id = frame_id
        self.reason = reason
        self.p_value = float(0.0 if p_value is None else p_value)
        self.persistence_count = int(persistence_count)
        self.is_real = bool(is_real)
        self.compactness = compactness
        self.gradient_density = gradient_density
        self.compactness_passed = bool(compactness_passed)
        self.metadata = dict(metadata or {})

    @property
    def bounding_box(self) -> tuple[int, int, int, int]:
        """Legacy alias for ``bbox``."""
        return self.bbox

    @bounding_box.setter
    def bounding_box(self, value: tuple[int, int, int, int]) -> None:
        self.bbox = value


# ---------------------------------------------------------------------------
# Full pipeline result
# ---------------------------------------------------------------------------

@dataclass(init=False)
class ChangeDetectionResult:
    """
    Complete output for one before/after image pair.

    Fields
    ------
    status : ``"OK"`` or ``"GEOMETRY_FAILURE"``.
    alignment_result : the AlignmentResult from Module 1 / Lock.py.
    ground_candidates : raw Path-A candidates (not yet confirmed).
    object_candidates : raw Path-B candidates (not yet confirmed).
    final_detections : fused, optionally tracker-confirmed detections.
    rejected_candidates : candidates removed during fusion / gating.
    debug_info : optional diagnostic payload.

    Legacy shim fields
    ------------------
    detections : alias for ``final_detections`` (old public API).
    real_detections : tracker-confirmed subset of detections.
    aligned_bgr : the warped new-image in reference space (BGR, uint8).
    diagnostics : alias for ``debug_info``.
    alignment : alias for ``alignment_result``.
    """

    status: str
    alignment_result: Any  # AlignmentResult — avoid circular import
    ground_candidates: list[ChangeDetection] = field(default_factory=list)
    object_candidates: list[ChangeDetection] = field(default_factory=list)
    final_detections: list[ChangeDetection] = field(default_factory=list)
    rejected_candidates: list[ChangeDetection] = field(default_factory=list)
    debug_info: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Legacy shim fields
    # ------------------------------------------------------------------
    real_detections: list[ChangeDetection] = field(default_factory=list)
    aligned_bgr: np.ndarray | None = None

    def __init__(
        self,
        status: str,
        alignment_result: Any | None = None,
        ground_candidates: list[ChangeDetection] | None = None,
        object_candidates: list[ChangeDetection] | None = None,
        final_detections: list[ChangeDetection] | None = None,
        rejected_candidates: list[ChangeDetection] | None = None,
        debug_info: dict[str, Any] | None = None,
        real_detections: list[ChangeDetection] | None = None,
        aligned_bgr: np.ndarray | None = None,
        *,
        alignment: Any | None = None,
        detections: list[ChangeDetection] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        if alignment_result is None:
            alignment_result = alignment
        elif alignment is not None and alignment_result is not alignment:
            raise ValueError("alignment_result and alignment disagree")
        if final_detections is None:
            final_detections = detections
        elif detections is not None and final_detections != detections:
            raise ValueError("final_detections and detections disagree")
        if debug_info is None:
            debug_info = diagnostics
        elif diagnostics is not None and debug_info != diagnostics:
            raise ValueError("debug_info and diagnostics disagree")

        self.status = status
        self.alignment_result = alignment_result
        self.ground_candidates = list(ground_candidates or [])
        self.object_candidates = list(object_candidates or [])
        self.final_detections = list(final_detections or [])
        self.rejected_candidates = list(rejected_candidates or [])
        self.debug_info = dict(debug_info or {})
        self.real_detections = list(real_detections or [])
        self.aligned_bgr = aligned_bgr

    # ------------------------------------------------------------------
    # Backward-compatibility properties
    # ------------------------------------------------------------------

    @property
    def detections(self) -> list[ChangeDetection]:
        """Legacy alias for ``final_detections``."""
        return self.final_detections

    @detections.setter
    def detections(self, value: list[ChangeDetection]) -> None:
        self.final_detections = value

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Legacy alias for ``debug_info``."""
        return self.debug_info

    @diagnostics.setter
    def diagnostics(self, value: dict[str, Any]) -> None:
        self.debug_info = value

    @property
    def alignment(self) -> Any:
        """Legacy alias for ``alignment_result``."""
        return self.alignment_result

    @alignment.setter
    def alignment(self, value: Any) -> None:
        self.alignment_result = value

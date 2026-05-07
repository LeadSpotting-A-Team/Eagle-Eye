"""
backend/change_tracker.py
--------------------------
Stateful temporal tracker that requires N consecutive successful detections
before a change is confirmed as real.

All coordinates are in reference-image space (i.e. after alignment / warping
into the reference frame) so that a moving camera does not cause spurious
track resets.

Design rules
~~~~~~~~~~~~
* A geometry-failure frame does NOT call ``update`` — the caller simply skips
  the frame.  The tracker therefore never increments persistence on bad frames.
* Two candidates are considered the "same change" only when:
    - their bounding boxes overlap at IoU >= ``iou_threshold``, AND
    - their ``change_type`` matches exactly.
* The EWMA bounding box smoothing is intentionally not applied: we keep the
  latest box so that growing/shrinking regions are tracked correctly.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from backend.change_types import ChangeDetection, ChangeType

LOGGER = logging.getLogger(__name__)


def _iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """Intersection-over-Union for two (x, y, w, h) bounding boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


class _Track:
    """Internal mutable state for one tracked candidate."""

    __slots__ = (
        "change_type",
        "bbox",
        "persistence_count",
        "last_ordinal",
    )

    def __init__(
        self,
        change_type: ChangeType,
        bbox: tuple[int, int, int, int],
        ordinal: int,
    ) -> None:
        self.change_type = change_type
        self.bbox = bbox
        self.persistence_count: int = 1
        self.last_ordinal: int = ordinal


class ChangeTracker:
    """
    Frame-to-frame persistence tracker in reference-image coordinates.

    Parameters
    ----------
    required_persistence : int
        Number of consecutive frames a detection must appear before it is
        classified as a real change (default 3).
    iou_threshold : float
        Minimum IoU to consider two bounding boxes the same region
        (default 0.35).
    """

    def __init__(
        self,
        required_persistence: int = 3,
        iou_threshold: float = 0.35,
    ) -> None:
        self.required_persistence = int(required_persistence)
        self.iou_threshold = float(iou_threshold)
        self._tracks: list[_Track] = []
        self._ordinal: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: list[ChangeDetection],
        frame_id: Any = None,
    ) -> list[ChangeDetection]:
        """
        Incorporate one frame of detections and return confirmed real changes.

        Call this method once per valid (non-geometry-failure) frame.
        Do not call it when the alignment result is ``GEOMETRY_FAILURE``.

        Parameters
        ----------
        detections : list[ChangeDetection]
            Fused candidates for this frame (in reference-image coordinates).
        frame_id : Any
            Opaque frame identifier that is forwarded to the returned objects.

        Returns
        -------
        list[ChangeDetection]
            Only those detections whose ``persistence_count`` has reached
            ``required_persistence``.  Each returned object has
            ``is_real=True`` and ``frame_id`` set.
        """
        ordinal = self._ordinal
        self._ordinal += 1

        matched_track_ids: set[int] = set()
        output: list[ChangeDetection] = []

        # Sort for determinism (top-left first, then by type value)
        sorted_detections = sorted(
            detections,
            key=lambda d: (d.bbox[1], d.bbox[0], d.change_type.value),
        )

        for det in sorted_detections:
            best_idx: int | None = None
            best_iou = 0.0

            for idx, track in enumerate(self._tracks):
                if idx in matched_track_ids:
                    continue
                if track.change_type != det.change_type:
                    continue
                overlap = _iou(track.bbox, det.bbox)
                if overlap >= self.iou_threshold and overlap > best_iou:
                    best_idx = idx
                    best_iou = overlap

            if best_idx is None:
                # New candidate — start a fresh track
                self._tracks.append(_Track(det.change_type, det.bbox, ordinal))
                best_idx = len(self._tracks) - 1
                count = 1
            else:
                track = self._tracks[best_idx]
                if track.last_ordinal == ordinal - 1:
                    # Consecutive frame → increment
                    count = track.persistence_count + 1
                else:
                    # Gap → reset
                    count = 1
                track.bbox = det.bbox
                track.persistence_count = count
                track.last_ordinal = ordinal

            matched_track_ids.add(best_idx)
            is_real = count >= self.required_persistence

            confirmed = replace(
                det,
                persistence_count=count,
                is_real=is_real,
                frame_id=frame_id,
            )
            # metadata is a plain dict; copy it so the original is not mutated
            confirmed.metadata = dict(confirmed.metadata)
            confirmed.metadata["frame_id"] = frame_id
            output.append(confirmed)

        # Prune stale tracks that have not been seen for required_persistence frames
        max_gap = max(1, self.required_persistence)
        self._tracks = [
            t for t in self._tracks
            if ordinal - t.last_ordinal <= max_gap
        ]

        LOGGER.debug(
            "ChangeTracker frame=%s detections=%d confirmed=%d tracks=%d",
            frame_id,
            len(detections),
            sum(1 for d in output if d.is_real),
            len(self._tracks),
        )

        return [d for d in output if d.is_real]

    def reset(self) -> None:
        """Discard all track history (e.g. scene change)."""
        self._tracks.clear()
        self._ordinal = 0

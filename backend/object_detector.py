"""
backend/object_detector.py
--------------------------
Adapter interface for per-image object detection used by Path B.

Design goals
~~~~~~~~~~~~
* **No hard dependency on AI packages.**  The core class ``ObjectDetector``
  and the ``NoOpDetector`` stub are pure-Python / OpenCV only.
* YOLO support is isolated in ``YoloDetector`` behind an explicit
  ``EAGLE_EYE_ENABLE_YOLO=1`` environment variable **and** a try/import
  guard so the rest of the project keeps working even if *ultralytics* is
  not installed.
* Callers always receive a ``list[ObjectDetection]`` regardless of which
  concrete detector is active.
"""

from __future__ import annotations

import logging
import os
from importlib import import_module
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.dependency_policy import check_policy

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detected object data
# ---------------------------------------------------------------------------

@dataclass
class ObjectDetection:
    """One object detected inside a single image (reference or aligned-new)."""

    bbox: tuple[int, int, int, int]   # (x, y, width, height) in image pixels
    class_id: int = -1
    class_name: str = "unknown"
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ObjectDetector:
    """
    Abstract adapter.  Subclasses must implement ``detect``.

    This interface keeps the rest of the pipeline independent from any
    specific AI framework.
    """

    def detect(self, image_bgr: np.ndarray) -> list[ObjectDetection]:
        """
        Run object detection on a BGR uint8 image.

        Returns
        -------
        list[ObjectDetection]
            One entry per detected object, in arbitrary order.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# No-op stub
# ---------------------------------------------------------------------------

class NoOpDetector(ObjectDetector):
    """
    Stub detector that always returns an empty list.

    Used when AI packages are not available or the project runs in
    classical-only mode.  The stub preserves the adapter interface so
    Path B can always call ``detector.detect(image)`` without conditional
    logic in the main pipeline.
    """

    def detect(self, image_bgr: np.ndarray) -> list[ObjectDetection]:  # noqa: ARG002
        return []


# ---------------------------------------------------------------------------
# YOLO detector (optional — guarded by env flag + try/import)
# ---------------------------------------------------------------------------

class YoloDetector(ObjectDetector):
    """
    YOLOv8-small detector backed by *ultralytics*.

    Activation
    ~~~~~~~~~~
    Set ``EAGLE_EYE_ENABLE_YOLO=1`` in the environment **and** install
    *ultralytics* (``pip install ultralytics``).  If either condition is
    not met the constructor raises ``RuntimeError`` and the caller should
    fall back to ``NoOpDetector``.

    Parameters
    ----------
    model_name : str
        Any YOLOv8 model identifier accepted by ``YOLO()``.
        Defaults to ``"yolov8s.pt"`` (small, good accuracy/speed balance).
    confidence_threshold : float
        Minimum detection confidence to include in results.
    device : str
        Inference device, e.g. ``"cpu"`` or ``"cuda:0"``.
    """

    def __init__(
        self,
        model_name: str = "yolov8s.pt",
        confidence_threshold: float = 0.35,
        device: str = "cpu",
    ) -> None:
        if not _yolo_enabled():
            raise RuntimeError(
                "YoloDetector is disabled.  "
                "Set EAGLE_EYE_ENABLE_YOLO=1 and install ultralytics to use it."
            )
        try:
            check_policy("ultralytics")
            yolo_module = import_module("ultralytics")
            YOLO = yolo_module.YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed.  Run: pip install ultralytics"
            ) from exc

        self._model = YOLO(model_name)
        self._model.to(device)
        self._conf = float(confidence_threshold)
        LOGGER.info("YoloDetector initialised with model=%s device=%s", model_name, device)

    def detect(self, image_bgr: np.ndarray) -> list[ObjectDetection]:
        results = self._model(image_bgr, conf=self._conf, verbose=False)
        detections: list[ObjectDetection] = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    cls_id = int(box.cls[0].item())
                    cls_name = (
                        result.names[cls_id]
                        if result.names and cls_id in result.names
                        else "unknown"
                    )
                    conf = float(box.conf[0].item())
                    detections.append(
                        ObjectDetection(
                            bbox=(x, y, w, h),
                            class_id=cls_id,
                            class_name=cls_name,
                            confidence=conf,
                        )
                    )
            elif result.obb is not None:
                for obb in result.obb:
                    corners = obb.xyxyxyxy[0].tolist()
                    xs = [p[0] for p in corners]
                    ys = [p[1] for p in corners]
                    x, y = int(min(xs)), int(min(ys))
                    w, h = int(max(xs) - x), int(max(ys) - y)
                    cls_id = int(obb.cls[0].item())
                    cls_name = (
                        result.names[cls_id]
                        if result.names and cls_id in result.names
                        else "unknown"
                    )
                    conf = float(obb.conf[0].item())
                    detections.append(
                        ObjectDetection(
                            bbox=(x, y, w, h),
                            class_id=cls_id,
                            class_name=cls_name,
                            confidence=conf,
                        )
                    )
        return detections


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def _yolo_enabled() -> bool:
    return os.environ.get("EAGLE_EYE_ENABLE_YOLO", "0").strip() == "1"


def build_object_detector(
    *,
    force_noop: bool = False,
    model_name: str = "yolov8s.pt",
    confidence_threshold: float = 0.35,
    device: str = "cpu",
) -> ObjectDetector:
    """
    Return the best available detector.

    If ``force_noop`` is True, or if YOLO is not enabled / not installed,
    returns a ``NoOpDetector``.  Otherwise returns a ``YoloDetector``.
    """
    if force_noop or not _yolo_enabled():
        LOGGER.debug("Using NoOpDetector (classical-only mode).")
        return NoOpDetector()
    try:
        detector = YoloDetector(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            device=device,
        )
        LOGGER.info("YoloDetector active.")
        return detector
    except RuntimeError as exc:
        LOGGER.warning("YoloDetector unavailable (%s).  Falling back to NoOpDetector.", exc)
        return NoOpDetector()

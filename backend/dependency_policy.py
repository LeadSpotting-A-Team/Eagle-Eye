"""
backend/dependency_policy.py
-----------------------------
Runtime policy enforcement for package-level dependencies.

Modes
~~~~~
CLASSICAL_ONLY  (default)
    No AI/ML packages (torch, keras, transformers, ultralytics) may be
    imported anywhere in the project (except inside .venv / __pycache__).
    The existing ``no_ai_guard.py`` scanner enforces this at test time.

HYBRID_AI
    YOLO or other AI packages may be used through the official adapter
    interface (``backend.object_detector.YoloDetector``).  Activating this
    mode also registers the YOLO env flag.

The mode can be changed programmatically via ``set_mode()``.  It is read by
``check_policy()`` at runtime so tests can assert the correct behaviour.
"""

from __future__ import annotations

import os
from enum import Enum


class DependencyMode(str, Enum):
    """Operating mode that governs which packages may be loaded."""

    CLASSICAL_ONLY = "classical_only"
    HYBRID_AI = "hybrid_ai"


# Module-level state (default: classical-only keeps existing tests green)
_current_mode: DependencyMode = DependencyMode.CLASSICAL_ONLY


def get_mode() -> DependencyMode:
    """Return the currently active dependency mode."""
    return _current_mode


def set_mode(mode: DependencyMode) -> None:
    """
    Switch the active dependency mode.

    Setting ``HYBRID_AI`` also sets ``EAGLE_EYE_ENABLE_YOLO=1`` so that
    the YOLO detector factory can activate without a separate env step.
    """
    global _current_mode  # noqa: PLW0603
    _current_mode = mode
    if mode == DependencyMode.HYBRID_AI:
        os.environ.setdefault("EAGLE_EYE_ENABLE_YOLO", "1")


def is_ai_allowed() -> bool:
    """Return True when AI packages may be imported / instantiated."""
    return _current_mode == DependencyMode.HYBRID_AI


def check_policy(package_name: str) -> None:
    """
    Raise ``RuntimeError`` if ``package_name`` is forbidden by the current mode.

    Use this at the top of any optional-AI code path so that classical tests
    receive a clear error rather than an obscure ``ImportError``.
    """
    forbidden_in_classical = {"torch", "keras", "transformers", "ultralytics"}
    if _current_mode == DependencyMode.CLASSICAL_ONLY:
        pkg_lower = package_name.lower()
        if pkg_lower in forbidden_in_classical:
            raise RuntimeError(
                f"Package '{package_name}' is forbidden in CLASSICAL_ONLY mode.  "
                "Call dependency_policy.set_mode(DependencyMode.HYBRID_AI) first."
            )

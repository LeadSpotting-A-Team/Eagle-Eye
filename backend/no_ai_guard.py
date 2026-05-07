from __future__ import annotations

from pathlib import Path
import re


FORBIDDEN_AI_PACKAGES = ("torch", "keras", "transformers", "ultralytics")
_IMPORT_RE = re.compile(
    r"^\s*(?:import|from)\s+(" + "|".join(FORBIDDEN_AI_PACKAGES) + r")\b",
    re.IGNORECASE,
)


def find_no_ai_policy_violations(project_root: str | Path) -> list[str]:
    root = Path(project_root)
    violations: list[str] = []

    requirements = root / "requirements.txt"
    if requirements.exists():
        for line_no, line in enumerate(requirements.read_text(encoding="utf-8").splitlines(), 1):
            package = re.split(r"[<>=!~\s]", line.strip(), maxsplit=1)[0].lower()
            if package in FORBIDDEN_AI_PACKAGES:
                violations.append(f"{requirements}:{line_no}: forbidden dependency {package}")

    ignored_parts = {".git", ".venv", "__pycache__"}
    for path in root.rglob("*.py"):
        if any(part in ignored_parts for part in path.parts):
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            match = _IMPORT_RE.match(line)
            if match:
                violations.append(f"{path}:{line_no}: forbidden import {match.group(1)}")

    return violations


def assert_no_ai_policy(project_root: str | Path) -> None:
    violations = find_no_ai_policy_violations(project_root)
    if violations:
        joined = "\n".join(violations)
        raise RuntimeError(f"Strict no-AI policy violation(s):\n{joined}")

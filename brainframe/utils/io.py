"""Filesystem and JSON helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if it does not exist and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    """Write ``obj`` as JSON to ``path`` (creating parent dirs)."""
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=indent, default=str), encoding="utf-8")
    return p


def load_json(path: str | Path) -> Any:
    """Read JSON from ``path``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))

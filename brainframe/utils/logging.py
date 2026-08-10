"""Lightweight logging wrapper."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = os.environ.get("BRAINFRAME_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S"
        )
    )
    root = logging.getLogger("brainframe")
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str = "brainframe") -> logging.Logger:
    """Return a configured logger under the ``brainframe`` namespace."""
    _configure_root()
    if not name.startswith("brainframe"):
        name = f"brainframe.{name}"
    return logging.getLogger(name)

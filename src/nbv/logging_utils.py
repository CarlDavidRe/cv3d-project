"""Consistent console and file logging for experiment scripts."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_path: str | Path, verbose: bool = False) -> logging.Logger:
    """Configure and return the project logger without duplicating handlers."""

    destination = Path(log_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("nbv")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    for handler in (logging.StreamHandler(), logging.FileHandler(destination)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger

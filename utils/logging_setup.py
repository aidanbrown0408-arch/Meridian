"""Standardized logging for Meridian Capital.

Every agent logs under its own callsign so the log reads like a desk transcript:

    2026-09-09 17:00:01 INFO  [Wong/data] Fetched SPY: 502 bars (yfinance)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False

FORMAT = "%(asctime)s %(levelname)-5s [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO", log_file: str | Path | None = None,
                  console: bool = True) -> None:
    """Configure root logging once. Safe to call repeatedly."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    formatter = logging.Formatter(FORMAT, datefmt=DATE_FORMAT)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    if console:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        root.addHandler(stream)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Third-party chatter is noise in a desk transcript.
    for noisy in ("yfinance", "urllib3", "peewee", "ccxt"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str, agent: str | None = None) -> logging.Logger:
    """Return a logger. If `agent` is given, the label becomes 'Agent/name'."""
    label = f"{agent}/{name}" if agent else name
    return logging.getLogger(label)

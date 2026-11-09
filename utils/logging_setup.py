"""Logging that writes to a file, the console and an in-memory list (for the UI)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional


class ListHandler(logging.Handler):
    def __init__(self, sink: List[str]):
        super().__init__()
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self.sink.append(self.format(record))


def get_logger(name: str = "promopilot", log_file: Optional[Path] = None,
               sink: Optional[List[str]] = None, console: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):  # fresh handlers per run -> no duplicate lines
        logger.removeHandler(h)
        h.close()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    if sink is not None:
        lh = ListHandler(sink)
        lh.setFormatter(fmt)
        logger.addHandler(lh)
    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

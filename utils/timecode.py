"""Timestamp helpers. All duration maths is done here in Python - never by an LLM."""
from __future__ import annotations

import re
from typing import Optional

_NUM = re.compile(r"^\d+(?:\.\d+)?$")


def parse_timecode(value: object) -> Optional[float]:
    """Parse 'MM:SS.ss', 'HH:MM:SS.ss' or 'SS.ss' (optionally wrapped in brackets)
    into seconds. Returns None when the value cannot be parsed."""
    if value is None:
        return None
    s = str(value).strip().strip("()[]").strip()
    if not s:
        return None
    parts = s.split(":")
    if len(parts) > 3 or not all(_NUM.match(p) for p in parts):
        return None
    seconds = 0.0
    for p in parts:  # works for 1, 2 or 3 parts
        seconds = seconds * 60 + float(p)
    return round(seconds, 3)


def format_timecode(seconds: Optional[float], decimals: int = 2) -> str:
    """Format seconds as 'MM:SS.ss' (or 'HH:MM:SS.ss' for long inputs)."""
    if seconds is None:
        return ""
    total = round(float(seconds), decimals)
    h = int(total // 3600)
    m = int((total % 3600) // 60)
    s = total - h * 3600 - m * 60
    width = 3 + decimals if decimals else 2
    sec_txt = f"{s:0{width}.{decimals}f}"
    return f"{h:02d}:{m:02d}:{sec_txt}" if h else f"{m:02d}:{sec_txt}"


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = overlap_seconds(a_start, a_end, b_start, b_end)
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0

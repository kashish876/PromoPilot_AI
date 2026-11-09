"""Dynamic input discovery: ZIP archives (also nested), folders, or single files.
Nothing is hardcoded about names or counts; ZIPs are read in memory (no extraction,
so no zip-slip risk)."""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Tuple

SUPPORTED_EXT = {".txt", ".md", ".text", ".screenplay"}
MAX_MEMBER_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 300 * 1024 * 1024


@dataclass
class RawScript:
    name: str
    text: str


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _from_bytes(name: str, data: bytes, scripts: List[RawScript], skipped: List[str], depth: int, budget: List[int]) -> None:
    base = name.replace("\\", "/")
    parts = base.split("/")
    if "__MACOSX" in parts or parts[-1].startswith(".") or parts[-1].startswith("~$"):
        return
    ext = Path(base).suffix.lower()
    budget[0] += len(data)
    if budget[0] > MAX_TOTAL_BYTES:
        skipped.append(f"{name}: skipped (total size limit reached)")
        return
    if ext == ".zip":
        if depth >= 2:
            skipped.append(f"{name}: nested ZIP too deep, skipped")
            return
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    if info.file_size > MAX_MEMBER_BYTES:
                        skipped.append(f"{info.filename}: skipped (>{MAX_MEMBER_BYTES // 1_000_000} MB)")
                        continue
                    _from_bytes(info.filename, zf.read(info), scripts, skipped, depth + 1, budget)
        except zipfile.BadZipFile:
            skipped.append(f"{name}: not a valid ZIP file")
        return
    if ext not in SUPPORTED_EXT:
        skipped.append(f"{name}: unsupported file type '{ext or 'none'}'")
        return
    if not data.strip():
        skipped.append(f"{name}: empty file")
        return
    scripts.append(RawScript(name=base, text=_decode(data)))


def load_scripts(sources: Iterable[Any]) -> Tuple[List[RawScript], List[str]]:
    """sources: paths (zip/dir/file), (name, bytes) tuples, or upload-like objects
    exposing .name and .getvalue()/.read(). Returns (scripts, skipped_notes)."""
    scripts: List[RawScript] = []
    skipped: List[str] = []
    budget = [0]
    for src in sources:
        if isinstance(src, (str, Path)):
            p = Path(src)
            if not p.exists():
                skipped.append(f"{p}: path not found")
            elif p.is_dir():
                for f in sorted(p.rglob("*"), key=lambda x: natural_key(str(x))):
                    if f.is_file():
                        _from_bytes(str(f.relative_to(p)), f.read_bytes(), scripts, skipped, 0, budget)
            else:
                _from_bytes(p.name, p.read_bytes(), scripts, skipped, 0, budget)
        elif isinstance(src, tuple) and len(src) == 2:
            _from_bytes(src[0], src[1], scripts, skipped, 0, budget)
        elif hasattr(src, "name") and (hasattr(src, "getvalue") or hasattr(src, "read")):
            data = src.getvalue() if hasattr(src, "getvalue") else src.read()
            _from_bytes(src.name, data, scripts, skipped, 0, budget)
        else:
            skipped.append(f"{src!r}: unsupported input object")
    scripts.sort(key=lambda s: natural_key(s.name))
    return scripts, skipped

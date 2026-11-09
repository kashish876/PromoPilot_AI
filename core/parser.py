"""Screenplay parser.

Format observed in the supplied pack (inspected, not assumed):

    EPISODE NUMBER: Episode 1
    EPISODE LOGLINE: ...
    CONTINUITY NOTES: ...
    --- SCRIPT START ---
    SCENE 01
    Scene heading: INT. CATHEDRAL - DAY
    Scene start: 00:00.00          (episode-global clock, MM:SS.ss)
    Scene end: 02:18.00
    Actions and dialogue:
    **LADY JULIANA (Confused, Breathless):**
    Start: (00:13.50) | End: (00:14.30)
    "Why?"
    ...
    --- SCRIPT END ---

It also tolerates variants seen in the pack or likely in new shows: upper-case
keys (episode 6), missing trailing newline, un-bolded labels, HH:MM:SS clocks,
missing scene headers / times, and CRLF line endings. Problems become warnings,
never crashes.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

from core.models import Episode, Event, Scene
from utils.timecode import parse_timecode

_HEADER_KEYS = {
    "episode number": "number", "episode": "number",
    "episode logline": "logline", "logline": "logline",
    "continuity notes": "continuity", "continuity": "continuity",
}
_HEADER_RE = re.compile(r"^\s*([A-Za-z ]+?)\s*:\s*(.*)$")
_SCENE_RE = re.compile(r"^\s*SCENE\s+([A-Za-z0-9._-]+)\s*$", re.I)
_SCENE_KV_RE = re.compile(r"^\s*scene\s+(heading|start|end)\s*:\s*(.*)$", re.I)
_BOLD_LABEL_RE = re.compile(r"^\s*\*\*\s*(.+?)\s*:?\s*\*\*\s*:?\s*$")
_CAPS_LABEL_RE = re.compile(r"^\s*([A-Z][A-Z0-9 .'&/-]*(?:\([^)]*\))?)\s*:\s*$")
_TIMING_RE = re.compile(
    r"^\s*start\s*:\s*\(?\s*([\d:.]+)\s*\)?\s*(?:[|,;\-–—]\s*)?(?:end\s*:\s*\(?\s*([\d:.]+)\s*\)?)?\s*$", re.I)
_END_MARK_RE = re.compile(r"^\s*-{2,}\s*SCRIPT\s+END\s*-{2,}\s*$", re.I)
_START_MARK_RE = re.compile(r"^\s*-{2,}\s*SCRIPT\s+START\s*-{2,}\s*$", re.I)
_LABEL_SPLIT_RE = re.compile(r"^(.*?)\s*\(([^)]*)\)\s*$")

_KIND_BY_LABEL = {
    "CAMERA": "camera", "SHOT": "camera",
    "ACTION": "action", "ACTIONS": "action",
    "SOUND FX": "sfx", "SFX": "sfx", "SOUND": "sfx", "MUSIC": "sfx", "SCORE": "sfx",
    "TRANSITION": "other", "TEXT": "other", "SUPER": "other", "TITLE": "other",
}


def _split_label(label: str) -> Tuple[str, str, str]:
    """'LADY JULIANA (Confused, Breathless)' -> (kind, speaker, delivery)."""
    label = label.strip().rstrip(":").strip()
    m = _LABEL_SPLIT_RE.match(label)
    base, delivery = (m.group(1).strip(), m.group(2).strip()) if m else (label, "")
    kind = _KIND_BY_LABEL.get(base.upper())
    if kind:
        return kind, "", delivery
    return "dialogue", base.upper(), delivery


def _clean_dialogue(text: str) -> str:
    text = " ".join(text.split())
    for a, b in (('"', '"'), ("\u201c", "\u201d"), ("'", "'")):
        if len(text) >= 2 and text.startswith(a) and text.endswith(b):
            text = text[1:-1].strip()
            break
    return text.replace("\u201c", '"').replace("\u201d", '"')


def _parse_heading(heading: str) -> Tuple[str, str, str]:
    """'INT. CATHEDRAL – DAY' -> ('INT', 'CATHEDRAL', 'DAY')."""
    m = re.match(r"^\s*(INT\.?/EXT\.?|EXT\.?/INT\.?|INT\.?|EXT\.?)\s*(.*)$", heading, re.I)
    int_ext, rest = (m.group(1).upper().rstrip("."), m.group(2)) if m else ("", heading)
    parts = re.split(r"\s+[–—-]\s+", rest.strip(), maxsplit=1)
    location = parts[0].strip()
    tod = parts[1].strip() if len(parts) > 1 else ""
    return int_ext, location, tod


def _episode_number(header_value: str, filename: str, fallback: int) -> int:
    for src in (header_value, Path(filename).stem):
        m = re.search(r"(?:episode|ep)[\s._-]*0*(\d+)", src, re.I) or (
            re.search(r"^0*(\d+)$", src.strip()) if src is header_value else None)
        if m:
            return int(m.group(1))
    m = re.search(r"(\d+)", Path(filename).stem)
    return int(m.group(1)) if m else fallback


def parse_episode_text(text: str, source_name: str = "episode.txt", fallback_number: int = 1) -> Episode:
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    warnings: List[str] = []
    header = {"number": "", "logline": "", "continuity": ""}

    # ---- header -----------------------------------------------------------------
    i = 0
    last_key: Optional[str] = None
    while i < len(lines):
        ln = lines[i]
        if _START_MARK_RE.match(ln) or _SCENE_RE.match(ln):
            break
        m = _HEADER_RE.match(ln)
        if m and m.group(1).strip().lower() in _HEADER_KEYS:
            last_key = _HEADER_KEYS[m.group(1).strip().lower()]
            header[last_key] = m.group(2).strip()
        elif ln.strip() and last_key:      # wrapped header value
            header[last_key] += " " + ln.strip()
        else:
            last_key = None if not ln.strip() else last_key
        i += 1
    ep_num = _episode_number(header["number"], source_name, fallback_number)
    if not header["number"]:
        warnings.append("Header 'EPISODE NUMBER' missing; number inferred from filename/order.")
    if not header["logline"]:
        warnings.append("Header 'EPISODE LOGLINE' missing.")

    # ---- body: scenes + blocks ----------------------------------------------------
    scenes: List[Scene] = []
    cur_scene: Optional[Scene] = None
    cur_event: Optional[dict] = None
    counter = 0

    def new_scene(sid: str) -> Scene:
        s = Scene(scene_id=sid)
        scenes.append(s)
        return s

    def flush_event() -> None:
        nonlocal cur_event, counter, cur_scene
        if cur_event is None:
            return
        ev = cur_event
        cur_event = None
        body = " ".join(b.strip() for b in ev["body"] if b.strip())
        if not body and ev["kind"] == "dialogue":
            warnings.append(f"Empty dialogue for label '{ev['label']}' near {ev['start_raw']}; skipped.")
            return
        if cur_scene is None:
            cur_scene = new_scene("00")
            warnings.append("Content found before any 'SCENE' header; implicit scene '00' created.")
        counter += 1
        text_val = _clean_dialogue(body) if ev["kind"] == "dialogue" else " ".join(body.split())
        cur_scene.events.append(Event(
            event_id=f"L{counter:03d}", kind=ev["kind"], text=text_val, start=ev["start"], end=ev["end"],
            speaker=ev["speaker"], delivery=ev["delivery"], raw_label=ev["label"], scene_id=cur_scene.scene_id))

    in_body_header_scene = False
    while i < len(lines):
        ln = lines[i]
        i += 1
        if _START_MARK_RE.match(ln):
            continue
        if _END_MARK_RE.match(ln):
            flush_event()
            break
        ms = _SCENE_RE.match(ln)
        if ms:
            flush_event()
            cur_scene = new_scene(ms.group(1))
            in_body_header_scene = True
            continue
        mkv = _SCENE_KV_RE.match(ln)
        if mkv and in_body_header_scene and cur_event is None and cur_scene is not None:
            key, val = mkv.group(1).lower(), mkv.group(2).strip()
            if key == "heading":
                cur_scene.heading = val
                cur_scene.int_ext, cur_scene.location, cur_scene.time_of_day = _parse_heading(val)
            elif key == "start":
                cur_scene.start = parse_timecode(val)
                if cur_scene.start is None:
                    warnings.append(f"Scene {cur_scene.scene_id}: unparseable scene start '{val}'.")
            else:
                cur_scene.end = parse_timecode(val)
                if cur_scene.end is None:
                    warnings.append(f"Scene {cur_scene.scene_id}: unparseable scene end '{val}'.")
            continue
        if re.match(r"^\s*actions?\s+and\s+dialogue\s*:?\s*$", ln, re.I):
            in_body_header_scene = False
            continue

        label = None
        mb = _BOLD_LABEL_RE.match(ln)
        if mb:
            label = mb.group(1)
        else:
            mc = _CAPS_LABEL_RE.match(ln)
            if mc:   # un-bolded label: accept when the next non-blank line is a timing line or a quoted line
                nxt = next((x for x in lines[i:i + 4] if x.strip()), "")
                if _TIMING_RE.match(nxt) or nxt.strip()[:1] in ('"', "\u201c"):
                    label = mc.group(1)
        if label is not None:
            flush_event()
            in_body_header_scene = False
            kind, speaker, delivery = _split_label(label)
            cur_event = {"kind": kind, "speaker": speaker, "delivery": delivery, "label": label.strip(),
                         "start": None, "end": None, "start_raw": "", "body": [], "have_timing": False}
            continue

        if cur_event is not None:
            mt = _TIMING_RE.match(ln)
            if mt and not cur_event["have_timing"]:
                cur_event["have_timing"] = True
                cur_event["start_raw"] = mt.group(1)
                cur_event["start"] = parse_timecode(mt.group(1))
                cur_event["end"] = parse_timecode(mt.group(2)) if mt.group(2) else None
                if cur_event["start"] is None:
                    warnings.append(f"Unparseable start time '{mt.group(1)}' for '{cur_event['label']}'.")
                continue
            if ln.strip():
                cur_event["body"].append(ln)
    flush_event()

    # ---- post-processing / repair --------------------------------------------------
    scenes = [s for s in scenes if s.events or s.heading]
    if not scenes:
        warnings.append("No scenes/events could be parsed from this file.")
    for s in scenes:
        timed = [e for e in s.events if e.start is not None]
        # missing/invalid event end -> next timed event start, else scene end (flagged as estimated)
        for idx, e in enumerate(s.events):
            if e.start is None:
                warnings.append(f"Scene {s.scene_id}: {e.kind} '{e.raw_label}' has no timestamp; excluded from promo windows.")
                continue
            if e.end is None or e.end < e.start:
                nxt = next((x.start for x in s.events[idx + 1:] if x.start is not None and x.start > e.start), None)
                e.end = nxt if nxt is not None else (s.end if s.end and s.end > e.start else e.start + 1.0)
                e.end_estimated = True
                warnings.append(f"Scene {s.scene_id}: end time for {e.event_id} missing/invalid; estimated {e.end:.2f}s.")
        if s.start is None and timed:
            s.start = min(e.start for e in timed)
            warnings.append(f"Scene {s.scene_id}: scene start missing; derived from first event.")
        if s.end is None and timed:
            s.end = max(e.end for e in timed if e.end is not None)
            warnings.append(f"Scene {s.scene_id}: scene end missing; derived from last event.")
        out_of_range = [e for e in timed if s.start is not None and s.end is not None
                        and (e.start < s.start - 0.5 or e.start > s.end + 0.5)]
        if out_of_range:
            warnings.append(f"Scene {s.scene_id}: {len(out_of_range)} event(s) fall outside the scene's time range.")
    return Episode(episode_id=f"Episode {ep_num}", episode_number=ep_num, source_file=source_name,
                   logline=header["logline"], continuity_notes=header["continuity"],
                   scenes=scenes, warnings=warnings)

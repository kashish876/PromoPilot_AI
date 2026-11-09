"""Python is the referee. Whatever a model returns is clamped, cross-checked against the
screenplay, and stripped of anything untraceable BEFORE it can influence a score."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from core.models import Candidate, Episode, Evaluation

CRITERIA = ("hook", "cliffhanger", "standalone", "emotion")
_QUOTE_RE = re.compile(r"[\"\u201c\u2018]([^\"\u201d\u2019]{6,})[\"\u201d\u2019]")


def normalize_text(s: str) -> str:
    s = s.lower().replace("\u2019", "'").replace("\u2018", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return " ".join(s.split())


def window_text(ep: Episode, cand: Candidate) -> str:
    ids = set(cand.segments[0].event_ids) if cand.segments else set()
    return normalize_text(" ".join(e.text for e in ep.events if e.event_id in ids))


def verify_quotes(reason: str, haystack_norm: str) -> (str, List[str]):
    """Replace any quoted fragment that is NOT found verbatim in the clip's screenplay text."""
    warnings: List[str] = []

    def repl(m: "re.Match[str]") -> str:
        frag = normalize_text(m.group(1))
        if frag and frag in haystack_norm:
            return m.group(0)
        warnings.append("model 'reason' contained a quote not found in the clip; quote removed")
        return "[untraceable quote removed]"

    return _QUOTE_RE.sub(repl, reason), warnings


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def normalize_evaluation(raw: Optional[Dict[str, Any]], cand: Candidate, ep: Episode) -> Evaluation:
    if not raw:
        return Evaluation(cand.cand_id, {k: 0.0 for k in CRITERIA}, warnings=["model returned nothing for this candidate"], ok=False)
    warnings: List[str] = []
    scores: Dict[str, float] = {}
    missing = []
    for k in CRITERIA:
        v = _num(raw.get(k))
        if v is None:
            missing.append(k)
        else:
            scores[k] = v
    if missing:
        return Evaluation(cand.cand_id, {k: scores.get(k, 0.0) for k in CRITERIA},
                          warnings=[f"model output missing/invalid scores: {', '.join(missing)}"], ok=False)
    if any(v > 10 for v in scores.values()) and all(v <= 100 for v in scores.values()):
        scores = {k: v / 10 for k, v in scores.items()}
        warnings.append("model scored on a 0-100 scale; rescaled to 0-10")
    scores = {k: round(max(0.0, min(10.0, v)), 2) for k, v in scores.items()}

    valid_ids = set(cand.segments[0].event_ids)

    def keep_id(x: Any) -> str:
        x = str(x or "").strip()
        if x and x not in valid_ids:
            warnings.append(f"model referenced line '{x}' outside the clip; ignored")
            return ""
        return x

    hook_id, cliff_id = keep_id(raw.get("hook_line_id")), keep_id(raw.get("cliffhanger_line_id"))
    keys = raw.get("key_line_ids") or []
    keys = [keep_id(k) for k in keys] if isinstance(keys, list) else []
    keys = [k for k in dict.fromkeys(keys) if k][:4]
    reason = str(raw.get("reason") or "").strip()[:600]
    reason, qw = verify_quotes(reason, window_text(ep, cand))
    warnings.extend(qw)
    model_warn = raw.get("warnings") or []
    if isinstance(model_warn, list):
        warnings.extend(str(w)[:160] for w in model_warn[:5])
    conf = _num(raw.get("confidence"))
    conf = 0.5 if conf is None else max(0.0, min(1.0, conf))
    return Evaluation(cand.cand_id, scores, hook_id, cliff_id, keys, reason, warnings, conf, True)


def validate_segments(segments: list, min_dur: float, max_dur: float) -> List[str]:
    """Duration + ordering checks, done in code. Works for single- and multi-segment promos."""
    problems: List[str] = []
    total = 0.0
    prev = None
    for s in segments:
        if s.end <= s.start:
            problems.append("segment has non-positive length")
        total += s.end - s.start
        key = (s.episode_number, s.start)
        if prev is not None and key < prev:
            problems.append("segments are not in chronological order")
        prev = key
    if total < min_dur - 1e-6:
        problems.append(f"total duration {total:.1f}s is below {min_dur:.0f}s")
    if total > max_dur + 1e-6:
        problems.append(f"total duration {total:.1f}s is above {max_dur:.0f}s")
    return problems

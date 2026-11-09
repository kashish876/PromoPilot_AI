"""Step 6 - assemble the final, fully traceable promo record."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import PipelineConfig
from core.models import Candidate, Episode, Evaluation, Event
from core.scoring import ScoreBreakdown
from core.signals import RESOLUTION
from core.validate import validate_segments
from utils.timecode import format_timecode


def line_dict(e: Event) -> Dict[str, Any]:
    return {"line_id": e.event_id, "kind": e.kind, "timestamp": format_timecode(e.start),
            "end_timestamp": format_timecode(e.end), "start_sec": e.start, "end_sec": e.end,
            "speaker": e.speaker, "delivery": e.delivery, "text": e.text}


@dataclass
class Promo:
    promo_id: str
    cand_id: str
    episode_number: int
    episode_id: str
    start: str
    end: str
    start_sec: float
    end_sec: float
    duration_sec: float
    segments: List[Dict[str, Any]]
    opening_hook: Dict[str, Any]
    cliffhanger: Dict[str, Any]
    key_dialogue: List[Dict[str, Any]]
    why_it_works: str
    model_scores: Dict[str, float]
    points: Dict[str, float]
    total_score: float
    score_explanation: str
    confidence_level: str
    confidence_value: float
    warnings: List[str]
    moment_tags: List[str]
    rule_based_signals: Dict[str, float]
    judged_by: Dict[str, str]
    evaluation_source: str
    duration_valid: bool
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    rank: int = 0
    selected: bool = False
    status: str = "candidate"
    status_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def _confidence(ev: Evaluation, py_warnings: List[str], pre: Dict[str, float], is_rule_model: bool) -> (float, str):
    s = ev.scores
    val = ev.model_confidence + 0.05 * (min(s["hook"], s["cliffhanger"]) - 5.0) - 0.1 * len(py_warnings)
    if not is_rule_model:  # agreement with the independent heuristics is only informative for a real LLM
        gap = max(abs(s["hook"] - pre.get("hook", s["hook"])), abs(s["cliffhanger"] - pre.get("cliffhanger", s["cliffhanger"])))
        val += 0.05 if gap <= 2.5 else (-0.1 if gap >= 4.5 else 0.0)
    val = round(max(0.0, min(1.0, val)), 2)
    return val, ("high" if val >= 0.65 else "medium" if val >= 0.45 else "low")


def build_promo(cand: Candidate, ev: Evaluation, bd: ScoreBreakdown, ep: Episode, cfg: PipelineConfig,
                judged_by: Dict[str, str], source: str) -> Promo:
    seg = cand.segments[0]
    ids = set(seg.event_ids)
    lines = [e for e in ep.events if e.event_id in ids and e.start is not None]
    lines.sort(key=lambda e: (e.start, e.end))
    by_id = {e.event_id: e for e in lines}
    py_warn: List[str] = []

    # --- opening hook: must sit inside the first N seconds, otherwise Python substitutes the first beat
    zone = [e for e in lines if e.kind in ("dialogue", "action") and e.start < cand.start + cfg.hook_window + 0.01]
    hook = by_id.get(ev.hook_line_id)
    if hook is None or hook not in zone:
        if hook is not None:
            py_warn.append("model's hook line was outside the first 3 s; replaced with the first on-screen beat")
        hook = zone[0] if zone else None
    if hook is None:
        py_warn.append("no spoken line or action in the first 3 s")

    # --- cliffhanger: the last spoken line (or the model's pick if it is in the final 12 s)
    dlg = [e for e in lines if e.kind == "dialogue"]
    cliff = by_id.get(ev.cliffhanger_line_id)
    if cliff is None or cliff.start < cand.end - 12.0:
        cliff = dlg[-1] if dlg else (lines[-1] if lines else None)
    if dlg and RESOLUTION.search(dlg[-1].text):
        py_warn.append("final line contains resolution-style wording; ending may feel resolved")
    if dlg and dlg[0].text.startswith(("...", "\u2026")):
        py_warn.append("clip opens on a continuation line (starts mid-sentence)")

    key_ids = [i for i in ev.key_line_ids if i in by_id]
    for extra in (hook, cliff):
        if extra is not None and extra.event_id not in key_ids and extra.kind == "dialogue":
            key_ids.append(extra.event_id)
    key = sorted((by_id[i] for i in dict.fromkeys(key_ids)), key=lambda e: e.start)[:5]

    dur = cand.duration
    problems = validate_segments(cand.segments, cfg.min_duration, cfg.max_duration)
    if problems:
        py_warn.extend(problems)
    if not problems and (dur < cfg.min_duration + 2 or dur > cfg.max_duration - 2):
        py_warn.append("duration is within 2 s of the allowed limit")
    first_scene_start = any(abs(cand.start - s.start) <= 2.0 for s in ep.scenes if s.start is not None)
    if not first_scene_start and len({e.speaker for e in dlg if e.speaker}) <= 2:
        py_warn.append("clip starts mid-scene; location may be unclear to a new viewer")
    pre = cand.pre_scores
    is_rule = judged_by.get("provider") == "rule_based"
    if not is_rule:
        if abs(ev.scores["hook"] - pre.get("hook", ev.scores["hook"])) >= 4.5 or \
           abs(ev.scores["cliffhanger"] - pre.get("cliffhanger", ev.scores["cliffhanger"])) >= 4.5:
            py_warn.append("model and rule-based heuristics disagree strongly on hook/cliffhanger; review manually")
    conf_val, conf_level = _confidence(ev, py_warn, pre, is_rule)

    warnings = list(dict.fromkeys(py_warn + ev.warnings))
    reason = ev.reason or "No explanation returned."
    return Promo(
        promo_id=f"EP{cand.episode_number:03d}_{int(round(cand.start * 100)):06d}",
        cand_id=cand.cand_id, episode_number=cand.episode_number, episode_id=ep.episode_id,
        start=format_timecode(cand.start), end=format_timecode(cand.end),
        start_sec=cand.start, end_sec=cand.end, duration_sec=dur,
        segments=[s.to_dict() for s in cand.segments],
        opening_hook=line_dict(hook) if hook else {}, cliffhanger=line_dict(cliff) if cliff else {},
        key_dialogue=[line_dict(e) for e in key], why_it_works=reason,
        model_scores=dict(ev.scores), points=bd.points, total_score=bd.total,
        score_explanation=bd.explain(ev.scores),
        confidence_level=conf_level, confidence_value=conf_val, warnings=warnings,
        moment_tags=cand.tags, rule_based_signals={k: round(v, 2) for k, v in pre.items()},
        judged_by=judged_by, evaluation_source=source, duration_valid=not problems,
        transcript=[line_dict(e) for e in lines if e.kind in ("dialogue", "action")],
    )

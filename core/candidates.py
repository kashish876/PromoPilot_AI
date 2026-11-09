"""Step 2 - candidate moment detection.

A promo must START on a beat (a spoken line or a visible action) and END on a
beat boundary, so candidates are built from *anchors* that exist in the
screenplay, never from invented times:

  start anchors : start time of every dialogue/action event
  end anchors   : end time   of every dialogue/action event
  candidate     : (start anchor, end anchor) with 20 <= duration <= 60 s that does
                  not slice through a spoken line and has >= 2 dialogue lines

Windows are pre-scored with the rule-based signals (hook in the first 3 s, how
unresolved the end is, standalone context, emotion), the best are kept after a
diversity filter, and only that shortlist goes to the (possibly paid) LLM.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from config import PipelineConfig
from core.models import Candidate, Episode, Event, Segment
from core.signals import (all_signals, build_window, cliffhanger_signal, hook_signal,
                          moment_tags, window_is_clean)
from utils.timecode import iou

PRE_WEIGHTS = {"hook": 0.30, "cliffhanger": 0.30, "standalone": 0.20, "emotion": 0.20}


def _timed_content(ep: Episode) -> List[Event]:
    return sorted([e for e in ep.events if e.kind in ("dialogue", "action", "sfx")
                   and e.start is not None and e.end is not None], key=lambda e: (e.start, e.end))


def generate_candidates(ep: Episode, cfg: PipelineConfig) -> List[Candidate]:
    events = _timed_content(ep)
    beats = [e for e in events if e.kind in ("dialogue", "action")]
    if len(beats) < 2:
        return []
    ep_end = ep.duration
    bounds = [(s.start, s.end) for s in ep.scenes if s.start is not None and s.end is not None]

    starts = sorted({round(e.start, 2) for e in beats})
    ends = sorted({round(e.end, 2) for e in beats})

    # cheap per-anchor scores (computed once per anchor, not once per pair)
    hook_by_start: Dict[float, float] = {}
    for s in starts:
        w = build_window(events, s, s + cfg.hook_window + 7.0, ep_end, bounds)
        hook_by_start[s] = hook_signal(w, cfg.hook_window)[0]
    cliff_by_end: Dict[float, float] = {}
    for e in ends:
        w = build_window(events, max(0.0, e - 10.0), e, ep_end, bounds)
        cliff_by_end[e] = cliffhanger_signal(w)[0]

    pairs: List[Tuple[float, float, float]] = []
    for s in starts:
        for e in ends:
            dur = e - s
            if dur < cfg.min_duration or dur > cfg.max_duration:
                continue
            pairs.append((0.5 * hook_by_start[s] + 0.5 * cliff_by_end[e], s, e))
    pairs.sort(reverse=True)

    scored: List[Tuple[float, float, float, Dict[str, float], List[str]]] = []
    for _, s, e in pairs[:250]:
        if not window_is_clean(events, s, e):
            continue
        w = build_window(events, s, e, ep_end, bounds)
        if len(w.dialogue) < 2:
            continue
        sig = all_signals(w, cfg.hook_window)
        pre = {k: v[0] for k, v in sig.items()}
        total = sum(PRE_WEIGHTS[k] * pre[k] for k in PRE_WEIGHTS)
        scored.append((total, s, e, pre, moment_tags(w)))
    scored.sort(key=lambda t: -t[0])

    shortlist: List[Tuple[float, float, float, Dict[str, float], List[str]]] = []
    for item in scored:
        if all(iou(item[1], item[2], k[1], k[2]) <= cfg.shortlist_iou for k in shortlist):
            shortlist.append(item)
        if len(shortlist) >= cfg.max_candidates_per_episode:
            break
    shortlist.sort(key=lambda t: t[1])

    out: List[Candidate] = []
    for n, (total, s, e, pre, tags) in enumerate(shortlist, 1):
        w = build_window(events, s, e, ep_end, bounds)
        pre = dict(pre, combined=round(total, 2))
        out.append(Candidate(
            cand_id=f"EP{ep.episode_number:03d}-C{n:02d}", episode_number=ep.episode_number,
            segments=[Segment(ep.episode_number, s, e, [x.event_id for x in w.events])],
            pre_scores=pre, tags=tags))
    return out

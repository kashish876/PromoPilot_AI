"""Step 4 (selection half) - pick final promos from the evaluated candidates.

Greedy, global, deterministic: best score first; a candidate is rejected (with a recorded
reason) if it is invalid, scores below the floor, exceeds the per-episode cap, overlaps a
higher-ranked promo from the same episode, or repeats the dialogue of one already chosen."""
from __future__ import annotations

import re
from typing import List, Set

from config import PipelineConfig
from core.promos import Promo
from utils.timecode import overlap_seconds


def _tokens(p: Promo) -> Set[str]:
    words: Set[str] = set()
    for l in p.transcript:
        if l["kind"] == "dialogue":
            words |= {w for w in re.findall(r"[a-z']{4,}", l["text"].lower())}
    return words


def _jaccard(a: Set[str], b: Set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def select_promos(all_promos: List[Promo], cfg: PipelineConfig) -> List[Promo]:
    order = sorted(all_promos, key=lambda p: (-p.total_score, -p.model_scores.get("cliffhanger", 0),
                                              -p.model_scores.get("hook", 0), p.episode_number, p.start_sec))
    chosen: List[Promo] = []
    per_ep = {}
    toks = {}
    for p in order:
        if not p.duration_valid:
            p.status, p.status_reason = "rejected", "invalid duration"
            continue
        if p.model_scores.get("hook", 0) < cfg.min_hook_score:
            p.status, p.status_reason = "rejected", f"hook gate: hook {p.model_scores.get('hook', 0):.1f} < {cfg.min_hook_score:g}"
            continue
        if p.model_scores.get("cliffhanger", 0) < cfg.min_cliffhanger_score:
            p.status, p.status_reason = "rejected", f"cliffhanger gate: {p.model_scores.get('cliffhanger', 0):.1f} < {cfg.min_cliffhanger_score:g}"
            continue
        if p.total_score < cfg.min_total_score:
            p.status, p.status_reason = "rejected", f"score {p.total_score:.1f} below minimum {cfg.min_total_score:.0f}"
            continue
        if per_ep.get(p.episode_number, 0) >= cfg.max_promos_per_episode:
            p.status, p.status_reason = "rejected", "per-episode cap reached"
            continue
        clash = next((c for c in chosen if c.episode_number == p.episode_number and
                      overlap_seconds(c.start_sec, c.end_sec, p.start_sec, p.end_sec) > cfg.max_overlap_seconds), None)
        if clash:
            p.status, p.status_reason = "rejected", f"overlaps higher-ranked {clash.promo_id}"
            continue
        t = toks.setdefault(p.promo_id, _tokens(p))
        dup = next((c for c in chosen if _jaccard(t, toks.setdefault(c.promo_id, _tokens(c))) >= cfg.near_duplicate_jaccard), None)
        if dup:
            p.status, p.status_reason = "rejected", f"near-duplicate dialogue of {dup.promo_id}"
            continue
        chosen.append(p)
        per_ep[p.episode_number] = per_ep.get(p.episode_number, 0) + 1
        p.selected, p.status, p.status_reason = True, "selected", ""
        p.rank = len(chosen)
    return chosen

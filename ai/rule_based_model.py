"""Free, offline fallback judge. NOT an LLM: it applies the heuristic signals in
core/signals.py. It exists so the whole app runs with no API key and so the LLM's
output can be sanity-checked against something deterministic."""
from __future__ import annotations

from typing import Any, Dict, List

from ai.base_model import BaseLLM, EpisodeRequest
from core.signals import (ACCUSE_REVEAL, THREAT, all_signals, build_window)


class RuleBasedModel(BaseLLM):
    provider = "rule_based"
    model = "heuristics-v1"

    def evaluate_episode(self, request: EpisodeRequest) -> List[Dict[str, Any]]:
        ep = request.episode
        events = [e for e in ep.events if e.start is not None and e.end is not None]
        bounds = [(s.start, s.end) for s in ep.scenes if s.start is not None and s.end is not None]
        out: List[Dict[str, Any]] = []
        for c in request.candidates:
            w = build_window(events, c.start, c.end, ep.duration, bounds)
            sig = all_signals(w, request.hook_seconds)
            zone = [e for e in w.events if e.kind in ("dialogue", "action") and e.start < c.start + request.hook_seconds]
            dlg = w.dialogue
            hook_id = zone[0].event_id if zone else ""
            cliff_id = dlg[-1].event_id if dlg else ""
            ranked = sorted((e for e in dlg if e.event_id not in (hook_id, cliff_id)),
                            key=lambda e: -(("?" in e.text) + ("!" in e.text) + 2 * bool(ACCUSE_REVEAL.search(e.text))
                                            + 2 * bool(THREAT.search(e.text))))
            key_ids = [x for x in (hook_id, cliff_id) if x] + [e.event_id for e in ranked[:2]]
            key_ids = sorted(set(key_ids), key=lambda i: next(e.start for e in w.events if e.event_id == i))
            bits = []
            for name in ("hook", "cliffhanger", "standalone", "emotion"):
                if sig[name][1]:
                    bits.append(f"{name}: " + "; ".join(sig[name][1][:2]))
            out.append({
                "cand_id": c.cand_id,
                "hook": sig["hook"][0], "cliffhanger": sig["cliffhanger"][0],
                "standalone": sig["standalone"][0], "emotion": sig["emotion"][0],
                "hook_line_id": hook_id, "cliffhanger_line_id": cliff_id, "key_line_ids": key_ids,
                "reason": " | ".join(bits) or "no strong signals detected",
                "warnings": ["scored by keyword/structure heuristics, not a language model"],
                "confidence": 0.5,
            })
        return out

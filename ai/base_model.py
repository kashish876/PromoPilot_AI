"""The ONE interface every AI backend implements.

The rest of the application (parser, candidate generator, validator, scorer, UI)
only ever calls `BaseLLM.evaluate_episode(request)` and receives plain dicts, so a
new model = one new subclass + one line in the factory. Nothing else changes.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from core.models import Candidate, Episode

# keys every backend must return per candidate (Python validates & clamps them)
EVAL_KEYS = ("cand_id", "hook", "cliffhanger", "standalone", "emotion", "hook_line_id",
             "cliffhanger_line_id", "key_line_ids", "reason", "warnings", "confidence")


@dataclass
class EpisodeRequest:
    """Everything a judge needs for one episode: the transcript (with stable line
    IDs), the shortlisted candidate windows, and the rubric settings."""
    episode: Episode
    candidates: List[Candidate]
    hook_seconds: float = 3.0
    min_duration: float = 20.0
    max_duration: float = 60.0

    def lines(self) -> List[Dict[str, Any]]:
        return [{"id": e.event_id, "t": f"{e.start:.1f}-{e.end:.1f}", "kind": e.kind,
                 "speaker": e.speaker, "delivery": e.delivery, "text": e.text}
                for e in self.episode.events if e.kind in ("dialogue", "action", "sfx") and e.start is not None]

    def candidate_payload(self) -> List[Dict[str, Any]]:
        out = []
        for c in self.candidates:
            ids = c.segments[0].event_ids
            out.append({"cand_id": c.cand_id, "start": round(c.start, 2), "end": round(c.end, 2),
                        "duration": c.duration, "first_line_id": ids[0] if ids else "",
                        "last_line_id": ids[-1] if ids else "", "line_ids": ids,
                        "rule_based_tags": c.tags})
        return out


class BaseLLM(ABC):
    """Subclass this to plug in any model (Gemini, OpenAI-compatible, Kuku's API, local...)."""
    provider: str = "base"
    model: str = ""

    @abstractmethod
    def evaluate_episode(self, request: EpisodeRequest) -> List[Dict[str, Any]]:
        """Return one dict per candidate with EVAL_KEYS. hook/cliffhanger/standalone/emotion
        are 0-10. Line references must be IDs from request.lines(), never copied text."""

    def evaluate_promo(self, episode: Episode, candidate: Candidate, hook_seconds: float = 3.0) -> Dict[str, Any]:
        """Convenience: judge a single candidate."""
        res = self.evaluate_episode(EpisodeRequest(episode, [candidate], hook_seconds))
        return res[0] if res else {}

    def describe(self) -> Dict[str, str]:
        return {"provider": self.provider, "model": self.model}


def extract_json(text: str) -> List[Dict[str, Any]]:
    """Tolerant JSON extraction: handles ```json fences, prose around JSON, and
    wrapper objects like {"evaluations": [...]}."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I).strip()
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"(\[.*\]|\{.*\})", t, re.S)
        if not m:
            raise ValueError("model output contained no JSON")
        data = json.loads(m.group(1))
    if isinstance(data, dict):
        for key in ("evaluations", "results", "candidates", "promos"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    if not isinstance(data, list):
        raise ValueError("model JSON is neither list nor object")
    return data

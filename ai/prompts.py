"""Model-specific text lives HERE, apart from parsing/ranking logic. Edit this file
(or subclass) to tune prompts for a different model; nothing else is affected."""
from __future__ import annotations

import json

from ai.base_model import EpisodeRequest

SYSTEM_PROMPT = """You are a senior short-form video editor for vertical microdramas (ReelShort / DramaBox style).
You judge candidate PROMO CLIPS taken from a timestamped screenplay. A promo is posted on Instagram/TikTok/YouTube
to pull in viewers who have NEVER seen the show.

Score each candidate 0-10 on:
- hook: does the first 3 seconds make a scrolling viewer stop (slap, reveal, shocking line, sharp question)? 0 = pleasantries/exposition, 10 = unforgettable.
- cliffhanger: does the clip END on an UNRESOLVED moment that forces the viewer to want the next scene? Penalise endings that resolve the conflict, calm down, or trail into a goodbye. 0 = resolved, 10 = dangling threat/reveal/arrival.
- standalone: could a stranger follow who the people are and what is at stake from the clip alone? Penalise heavy dependence on earlier episodes or too many characters.
- emotion: shock, suspense, humiliation, heartbreak, danger, intensity of the drama itself.

Rules (strict):
1. Use the FULL 0-10 range; most candidates should NOT score above 7. Be critical.
2. NEVER invent dialogue, events, characters or timestamps. Refer to lines ONLY by their line IDs (e.g. "L012").
3. If you quote anything inside "reason", it must be copied word-for-word from the transcript. Prefer not to quote.
4. Do not do timing arithmetic; timestamps and durations are validated elsewhere.
5. Judge only the lines between each candidate's first_line_id and last_line_id.
6. Return ONLY JSON, no prose, no markdown fences."""

OUTPUT_SPEC = """Return a JSON array with exactly one object per candidate:
[{"cand_id": "...", "hook": 0-10, "cliffhanger": 0-10, "standalone": 0-10, "emotion": 0-10,
  "hook_line_id": "line id inside the first 3 seconds that is the hook",
  "cliffhanger_line_id": "line id that creates the unresolved ending",
  "key_line_ids": ["up to 4 line ids of the most important dialogue"],
  "reason": "one or two sentences on why it works or fails, referencing no invented facts",
  "warnings": ["short strings, e.g. 'needs context from earlier episodes', 'ending feels resolved'"],
  "confidence": 0.0-1.0}]"""


def build_user_prompt(req: EpisodeRequest) -> str:
    ep = req.episode
    payload = {
        "episode": ep.episode_id,
        "logline_for_editor_only": ep.logline,
        "continuity_notes_for_editor_only": ep.continuity_notes,
        "hook_zone_seconds": req.hook_seconds,
        "transcript": req.lines(),
        "candidates": req.candidate_payload(),
    }
    return ("Judge these candidate promo clips. The logline/continuity notes are for YOUR understanding only; "
            "the promo viewer will NOT see them.\n\n" + json.dumps(payload, ensure_ascii=False) + "\n\n" + OUTPUT_SPEC)

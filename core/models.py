"""Plain dataclasses shared by every layer. No logic about AI or UI lives here."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from utils.timecode import format_timecode

# Event kinds that carry "story content" a viewer would see/hear in a promo.
CONTENT_KINDS = ("dialogue", "action", "sfx")


@dataclass
class Event:
    event_id: str                 # stable per episode, e.g. "L012"
    kind: str                     # dialogue | action | camera | sfx | other
    text: str
    start: Optional[float]
    end: Optional[float]
    speaker: str = ""             # dialogue only
    delivery: str = ""            # e.g. "Cold, Commanding"
    raw_label: str = ""
    scene_id: str = ""
    end_estimated: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start_tc"] = format_timecode(self.start)
        d["end_tc"] = format_timecode(self.end)
        return d


@dataclass
class Scene:
    scene_id: str
    heading: str = ""
    location: str = ""
    int_ext: str = ""
    time_of_day: str = ""
    start: Optional[float] = None
    end: Optional[float] = None
    events: List[Event] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id, "heading": self.heading, "location": self.location,
            "int_ext": self.int_ext, "time_of_day": self.time_of_day,
            "start_timestamp": format_timecode(self.start), "end_timestamp": format_timecode(self.end),
            "start_sec": self.start, "end_sec": self.end,
            "actions": [e.to_dict() for e in self.events if e.kind == "action"],
            "dialogues": [e.to_dict() for e in self.events if e.kind == "dialogue"],
            "camera": [e.to_dict() for e in self.events if e.kind == "camera"],
            "sound_fx": [e.to_dict() for e in self.events if e.kind == "sfx"],
            "events_in_order": [e.event_id for e in self.events],
        }


@dataclass
class Episode:
    episode_id: str
    episode_number: int
    source_file: str
    logline: str = ""
    continuity_notes: str = ""
    scenes: List[Scene] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def events(self) -> List[Event]:
        return [e for s in self.scenes for e in s.events]

    @property
    def duration(self) -> float:
        ends = [e.end for e in self.events if e.end is not None]
        return max(ends) if ends else 0.0

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id, "episode_number": self.episode_number,
            "source_file": self.source_file, "logline": self.logline,
            "continuity_notes": self.continuity_notes,
            "duration_sec": self.duration,
            "n_scenes": len(self.scenes), "n_events": len(self.events),
            "n_dialogue_lines": sum(1 for e in self.events if e.kind == "dialogue"),
            "parse_warnings": self.warnings,
            "scenes": [s.to_dict() for s in self.scenes],
        }


@dataclass
class Segment:
    """One continuous stretch of ONE episode. A promo is a list of segments."""
    episode_number: int
    start: float
    end: float
    event_ids: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    def to_dict(self) -> dict:
        return {"episode_number": self.episode_number,
                "start": format_timecode(self.start), "end": format_timecode(self.end),
                "start_sec": self.start, "end_sec": self.end, "duration_sec": self.duration}


@dataclass
class Candidate:
    cand_id: str
    episode_number: int
    segments: List[Segment]
    pre_scores: Dict[str, float] = field(default_factory=dict)   # rule-based 0-10 signals
    tags: List[str] = field(default_factory=list)                # e.g. ["confrontation", "betrayal"]

    @property
    def start(self) -> float:
        return self.segments[0].start

    @property
    def end(self) -> float:
        return self.segments[-1].end

    @property
    def duration(self) -> float:
        return round(sum(s.duration for s in self.segments), 3)


@dataclass
class Evaluation:
    """A model's judgement of one candidate AFTER Python validation."""
    cand_id: str
    scores: Dict[str, float]               # hook/cliffhanger/standalone/emotion, each 0-10
    hook_line_id: str = ""
    cliffhanger_line_id: str = ""
    key_line_ids: List[str] = field(default_factory=list)
    reason: str = ""
    warnings: List[str] = field(default_factory=list)
    model_confidence: float = 0.5
    ok: bool = True                        # False when the model gave nothing usable

"""Central, configurable settings. Everything can be overridden by environment
variables (see .env.example) or a JSON config file passed to main.py --config."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, Optional


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no extra dependency). Existing env vars win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass
class ScoringConfig:
    """Transparent, configurable rubric. Weights are PROPOSED, not Kuku's official ones."""
    weights: Dict[str, float] = field(default_factory=lambda: {
        "hook": 30.0, "cliffhanger": 30.0, "standalone": 20.0, "emotion": 15.0, "duration": 5.0})
    sweet_spot: tuple = (30.0, 45.0)      # seconds that earn full duration points
    edge_credit: float = 0.5              # fraction of duration points awarded at 20s / 60s

    def total_weight(self) -> float:
        return sum(self.weights.values())


@dataclass
class PipelineConfig:
    min_duration: float = 20.0
    max_duration: float = 60.0
    hook_window: float = 3.0               # the hook must land inside the first N seconds
    max_candidates_per_episode: int = 8    # shortlist sent to the model per episode
    max_promos_per_episode: int = 3
    min_total_score: float = 45.0          # promos below this are reported as "rejected"
    min_hook_score: float = 4.0            # gate (0-10): the brief REQUIRES a hook in the first 3 s
    min_cliffhanger_score: float = 3.5     # gate (0-10): the brief REQUIRES an unresolved ending
    max_overlap_seconds: float = 0.0       # selected promos of one episode may not overlap
    shortlist_iou: float = 0.5             # shortlist diversity (candidates may overlap; picks may not)
    near_duplicate_jaccard: float = 0.8    # cross-episode duplicate-dialogue filter
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        cfg = cls()
        for f in fields(cls):
            key = "PROMOPILOT_" + f.name.upper()
            if key in os.environ and f.name != "scoring":
                caster = type(getattr(cfg, f.name))
                setattr(cfg, f.name, caster(os.environ[key]))
        if "PROMOPILOT_WEIGHTS" in os.environ:  # e.g. hook=30,cliffhanger=30,...
            for pair in os.environ["PROMOPILOT_WEIGHTS"].split(","):
                k, v = pair.split("=")
                cfg.scoring.weights[k.strip()] = float(v)
        return cfg

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = cls.from_env()
        scoring = raw.pop("scoring", {})
        for k, v in raw.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        for k, v in scoring.items():
            if hasattr(cfg.scoring, k):
                setattr(cfg.scoring, k, tuple(v) if k == "sweet_spot" else v)
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)


def model_settings_from_env() -> dict:
    """Model selection is configuration, not code."""
    return {
        "provider": os.getenv("PROMOPILOT_MODEL", "auto"),
        "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "openai_base_url": os.getenv("OPENAI_COMPAT_BASE_URL", ""),
        "openai_api_key": os.getenv("OPENAI_COMPAT_API_KEY", ""),
        "openai_model": os.getenv("OPENAI_COMPAT_MODEL", ""),
    }

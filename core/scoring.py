"""Step 5 - transparent, configurable scoring. Model gives 0-10 per criterion; Python
turns that into weighted points. Duration suitability is computed purely in Python."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from config import PipelineConfig, ScoringConfig


def duration_points(duration: float, cfg: PipelineConfig) -> float:
    """Full credit inside the sweet spot, linearly down to `edge_credit` at the limits, 0 outside."""
    sc = cfg.scoring
    w = sc.weights.get("duration", 0.0)
    lo, hi = sc.sweet_spot
    if duration < cfg.min_duration or duration > cfg.max_duration:
        return 0.0
    if lo <= duration <= hi:
        return w
    if duration < lo:
        frac = (duration - cfg.min_duration) / max(lo - cfg.min_duration, 1e-9)
    else:
        frac = (cfg.max_duration - duration) / max(cfg.max_duration - hi, 1e-9)
    return round(w * (sc.edge_credit + (1 - sc.edge_credit) * frac), 2)


@dataclass
class ScoreBreakdown:
    points: Dict[str, float]
    max_points: Dict[str, float]
    total: float                   # normalised to 0-100

    def explain(self, raw: Dict[str, float]) -> str:
        parts = []
        for k in ("hook", "cliffhanger", "standalone", "emotion"):
            if k in self.points:
                parts.append(f"{k} {raw[k]:.1f}/10 -> {self.points[k]:.1f}/{self.max_points[k]:.0f}")
        if "duration" in self.points:
            parts.append(f"duration -> {self.points['duration']:.1f}/{self.max_points['duration']:.0f}")
        weakest = min((k for k in raw), key=lambda k: raw[k])
        return "; ".join(parts) + f". Total {self.total:.1f}/100. Weakest criterion: {weakest}."


def score_promo(raw_scores: Dict[str, float], duration: float, cfg: PipelineConfig) -> ScoreBreakdown:
    sc: ScoringConfig = cfg.scoring
    points = {k: round(sc.weights.get(k, 0.0) * raw_scores.get(k, 0.0) / 10.0, 2)
              for k in ("hook", "cliffhanger", "standalone", "emotion")}
    points["duration"] = duration_points(duration, cfg)
    max_points = {k: sc.weights.get(k, 0.0) for k in points}
    tw = sc.total_weight() or 1.0
    total = round(sum(points.values()) / tw * 100.0, 2)
    return ScoreBreakdown(points, max_points, total)

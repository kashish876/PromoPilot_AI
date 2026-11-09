"""Orchestration: parse -> candidates -> model evaluation -> validation -> scoring ->
selection. Episodes are independent; one bad episode never stops the run."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ai.base_model import BaseLLM, EpisodeRequest
from ai.rule_based_model import RuleBasedModel
from config import PipelineConfig
from core.candidates import generate_candidates
from core.loader import RawScript
from core.models import Episode
from core.parser import parse_episode_text
from core.promos import Promo, build_promo
from core.scoring import score_promo
from core.selection import select_promos
from core.validate import normalize_evaluation

ProgressFn = Callable[[int, int, str], None]


@dataclass
class RunResult:
    episodes: List[Episode]
    parse_report: Dict[str, Any]
    all_promos: List[Promo]                  # every evaluated candidate, with status
    promos: List[Promo]                      # selected, ranked
    raw_model_outputs: Dict[str, Any]
    logs: List[str]
    stats: Dict[str, Any]
    config: Dict[str, Any]
    model_info: Dict[str, str]


def parse_scripts(scripts: Sequence[RawScript], log: logging.Logger) -> Tuple[List[Episode], Dict[str, Any]]:
    episodes: List[Episode] = []
    failures: List[Dict[str, str]] = []
    used = set()
    for idx, sc in enumerate(scripts, 1):
        try:
            ep = parse_episode_text(sc.text, sc.name, idx)
        except Exception as e:  # parser should never raise, but a bad file must not kill the run
            failures.append({"file": sc.name, "error": f"{type(e).__name__}: {e}"})
            log.error("PARSE FAILED %s: %s", sc.name, e)
            continue
        if not ep.events:
            failures.append({"file": sc.name, "error": "no timestamped events found (unsupported format?)"})
            log.error("PARSE FAILED %s: no events", sc.name)
            continue
        if ep.episode_number in used:
            new = max(used) + 1
            ep.warnings.append(f"episode number {ep.episode_number} already used; reassigned to {new}")
            log.warning("%s: duplicate episode number %d -> %d", sc.name, ep.episode_number, new)
            ep.episode_number, ep.episode_id = new, f"Episode {new}"
        used.add(ep.episode_number)
        episodes.append(ep)
        log.info("parsed %-45s %s scenes=%d events=%d dur=%.1fs warnings=%d", sc.name, ep.episode_id,
                 len(ep.scenes), len(ep.events), ep.duration, len(ep.warnings))
    episodes.sort(key=lambda e: e.episode_number)
    report = {
        "files_received": len(scripts), "episodes_parsed": len(episodes), "parse_failures": failures,
        "parse_success_rate": round(len(episodes) / len(scripts), 4) if scripts else 0.0,
        "episodes_with_warnings": sum(1 for e in episodes if e.warnings),
        "total_warnings": sum(len(e.warnings) for e in episodes),
    }
    return episodes, report


class PromoPipeline:
    def __init__(self, model: BaseLLM, cfg: Optional[PipelineConfig] = None, fallback: Optional[BaseLLM] = None,
                 logger: Optional[logging.Logger] = None, progress: Optional[ProgressFn] = None):
        self.model = model
        self.cfg = cfg or PipelineConfig.from_env()
        self.fallback = fallback if fallback is not None else (None if isinstance(model, RuleBasedModel) else RuleBasedModel())
        self.log = logger or logging.getLogger("promopilot")
        self.progress = progress or (lambda i, n, m: None)

    # ------------------------------------------------------------------ one episode
    def evaluate_episode(self, ep: Episode, raw_sink: Dict[str, Any]) -> List[Promo]:
        cfg = self.cfg
        cands = generate_candidates(ep, cfg)
        if not cands:
            self.log.warning("%s: no valid 20-60 s candidate windows (episode duration %.1fs)", ep.episode_id, ep.duration)
            raw_sink[ep.episode_id] = {"candidates": 0}
            return []
        req = EpisodeRequest(ep, cands, cfg.hook_window, cfg.min_duration, cfg.max_duration)
        t0, raw, err = time.time(), None, None
        try:
            raw = self.model.evaluate_episode(req)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            self.log.error("%s: model call failed (%s) -> using fallback", ep.episode_id, err)
        by_id = {r.get("cand_id"): r for r in (raw or []) if isinstance(r, dict)}
        fb_by_id: Dict[str, Any] = {}
        fb_loaded = False
        promos: List[Promo] = []
        n_fb = 0
        for c in cands:
            ev = normalize_evaluation(by_id.get(c.cand_id), c, ep)
            judged, source = self.model.describe(), "primary"
            if not ev.ok:
                if self.fallback is not None:
                    if not fb_loaded:
                        fb_by_id = {r["cand_id"]: r for r in self.fallback.evaluate_episode(req)}
                        fb_loaded = True
                    why = "; ".join(ev.warnings)
                    ev = normalize_evaluation(fb_by_id.get(c.cand_id), c, ep)
                    ev.warnings.insert(0, f"primary model unusable for this candidate ({why}); rule-based fallback used")
                    judged, source = self.fallback.describe(), "fallback"
                    n_fb += 1
                else:
                    self.log.warning("%s %s: unusable model output and no fallback", ep.episode_id, c.cand_id)
                    continue
            bd = score_promo(ev.scores, c.duration, cfg)
            promos.append(build_promo(c, ev, bd, ep, cfg, judged, source))
        raw_sink[ep.episode_id] = {"candidates": [c.cand_id for c in cands], "error": err,
                                   "response": raw, "fallback_used_for": n_fb,
                                   "seconds": round(time.time() - t0, 3)}
        self.log.info("%s: %d candidates evaluated by %s (fallback for %d) in %.2fs", ep.episode_id, len(promos),
                      self.model.describe()["provider"], n_fb, time.time() - t0)
        return promos

    # ------------------------------------------------------------------ full run
    def run(self, scripts: Sequence[RawScript], only_episodes: Optional[Sequence[int]] = None,
            skipped_notes: Optional[List[str]] = None) -> RunResult:
        started = time.time()
        logs = getattr(self, "_log_sink", [])
        episodes, parse_report = parse_scripts(scripts, self.log)
        parse_report["skipped_inputs"] = skipped_notes or []
        for note in parse_report["skipped_inputs"]:
            self.log.warning("skipped input: %s", note)
        if only_episodes:
            episodes = [e for e in episodes if e.episode_number in set(only_episodes)]
        raw_outputs: Dict[str, Any] = {}
        all_promos: List[Promo] = []
        errors: List[str] = []
        for i, ep in enumerate(episodes, 1):
            self.progress(i - 1, len(episodes), f"Analysing {ep.episode_id}")
            try:
                all_promos.extend(self.evaluate_episode(ep, raw_outputs))
            except Exception as e:   # isolation: log and continue with the next episode
                errors.append(f"{ep.episode_id}: {type(e).__name__}: {e}")
                self.log.exception("%s failed unexpectedly", ep.episode_id)
        self.progress(len(episodes), len(episodes), "Ranking promos")
        selected = select_promos(all_promos, self.cfg)
        elapsed = round(time.time() - started, 2)
        covered = {p.episode_number for p in selected}
        stats = {
            "episodes_detected": len(episodes), "episodes_with_candidates": len({p.episode_number for p in all_promos}),
            "episodes_with_selected_promo": len(covered), "candidates_evaluated": len(all_promos),
            "promos_selected": len(selected), "rejected": len(all_promos) - len(selected),
            "episode_errors": errors, "fallback_evaluations": sum(1 for p in all_promos if p.evaluation_source == "fallback"),
            "runtime_seconds": elapsed, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.log.info("DONE episodes=%d candidates=%d selected=%d in %.1fs", len(episodes), len(all_promos), len(selected), elapsed)
        return RunResult(episodes, parse_report, all_promos, selected, raw_outputs, logs, stats,
                         self.cfg.to_dict(), self.model.describe())

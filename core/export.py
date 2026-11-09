"""Write every artefact: parsed JSON, raw model I/O, ranked promos (JSON+CSV), per-episode
results and the processing log. Files are written exactly as produced by the run."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List

from core.pipeline import RunResult
from core.promos import Promo

CSV_COLUMNS = ["rank", "promo_id", "episode_number", "start", "end", "duration_sec", "total_score",
               "confidence_level", "hook_score", "cliffhanger_score", "standalone_score", "emotion_score",
               "hook_points", "cliffhanger_points", "standalone_points", "emotion_points", "duration_points",
               "opening_hook", "cliffhanger", "key_dialogue", "why_it_works", "moment_tags", "warnings",
               "segments", "judged_by"]


def promo_row(p: Promo) -> Dict[str, Any]:
    q = lambda l: (f"[{l['timestamp']}] {l['speaker']}: {l['text']}" if l.get("speaker") else f"[{l['timestamp']}] ({l.get('kind','')}) {l['text']}") if l else ""
    return {
        "rank": p.rank, "promo_id": p.promo_id, "episode_number": p.episode_number, "start": p.start, "end": p.end,
        "duration_sec": p.duration_sec, "total_score": p.total_score, "confidence_level": p.confidence_level,
        "hook_score": p.model_scores["hook"], "cliffhanger_score": p.model_scores["cliffhanger"],
        "standalone_score": p.model_scores["standalone"], "emotion_score": p.model_scores["emotion"],
        "hook_points": p.points["hook"], "cliffhanger_points": p.points["cliffhanger"],
        "standalone_points": p.points["standalone"], "emotion_points": p.points["emotion"],
        "duration_points": p.points["duration"],
        "opening_hook": q(p.opening_hook), "cliffhanger": q(p.cliffhanger),
        "key_dialogue": " || ".join(q(l) for l in p.key_dialogue), "why_it_works": p.why_it_works,
        "moment_tags": ", ".join(p.moment_tags), "warnings": " | ".join(p.warnings),
        "segments": json.dumps([{k: s[k] for k in ("episode_number", "start", "end")} for s in p.segments]),
        "judged_by": f"{p.judged_by.get('provider')}:{p.judged_by.get('model')}",
    }


def promos_to_csv(promos: List[Promo]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
    w.writeheader()
    for p in promos:
        w.writerow(promo_row(p))
    return buf.getvalue()


def promos_to_json(result: RunResult, promos: List[Promo] = None) -> str:
    promos = result.promos if promos is None else promos
    return json.dumps({"model": result.model_info, "config": result.config, "stats": result.stats,
                       "promos": [p.to_dict() for p in promos]}, indent=2, ensure_ascii=False)


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_outputs(result: RunResult, out_dir: str) -> Dict[str, str]:
    out = Path(out_dir)
    for sub in ("raw", "parsed", "ranked", "reports"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    # parsed
    for ep in result.episodes:
        _dump(out / "parsed" / f"episode_{ep.episode_number:03d}.json", ep.to_dict())
    _dump(out / "parsed" / "parse_report.json", result.parse_report)
    # raw: exactly what the model returned + every evaluated candidate (selected or not)
    _dump(out / "raw" / "model_responses.json", result.raw_model_outputs)
    _dump(out / "raw" / "candidates_all.json", {"model": result.model_info, "stats": result.stats,
                                                "candidates": [p.to_dict() for p in sorted(result.all_promos, key=lambda p: (p.episode_number, p.start_sec))]})
    _dump(out / "raw" / "run_manifest.json", {"model": result.model_info, "config": result.config, "stats": result.stats})
    # ranked
    (out / "ranked" / "promos_ranked.json").write_text(promos_to_json(result), encoding="utf-8")
    (out / "ranked" / "promos_ranked.csv").write_text(promos_to_csv(result.promos), encoding="utf-8")
    by_ep: Dict[int, List[Promo]] = {}
    for p in result.promos:
        by_ep.setdefault(p.episode_number, []).append(p)
    for ep in result.episodes:
        _dump(out / "ranked" / "by_episode" / f"episode_{ep.episode_number:03d}.json",
              {"episode_id": ep.episode_id, "promos": [p.to_dict() for p in by_ep.get(ep.episode_number, [])],
               "rejected_candidates": [{"cand_id": p.cand_id, "start": p.start, "end": p.end, "score": p.total_score,
                                        "reason": p.status_reason} for p in result.all_promos
                                       if p.episode_number == ep.episode_number and not p.selected]})
    (out / "raw" / "processing.log").write_text("\n".join(result.logs), encoding="utf-8")
    return {"out_dir": str(out)}

#!/usr/bin/env python3
"""PromoPilot AI - command line.

  python main.py --input path/to/scripts.zip --out outputs
  python main.py --input ./my_episodes_folder --model gemini --top 10
  python main.py --input sample_data/synthetic_show --model rule_based --episodes 1,2
"""
from __future__ import annotations

import argparse
import sys

from ai.model_factory import available_models
from config import PipelineConfig
from core.export import write_outputs
from core.service import run_promopilot
from utils.timecode import format_timecode


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PromoPilot AI - find and rank promo clips in microdrama screenplays")
    p.add_argument("--input", "-i", nargs="+", required=True, help="ZIP file(s), folder(s) or screenplay .txt/.md file(s)")
    p.add_argument("--out", "-o", default="outputs", help="output directory (default: outputs)")
    p.add_argument("--model", "-m", default=None, help=f"one of {available_models()} (default: $PROMOPILOT_MODEL or auto)")
    p.add_argument("--config", help="optional JSON file overriding PipelineConfig / scoring weights")
    p.add_argument("--episodes", help="comma-separated episode numbers to process, e.g. 1,2,3")
    p.add_argument("--top", type=int, default=10, help="how many ranked promos to print")
    p.add_argument("--max-per-episode", type=int, help="override the per-episode promo cap")
    p.add_argument("--min-score", type=float, help="override the minimum total score")
    p.add_argument("--no-write", action="store_true", help="do not write output files")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    cfg = PipelineConfig.from_json(a.config) if a.config else PipelineConfig.from_env()
    if a.max_per_episode is not None:
        cfg.max_promos_per_episode = a.max_per_episode
    if a.min_score is not None:
        cfg.min_total_score = a.min_score
    only = [int(x) for x in a.episodes.split(",")] if a.episodes else None
    try:
        res = run_promopilot(a.input, model=a.model, cfg=cfg, only_episodes=only,
                             log_file=None if a.no_write else f"{a.out}/raw/processing.log", console=True)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if not res.episodes:
        print("No screenplay episodes could be parsed from the input.", file=sys.stderr)
        return 1
    if not a.no_write:
        write_outputs(res, a.out)
    s = res.stats
    print(f"\nModel: {res.model_info['provider']} / {res.model_info['model']}")
    print(f"Episodes: {s['episodes_detected']} | candidates evaluated: {s['candidates_evaluated']} | promos selected: {s['promos_selected']} | {s['runtime_seconds']}s")
    print(f"\n{'#':>3} {'promo':<14} {'ep':>3} {'start':>8} {'end':>8} {'dur':>5} {'score':>6} conf")
    for p in res.promos[: a.top]:
        print(f"{p.rank:>3} {p.promo_id:<14} {p.episode_number:>3} {p.start:>8} {p.end:>8} {p.duration_sec:>5.1f} {p.total_score:>6.1f} {p.confidence_level}")
        print(f"      hook: {p.opening_hook.get('text','')[:90]}")
    if not a.no_write:
        print(f"\nOutputs written to {a.out}/ (raw, parsed, ranked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

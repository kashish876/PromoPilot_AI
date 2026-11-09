#!/usr/bin/env python3
"""Independent quality checker. Re-reads the ORIGINAL screenplay text with its own
regexes (it does not import the parser) and verifies what the pipeline wrote:

  python tools/quality_check.py --input scripts.zip --out outputs

Writes outputs/reports/quality_metrics.json and quality_report_auto.md.
Only things that can be measured are measured: no precision/recall/engagement claims."""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.loader import load_scripts  # noqa: E402  (only used to read raw text)

BLOCK = re.compile(r"Start:\s*\(?([\d:.]+)\)?\s*[|,\-\u2013\u2014]?\s*End:\s*\(?([\d:.]+)\)?\s*\n(.*)", re.I)
EP_NO = re.compile(r"EPISODE NUMBER:\s*Episode\s*(\d+)", re.I)


def sec(tc: str) -> float:
    parts = [float(x) for x in tc.split(":")]
    out = 0.0
    for p in parts:
        out = out * 60 + p
    return round(out, 2)


def norm(s: str) -> str:
    return " ".join(re.sub(r"[\u201c\u201d\"]", "", s).split()).lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    out = Path(a.out)
    promos_doc = json.loads((out / "ranked/promos_ranked.json").read_text(encoding="utf-8"))
    cands_doc = json.loads((out / "raw/candidates_all.json").read_text(encoding="utf-8"))
    parse_rep = json.loads((out / "parsed/parse_report.json").read_text(encoding="utf-8"))
    promos, cands = promos_doc["promos"], cands_doc["candidates"]
    scripts, skipped = load_scripts(a.input)

    raw = {}   # episode number -> {starts, ends, dialogue_blocks:[(s,e,text)], text}
    for i, s in enumerate(scripts, 1):
        m = EP_NO.search(s.text)
        n = int(m.group(1)) if m else i
        blocks = [(sec(x.group(1)), sec(x.group(2)), x.group(3).strip()) for x in BLOCK.finditer(s.text)]
        raw[n] = {"blocks": blocks, "text": norm(s.text)}

    checks = Counter()
    failures = []
    for p in promos:
        n = p["episode_number"]
        r = raw.get(n)
        checks["promos"] += 1
        if r is None:
            failures.append((p["promo_id"], "episode not found in source"))
            continue
        dur = round(p["end_sec"] - p["start_sec"], 2)
        checks["duration_ok"] += 20.0 <= dur <= 60.0
        checks["segment_sum_ok"] += abs(sum(s["duration_sec"] for s in p["segments"]) - p["duration_sec"]) < 0.05
        starts = {b[0] for b in r["blocks"] if not b[2].startswith("Cut")}
        ends = {b[1] for b in r["blocks"]}
        checks["start_exists_in_source"] += round(p["start_sec"], 2) in starts
        checks["end_exists_in_source"] += round(p["end_sec"], 2) in ends
        # no spoken line (quoted block) is sliced by the cut points
        sliced = [b for b in r["blocks"] if b[2].startswith(('"', "\u201c")) and
                  ((b[0] < p["start_sec"] - 0.05 < b[1] - 0.05) or (b[0] < p["end_sec"] - 0.05 and b[1] > p["end_sec"] + 0.05))]
        checks["no_line_sliced"] += not sliced
        if sliced:
            failures.append((p["promo_id"], "a spoken line is cut by the start/end"))
        hook = p["opening_hook"]
        checks["hook_within_3s"] += bool(hook) and hook["start_sec"] - p["start_sec"] <= 3.01
        quote_lines = [l for l in [p["opening_hook"], p["cliffhanger"]] + p["key_dialogue"] if l and l["kind"] == "dialogue"]
        ok_q = all(norm(l["text"]) in r["text"] for l in quote_lines)
        checks["quotes_verbatim_in_source"] += ok_q
        checks["quote_timestamps_match_source"] += all(
            any(abs(b[0] - l["start_sec"]) < 0.02 and norm(l["text"]) in norm(b[2]) for b in r["blocks"]) for l in quote_lines)
        if not ok_q:
            failures.append((p["promo_id"], "a quoted line was not found verbatim in the source"))
        checks["score_reconstructs"] += abs(sum(p["points"].values()) - p["total_score"]) < 0.1
    overlaps = 0
    for i, x in enumerate(promos):
        for y in promos[i + 1:]:
            if x["episode_number"] == y["episode_number"] and min(x["end_sec"], y["end_sec"]) - max(x["start_sec"], y["start_sec"]) > 0:
                overlaps += 1
    pct = lambda k: round(100.0 * checks[k] / checks["promos"], 1) if checks["promos"] else None

    detected = {e for e in (c["episode_number"] for c in cands)}
    ends_q = sum(1 for p in promos if p["cliffhanger"].get("text", "").rstrip().endswith("?"))
    metrics = {
        "files_received": parse_rep["files_received"], "episodes_parsed": parse_rep["episodes_parsed"],
        "parse_success_rate_pct": round(100 * parse_rep["parse_success_rate"], 1),
        "parse_warnings_total": parse_rep["total_warnings"],
        "source_files_independently_read": len(scripts), "skipped_inputs": skipped,
        "episodes_with_candidates": len(detected), "episodes_without_candidates": sorted(set(raw) - detected),
        "episodes_with_selected_promo": len({p["episode_number"] for p in promos}),
        "episodes_without_selected_promo": sorted(set(raw) - {p["episode_number"] for p in promos}),
        "candidates_evaluated": len(cands), "promos_selected": len(promos),
        "duration_valid_pct": pct("duration_ok"), "segment_sum_valid_pct": pct("segment_sum_ok"),
        "start_time_exists_in_source_pct": pct("start_exists_in_source"), "end_time_exists_in_source_pct": pct("end_exists_in_source"),
        "no_spoken_line_sliced_pct": pct("no_line_sliced"), "hook_within_first_3s_pct": pct("hook_within_3s"),
        "quotes_verbatim_in_source_pct": pct("quotes_verbatim_in_source"), "quote_timestamps_match_source_pct": pct("quote_timestamps_match_source"),
        "score_reconstructs_pct": pct("score_reconstructs"), "overlapping_pairs_within_episode": overlaps,
        "failures": failures,
        "score_min_median_max": [min(p["total_score"] for p in promos), statistics.median(p["total_score"] for p in promos), max(p["total_score"] for p in promos)] if promos else None,
        "duration_min_median_max": [min(p["duration_sec"] for p in promos), statistics.median(p["duration_sec"] for p in promos), max(p["duration_sec"] for p in promos)] if promos else None,
        "confidence_levels": dict(Counter(p["confidence_level"] for p in promos)),
        "promos_ending_on_a_question": ends_q,
        "promos_with_warnings_besides_model_note": sum(1 for p in promos if any("heuristics" not in w for w in p["warnings"])),
        "judged_by": promos_doc["model"],
        "rejection_reasons": dict(Counter(
            ("overlaps a higher-ranked promo" if c["status_reason"].startswith("overlaps") else
             "score below minimum" if c["status_reason"].startswith("score") else
             "hook/cliffhanger gate" if "gate" in c["status_reason"] else
             "per-episode cap" if c["status_reason"].startswith("per-episode") else
             "near-duplicate dialogue" if c["status_reason"].startswith("near-dup") else c["status_reason"])
            for c in cands if not c["selected"])),
    }
    (out / "reports").mkdir(parents=True, exist_ok=True)
    (out / "reports/quality_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    md = ["# Automated quality check (independent re-read of the source screenplay)", "",
          f"Model under test: `{metrics['judged_by']['provider']} / {metrics['judged_by']['model']}`", "",
          "| Check | Result |", "|---|---|"]
    rows = [("Files received / episodes parsed", f"{metrics['files_received']} / {metrics['episodes_parsed']} ({metrics['parse_success_rate_pct']}%)"),
            ("Parse warnings", metrics["parse_warnings_total"]),
            ("Episodes with >=1 selected promo", f"{metrics['episodes_with_selected_promo']} / {metrics['episodes_parsed']}"),
            ("Episodes with NO selected promo", metrics["episodes_without_selected_promo"] or "none"),
            ("Candidates evaluated / promos selected", f"{metrics['candidates_evaluated']} / {metrics['promos_selected']}"),
            ("Duration within 20-60 s", f"{metrics['duration_valid_pct']}%"),
            ("Start time exists in source", f"{metrics['start_time_exists_in_source_pct']}%"),
            ("End time exists in source", f"{metrics['end_time_exists_in_source_pct']}%"),
            ("No spoken line sliced by the cut", f"{metrics['no_spoken_line_sliced_pct']}%"),
            ("Hook line inside first 3 s", f"{metrics['hook_within_first_3s_pct']}%"),
            ("Quoted lines verbatim in source", f"{metrics['quotes_verbatim_in_source_pct']}%"),
            ("Quote timestamps match source", f"{metrics['quote_timestamps_match_source_pct']}%"),
            ("Score = sum of component points", f"{metrics['score_reconstructs_pct']}%"),
            ("Overlapping promo pairs inside one episode", metrics["overlapping_pairs_within_episode"]),
            ("Score min / median / max", metrics["score_min_median_max"]),
            ("Duration min / median / max (s)", metrics["duration_min_median_max"]),
            ("Confidence levels", metrics["confidence_levels"]),
            ("Promos whose last spoken line is a question", f"{ends_q} / {metrics['promos_selected']}"),
            ("Candidate rejection reasons", metrics["rejection_reasons"]),
            ("Failures", metrics["failures"] or "none")]
    md += [f"| {k} | {v} |" for k, v in rows]
    md += ["", "These are **mechanical validity checks**. They do not measure whether a promo is *good*; "
           "no precision/recall or engagement figure is claimed. See `manual_review.md` for the human read-through."]
    (out / "reports/quality_report_auto.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

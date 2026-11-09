#!/usr/bin/env python3
"""Builds docs/solution_document.pdf (2 pages). Numbers are read from outputs/ so they match the run."""
import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "outputs"
M = json.loads((out / "reports/quality_metrics.json").read_text())
man = json.loads((out / "raw/run_manifest.json").read_text())
P = json.loads((out / "ranked/promos_ranked.json").read_text())["promos"]
cfg = man["config"]
top = P[:3]
F = 8.6
body = ParagraphStyle("b", fontName="Helvetica", fontSize=F, leading=F + 1.7, alignment=TA_LEFT, spaceAfter=1.5)
h = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9.6, leading=11.5, textColor=colors.HexColor("#7a1f3d"), spaceBefore=3.5, spaceAfter=1)
title = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=13, leading=15)
sub = ParagraphStyle("s", fontName="Helvetica", fontSize=8, leading=10, textColor=colors.grey, spaceAfter=3)
cell = ParagraphStyle("c", fontName="Helvetica", fontSize=7.8, leading=9.4)
bul = ParagraphStyle("bl", parent=body, leftIndent=8, bulletIndent=0)


def B(items):
    return [Paragraph(i, bul, bulletText="\u2022") for i in items]


def tbl(rows, widths):
    t = Table([[Paragraph(str(c), cell) for c in r] for r in rows], colWidths=widths)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1e6ea")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]))
    return t


s = []
s += [Paragraph("PromoPilot AI \u2013 Problem 1: Find the Promos", title),
      Paragraph("Kuku AI Generalist assignment \u00b7 AI-powered microdrama promo discovery and ranking \u00b7 supplied 50-episode input pack", sub)]

s += [Paragraph("1. Problem chosen and why", h),
      Paragraph("Problem 1. It has a real input pack, is a weekly repeating editor task (new shows every week), and its output can be checked line by line against the screenplay "
                "(timestamps, quotes, durations). Problems 2 and 3 could not be verified as concretely in 2\u20133 hours.", body)]

s += [Paragraph("2. Framing and assumptions", h)] + B([
    "Timestamps are one <b>episode-global</b> clock (scene 2 continues from scene 1), format MM:SS.ss, with start/end for every dialogue, action, camera and sound line.",
    "Input is text only (no video/audio), so 'hook' is judged from lines, delivery notes (e.g. 'Furious') and action text, not from performance.",
    "A promo is one continuous clip that starts at the start of a spoken line/action and ends at the end of one (never cuts mid-line, no padding). Multi-segment output is supported in the data structure and validation, but not generated.",
    "Not every episode ends on a cliffhanger (episodes 10 and 25 end calmly), so 'end of episode' is <b>not</b> treated as one. Rubric weights (30/30/20/15/5) are my proposal, not Kuku's; the score is an editorial rubric, <b>not</b> predicted engagement."])

s += [Paragraph("3\u20135. Solution, architecture and what Python vs AI does", h),
      Paragraph("Pipeline: <b>parse</b> \u2192 <b>candidate windows</b> \u2192 <b>judge</b> \u2192 <b>validate</b> \u2192 <b>score &amp; rank</b> \u2192 <b>output</b>. Streamlit dashboard + CLI share one service function; "
                "ZIP/folder/single-file inputs are discovered dynamically.", body),
      tbl([["Python (deterministic, tested)", "AI model (swappable) "],
           ["Parse scenes/timestamps/speakers; build 20\u201360 s windows from real line boundaries (never slicing a spoken line, \u22652 dialogue lines); rule-based moment tags (betrayal, threat, arrival, reveal\u2026); "
            "all duration/timestamp maths; clamp scores; drop invented line IDs; strip quotes not found verbatim; gates, weighted score, ranking, de-overlap, de-duplication; confidence + warnings.",
            "Per episode, one call: reads the transcript (with line IDs) and the shortlisted windows; returns 0\u201310 for hook, cliffhanger, standalone context, emotion + the hook/cliffhanger/key line <b>IDs</b> + a one-line reason. "
            "It never returns timestamps. Default fallback = rule-based judge (heuristics, <b>not</b> an LLM), so everything runs with no API key."]], [8.9 * cm, 8.9 * cm])]

s += [Paragraph("6. Selection and ranking method", h),
      Paragraph(f"Windows are pre-scored by structural + lexical signals (opening question/threat/arrival in the first 3 s, unresolved last lines, speakers named, escalation, turn-taking), "
                f"shortlisted (\u2264{cfg['max_candidates_per_episode']}/episode, diverse) and judged. Score = hook 30 + cliffhanger 30 + standalone 20 + emotion 15 + duration 5 (full points 30\u201345 s), normalised to 100, configurable. "
                f"Gates reflect the brief's must-haves: hook \u2265 {cfg['min_hook_score']:g}, cliffhanger \u2265 {cfg['min_cliffhanger_score']:g}, total \u2265 {cfg['min_total_score']:g}; \u2264{cfg['max_promos_per_episode']} per episode, no overlap inside an episode. "
                "Each promo carries a score breakdown, confidence (high/medium/low from model confidence, weakest of hook/cliffhanger, warnings, model-vs-heuristic agreement) and warnings.", body)]

s += [Paragraph("7. Tools", h),
      Paragraph("Python 3 (stdlib only for the pipeline: re, json, zipfile, pathlib, logging, urllib), pandas + Streamlit (dashboard), reportlab (this PDF), unittest (38 tests). Model adapters: Gemini (free tier), generic OpenAI-compatible endpoint (slot for a Kuku API), rule-based fallback.", body)]

dist = M["score_min_median_max"]
s += [Paragraph("8. Actual results on the supplied 50 episodes (rule-based judge, no API key)", h)] + B([
    f"<b>{M['episodes_parsed']}/{M['files_received']}</b> episodes parsed, {M['parse_warnings_total']} parse warnings (episode 6 uses an upper-case header variant; handled). Counts match an independent grep: 1,754 camera, 1,487 dialogue, 372 action, 114 sound lines.",
    f"{M['candidates_evaluated']} shortlisted candidates judged \u2192 <b>{M['promos_selected']} promos</b> selected, covering {M['episodes_with_selected_promo']}/50 episodes; runtime {man['stats']['runtime_seconds']} s. Scores {dist[0]:.1f} / {dist[1]:.1f} / {dist[2]:.1f} (min/median/max); durations {M['duration_min_median_max'][0]:.1f}\u2013{M['duration_min_median_max'][2]:.1f} s. Confidence: {M['confidence_levels']}.",
    "Top 3: " + "; ".join(f"#{p['rank']} Ep {p['episode_number']} {p['start']}\u2013{p['end']} ({p['duration_sec']:.0f}s, {p['total_score']:.0f}) hook \u201c{p['opening_hook']['text'][:40]}\u201d" for p in top) +
    ". Post first: #1 (stakes + unanswered 'When?'); #3 is a clean reunion twist. Hold #6 (Ep 3): exciting but ends on a resolving rescue."])

s += [Paragraph("9. Quality checking", h),
      Paragraph(f"(a) <b>Independent mechanical check</b> (tools/quality_check.py re-reads the raw screenplay with its own regexes): duration 20\u201360 s {M['duration_valid_pct']}%; start/end exist in source {M['start_time_exists_in_source_pct']}/{M['end_time_exists_in_source_pct']}%; no spoken line sliced {M['no_spoken_line_sliced_pct']}%; "
                f"hook inside first 3 s {M['hook_within_first_3s_pct']}%; every quoted line verbatim at the cited time {M['quote_timestamps_match_source_pct']}%; overlapping pairs {M['overlapping_pairs_within_episode']}. "
                "(b) <b>Human read-through</b> of ranks 1\u20136, 12 and the bottom four (outputs/reports/manual_review.md): one reader, no editor labels. "
                "(c) 38 unit tests: subset ZIP, single file, messy second format, junk files, hallucinating/failing/garbage models, model swap, UI smoke test. "
                "<b>Not measured:</b> precision, recall, accuracy, engagement \u2013 there are no labels, and I do not claim any.", body)]

s += [Paragraph("10. Where it fails (real, found by reading outputs)", h)] + B([
    "<b>Mic-drop \u2260 cliffhanger:</b> #4 (Ep 31) ends on a witty comeback + frozen reaction and got cliffhanger 8.5; an editor would call that beat closed. <b>Resolved ending ranked #6</b> (Ep 3, rescue speech trailing off).",
    "<b>Cold-viewer context is approximated</b> (speakers, roles, back-references); it cannot know a late-episode clip depends on earlier ones, and there is <b>no spoiler flag</b>. Action-only cliffhangers are under-detected (16/79 promos end on action).",
    "Early versions rewarded a bare 'Yes.' as a hook and let 20-second silent openers through on total score alone \u2192 fixed with a reactive-opener penalty and hook/cliffhanger gates. Lowest picks (#76\u201379) sit right at the hook gate and would not be posted.",
    "<b>Not run live:</b> Gemini/OpenAI-compatible calls and the real Streamlit server were not executed in the build sandbox (no network); adapters are tested with fake transports only."])

s += [Paragraph("11\u201313. Scalability, model swap, improvements", h)] + B([
    "<b>New show/volume:</b> episodes are independent; one model call per episode. Measured prompt size \u2248 143k input tokens / 50 calls for the whole pack (\u2248 2.9k tokens per episode), so a weekly show fits free tiers; Python pre-filtering keeps LLM spend small. New format: parser tolerates un-bolded labels, HH:MM:SS, missing headers/ends (warns, never crashes); a truly new layout = one parser function.",
    "<b>Swap model:</b> implement <font face='Courier'>BaseLLM.evaluate_episode</font>, register it (or set PROMOPILOT_MODEL / dropdown / base URL+key+model for an OpenAI-compatible Kuku API). Parser, scoring and UI are untouched (demonstrated by a test). Failed/garbage output falls back per candidate and is labelled.",
    "<b>With stronger models/data:</b> LLM judge over all windows, spoiler/continuity flag, multi-segment and cross-episode stitching, calibrating weights on editor picks and post performance, audio/video cues; editors stay in the loop on low-confidence/disagreement flags."])

s += [Paragraph("14. AI tools used and where they got it wrong", h),
      Paragraph("<b>ChatGPT</b>: assignment breakdown and problem choice. <b>Claude</b>: wrote and iterated the code, tests and this document under my direction; I own the design decisions and can explain them. "
                "<b>Real errors:</b> (1) Claude's first shell statistics over the pack (concatenating files with cat) undercounted headers because the files have no trailing newline, so lines merged across files; caught by an implausible count and redone per file. "
                "(2) Its first parser silently merged an un-bolded speaker line into the previous line's text; only a test on a messy sample caught it. "
                "(3) Its first hook heuristic scored a bare reply 'Yes.' as a good hook (6/10). All three were fixed and are covered by tests.", body)]

doc = SimpleDocTemplate(str(ROOT / "docs/solution_document.pdf"), pagesize=A4, leftMargin=1.3 * cm, rightMargin=1.3 * cm, topMargin=1.1 * cm, bottomMargin=1.0 * cm,
                        title="PromoPilot AI - Solution Document", author="")
doc.build(s)
print("built")

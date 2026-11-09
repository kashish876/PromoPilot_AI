"""PromoPilot AI - Streamlit dashboard.   Run:  streamlit run app.py

The UI is a thin layer: every decision (parsing, candidates, model call, validation,
scoring, ranking) happens in core/ and ai/ via core.service.run_promopilot, the same
function the CLI uses. Swapping the model changes nothing in this file except the dropdown.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from ai.model_factory import available_models
from config import PipelineConfig
from core.export import promos_to_csv, promos_to_json
from core.loader import load_scripts
from core.parser import parse_episode_text
from core.service import run_promopilot

SAMPLE_DIR = Path(__file__).parent / "sample_data" / "synthetic_show"

st.set_page_config(page_title="PromoPilot AI", page_icon="🎬", layout="wide")


def secret(name: str, default: str = "") -> str:
    """Streamlit secrets first, then environment variables. Never hardcoded."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


# ------------------------------------------------------------------------------ header
st.title("🎬 PromoPilot AI")
st.caption("AI-powered microdrama promo discovery and ranking engine")
st.markdown(
    "Upload timestamped screenplays and get **ranked 20-60 s promo clips**: a hook in the first 3 s, an unresolved ending, "
    "and enough context for a viewer who has never seen the show. Python parses, validates and scores; an interchangeable "
    "model (or the free rule-based fallback) judges hook, cliffhanger, context and emotion. "
    "_Scores are a transparent editorial rubric, **not** predicted audience engagement._")

# ------------------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Model")
    provider = st.selectbox("AI model", available_models(), index=0,
                            help="auto = Gemini if a key is configured, else an OpenAI-compatible endpoint, else the free rule-based fallback")
    overrides = {}
    if provider in ("auto", "gemini"):
        overrides["gemini_api_key"] = st.text_input("Gemini API key", value=secret("GEMINI_API_KEY"), type="password")
        overrides["gemini_model"] = st.text_input("Gemini model", value=secret("GEMINI_MODEL", "gemini-2.5-flash"))
    if provider in ("auto", "openai_compatible"):
        with st.expander("OpenAI-compatible endpoint (e.g. a Kuku-provided API)"):
            overrides["openai_base_url"] = st.text_input("Base URL", value=secret("OPENAI_COMPAT_BASE_URL"))
            overrides["openai_api_key"] = st.text_input("API key", value=secret("OPENAI_COMPAT_API_KEY"), type="password")
            overrides["openai_model"] = st.text_input("Model name", value=secret("OPENAI_COMPAT_MODEL"))
    st.header("Selection rules")
    cfg = PipelineConfig.from_env()
    cfg.max_promos_per_episode = st.slider("Max promos per episode", 1, 5, cfg.max_promos_per_episode)
    cfg.min_total_score = st.slider("Minimum total score", 0, 90, int(cfg.min_total_score))
    cfg.min_hook_score = st.slider("Hook gate (0-10)", 0.0, 9.0, float(cfg.min_hook_score), 0.5)
    cfg.min_cliffhanger_score = st.slider("Cliffhanger gate (0-10)", 0.0, 9.0, float(cfg.min_cliffhanger_score), 0.5)
    cfg.max_candidates_per_episode = st.slider("Candidates sent to the model per episode", 3, 15, cfg.max_candidates_per_episode)
    with st.expander("Scoring weights (total is normalised to 100)"):
        for k in list(cfg.scoring.weights):
            cfg.scoring.weights[k] = st.number_input(k, 0.0, 100.0, float(cfg.scoring.weights[k]), 1.0)

# ------------------------------------------------------------------------------ input
st.subheader("1 · Input")
uploads = st.file_uploader("Screenplay ZIP or individual .txt / .md files", type=["zip", "txt", "md"], accept_multiple_files=True)
use_sample = st.checkbox("Use the bundled synthetic sample show (3 episodes, safe to share)", value=not uploads)

sources = []
if uploads:
    sources = [(u.name, u.getvalue()) for u in uploads]
elif use_sample and SAMPLE_DIR.exists():
    sources = [SAMPLE_DIR]


@st.cache_data(show_spinner=False)
def detect(sig: str, payload):
    scripts, skipped = load_scripts(payload)
    eps = [parse_episode_text(s.text, s.name, i + 1) for i, s in enumerate(scripts)]
    return [{"file": e.source_file, "episode": e.episode_id, "scenes": len(e.scenes), "events": len(e.events),
             "duration_s": round(e.duration, 1), "warnings": len(e.warnings)} for e in eps], skipped


if sources:
    sig = hashlib.md5(repr([(s if isinstance(s, Path) else (s[0], len(s[1]))) for s in sources]).encode()).hexdigest()
    detected, skipped = detect(sig, sources)
    good = [d for d in detected if d["events"] > 0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Episodes detected", len(good))
    c2.metric("Files skipped / unparseable", len(skipped) + len(detected) - len(good))
    c3.metric("Parse warnings", sum(d["warnings"] for d in detected))
    with st.expander("Detected episodes and skipped files"):
        st.dataframe(pd.DataFrame(detected), use_container_width=True)
        for note in skipped:
            st.warning(note)
else:
    st.info("Upload a ZIP/screenplay files, or tick the sample checkbox, to begin.")

# ------------------------------------------------------------------------------ run
st.subheader("2 · Run")
if st.button("Run promo detection", type="primary", disabled=not sources):
    bar = st.progress(0.0, text="Starting...")

    def progress(i: int, n: int, msg: str) -> None:
        bar.progress(min(1.0, i / max(n, 1)), text=f"{msg} ({i}/{n})")

    try:
        with st.status("Processing screenplays...", expanded=False) as status:
            st.session_state["result"] = run_promopilot(sources, model=provider, model_overrides=overrides, cfg=cfg, progress=progress)
            status.update(label="Done", state="complete")
        bar.progress(1.0, text="Done")
    except Exception as e:   # e.g. chosen model without credentials
        st.error(f"Could not run: {e}")

res = st.session_state.get("result")
if res is None:
    st.stop()

# ------------------------------------------------------------------------------ results
st.subheader("3 · Ranked promos")
s = res.stats
m = st.columns(5)
m[0].metric("Episodes processed", s["episodes_detected"])
m[1].metric("Candidates evaluated", s["candidates_evaluated"])
m[2].metric("Promos discovered", s["promos_selected"])
m[3].metric("Runtime (s)", s["runtime_seconds"])
m[4].metric("Model", f"{res.model_info['provider']}")
if s["fallback_evaluations"]:
    st.warning(f"{s['fallback_evaluations']} evaluation(s) used the rule-based fallback because the chosen model failed or returned unusable output.")
if res.parse_report["parse_failures"]:
    st.error(f"{len(res.parse_report['parse_failures'])} file(s) could not be parsed: " + ", ".join(f["file"] for f in res.parse_report["parse_failures"]))

eps_all = sorted({p.episode_number for p in res.promos})
f1, f2, f3 = st.columns([2, 1, 1])
pick_eps = f1.multiselect("Filter by episode", eps_all, default=[])
min_view = f2.slider("Min score", 0, 100, 0)
confs = f3.multiselect("Confidence", ["high", "medium", "low"], default=["high", "medium", "low"])
shown = [p for p in res.promos if (not pick_eps or p.episode_number in pick_eps) and p.total_score >= min_view and p.confidence_level in confs]

st.dataframe(pd.DataFrame([{
    "Rank": p.rank, "Promo": p.promo_id, "Episode": p.episode_number, "Start": p.start, "End": p.end,
    "Dur (s)": p.duration_sec, "Score": p.total_score, "Confidence": p.confidence_level,
    "Hook": p.opening_hook.get("text", ""), "Ends on": p.cliffhanger.get("text", "")} for p in shown]),
    use_container_width=True, hide_index=True)

d1, d2, _ = st.columns([1, 1, 4])
d1.download_button("⬇ CSV", promos_to_csv(shown), "promos_ranked.csv", "text/csv")
d2.download_button("⬇ JSON", promos_to_json(res, shown), "promos_ranked.json", "application/json")

# ------------------------------------------------------------------------------ detail
if shown:
    st.subheader("4 · Promo detail")
    choice = st.selectbox("Select a promo", shown, format_func=lambda p: f"#{p.rank} · {p.promo_id} · Ep {p.episode_number} · {p.total_score:.1f}")
    a, b = st.columns([3, 2])
    with a:
        st.markdown(f"**Episode {choice.episode_number} · {choice.start} → {choice.end} · {choice.duration_sec:.1f}s · "
                    f"score {choice.total_score:.1f}/100 · confidence {choice.confidence_level}**")
        for i, sg in enumerate(choice.segments, 1):
            st.write(f"Segment {i}: Ep {sg['episode_number']} {sg['start']} → {sg['end']} ({sg['duration_sec']}s)")
        h, c = choice.opening_hook, choice.cliffhanger
        st.markdown(f"**🎣 Opening hook** `{h.get('timestamp', '')}` {h.get('speaker', '')}: {h.get('text', '')}")
        st.markdown(f"**🪝 Cliffhanger** `{c.get('timestamp', '')}` {c.get('speaker', '')}: {c.get('text', '')}")
        st.markdown(f"**Why it works / fails:** {choice.why_it_works}")
        if choice.moment_tags:
            st.caption("Rule-based moment tags: " + ", ".join(choice.moment_tags))
        for w in choice.warnings:
            st.warning(w)
    with b:
        st.markdown("**Score breakdown**")
        df = pd.DataFrame({"points": choice.points}).rename_axis("criterion")
        st.bar_chart(df)
        st.caption(choice.score_explanation)
        st.caption(f"Judged by {choice.judged_by.get('provider')} / {choice.judged_by.get('model')} ({choice.evaluation_source})")
    with st.expander("Timestamped transcript of the clip (from the screenplay)", expanded=True):
        st.dataframe(pd.DataFrame(choice.transcript)[["timestamp", "end_timestamp", "kind", "speaker", "delivery", "text"]],
                     use_container_width=True, hide_index=True)

with st.expander("Rejected candidates"):
    st.dataframe(pd.DataFrame([{"Episode": p.episode_number, "Start": p.start, "End": p.end, "Score": p.total_score, "Reason": p.status_reason}
                               for p in res.all_promos if not p.selected]), use_container_width=True, hide_index=True)
with st.expander("Processing log"):
    st.code("\n".join(res.logs[-300:]) or "(empty)")

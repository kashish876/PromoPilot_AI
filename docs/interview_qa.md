# Likely interview questions (answer in your own words; verify against the code)

1. **Why Problem 1?** Real input pack, repeating weekly workload, output checkable line-by-line against the script; problems 2/3 couldn't be verified as concretely in 2-3 h.
2. **What does the AI do vs Python?** Python: parsing, candidate windows, duration/timestamp maths, validation, scoring, ranking. Model: judging 4 qualitative criteria and writing the reason, via line IDs only.
3. **How do you stop hallucinated quotes/timestamps?** Model gets a transcript with line IDs and returns IDs; Python maps IDs to real text/times, discards IDs outside the clip, strips quoted fragments not found verbatim. `tools/quality_check.py` re-reads the source independently: 100% of quotes/times verified.
4. **How is the score computed?** 0-10 per criterion x configurable weights (30/30/20/15), duration points in Python (5), normalised to 100. Gates: hook >= 4, cliffhanger >= 3.5. It is a rubric, not predicted engagement.
5. **How do you swap models?** `BaseLLM.evaluate_episode`; factory registry; config/env/dropdown. `openai_compatible` fits a Kuku API. Tests register a fake model.
6. **What if the API fails?** Per-episode/per-candidate fallback to the rule-based judge, labelled in the output.
7. **What did you NOT test?** Live Gemini/OpenAI calls and the real Streamlit server (sandbox had no network/Streamlit). Say so, and run them once before the interview.
8. **Where does it fail?** See `outputs/reports/manual_review.md`: mic-drop endings read as cliffhangers, resolved rescue ranked #6, no spoiler awareness, action-only cliffhangers under-detected.
9. **Cost at scale?** Measured: ~143k input tokens, 50 calls for 50 episodes (one call per episode, ~2.9k tokens each). Candidate filtering in Python keeps LLM spend small; rule-based pass is free.
10. **Human in the loop?** Editors pick from ranked list with warnings/confidence; low-confidence and disagreement flags tell them where to look.
11. **New show with different format?** Parser tolerates variants and warns; new layout = one parser function. Show `sample_data/synthetic_show_variant`.
12. **How would you improve it?** Stronger LLM judge, editor-labelled data to calibrate weights, spoiler/continuity flag, multi-segment stitching, audio/video cues, A/B results feedback.

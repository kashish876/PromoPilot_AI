# 3-minute walkthrough script (screen recording, your own voice)

**0:00-0:20 Problem.** "Editors watch every episode to find 20-60 s promos: hook in 3 s, cliffhanger ending, makes sense cold. New shows weekly, so I built PromoPilot AI."
**0:20-0:50 Run on the real pack.** Terminal: `python main.py --input scripts.zip --out outputs`. Point out: 50/50 parsed, ~5 s, 79 promos, no API key.
**0:50-1:30 Dashboard.** `streamlit run app.py` -> upload ZIP -> episodes detected -> Run. Show ranked table, filter by episode, open the #1 promo: segments, hook, cliffhanger, reason, score breakdown, warnings, transcript.
**1:30-2:00 Trust.** "The model returns line IDs, never timestamps. Python checks duration, drops invented lines and untraceable quotes." Show `tools/quality_check.py` output (100% mechanical checks) and say what it does NOT prove.
**2:00-2:30 Swap the model.** Sidebar dropdown rule_based -> gemini (paste key) -> run again; or show the 10-line `register_model` example and the test.
**2:30-3:00 Honest failures + next steps.** Show `manual_review.md`: #4 mic-drop ending mistaken for cliffhanger; no spoiler flag. "With a stronger LLM and editor labels I'd calibrate the rubric."

# PromoPilot AI - microdrama promo discovery and ranking

Reads timestamped microdrama screenplays and returns **ranked 20-60 s promo clips**: hook in the first 3 s,
unresolved ending, enough context for a stranger. Python parses/validates/scores; a **swappable model** (or a free
rule-based fallback) judges hook, cliffhanger, standalone context and emotion.

> **Confidential data:** Kuku's input pack and `outputs/` (which quote it) must never be published. `.gitignore` excludes them.
> The public demo uses the synthetic show in `sample_data/` (original text).

## Run it (laptop or Colab; Python 3.10+)
```bash
pip install -r requirements.txt            # CLI needs nothing beyond the standard library
python main.py --input path/to/scripts.zip --out outputs            # free rule-based judge, ~5 s for 50 episodes
python main.py --input folder_or_file.txt --episodes 1,2,3 --top 10
GEMINI_API_KEY=... python main.py --input scripts.zip --model gemini
streamlit run app.py                       # dashboard: upload ZIP/files, pick model, filter, detail view, CSV/JSON download
python -m unittest discover -s tests -t .  # 38 tests
python tools/quality_check.py --input scripts.zip --out outputs      # independent source-vs-output verification
```
Colab: upload the folder, `!pip install -r requirements.txt`, then `!python main.py --input /content/scripts.zip`.
Inputs accepted: ZIP (nested ok), folder, or individual `.txt/.md` files. Nothing is hardcoded about names or episode counts.

## Pipeline (core/)
| Step | Module | Who does it |
|---|---|---|
| 1 Parse (scenes, timestamps, speakers, actions, SFX) | `parser.py`, `loader.py` | Python |
| 2 Candidate windows (start/end only on real beats, 20-60 s, never slicing a spoken line, rule-based tags) | `candidates.py`, `signals.py` | Python rules + light NLP |
| 3 Judge hook / cliffhanger / standalone / emotion (0-10) + reason, by **line IDs** | `ai/*` | Model (or rule-based fallback) |
| 4 Validate: clamp scores, drop invented line IDs, strip untraceable quotes, check durations | `validate.py` | Python |
| 5 Score (configurable weights 30/30/20/15/5), gates, rank, de-overlap, de-duplicate | `scoring.py`, `selection.py` | Python |
| 6 Output: segments, hook, cliffhanger, key lines, reason, breakdown, confidence, warnings | `promos.py`, `export.py` | Python |

The model never produces timestamps; it refers to lines by ID, and Python looks up the real text and times.

## Swap the model (no other file changes)
Config: `PROMOPILOT_MODEL=auto|rule_based|gemini|openai_compatible` or `--model`, or the dashboard dropdown.
`openai_compatible` + `OPENAI_COMPAT_BASE_URL/API_KEY/MODEL` is the slot for a Kuku-provided API. A brand-new backend:
```python
from ai.base_model import BaseLLM
from ai.model_factory import register_model
class KukuModel(BaseLLM):
    provider, model = "kuku", "v1"
    def evaluate_episode(self, request):      # return one dict per candidate (see ai/base_model.EVAL_KEYS)
        ...
register_model("kuku", KukuModel)             # then: --model kuku
```
`tests/test_pipeline_and_models.py::SwapTests` does exactly this. If a model fails or returns junk, the affected
candidates fall back to the rule-based judge and are labelled `evaluation_source: fallback`.

## New show during an interview
`python main.py --input new_show.zip --out outputs/new_show` (or upload in the dashboard). If the format differs slightly,
the parser tolerates un-bolded labels, HH:MM:SS clocks, missing headers/end times and warns instead of failing
(`sample_data/synthetic_show_variant` + tests). A completely different layout needs one new function in `core/parser.py`.

## Layout
`app.py` dashboard | `main.py` CLI | `config.py` | `core/` | `ai/` | `utils/` | `tests/` | `tools/` | `sample_data/` (synthetic) |
`outputs/{raw,parsed,ranked,reports}` (run on the real 50 episodes; confidential) | `docs/` | `demo/`

## Honest limits
- Live Gemini/OpenAI-compatible calls were **not executed in the build sandbox** (no network); adapters are tested only with fake transports. Run once with your key.
- Streamlit was not installable in the build sandbox; `app.py` is smoke-tested against a stub. Run `streamlit run app.py` yourself.
- The default judge is heuristic, not an LLM. Scores are a rubric, **not** predicted engagement. No precision/recall exists (no editor labels).
- Multi-segment promos are supported in the output structure and validation, but the generator emits single continuous clips only; no cross-episode joins.

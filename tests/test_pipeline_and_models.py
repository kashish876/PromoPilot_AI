import json, unittest
from tests.helpers import SAMPLE, zip_of
from ai.base_model import BaseLLM, EpisodeRequest, extract_json
from ai.gemini_model import GeminiModel
from ai.openai_compatible_model import OpenAICompatibleModel
from ai.model_factory import create_model, register_model, REGISTRY
from ai.rule_based_model import RuleBasedModel
from config import PipelineConfig
from core.candidates import generate_candidates
from core.export import promos_to_csv, promos_to_json, write_outputs
from core.parser import parse_episode_text
from core.scoring import duration_points, score_promo
from core.service import run_promopilot
from core.models import Segment
from core.validate import validate_segments, normalize_evaluation, verify_quotes, normalize_text

SRC = [("show.zip", zip_of(SAMPLE / "synthetic_show"))]


class ScoringTests(unittest.TestCase):
    def test_default_weights_sum_to_100(self):
        self.assertEqual(PipelineConfig().scoring.total_weight(), 100.0)

    def test_perfect_scores_give_100(self):
        bd = score_promo({"hook": 10, "cliffhanger": 10, "standalone": 10, "emotion": 10}, 40.0, PipelineConfig())
        self.assertEqual(bd.total, 100.0)

    def test_duration_points_shape(self):
        cfg = PipelineConfig()
        self.assertEqual(duration_points(35, cfg), 5.0)
        self.assertEqual(duration_points(20, cfg), 2.5)
        self.assertEqual(duration_points(60, cfg), 2.5)
        self.assertEqual(duration_points(19.9, cfg), 0.0)
        self.assertEqual(duration_points(60.1, cfg), 0.0)

    def test_custom_weights_normalise_to_100(self):
        cfg = PipelineConfig()
        cfg.scoring.weights = {"hook": 1, "cliffhanger": 1, "standalone": 0, "emotion": 0, "duration": 0}
        self.assertEqual(score_promo({"hook": 10, "cliffhanger": 5, "standalone": 0, "emotion": 0}, 30, cfg).total, 75.0)

    def test_multi_segment_validation(self):
        ok = [Segment(1, 10, 25), Segment(1, 40, 60)]
        self.assertEqual(validate_segments(ok, 20, 60), [])
        self.assertTrue(validate_segments([Segment(1, 0, 10)], 20, 60))
        self.assertTrue(validate_segments([Segment(1, 0, 40), Segment(1, 50, 80)], 20, 60))
        self.assertTrue(validate_segments([Segment(1, 40, 60), Segment(1, 10, 25)], 20, 60))


class PipelineInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = run_promopilot(SRC, model="rule_based")

    def test_all_episodes_processed(self):
        self.assertEqual(self.res.stats["episodes_detected"], 3)
        self.assertEqual(self.res.parse_report["parse_success_rate"], 1.0)

    def test_every_promo_valid(self):
        self.assertTrue(self.res.promos)
        for p in self.res.promos:
            self.assertTrue(20 <= p.duration_sec <= 60, p)
            self.assertTrue(p.duration_valid)
            self.assertGreater(p.end_sec, p.start_sec)
            self.assertEqual(p.segments[0]["start_sec"], p.start_sec)

    def test_hook_inside_first_three_seconds(self):
        for p in self.res.promos:
            if p.opening_hook:
                self.assertLessEqual(p.opening_hook["start_sec"] - p.start_sec, 3.01)

    def test_no_overlap_within_episode(self):
        for a in self.res.promos:
            for b in self.res.promos:
                if a is not b and a.episode_number == b.episode_number:
                    self.assertTrue(a.end_sec <= b.start_sec or b.end_sec <= a.start_sec)

    def test_ranking_sorted_and_score_reconstructs(self):
        scores = [p.total_score for p in self.res.promos]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual([p.rank for p in self.res.promos], list(range(1, len(scores) + 1)))
        for p in self.res.promos:
            self.assertAlmostEqual(sum(p.points.values()), p.total_score, places=1)

    def test_quotes_are_traceable_to_source(self):
        src = {e.event_id: e for ep in self.res.episodes for e in ep.events}
        texts = {(ep.episode_number, e.event_id): e.text for ep in self.res.episodes for e in ep.events}
        for p in self.res.promos:
            for l in [p.opening_hook, p.cliffhanger] + p.key_dialogue:
                if l:
                    self.assertEqual(texts[(p.episode_number, l["line_id"])], l["text"])

    def test_exports(self):
        csv_text = promos_to_csv(self.res.promos)
        self.assertEqual(len(csv_text.strip().splitlines()), len(self.res.promos) + 1)
        data = json.loads(promos_to_json(self.res))
        self.assertEqual(len(data["promos"]), len(self.res.promos))

    def test_subset_and_single_file_inputs(self):
        sub = run_promopilot([("s.zip", zip_of(SAMPLE / "synthetic_show", only=["001"]))], model="rule_based")
        self.assertEqual(sub.stats["episodes_detected"], 1)
        one = run_promopilot([SAMPLE / "synthetic_show" / "episode-002-Synthetic 2.txt"], model="rule_based")
        self.assertEqual(one.stats["episodes_detected"], 1)
        self.assertEqual(one.episodes[0].episode_number, 2)

    def test_messy_variant_still_produces_promos(self):
        res = run_promopilot([SAMPLE / "synthetic_show_variant"], model="rule_based")
        self.assertEqual(res.stats["episodes_detected"], 3)
        self.assertTrue(res.promos)
        self.assertGreater(res.parse_report["total_warnings"], 0)

    def test_deterministic(self):
        again = run_promopilot(SRC, model="rule_based")
        self.assertEqual([(p.promo_id, p.total_score) for p in again.promos], [(p.promo_id, p.total_score) for p in self.res.promos])

    def test_write_outputs(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            write_outputs(self.res, d)
            for rel in ("ranked/promos_ranked.json", "ranked/promos_ranked.csv", "parsed/episode_001.json", "raw/candidates_all.json",
                        "raw/model_responses.json", "raw/processing.log", "ranked/by_episode/episode_002.json"):
                self.assertTrue(os.path.exists(os.path.join(d, rel)), rel)

    def test_unparseable_input_gives_empty_result_not_crash(self):
        res = run_promopilot([("junk.txt", b"nothing useful here")], model="rule_based")
        self.assertEqual(res.stats["episodes_detected"], 0)
        self.assertEqual(res.parse_report["parse_failures"][0]["file"], "junk.txt")


class HallucinationGuardTests(unittest.TestCase):
    def setUp(self):
        text = (SAMPLE / "synthetic_show" / "episode-001-Synthetic 1.txt").read_text()
        self.ep = parse_episode_text(text, "e1.txt")
        self.cand = generate_candidates(self.ep, PipelineConfig())[0]

    def test_invented_line_ids_and_quotes_are_removed(self):
        raw = {"cand_id": self.cand.cand_id, "hook": 9, "cliffhanger": 9, "standalone": 9, "emotion": 9,
               "hook_line_id": "L999", "cliffhanger_line_id": "L998", "key_line_ids": ["L997"],
               "reason": 'The line "I will bury you all tonight, Mara" is devastating.', "warnings": [], "confidence": 0.9}
        ev = normalize_evaluation(raw, self.cand, self.ep)
        self.assertEqual(ev.hook_line_id, "")
        self.assertNotIn("bury you all", ev.reason)
        self.assertTrue(any("quote" in w for w in ev.warnings))
        self.assertTrue(any("outside the clip" in w for w in ev.warnings))

    def test_real_quote_is_kept(self):
        self.assertIn("Watch me", verify_quotes('She says "Watch me." clearly.', normalize_text("Mara: Watch me. Daniel: no"))[0])

    def test_scale_and_clamp(self):
        raw = {"cand_id": self.cand.cand_id, "hook": 80, "cliffhanger": 90, "standalone": 70, "emotion": 60}
        ev = normalize_evaluation(raw, self.cand, self.ep)
        self.assertEqual(ev.scores["hook"], 8.0)
        raw = {"cand_id": self.cand.cand_id, "hook": -5, "cliffhanger": "x", "standalone": 1, "emotion": 1}
        self.assertFalse(normalize_evaluation(raw, self.cand, self.ep).ok)

    def test_python_never_trusts_model_timestamps(self):
        raw = {"cand_id": self.cand.cand_id, "hook": 5, "cliffhanger": 5, "standalone": 5, "emotion": 5,
               "start": "00:00.00", "end": "09:59.00", "duration": 12}
        ev = normalize_evaluation(raw, self.cand, self.ep)
        self.assertTrue(ev.ok)
        self.assertFalse(hasattr(ev, "end"))


class FakeLLM(BaseLLM):
    """Demonstrates swappability: a brand-new backend in ~10 lines, zero changes elsewhere."""
    provider, model = "fake", "fake-1"

    def evaluate_episode(self, request: EpisodeRequest):
        return [{"cand_id": c.cand_id, "hook": 9, "cliffhanger": 8, "standalone": 7, "emotion": 6,
                 "hook_line_id": c.segments[0].event_ids[0], "cliffhanger_line_id": c.segments[0].event_ids[-1],
                 "key_line_ids": [], "reason": "fake", "warnings": [], "confidence": 0.9} for c in request.candidates]


class FlakyLLM(BaseLLM):
    provider, model = "flaky", "x"

    def evaluate_episode(self, request):
        raise RuntimeError("API down")


class SwapTests(unittest.TestCase):
    def test_register_and_run_custom_model(self):
        register_model("fake", FakeLLM)
        try:
            res = run_promopilot(SRC, model="fake")
            self.assertEqual(res.model_info["provider"], "fake")
            self.assertTrue(all(p.judged_by["provider"] == "fake" for p in res.promos))
            self.assertTrue(res.promos[0].total_score > 70)
        finally:
            REGISTRY.pop("fake", None)

    def test_failing_model_falls_back_to_rules(self):
        res = run_promopilot(SRC, model_instance=FlakyLLM())
        self.assertTrue(res.promos)
        self.assertTrue(all(p.evaluation_source == "fallback" for p in res.promos))
        self.assertIn("rule-based fallback", " ".join(res.promos[0].warnings))

    def test_gemini_adapter_with_fake_transport(self):
        calls = {}
        def transport(url, payload, headers):
            calls.update(url=url, payload=payload, headers=headers)
            req_ids = json.loads(payload["contents"][0]["parts"][0]["text"].split("\n\n", 1)[1].rsplit("\n\n", 1)[0])
            ids = [c["cand_id"] for c in req_ids["candidates"]]
            out = [{"cand_id": i, "hook": 7, "cliffhanger": 7, "standalone": 7, "emotion": 7, "key_line_ids": [], "reason": "ok", "warnings": [], "confidence": 0.8} for i in ids]
            return {"candidates": [{"content": {"parts": [{"text": "```json\n" + json.dumps(out) + "\n```"}]}}]}
        m = GeminiModel(api_key="test-key", model="gemini-test", transport=transport)
        res = run_promopilot(SRC, model_instance=m)
        self.assertIn("gemini-test:generateContent", calls["url"])
        self.assertEqual(calls["headers"]["x-goog-api-key"], "test-key")
        self.assertNotIn("test-key", json.dumps(calls["payload"]))             # key never enters the prompt
        self.assertTrue(res.promos and res.promos[0].judged_by["provider"] == "gemini")

    def test_openai_compatible_adapter_with_fake_transport(self):
        def transport(url, payload, headers):
            self.assertTrue(url.endswith("/chat/completions"))
            self.assertEqual(headers["Authorization"], "Bearer k")
            return {"choices": [{"message": {"content": "not json at all"}}]}
        m = OpenAICompatibleModel(base_url="https://kuku.example/v1", api_key="k", model="strong", transport=transport)
        res = run_promopilot(SRC, model_instance=m)       # garbage output -> per-candidate fallback, no crash
        self.assertTrue(res.promos)
        self.assertTrue(all(p.evaluation_source == "fallback" for p in res.promos))

    def test_factory_auto_and_errors(self):
        self.assertIsInstance(create_model("auto", gemini_api_key="", openai_base_url=""), (RuleBasedModel, GeminiModel, OpenAICompatibleModel))
        self.assertIsInstance(create_model("mock"), RuleBasedModel)
        with self.assertRaises(ValueError):
            create_model("nope")
        with self.assertRaises(ValueError):
            create_model("gemini", gemini_api_key="")

    def test_extract_json_variants(self):
        self.assertEqual(extract_json('[{"a":1}]'), [{"a": 1}])
        self.assertEqual(extract_json('Sure!\n```json\n{"evaluations":[{"a":2}]}\n```'), [{"a": 2}])
        with self.assertRaises(ValueError):
            extract_json("no json here")


if __name__ == "__main__":
    unittest.main()

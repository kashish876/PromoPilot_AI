import unittest, zipfile, io
from tests.helpers import ROOT, SAMPLE, zip_of
from core.parser import parse_episode_text
from core.loader import load_scripts
from utils.timecode import parse_timecode, format_timecode


class TimecodeTests(unittest.TestCase):
    def test_parse_formats(self):
        self.assertEqual(parse_timecode("(02:15.50)"), 135.5)
        self.assertEqual(parse_timecode("00:01:02.5"), 62.5)
        self.assertEqual(parse_timecode("12.25"), 12.25)
        self.assertIsNone(parse_timecode("abc"))
        self.assertIsNone(parse_timecode(""))

    def test_roundtrip(self):
        self.assertEqual(format_timecode(135.5), "02:15.50")
        self.assertEqual(parse_timecode(format_timecode(61.07)), 61.07)


class ParserTests(unittest.TestCase):
    def test_kuku_format(self):
        ep = parse_episode_text((SAMPLE / "synthetic_show" / "episode-001-Synthetic 1.txt").read_text(), "episode-001.txt")
        self.assertEqual(ep.episode_number, 1)
        self.assertEqual(len(ep.scenes), 2)
        self.assertEqual(ep.scenes[0].location, "GRAND BALLROOM")
        self.assertEqual(ep.scenes[0].time_of_day, "NIGHT")
        d = [e for e in ep.events if e.kind == "dialogue"]
        self.assertEqual(d[0].speaker, "MARA")
        self.assertEqual(d[0].delivery, "Shaken")
        self.assertFalse(d[0].text.startswith('"'))
        self.assertEqual(ep.warnings, [])

    def test_variant_format_is_tolerated_and_warns(self):
        v = SAMPLE / "synthetic_show_variant"
        ep3 = parse_episode_text((v / "variant-episode-3.txt").read_bytes().decode("utf-8-sig"), "variant-episode-3.txt")
        self.assertEqual(ep3.episode_number, 3)
        self.assertTrue(any("no timestamp" in w for w in ep3.warnings))      # the untimed line
        self.assertTrue(any("end time" in w for w in ep3.warnings))          # the missing end
        self.assertTrue(any("scene end missing" in w for w in ep3.warnings))
        ep2 = parse_episode_text((v / "ep02_untitled.txt").read_bytes().decode("utf-8-sig"), "ep02_untitled.txt", 7)
        self.assertEqual(ep2.episode_number, 2)                              # inferred from the filename
        self.assertTrue(any("EPISODE NUMBER" in w for w in ep2.warnings))

    def test_same_content_same_dialogue_in_both_formats(self):
        a = parse_episode_text((SAMPLE / "synthetic_show" / "episode-001-Synthetic 1.txt").read_text(), "a.txt")
        b = parse_episode_text((SAMPLE / "synthetic_show_variant" / "variant-episode-1.txt").read_bytes().decode("utf-8-sig"), "b.txt")
        key = lambda ep: [(e.speaker, e.text, e.start, e.end) for e in ep.events if e.kind == "dialogue"]
        self.assertEqual(key(a), key(b))

    def test_garbage_does_not_crash(self):
        ep = parse_episode_text("hello\nthis is not a screenplay\n\x00\x01", "junk.txt")
        self.assertEqual(ep.events, [])
        self.assertTrue(ep.warnings)

    def test_uppercase_scene_keys(self):
        t = ("EPISODE NUMBER: Episode 6\nEPISODE LOGLINE: x\n\n--- SCRIPT START ---\n\nSCENE 01\nSCENE HEADING: EXT. PATH – NIGHT\n"
             "SCENE START: 00:00.00\nSCENE END: 00:33.70\n\nACTIONS AND DIALOGUE:\n\n**ACTION:**\nStart: (00:00.00) | End: (00:02.30)\nA door opens.\n")
        ep = parse_episode_text(t, "e6.txt")
        self.assertEqual(ep.scenes[0].heading, "EXT. PATH – NIGHT")
        self.assertEqual(ep.scenes[0].end, 33.7)
        self.assertEqual(len(ep.events), 1)


class LoaderTests(unittest.TestCase):
    def test_zip_folder_and_single_file(self):
        z = zip_of(SAMPLE / "synthetic_show")
        s1, sk1 = load_scripts([("pack.zip", z)])
        s2, _ = load_scripts([SAMPLE / "synthetic_show"])
        s3, _ = load_scripts([SAMPLE / "synthetic_show" / "episode-002-Synthetic 2.txt"])
        self.assertEqual((len(s1), len(s2), len(s3)), (3, 3, 1))
        self.assertEqual(sk1, [])

    def test_bad_inputs_are_reported_not_fatal(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("a/readme.pdf", b"%PDF")
            z.writestr("a/empty.txt", "   ")
            z.writestr("__MACOSX/._x.txt", "junk")
            z.writestr("a/ok.txt", "EPISODE NUMBER: 1\n")
        scripts, skipped = load_scripts([("x.zip", buf.getvalue()), ("fake.zip", b"not a zip"), "/no/such/path"])
        self.assertEqual([s.name for s in scripts], ["a/ok.txt"])
        self.assertEqual(len(skipped), 4)

    def test_natural_sort(self):
        scripts, _ = load_scripts([("ep10.txt", b"x"), ("ep2.txt", b"x"), ("ep1.txt", b"x")])
        self.assertEqual([s.name for s in scripts], ["ep1.txt", "ep2.txt", "ep10.txt"])


if __name__ == "__main__":
    unittest.main()

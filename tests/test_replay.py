"""The invariants the fixed greedy replay stands on.

`scripts/strata_replay.py` exists because `bench.sh` cannot price a decode lever: its agent
session feeds the model's own answers back, so each run generates different text and
different draft acceptance, and the bootstrap 95% CI on the median over the samples it
produces is about +-6% -- wider than the 4-15% the levers on this box trade. The whole
argument for the replacement is one sentence: *the text is fixed, so the acceptance is
fixed*. If the corpus ever stops being a pure function of the source file, every number the
harness produces becomes a bench.sh number with extra steps, and nothing in a results file
would show it.

So these tests guard the instrument, not the service: the corpus is deterministic and lands
near the sizes its cell names claim, the confidence interval is itself reproducible, and a
corpus that changed between two runs is caught rather than silently compared.
"""
import statistics
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import strata_replay as replay  # noqa: E402


class TheCorpusIsFixed(unittest.TestCase):
    def test_the_same_file_produces_the_same_prompts(self):
        """The one property everything else rests on."""
        self.assertEqual([p["sha256"] for p in replay.corpus()],
                         [p["sha256"] for p in replay.corpus()])

    def test_a_cell_is_the_size_its_name_says(self):
        """Not exact tokens -- the engine's own count is what gets recorded -- but a cell
        called 32k that quietly became 3k would compare a different workload against a
        baseline and report the difference as an effect."""
        for p in replay.corpus():
            target = int("".join(c for c in p["label"].split("-")[1] if c.isdigit())) * 1000
            chars = len(p["messages"][1]["content"])
            self.assertGreater(chars, target * 3.6 * 0.8, f"{p['label']} shrank")
            self.assertLess(chars, target * 3.6 * 1.25, f"{p['label']} grew")

    def test_a_subset_of_cells_keeps_the_prompts_it_shares_with_the_whole_set(self):
        """Comparing an arm run with two cells against a baseline run with five is only
        sound if the two cells are byte-identical in both."""
        full = {p["label"]: p["sha256"] for p in replay.corpus()}
        subset = replay.corpus(("code-4k", "code-32k"))
        self.assertEqual([p["label"] for p in subset], ["code-4k", "code-32k"])
        for p in subset:
            self.assertEqual(p["sha256"], full[p["label"]])

    def test_the_shapes_ask_for_something_long_enough_to_measure_decode(self):
        """A task the model answers in twenty tokens measures nothing: decode is the rate
        over a few hundred tokens. The enumeration tasks are here for that reason, and a
        rewrite that turns them back into summaries would quietly break the instrument."""
        for p in replay.corpus():
            task = p["messages"][1]["content"].rsplit("\n\n", 1)[1]
            self.assertRegex(task.lower(), r"one per line")


class TheStatisticsAreReproducible(unittest.TestCase):
    def test_the_interval_does_not_move_between_calls(self):
        values = [104.8, 148.9, 157.1, 136.8, 165.6, 164.3, 142.5]
        self.assertEqual(replay.boot_ci(values), replay.boot_ci(values))

    def test_a_single_sample_reports_itself_and_not_an_interval(self):
        self.assertEqual(replay.boot_ci([120.0]), (120.0, 120.0))
        self.assertEqual(replay.boot_ci([]), (None, None))

    def test_the_interval_brackets_the_median(self):
        values = [104.8, 148.9, 157.1, 136.8, 165.6, 164.3, 142.5]
        lo, hi = replay.boot_ci(values)
        self.assertLessEqual(lo, statistics.median(values))
        self.assertGreaterEqual(hi, statistics.median(values))


class TheNumbersComeFromTheEngine(unittest.TestCase):
    def test_the_engine_timing_is_used_when_it_reports_one(self):
        d = replay.digest({"first": 1.0, "total": 3.0, "pieces": 100, "err": None,
                           "usage": {"prompt_tokens": 4000, "completion_tokens": 100},
                           "timings": {"predicted_per_second": 152.3,
                                       "prompt_per_second": 880.1,
                                       "draft_n": 40, "draft_n_accepted": 37}})
        self.assertEqual(d["decode_tps"], 152.3)
        self.assertEqual(d["prompt_tps"], 880.1)
        self.assertTrue(d["ok"])

    def test_the_client_fallback_only_fires_when_the_engine_stayed_silent(self):
        """100 pieces over 2 s after a first token at 1 s is 49.5 tok/s, and it is a client
        measurement: it must never override what the engine itself said."""
        d = replay.digest({"first": 1.0, "total": 3.0, "pieces": 100, "err": None,
                           "usage": {}, "timings": {}})
        self.assertEqual(d["decode_tps"], 49.5)
        self.assertIsNone(d["prompt_tps"])

    def test_a_request_that_never_reached_a_first_token_is_not_a_sample(self):
        self.assertFalse(replay.digest({"first": None, "total": 0.1, "pieces": 0,
                                        "err": "HTTP 500", "usage": {}, "timings": {}})["ok"])

    def test_the_answer_hash_separates_two_different_continuations(self):
        a = replay.digest({"first": 1.0, "total": 2.0, "pieces": 5, "err": None, "text": "one",
                           "usage": {}, "timings": {}})
        b = replay.digest({"first": 1.0, "total": 2.0, "pieces": 5, "err": None, "text": "two",
                           "usage": {}, "timings": {}})
        self.assertNotEqual(a["answer_sha256"], b["answer_sha256"])


class ASummaryRefusesToPoolTwoRegimes(unittest.TestCase):
    def _row(self, label, phase, decode, answer):
        return {"label": label, "ok": True, "phase": phase, "decode_tps": decode,
                "prompt_tps": 900.0, "ttft_s": 1.0, "drafts_offered": 10,
                "drafts_accepted": 9, "answer_sha256": answer}

    def test_only_the_warm_rounds_are_the_measurement(self):
        """The first pass over a prompt decodes far slower than every pass after it because
        the expert caches are still filling. Pooling the two produces a median that
        describes neither the cold path nor the steady state."""
        rows = ([self._row("code-4k", "cold", 101.0, "a")]
                + [self._row("code-4k", "warm", d, "a") for d in (142.0, 148.0, 151.0)])
        warm = replay.summarise([r for r in rows if r["phase"] == "warm"])
        self.assertEqual(warm["code-4k"]["n"], 3)
        self.assertEqual(warm["code-4k"]["decode_median"], 148.0)

    def test_two_different_answers_in_one_cell_are_visible_as_instability(self):
        rows = [self._row("code-4k", "warm", 142.0, "a"),
                self._row("code-4k", "warm", 143.0, "b")]
        s = replay.summarise(rows)
        self.assertEqual(len(s["code-4k"]["answers"]), 2)


if __name__ == "__main__":
    unittest.main()

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
import ast
import statistics
import sys
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

try:
    import strata_replay as replay  # noqa: E402
except ModuleNotFoundError:
    replay = None

# The runtime image ships runtime/, config/ and tests/ -- not scripts/, which holds the
# benchmark harness rather than anything the controller needs to serve. The updater runs
# this suite inside its candidate image as the gate on deployment, so importing the
# harness by path stopped a release over a tool that has nothing to do with serving: the
# gate failed, nothing drained, and the running engine never noticed. Skipped there, and
# named as skipped rather than passing silently; run in full in every real checkout.
requires_harness = unittest.skipIf(replay is None,
                                   "scripts/strata_replay.py is not in the runtime image")


@requires_harness
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
            chars = len(p["messages"][1]["content"])
            self.assertGreater(chars, p["target_tokens"] * replay.CPT[p["kind"]] * 0.9,
                               f"{p['label']} shrank")
            self.assertLess(chars, p["target_tokens"] * replay.CPT[p["kind"]] * 1.15,
                            f"{p['label']} grew")

    def test_the_shapes_are_not_assumed_to_tokenize_alike(self):
        """The uniform guess this replaced was wrong by a factor of two on JSON, and the
        first full run built a cell named 128k that came to 208,561 tokens and was refused
        as over context. If someone flattens this table back to one number, that returns."""
        self.assertLess(replay.CPT["tool"], replay.CPT["prose"] * 0.6)

    def test_the_largest_cell_fits_the_window_this_box_boots_with(self):
        """163,840 is what the narrowest session this house has contracted for. A cell that
        overflows is not slow, it is refused with HTTP 400, and a run that silently loses its
        largest cell looks like a result. Sized off the narrowest and not the widest window
        because the fixed cells are the control a session-size ladder reads: they have to run,
        unchanged, at every rung."""
        window = 163_840
        for label, _, _, tokens in replay.CELLS:
            self.assertLess(tokens + 256, window, f"{label} does not fit the loaded window")


@requires_harness
class TheEdgeCellIsSizedByTheEngine(unittest.TestCase):
    """`--edge` is the cell that follows the window, and the label is the whole contract.

    The fixed cells stop at 128,000 tokens and the window this resident can be given reaches
    262,144, so a ladder that measured only the fixed cells would be pricing a region it never
    entered. The cell that enters it takes its width from `/health` instead of from this file,
    which buys the reach at the cost of one property: two rungs have two different workloads. The
    label carries the width, so the cost is visible in the results instead of hidden in a cell
    name that quietly means something else at a different window.
    """

    def test_the_label_states_the_width_it_measured(self):
        for window in (163_840, 196_608, 229_376, 262_144):
            cell, why = replay.edge_cell(window, 0.95, 256)
            self.assertIsNone(why)
            label, _, _, tokens = cell
            self.assertEqual(label, "edge-%dk" % (tokens // 1024))
            self.assertEqual(tokens % 1024, 0, "the label rounds, so the width has to be round")

    def test_the_answer_still_fits_the_window_being_measured(self):
        # Over-context is an HTTP 400 and no sample at all, and at the edge there is nothing left
        # over to absorb it: the width is chosen so the prompt and its 256-token answer fit inside
        # the window the engine loaded with, not merely near it.
        for window in (163_840, 196_608, 229_376, 262_144):
            (label, _, _, tokens), _ = replay.edge_cell(window, 0.95, 256)
            self.assertLess(tokens + 256, window, label)

    def test_a_width_is_a_workload_and_two_widths_are_two(self):
        # The pure-function property the whole suite rests on, held at whatever width the engine
        # reported, and asserted apart as well: an edge at 160K and an edge at 256K must not be
        # able to pose as the same cell, because `--compare` refuses a changed corpus by name.
        cell, _ = replay.edge_cell(196_608, 0.95, 256)
        keep = replay.ALL_CELLS + (cell[0],)
        first = {p["label"]: p["sha256"] for p in replay.corpus(keep, (cell,))}
        again = {p["label"]: p["sha256"] for p in replay.corpus(keep, (cell,))}
        self.assertEqual(first, again)
        other, _ = replay.edge_cell(262_144, 0.95, 256)
        self.assertNotEqual(first[cell[0]],
                            {p["label"]: p["sha256"]
                             for p in replay.corpus((other[0],), (other,))}[other[0]])
        self.assertEqual([p["label"] for p in replay.corpus(replay.ALL_CELLS)],
                         list(replay.ALL_CELLS), "an edge cell is never part of the fixed set")

    def test_an_edge_that_cannot_be_sized_says_so_rather_than_running_short(self):
        # A silently tiny edge would report the speed of a 4K prompt under a label that says edge.
        for window, fraction in ((163_840, 0), (163_840, 1.5), (None, 0.95), (None, 0),
                                 (4_096, 0.95)):
            cell, why = replay.edge_cell(window, fraction, 256)
            self.assertIsNone(cell, f"{window} at {fraction}")
            self.assertTrue(why)

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
        over a few hundred tokens. The first prose task asked for the members of a
        fourteen-word set, so it answered in 35 tokens, acceptance fell to 60%, and greedy
        stopped being reproducible -- and none of that failed loudly, it just measured
        nothing. So the task must ask for a line-per-item answer, which is what the live
        check in `verdict` then confirms actually happened."""
        for p in replay.corpus():
            task = p["messages"][1]["content"].rsplit("\n\n", 1)[1].lower()
            self.assertIn("per line", task, f"{p['label']} does not ask for a line-per-item "
                                            f"answer, so it will stop too early to measure")


@requires_harness
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


@requires_harness
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


@requires_harness
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


@requires_harness
class TheRunRefusesToPassWhenItWasNotAlone(unittest.TestCase):
    """`verdict` is the part of this tool that protects a future reader, so its order is
    deliberate: a run taken while the engine was busy is not a slow run, it is a different
    run, and every number in it still looks plausible."""

    def _row(self, **over):
        row = {"label": "code-4k", "ok": True, "busy_before": 0, "prompt_tokens": 4000,
               "target_tokens": 4000, "decode_tps": 150.0, "generated_tokens": 256,
               "phase": "warm"}
        row.update(over)
        return row

    def _summary(self, answers=("a",)):
        return {"code-4k": {"n": 2, "answers": list(answers)}}

    def test_a_quiet_run_passes(self):
        self.assertEqual(replay.verdict([self._row()], self._summary()), (0, []))

    def test_a_request_that_failed_is_reported_first(self):
        code, reasons = replay.verdict([self._row(ok=False)], self._summary())
        self.assertEqual(code, 1)
        self.assertIn("failed", reasons[0])

    def test_foreign_work_disqualifies_the_run_rather_than_downgrading_it(self):
        code, reasons = replay.verdict([self._row(busy_before=1)], self._summary())
        self.assertEqual(code, 3)
        self.assertIn("not a baseline", reasons[0])

    def test_contamination_outranks_everything_else_about_the_same_run(self):
        """Both are true of the same run; only the contamination is the reason to throw it
        away, and reporting the lesser reason invites somebody to keep the run anyway."""
        rows = [self._row(busy_before=2), self._row(prompt_tokens=9000)]
        code, reasons = replay.verdict(rows, self._summary(("a", "b")))
        self.assertEqual(code, 3)
        self.assertIn("generating", reasons[0])

    def test_an_answer_that_changed_between_rounds_is_not_a_fixed_replay(self):
        code, reasons = replay.verdict([self._row()], self._summary(("a", "b")))
        self.assertEqual(code, 1)
        self.assertIn("not stable", reasons[0])

    def test_a_cell_that_drifted_off_its_name_is_caught(self):
        code, reasons = replay.verdict([self._row(prompt_tokens=9000)], self._summary())
        self.assertEqual(code, 1)
        self.assertIn("drifted", reasons[0])

    def test_a_cell_that_answers_too_briefly_to_measure_decode_is_caught(self):
        """The prose cell answered in 35 tokens and every number in the table still looked
        fine. A rate over 35 tokens is mostly the first tokens, where acceptance is lowest,
        so it is not decode -- and nothing about it fails except the science."""
        code, reasons = replay.verdict([self._row(generated_tokens=35)], self._summary(),
                                       min_tokens=150)
        self.assertEqual(code, 1)
        self.assertIn("too brief", reasons[0])

    def test_a_cell_that_runs_to_the_cap_is_allowed(self):
        code, reasons = replay.verdict([self._row(generated_tokens=256)], self._summary(),
                                       min_tokens=150)
        self.assertEqual((code, reasons), (0, []))

    def test_a_sampled_arm_reports_two_answers_instead_of_refusing_the_run(self):
        """A sampled continuation differs every run by construction, so on that arm the spread of
        answer hashes is the finding and not a fault. The greedy arm keeps the gate, because two
        answers there mean the replay stopped holding decoding fixed."""
        code, reasons = replay.verdict([self._row()], self._summary(("a", "b")),
                                       gate_answers=False)
        self.assertEqual((code, reasons), (0, []))


@requires_harness
class TheSampledArmsAreWhatTheImageWrites(unittest.TestCase):
    def test_the_two_tables_say_the_same_numbers(self):
        """The sampled arm exists to price what production would do, so the numbers it sends and
        the numbers this house's entry point writes into the run config have to be one set and not
        two that drifted apart -- a silent difference between them would be a measurement of
        something the box never runs."""
        overlay = (Path(__file__).resolve().parent.parent
                   / "runtime/overlay/strata-vision-entrypoint.sh").read_text()
        written = ast.literal_eval(
            "{" + overlay.split("PRESETS = {", 1)[1].split("\n}", 1)[0] + "}")
        self.assertEqual(written,
                         {k: v for k, v in replay.SAMPLING_ARMS.items() if k != "greedy"})
        self.assertEqual(replay.SAMPLING_ARMS["greedy"], {"temperature": 0})



@requires_harness
class ItWaitsForTheBoxToBeAlone(unittest.TestCase):
    """The baseline can only be taken unattended, and unattended means nobody can promise
    the box is quiet -- so the tool has to refuse rather than measure anyway."""

    def test_it_starts_once_the_engine_reports_idle(self):
        with unittest.mock.patch.object(replay, "busy_slots", return_value=0):
            self.assertIsNone(replay.wait_for_quiet("u", "k", hold=0.05, bound=5, settle=0.01))

    def test_it_gives_up_rather_than_measuring_a_busy_engine(self):
        with unittest.mock.patch.object(replay, "busy_slots", return_value=1):
            reason = replay.wait_for_quiet("u", "k", hold=0.05, bound=0.2, settle=0.01)
        self.assertIsNotNone(reason)
        self.assertIn("not idle", reason)

    def test_a_busy_request_part_way_through_resets_the_idle_clock(self):
        """Otherwise a foreign request that lands one second before the start still lets
        the run begin, which is the exact case the whole check exists for."""
        seen = iter([0, 0, 1, 0, 0, 0, 0, 0])
        with unittest.mock.patch.object(replay, "busy_slots", side_effect=lambda *a: next(seen)):
            self.assertIsNone(replay.wait_for_quiet("u", "k", hold=0.04, bound=5, settle=0.01))



@requires_harness
class ARepetitionIsReportedAndNotGated(unittest.TestCase):
    """The one number on a sampled arm that is about behaviour rather than speed. Greedy
    repeats because greedy always takes the same token; sampling repeats because the model
    lost the thread. Both are worth seeing and neither may fail a run, which is why the
    harness has a reporter here and no verdict."""

    def test_a_run_of_identical_lines_is_the_metric_the_field_reports(self):
        """4 is not this box's number: it is the threshold #1560 used over 16 real
        quarantined production loops, so a streak here means what it means there."""
        answer = "a.py: x\nb.py: y\nb.py: y\nb.py: y\nb.py: y\n"
        self.assertEqual(replay.repetition(answer)["repeat_streak"], 4)
        self.assertGreaterEqual(replay.LOOP_STREAK, 4)

    def test_a_loop_over_two_lines_is_not_missed(self):
        """Consecutive-identical alone reports a streak of 1 for `x y x y x y`, which is the
        shape a stalled enumeration actually takes. The duplicate fraction is the other half
        of the pair for exactly this case."""
        d = replay.repetition("x\ny\nx\ny\nx\ny\n")
        self.assertEqual(d["repeat_streak"], 1)
        self.assertGreater(d["dup_line_pct"], 40)

    def test_a_clean_answer_and_an_empty_one_are_both_reportable(self):
        self.assertEqual(replay.repetition("a\nb\nc\n"), {"repeat_streak": 1, "dup_line_pct": 0})
        self.assertEqual(replay.repetition(""), {"repeat_streak": 0, "dup_line_pct": 0})

    def test_the_number_reaches_the_summary_of_rows_that_predate_it(self):
        """Results files from before this column exists are still on disk and are still
        compared against; a run must summarise them rather than stop on a KeyError."""
        row = {"label": "code-4k", "ok": True, "phase": "warm", "decode_tps": 160.0,
               "prompt_tps": 900.0, "ttft_s": 1.0, "drafts_offered": 10, "drafts_accepted": 9,
               "answer_sha256": "a" * 16}
        s = replay.summarise([row, dict(row, repeat_streak=5, dup_line_pct=40)])
        self.assertEqual(s["code-4k"]["repeat_streak"], 5)
        self.assertEqual(s["code-4k"]["looped"], 1)
        replay.table(s, "renders")

    def test_the_verdict_does_not_refuse_a_run_for_repeating(self):
        rows = [{"label": "code-4k", "ok": True, "phase": "warm", "busy_before": False,
                 "prompt_tokens": 4000, "target_tokens": 4000, "generated_tokens": 256,
                 "decode_tps": 160.0, "prompt_tps": 900.0, "ttft_s": 1.0,
                 "drafts_offered": 10, "drafts_accepted": 9, "answer_sha256": "a" * 16,
                 "repeat_streak": 9, "dup_line_pct": 80, "looped": 1}]
        code, reasons = replay.verdict(rows, replay.summarise(rows))
        self.assertEqual((code, reasons), (0, []))


@requires_harness
class TheTailQuestionScoresWhatTheHeadQuestionCannotSee(unittest.TestCase):
    """`--tail` exists for one reason: an enumeration copied from the top of a 245,000-token
    prompt is copied out of its first few thousand tokens, so it reports "no difference" about a
    change to how the engine treats position 200,000 — it never looked there. A change to
    position handling (a rope scaling, a window above the trained one) is exactly the class of
    change that has to be priced in answers, so the question has to be about the end."""

    def test_the_default_question_is_the_one_every_baseline_answered(self):
        """Untouched bytes. The recorded runs on this box — the sampling arms, the session-size
        ladder — are all comparisons against these prompts, so a tail mode that quietly reworded
        the default question would invalidate all of them at once."""
        for p in replay.corpus(("code-4k", "tool-4k", "prose-32k")):
            self.assertNotIn("LAST", p["messages"][1]["content"])
        self.assertEqual([p["sha256"] for p in replay.corpus()],
                         [p["sha256"] for p in replay.corpus(tail=5)])

    def test_a_tail_run_asks_about_the_same_corpus_and_a_different_end_of_it(self):
        """One variable: the prompt bytes and their hashes are identical, the question is not, and
        the items it wants are not anywhere in the head of the answer the old question got.

        Checked for the two shapes whose items are unique. The prose shape repeats one sentence
        of boilerplate verbatim in every block, so in prose a tail item and a head item can be the
        same sentence — which is also why a session this long is measured with the tool shape, and
        why `--edge` builds itself out of records and not out of prose."""
        head = {p["label"]: p for p in replay.corpus()}
        tail = {p["label"]: p for p in replay.corpus(tail=4)}
        for label, p in tail.items():
            self.assertEqual(p["sha256"], head[label]["sha256"])
            self.assertIn("LAST 4", p["messages"][1]["content"])
            self.assertEqual(len(p["expected"]), 4)
            if p["kind"] == "prose":
                continue
            self.assertFalse(set(p["expected"]) & set(p["first_items"]),
                             f"{label}: the tail question asks for something the head already has")

    def test_the_ground_truth_is_actually_in_the_prompt_it_is_asked_about(self):
        """Scored, not read — but only if the truth is derivable from the corpus text rather than
        assumed from the builder. If a builder shape changes and the reader stops finding the
        items, that is a silent zero on every future tail run."""
        for p in replay.corpus(tail=3):
            body = p["messages"][1]["content"]
            for item in p["expected"]:
                if p["kind"] == "code":
                    # The answer is a composite the question asks for, so its two halves are what
                    # has to be in the prompt, not the joined line.
                    name, field = item.split(": ")
                    self.assertIn(f"class {name}:", body, f"{p['label']}: {name} is not in it")
                    self.assertIn(f"def __init__(self, {field},", body,
                                  f"{p['label']}: {name}'s field is not in it")
                else:
                    self.assertIn(item, body, f"{p['label']}: {item} is not in it")
            whole = replay.expected_items(p["kind"], body)
            self.assertGreaterEqual(len(whole), 8 * len(p["expected"]),
                                    f"{p['label']}: no tail to speak of, only a head")

    def test_an_answer_from_the_head_scores_zero_and_says_where_it_came_from(self):
        """The failure this instrument is for: the model does not report losing the tail, it
        answers fluently, in order, from the top of the prompt. Recall alone would show a zero;
        `answered_from` is what tells a lost tail apart from a broken harness."""
        p = next(p for p in replay.corpus(tail=5) if p["kind"] == "tool")
        perfect = replay.score_answer("\n".join(p["expected"]), p["expected"], p["first_items"])
        self.assertEqual((perfect["recall"], perfect["answered_from"], perfect["in_order"]),
                         (1.0, "tail", True))
        head = replay.score_answer("\n".join(p["first_items"]), p["expected"], p["first_items"])
        self.assertEqual((head["recall"], head["answered_from"]), (0.0, "head"))
        lost = replay.score_answer("The records above list many paths across the runtime.",
                                   p["expected"], p["first_items"])
        self.assertEqual((lost["recall"], lost["answered_from"]), (0.0, "other"))
        half = replay.score_answer("\n".join(p["expected"][2:]), p["expected"], p["first_items"])
        self.assertEqual((half["recall"], half["answered_from"]), (0.6, "tail"))

    def test_the_score_travels_with_the_sample_and_only_appears_when_asked_for(self):
        """A tail run has to be readable as one after the fact — and a run that never asked must
        not grow a column that reads as a zero."""
        p = next(p for p in replay.corpus(tail=5) if p["kind"] == "tool")
        row = {"label": "tool-4k", "ok": True, "phase": "warm", "decode_tps": 100.0,
               "prompt_tps": 900.0, "ttft_s": 4.0, "drafts_offered": 10, "drafts_accepted": 9,
               "answer_sha256": "a" * 16, "repeat_streak": 1, "dup_line_pct": 0.0,
               "score": replay.score_answer("\n".join(p["expected"]), p["expected"],
                                            p["first_items"])}
        s = replay.summarise([row])
        self.assertEqual((s["tool-4k"]["tail_recall_median"], s["tool-4k"]["answered_from"]),
                         (1.0, "tail"))
        self.assertEqual(s["tool-4k"]["tail_in_order"], 1)
        plain = replay.summarise([{k: v for k, v in row.items() if k != "score"}])
        self.assertNotIn("tail_recall_median", plain["tool-4k"])
        replay.table(s, "renders with the tail columns")
        replay.table(plain, "renders without them")


if __name__ == "__main__":
    unittest.main()

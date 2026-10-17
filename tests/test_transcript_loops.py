"""The invariants the transcript loop scan stands on.

`scripts/transcript_loops.py` is the observation half of the sampling decision: the replay
harness measures decode against a synthetic corpus, and the question "did the greedy profile
actually make this box's agents loop?" can only be answered from the session store, where the
calls the agents really made are recorded in order. It is read-only, so nothing here protects
a service. What it protects is the *reading*, because a loop count is the kind of number that
is easy to compute wrong and impossible to glance at and distrust.

Three ways to get it wrong, all of them silent:

* Comparing arguments as text. The same call written with its keys in another order is the
  same call, and a streak count that missed it would be a count of formatting.
* Counting a poll as a loop. The longest runs of identical calls in this house's transcripts
  are `job_output(wait=true)`, which is an agent waiting, not an agent looping; only a run
  that also answers identically is a stall.
* Filing a session under no model. Most sessions never write a selection record -- they ran
  the deployment default -- so a filter that treats them as unknown silently drops the
  majority of the traffic it is meant to be measuring.
"""
import io
import json
import sys
import types
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

try:
    import transcript_loops as loops  # noqa: E402
except ModuleNotFoundError:
    loops = None

requires_tool = unittest.skipIf(loops is None, "scripts/transcript_loops.py is missing")


def call(name, arguments, turn=1, result="r", time=1_700_000_000_000, _n=[0]):
    """One tool call and its result, as the session store writes the pair. The callId is
    unique per pair because in the real store it is: two identical calls are two ids, and
    that is what lets a result be matched to the call that asked for it."""
    _n[0] += 1
    cid = f"c{_n[0]}"
    return [{"type": "tool/call", "time": time,
             "data": {"turn": turn, "step": 1, "callId": cid, "name": name,
                      "arguments": json.dumps(arguments)}},
            {"type": "tool/result", "time": time,
             "data": {"turn": turn, "toolCallId": cid,
                      "message": {"role": "tool", "content": [{"type": "text", "text": result}]}}}]


def transcript(*pairs, model=None):
    rs = [{"type": "session", "data": {}}]
    if model:
        rs.append({"type": "model/selection", "data": {"provider": "local-ollama", "model": model}})
    for i, pair in enumerate(pairs):
        rs.extend(pair if isinstance(pair, list) else [pair])
    return rs


@requires_tool
class IdenticalCallsAreIdentical(unittest.TestCase):
    def test_arguments_are_compared_as_data_not_as_text(self):
        """A stream that wrote the keys the other way round made the same call. Counting that
        as a different call is how a loop count quietly becomes a formatting count."""
        a = loops.canonical('{"b": 2, "a": 1}')
        b = loops.canonical({"a": 1, "b": 2})
        c = loops.canonical('{"a":1,"b":2}')
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertNotEqual(a, loops.canonical('{"a": 1, "b": 3}'))

    def test_arguments_that_are_not_json_are_taken_literally(self):
        """Some harness tools take a bare string. Dropping them would drop the call."""
        self.assertEqual(loops.canonical("not json at all"), "not json at all")

    def test_a_result_is_matched_to_the_call_that_asked_for_it(self):
        rs = transcript(call("bash", {"command": "x"}), call("bash", {"command": "x"}, result="y"))
        seq = loops.calls(rs)
        self.assertEqual(len(seq), 2)
        self.assertEqual([c["name"] for c in seq], ["bash", "bash"])
        self.assertNotEqual(seq[0]["result"], seq[1]["result"])

    def test_a_selection_record_wins_and_the_default_covers_the_rest(self):
        """169 of this store's sessions wrote no selection record and ran the deployment
        default. Treating them as unknown would have thrown away most of the traffic and
        reported the result as a clean bill of health."""
        self.assertEqual(loops.served_model(transcript(), "daytime"), "daytime")
        self.assertIsNone(loops.served_model(transcript(), ""))
        self.assertEqual(loops.served_model(transcript(model="nighttime"), "daytime"), "nighttime")
        switched = [{"type": "model/selection", "data": {"model": "daytime"}},
                    {"type": "model/selection", "data": {"model": "nighttime"}}]
        self.assertEqual(loops.served_model(switched, "daytime"), "nighttime")


@requires_tool
class ALoopIsNotAPoll(unittest.TestCase):
    def test_the_field_definition_counts_identical_calls(self):
        seq = loops.calls(transcript(*[call("bash", {"command": "ls"}) for _ in range(4)]))
        s = loops.streaks(seq)
        self.assertEqual(s["max_streak"], 4)
        self.assertTrue(s["looped"])

    def test_waiting_for_a_job_is_not_a_stall(self):
        """`job_output(wait=true)` four times, each returning something new: the calls are
        identical because waiting is the operation. This is the top streak class on this box,
        so a metric that cannot see the difference reports waiting as looping."""
        seq = loops.calls(transcript(*[call("job_output", {"job_id": "j", "wait": True},
                                            result=f"still running {i}") for i in range(4)]))
        s = loops.streaks(seq)
        self.assertEqual(s["max_streak"], 4)
        self.assertEqual(s["max_stalled_streak"], 1)
        self.assertFalse(s["stalled"])

    def test_answering_itself_identically_is_a_stall(self):
        seq = loops.calls(transcript(*[call("bash", {"command": "ls"}, result="same")
                                       for _ in range(4)]))
        s = loops.streaks(seq)
        self.assertEqual(s["max_streak"], 4)
        self.assertGreaterEqual(s["max_stalled_streak"], 4)
        self.assertTrue(s["stalled"])

    def test_a_run_broken_by_one_different_call_is_not_one_run(self):
        seq = loops.calls(transcript(call("bash", {"command": "a"}), call("bash", {"command": "b"}),
                                     call("bash", {"command": "a"}), call("bash", {"command": "a"})))
        self.assertEqual(loops.streaks(seq)["max_streak"], 2)

    def test_the_same_call_in_another_turn_is_counted_and_said_so(self):
        across = loops.calls(transcript(call("read", {"file_path": "a"}, turn=1),
                                        call("read", {"file_path": "a"}, turn=2),
                                        call("read", {"file_path": "a"}, turn=3)))
        s = loops.streaks(across)
        self.assertEqual(s["max_streak"], 3)
        self.assertEqual(s["max_streak_within_turn"], 1)

    def test_an_empty_transcript_is_zero_and_not_an_error(self):
        s = loops.streaks([])
        self.assertEqual((s["calls"], s["max_streak"], s["looped"]), (0, 0, False))


@requires_tool
class TheDecoderFindsAWayOrSaysSo(unittest.TestCase):
    def test_frames_are_split_at_the_magic_not_at_the_first_frame(self):
        """What the Node route needs: one suffix per frame boundary, each of which its
        decoder reads as exactly one frame. A streaming decompressor handed the whole file
        stops at the end of the first frame and returns 217 bytes of a 349 KB session without
        raising, which is the failure this function exists to avoid."""
        blob = b"junk" + loops.MAGIC + b"\x00one" + loops.MAGIC + b"\x01two"
        self.assertEqual(loops.frames(blob), [loops.MAGIC + b"\x00one" + loops.MAGIC + b"\x01two",
                                              loops.MAGIC + b"\x01two"])

    def test_no_decoder_is_named_rather_than_reported_as_an_empty_session(self):
        """Every session would decode to nothing and the report would say no loops."""
        with unittest.mock.patch.dict(sys.modules, {"compression": None, "zstandard": None}), \
                unittest.mock.patch.object(loops.shutil, "which", return_value=None):
            with self.assertRaises(RuntimeError) as e:
                loops.decode(b"anything")
        self.assertIn("no zstd decoder", str(e.exception))

    def test_a_whole_stream_reader_is_preferred_over_the_frame_splitter(self):
        """Python's and the CLI's readers walk a concatenation of frames themselves; only
        Node needs the suffixes. Reversing the order would make every Python run depend on a
        binary that is not part of the language."""
        stub = types.SimpleNamespace(ZstdFile=lambda f, mode: io.BytesIO(b"WHOLE-STREAM"))
        with unittest.mock.patch.dict(sys.modules, {"compression": None, "zstandard": stub}):
            self.assertEqual(loops.decode(b"anything"), b"WHOLE-STREAM")

    def test_a_route_that_raises_falls_through_rather_than_failing_the_run(self):
        """Half-installed bindings are ordinary. What must not happen is the whole store
        reading as empty, which reports as a clean bill of health."""
        def boom(*a, **k):
            raise ValueError("half-installed binding")

        stub = types.SimpleNamespace(ZstdFile=boom)
        with unittest.mock.patch.dict(sys.modules, {"compression": None, "zstandard": stub}), \
                unittest.mock.patch.object(loops.shutil, "which",
                                           side_effect=lambda x: "/usr/bin/zstd" if x == "zstd" else None), \
                unittest.mock.patch.object(loops.subprocess, "run",
                                           return_value=types.SimpleNamespace(stdout=b"FROM-CLI")):
            self.assertEqual(loops.decode(b"anything"), b"FROM-CLI")


def message(text=None, reasoning=None, name=None, arguments=None, turn=1, step=1,
            time=1_700_000_000_000):
    """One assistant message as the harness records what the client was handed: reasoning,
    prose and calls are blocks in one content list, and only the prose is what the engine's
    parser served."""
    blocks = []
    if reasoning is not None:
        blocks.append({"type": "reasoning", "text": reasoning})
    if text is not None:
        blocks.append({"type": "text", "text": text})
    if name is not None:
        blocks.append({"type": "tool-call", "id": "c1", "name": name,
                       "arguments": json.dumps(arguments or {})})
    return {"type": "assistant/message", "time": time,
            "data": {"turn": turn, "step": step,
                     "message": {"role": "assistant", "content": blocks}}}


def failure(text, time=1_700_000_000_000):
    return {"type": "llm/retry", "time": time,
            "data": {"turn": 1, "step": 2, "failure": {"message": text, "code": "SERVER"}}}


DRIFT = "Let me look.\n<tool_call>search\n<parameter=query>strata sampling</parameter>"


@requires_tool
class ACallTheClientNeverGot(unittest.TestCase):
    """A dropped tool call leaves no call record, because it is the call that never became
    one. What it leaves is its markup in the prose, or the parser's own error -- and the one
    thing that is not evidence of it is an agent quoting the format while writing about it."""

    def test_markup_in_prose_with_no_call_is_a_dropped_call(self):
        d = loops.dropped(transcript(message(text=DRIFT)))
        self.assertEqual((d["messages"], d["with_call"], d["drops"], d["asks"]), (1, 0, 1, 1))
        self.assertIn("<parameter=", d["examples"][0]["text"])

    def test_a_quoted_format_is_an_agent_talking_about_the_parser(self):
        """Every session in which anybody explains this parser quotes it. Counting quotes as
        drops would report the documentation as an outage, and this house has more of the
        first than the second."""
        d = loops.dropped(transcript(
            message(text="The drift looks like this:\n```\n" + DRIFT + "\n```\nand is dropped.")))
        # Two pieces of markup in the fixture: the wrapper and the `<parameter=` opener.
        self.assertEqual((d["markup"], d["quoted"], d["drops"], d["asks"]), (2, 2, 0, 0))

    def test_an_inline_span_quotes_it_too(self):
        d = loops.dropped(transcript(message(text="The opener is `<parameter=NAME>`, not `<function=`.")))
        self.assertEqual(d["drops"], 0)
        self.assertGreaterEqual(d["quoted"], 2)

    def test_an_unterminated_fence_still_counts_as_a_quote(self):
        """A model that opens a fence and forgets to close it is quoting, not calling. The
        guessing runs toward keeping the markup out of the drop count."""
        d = loops.dropped(transcript(message(text="Watch:\n```" + DRIFT)))
        self.assertEqual(d["drops"], 0)

    def test_the_reasoning_channel_is_not_the_output(self):
        """An agent *thinking* about a `<parameter>` tag is the commonest source of the markup
        in this store, and the engine's tool-call parser never sees that channel."""
        d = loops.dropped(transcript(message(reasoning="It wrote <parameter=query> instead.",
                                             text="Done.")))
        self.assertEqual((d["markup"], d["drops"]), (0, 0))

    def test_a_call_that_arrived_beside_the_markup_is_not_a_drop(self):
        d = loops.dropped(transcript(message(text=DRIFT, name="search")))
        self.assertEqual((d["drops"], d["drops_beside_call"], d["with_call"]), (0, 1, 1))

    def test_two_drifted_calls_in_one_message_are_one_dropped_turn(self):
        """The rate is a rate over turns, so it is the message that counts, not the number of
        tags in it. Counting tags could put the figure over 100%."""
        d = loops.dropped(transcript(message(text=DRIFT + "\n" + DRIFT)))
        self.assertEqual((d["drops"], d["markup"]), (1, 4))

    def test_the_parsers_own_error_counts_as_a_lost_call(self):
        d = loops.dropped(transcript(
            message(text="working"),
            failure('local-ollama API error (500): malformed tool call: <function=bash<parameter=')))
        self.assertEqual(d["malformed"], 1)
        self.assertEqual(d["messages"], 1)

    def test_a_router_draining_is_not_a_parse_failure(self):
        """The commonest failure in this store is a configuration switch, and a metric that
        read it as a lost call would be tuned to the deploy rather than to the parser."""
        d = loops.dropped(transcript(
            message(text="working"),
            failure("The router is switching configuration (BACKEND_DRAINING); Harness waits "
                    "and retries Daytime without changing models.")))
        self.assertEqual(d["malformed"], 0)

    def test_an_ordinary_answer_is_not_counted_as_a_turn_that_wanted_a_tool(self):
        d = loops.dropped(transcript(message(text="The answer is 42."),
                                     message(text="Now the file.", name="read")))
        self.assertEqual((d["messages"], d["with_call"], d["asks"], d["drops"]), (2, 1, 1, 0))

    def test_a_transcript_with_no_messages_reports_zeros_rather_than_raising(self):
        d = loops.dropped(transcript(*[call("bash", {"command": "ls"})]))
        self.assertEqual((d["messages"], d["asks"], d["drops"], d["malformed"]), (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()

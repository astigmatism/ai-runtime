#!/usr/bin/env python3
"""How often this box's own agent sessions repeat a tool call, read off the transcripts.

The sampling question was settled upstream by argument and in the field by replay: greedy
produces repetition loops, the model card's sampling removes them (Niko1221/Strata#1560
reproduced 11 of 16 real quarantined loops under greedy and 0 of 16 at the card's set). What
neither has is this box, and the replay harness cannot supply it -- its corpus is synthetic
code, tool records and prose, and it measures decode. The evidence that a profile is *better
for the work* lives in the session store: every agent turn this house has ever run, with the
tool calls it actually made, in order, with timestamps.

So this is the observation half of the pair. It walks the DeepSeek Harness session store,
decodes each transcript, takes the tool calls in the order the agent made them, and reports
the longest run of identical consecutive calls -- identical meaning the same tool with the
same arguments, since a retry of a *different* command is an agent learning rather than an
agent looping. The threshold is #1560's: a streak of 4 is a loop, not persistence.

It is deliberately read-only and deliberately not a gate. Two sessions of the same shape can
differ because the tasks differ, and a session that ends after a repeated call has not looped
the way a stuck one has; this is for noticing that a change moved something, and for pointing
at the sessions to read. A streak count without the transcript open behind it is a number
about a story somebody guessed.

    python3 scripts/transcript_loops.py --flip 2026-10-17T05:32:00Z

`--dropped` answers the other question this box has about the engine, and it is answered
differently because the thing being counted is an absence. A dropped tool call is an assistant
turn that meant to call a tool and was served as prose, so there is no call record to count;
what is left behind is the call's *markup*, sitting in the message text the client was given.
Upstream's parser is where that markup is either honoured or left in the text (a body it cannot
take is content under `tool_call_recovery` and a `malformed tool call` request error without it,
`serve/frontend.py`), and the harness records whichever of those arrived without interpreting
them, so both traces are visible here and neither needs the engine to cooperate. Prose that
*quotes* the format -- the thing every agent writing about this parser does -- is excluded by
looking at whether the markup arrived inside a code fence or a backtick span.

    python3 scripts/transcript_loops.py --dropped --flip 2026-10-17T16:42:09Z

Session files are append-only zstd streams, one frame per flush, which is why decoding is a
small cascade rather than one call: Python 3.14's stdlib zstd, then the `zstandard` module,
then a `zstd` binary, then Node's built-in decoder -- in this house the last is the one that
is always there, because the harness that writes them runs on Node.

Stdlib only, for the same reason the replay harness is: it runs wherever a transcript is.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import hashlib
import io
import json
import os
import re
import shutil
import statistics
import subprocess
import sys

ROOT = os.path.expanduser("~/deployments/deepseek-harness/data/dsh/sessions")
FILE = "session.v4.jsonl.zstd"
MAGIC = b"\x28\xb5\x2f\xfd"
LOOP_STREAK = 4

# The engine's tool-call markup, as it survives in a message the parser served as prose.
CALL_MARKUP = (
    ("parameter", re.compile(r"<parameter[ =]")),
    ("function", re.compile(r"<function[ =]")),
    ("tool_call", re.compile(r"<tool_call[ >]")),
    ("tool_response", re.compile(r"<tool_response[ >]")),
)
# What the engine says when a call body reaches it, is entered as a call, and is not one:
# `parse_tool_call` raises and, without `tool_call_recovery`, that error ends the request.
MALFORMED = re.compile(r"malformed tool call")

# One frame per flush means a whole transcript is concatenated frames, and neither Python
# route streams across them. Scan for magic bytes instead: at 16 MiB the chance a compressed
# payload happens to contain one is about one in 260, and a false positive fails to
# decompress and is dropped.
_JS = """
const fs=require('fs'),z=require('zlib'),a=process.argv.length>1?process.argv[process.argv.length-1]:'-';
const b=a==='-'?fs.readFileSync(0):fs.readFileSync(a);
const M=Buffer.from([0x28,0xb5,0x2f,0xfd]);const out=[];let i=0;
for(;;){const a=b.indexOf(M,i);if(a<0)break;try{out.push(z.zstdDecompressSync(b.subarray(a)))}
catch(e){}i=a+4}
fs.writeSync(1,Buffer.concat(out));
"""


def frames(blob: bytes) -> list:
    """Every zstd frame boundary in a buffer, as the suffix starting at it, in order.

    Only the Node route needs this: its synchronous decoder stops at the end of the first
    frame and its stream decoder does the same, so a multi-frame transcript has to be handed
    over one suffix at a time and each suffix decodes to exactly its own frame. Python's and
    the CLI's whole-stream readers walk the concatenation themselves, which is why they get
    the blob.
    """
    out, at = [], 0
    while True:
        start = blob.find(MAGIC, at)
        if start < 0:
            return out
        out.append(blob[start:])
        at = start + len(MAGIC)


def decode_node(blob: bytes) -> bytes:
    proc = subprocess.run(["node", "-e", _JS, "-"], input=blob,
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=300)
    if proc.returncode:
        raise RuntimeError("node zstd decoder failed")
    return proc.stdout


def decode(blob: bytes) -> bytes:
    """Transcript bytes, by whatever zstd this machine happens to have.

    Whole-stream readers first, in the order they are likely to exist: Python 3.14's stdlib
    zstd, the `zstandard` binding, the `zstd` binary, and then Node, which is nearly always
    present in this house because the harness that writes these files runs on it. A route
    that raises part way falls through rather than failing the run, and a file that still
    cannot be read decodes to nothing and is counted as undecodable -- the one thing that
    must not happen is a store that looks loop-free because it could not be read.
    """
    try:                                            # Python 3.14+
        from compression import zstd

        return zstd.ZstdFile(io.BytesIO(blob), "r").read()
    except Exception:                               # noqa: BLE001
        pass
    try:                                            # the usual third-party binding
        import zstandard

        return zstandard.ZstdFile(io.BytesIO(blob), "r").read()
    except Exception:                               # noqa: BLE001
        pass
    if shutil.which("zstd"):                        # the CLI walks concatenated frames
        return subprocess.run(["zstd", "-dqc"], input=blob, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, check=True).stdout
    if shutil.which("node"):
        out = decode_node(blob)
        if out:
            return out
    raise RuntimeError("no zstd decoder: need Python 3.14, zstandard, zstd, or node")


def records(path: str) -> list:
    try:
        blob = open(path, "rb").read()
    except OSError:
        return []
    try:
        text = decode(blob)
    except Exception:                               # noqa: BLE001
        return []
    out = []
    for line in text.splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def canonical(arguments) -> str:
    """Arguments compared as data, because a stream that wrote `{"a": 1, "b": 2}` and one that
    wrote the same object with the keys in the other order is the same call."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return arguments.strip()
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"))


def result_sha(msg) -> str:
    """A digest of what a tool answered with, so a repeated call can be told apart from a
    repeated call that answered the same way."""
    if not isinstance(msg, dict):
        return ""
    return hashlib.sha256(json.dumps(msg.get("content"), sort_keys=True,
                                     default=str).encode()).hexdigest()[:16]


def calls(rs: list) -> list:
    """The tool calls of one transcript, in the order the agent made them, each tagged with
    the digest of its result. The result is what separates the two things that look the same
    in a call log: an agent polling a job that is moving, and an agent stuck."""
    answers = {}
    for r in rs:
        if r.get("type") != "tool/result":
            continue
        d = r.get("data") or {}
        cid = d.get("toolCallId") or (d.get("message") or {}).get("toolCallId")
        if cid:
            answers[cid] = result_sha(d.get("message"))
    out = []
    for r in rs:
        if r.get("type") != "tool/call":
            continue
        d = r.get("data") or {}
        if not d.get("name"):
            continue
        out.append({"name": d["name"], "arguments": canonical(d.get("arguments")),
                    "result": answers.get(d.get("callId"), ""),
                    "turn": d.get("turn"), "step": d.get("step"), "time": r.get("time")})
    return out


def code_spans(text: str) -> list:
    """The ranges of a message that are code: ``` fences and `inline` spans.

    An agent that writes about this parser quotes it, and a quote is not a dropped call. The
    fence test is deliberately dumb -- pair up the delimiters, take what is between them --
    because the alternative is a real markdown parser guessing at a model's unbalanced fence.
    An unterminated fence runs to the end of the message, which is the direction that keeps a
    quote out of the count.
    """
    spans = []
    pos = 0
    while True:
        start = text.find("```", pos)
        if start < 0:
            break
        end = text.find("```", start + 3)
        if end < 0:
            spans.append((start, len(text)))
            break
        spans.append((start, end + 3))
        pos = end + 3
    for m in re.finditer(r"`[^`\n]+`", text):
        spans.append((m.start(), m.end()))
    return spans


def call_forms(text: str) -> list:
    """Every place a message carries the engine's tool-call markup, marked quoted or not."""
    spans = code_spans(text or "")
    found = []
    for name, pattern in CALL_MARKUP:
        for m in pattern.finditer(text or ""):
            found.append({"form": name, "at": m.start(),
                          "quoted": any(a <= m.start() < b for a, b in spans),
                          "text": (text or "")[max(0, m.start() - 40):m.start() + 120]})
    return found


def messages(rs: list) -> list:
    """The assistant messages of one transcript, as the client saw them.

    Only the `text` blocks are read for markup. `reasoning` is a different channel that the
    engine's tool-call parser never sees, and an agent thinking out loud about a `<parameter>`
    tag is the single most common source of the markup in this house's transcripts.
    """
    out = []
    for r in rs:
        if r.get("type") != "assistant/message":
            continue
        d = r.get("data") or {}
        message = d.get("message") or {}
        blocks = message.get("content") or []
        prose = "".join(b.get("text", "") for b in blocks
                        if isinstance(b, dict) and b.get("type") == "text")
        out.append({"turn": d.get("turn"), "step": d.get("step"), "time": r.get("time"),
                    "has_call": any(isinstance(b, dict) and b.get("type") == "tool-call"
                                    for b in blocks),
                    "forms": call_forms(prose)})
    return out


def dropped(rs: list, top: int = 8) -> dict:
    """What a transcript would have to say if the engine had dropped one of its tool calls.

    There is no call record to count, because a dropped call is precisely the call that never
    became one, so this counts the two things a drop cannot help leaving behind. One: the
    call's markup in the prose the client was handed, outside any code span -- which is what
    the parser does with a wrapper it will not take (`serve/frontend.py`: the text it is).
    Two: a step that ended with the parser's own words, `malformed tool call`, which is what
    the same body does when it is entered as a call and still is not one, and which
    `tool_call_recovery` turns into the first case instead of an error.

    `asks` is the denominator a rate wants: turns that wanted a tool, counted as the messages
    that got a call plus the messages that carried markup instead of one.
    """
    msgs = messages(rs)
    markup = [f for m in msgs for f in m["forms"]]
    drops = [m for m in msgs if not m["has_call"]
             and any(not f["quoted"] for f in m["forms"])]
    beside = [m for m in msgs if m["has_call"]
              and any(not f["quoted"] for f in m["forms"])]
    malformed = []
    for r in rs:
        if r.get("type") not in ("llm/retry", "llm/retry-started"):
            continue
        failure = (r.get("data") or {}).get("failure") or {}
        text = " ".join(str(failure.get(key) or "") for key in ("message", "code", "error"))
        if MALFORMED.search(text):
            malformed.append({"time": r.get("time"), "text": text})
    examples = ([{"time": m["time"], "text": m["forms"][0]["text"]} for m in drops + beside]
                + [{"time": m["time"], "text": m["text"]} for m in malformed])[:top]
    return {"messages": len(msgs), "with_call": sum(1 for m in msgs if m["has_call"]),
            "markup": len(markup), "quoted": sum(1 for f in markup if f["quoted"]),
            "asks": sum(1 for m in msgs if m["has_call"]) + len(drops),
            "drops": len(drops), "drops_beside_call": len(beside),
            "malformed": len(malformed), "examples": examples}


def served_model(rs: list, default: str = ""):
    """The model a session was served by.

    The session store records a selection only when one is *made*: a session that never
    switched writes nothing, and it ran whatever the deployment's `agent-default-model` was.
    That is not the same as unknown -- the default is in the harness settings, which is why it
    is an argument here and is printed in the header instead of being folded in quietly. A
    session that switched mid-way is filed under the model it was using when it stopped.
    """
    model = None
    for r in rs:
        if r.get("type") == "model/selection":
            model = (r.get("data") or {}).get("model")
    return model or (default or None)


def streaks(seq: list, threshold: int = LOOP_STREAK) -> dict:
    """Longest run of identical consecutive calls, overall and within a single turn.

    Three readings, because one number cannot carry the question. `max_streak` is the field
    definition from #1560: identical consecutive calls, allowed to cross a turn boundary.
    `max_streak_within_turn` is the same run not allowed to leave a turn. The cheap one lies
    in a specific direction -- on this box's own transcripts the longest runs are
    `job_output(wait=true)` polls of a running job, identical because waiting is the
    operation, and the answers differ every time -- so `max_stalled_streak` continues a run
    only when the call also came back the same way, which is the model answering itself.
    """
    best = best_stall = best_turn = streak = stall = turn_streak = 0
    adjacent = 0
    previous = previous_turn = previous_result = None
    example = None
    for c in seq:
        identical = previous is not None and (c["name"], c["arguments"]) == previous
        same_answer = identical and bool(c["result"]) and c["result"] == previous_result
        same_turn = identical and c.get("turn") == previous_turn
        streak = streak + 1 if identical else 1
        stall = stall + 1 if same_answer else 1
        turn_streak = turn_streak + 1 if same_turn else 1
        if identical:
            adjacent += 1
        if streak > best:
            best, example = streak, c
        best_stall = max(best_stall, stall)
        best_turn = max(best_turn, turn_streak)
        previous, previous_turn = (c["name"], c["arguments"]), c.get("turn")
        previous_result = c["result"]
    return {"calls": len(seq), "max_streak": best, "max_stalled_streak": best_stall,
            "max_streak_within_turn": best_turn, "repeat_adjacent": adjacent,
            "looped": best >= threshold, "stalled": best_stall >= threshold,
            "example": example}


def sessions(root: str) -> list:
    for dirpath, _, files in os.walk(root):
        if FILE in files:
            yield os.path.join(dirpath, FILE)


def when(t):
    """A record timestamp, which is epoch milliseconds, as an aware UTC datetime."""
    if not t:
        return None
    return datetime.datetime.fromtimestamp(t / 1000 if t > 10 ** 12 else t,
                                           datetime.timezone.utc)


def parse_instant(text: str):
    """A `--flip` or `--since` argument. Naive means UTC, because every timestamp this tool
    compares against was written in UTC and a local-time interpretation would silently move
    the split by hours on a box whose clock is not UTC."""
    if not text:
        return None
    text = text.strip().replace("Z", "+00:00")
    stamp = datetime.datetime.fromisoformat(text)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=datetime.timezone.utc)
    return stamp


def report_dropped(rows: list, bucket, args) -> int:
    """Print the dropped-call reading. See `dropped` for what each column is."""
    def rate(g, key, over):
        total = sum(x[over] for x in g)
        return f"{100 * sum(x[key] for x in g) / total:.1f}%" if total else "n/a"

    print(f"{len(rows)} sessions of model '{args.model or 'any'}' under {args.root}"
          + (f", active since {args.since}" if args.since else "")
          + "\n  what counts as a dropped call: the engine's own tool-call markup, in an "
            "assistant\n  message that got no call, outside a code fence or a backtick span — "
            "a call the\n  client was handed as prose. Plus any step that ended in the "
            "parser's error.")
    groups = {}
    for r in rows:
        groups.setdefault(bucket(r), []).append(r)
    for name in sorted(groups):
        g = groups[name]
        total = {key: sum(x[key] for x in g) for key in
                 ("messages", "with_call", "asks", "drops", "drops_beside_call", "quoted",
                  "malformed")}
        print(f"\n{'last active ' + name + ' ' + args.flip if args.flip else 'all sessions'}: "
              f"{len(g)} sessions, {total['messages']} assistant messages")
        print(f"  turns that wanted a tool: {total['asks']} — {total['with_call']} arrived as a "
              f"call, {total['drops']} arrived as prose carrying the call's markup")
        print(f"  dropped-call rate: {rate(g, 'drops', 'asks')} of what wanted a tool "
              f"(upstream, over 1,462 turns of this model under Claude Code: 7.1% of turns lost "
              f"a call without the switch, 1.7% with it)")
        print(f"  markup an agent quoted in a fence or backticks, excluded: {total['quoted']}"
              f"; beside a call that did arrive: {total['drops_beside_call']}")
        print(f"  steps that ended in \"malformed tool call\": {total['malformed']} "
              f"{rate(g, 'malformed', 'asks')}")
    notable = [r for r in rows if r["drops"] or r["malformed"]]
    if notable:
        print(f"\n{len(notable)} sessions to read, not to count:")
        for r in sorted(notable, key=lambda r: -(r["drops"] + r["malformed"]))[:args.top]:
            example = (r["examples"] or [{}])[0].get("text", "").replace("\n", " | ")
            stamp = f"{r['last']:%Y-%m-%d %H:%M}Z" if r["last"] else "undated"
            print(f"  {r['drops']} in prose, {r['malformed']} errors — {r['session']}, {stamp}\n"
                  f"      {example[:130]}")
    else:
        print("\nNothing to read: no assistant message carries the markup outside a code span, "
              "and no step ended in the parser's error.")
    quoted = sum(r["quoted"] for r in rows)
    print(f"\nThat zero is worth what the instrument is worth, so here is its calibration: the "
          f"store does contain the markup — {quoted} of it, quoted by agents writing about this "
          f"very format — and all of it landed in the quoted column instead of the drop column. "
          f"A store that shows the format and none of it acting like a call is a store that does "
          f"not have the problem, which is a different finding from an instrument that cannot see "
          f"it.")
    if args.flip:
        print("\nBoth sides of the split are not automatically the same measurement: a switch at "
              "the split changes what the prose is allowed to contain. Before this box's "
              "tool_call_recovery deploy, markup in prose was a call the parser threw away; after "
              "it, most of those become a tool_call before the client sees a byte, so whatever "
              "still arrives in prose on the later side is a form recovery does not know — a lead "
              "to read, not a regression.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Loop streaks in real agent transcripts.")
    ap.add_argument("--root", default=ROOT, help="The DSH session store")
    ap.add_argument("--model", default="daytime",
                    help="Only sessions served by this model (empty for all)")
    ap.add_argument("--default-model", default="daytime",
                    help="What a session that never selected a model ran: this deployment's "
                         "agent-default-model in its settings.yaml (empty to exclude them)")
    ap.add_argument("--flip", default="", help="ISO instant to split before/after, e.g. a deploy")
    ap.add_argument("--since", default="", help="ISO lower bound on session activity")
    ap.add_argument("--threshold", type=int, default=LOOP_STREAK,
                    help="Identical consecutive calls that count as a loop")
    ap.add_argument("--top", type=int, default=8, help="Worst sessions to name")
    ap.add_argument("--workspace", default="", help="Substring filter on the session's directory")
    ap.add_argument("--dropped", action="store_true",
                    help="Report tool calls that never arrived instead of runs of repeats")
    args = ap.parse_args()

    flip = parse_instant(args.flip)
    since = parse_instant(args.since)

    rows = []
    drop_rows = []
    for path in sessions(args.root):
        if args.workspace and args.workspace not in path:
            continue
        rs = records(path)
        if not rs:
            continue
        model = served_model(rs, args.default_model)
        if args.model and model != args.model:
            continue
        if args.dropped:
            d = dropped(rs, args.top)
            stamp = when(max([r.get("time") or 0 for r in rs]) or 0)
            if d["messages"] and not (since and (stamp is None or stamp < since)):
                d.update(session=os.path.relpath(path, args.root).split(os.sep)[0],
                         model=model, last=stamp)
                drop_rows.append(d)
        seq = calls(rs)
        if not seq:
            continue
        last = when(seq[-1]["time"])
        if since and (last is None or last < since):
            continue
        s = streaks(seq, args.threshold)
        s.update(session=os.path.relpath(path, args.root).split(os.sep)[0], model=model,
                 last=last)
        rows.append(s)

    def bucket(r):
        if not flip or r["last"] is None:
            return "all"
        return "after" if r["last"] >= flip else "before"

    if args.dropped:
        return report_dropped(drop_rows, bucket, args)

    print(f"{len(rows)} sessions of model '{args.model or 'any'}' under {args.root}"
          + (f", active since {since:%Y-%m-%d %H:%M}Z" if since else "")
          + (f"\n  sessions with no selection record counted as '{args.default_model}' "
             f"(the deployment default)" if args.default_model else ""))
    groups = {}
    for r in rows:
        groups.setdefault(bucket(r), []).append(r)
    for name in sorted(groups):
        g = groups[name]
        label = "all sessions" if not flip else f"last active {name} {args.flip}"
        looping = [x for x in g if x["looped"]]
        stalled = [x for x in g if x["stalled"]]
        inside = [x for x in g if x["max_streak_within_turn"] >= args.threshold]
        print(f"\n{label}: {len(g)} sessions, {sum(x['calls'] for x in g)} tool calls")
        print(f"  a run of {args.threshold} identical consecutive calls: "
              f"{len(looping)} sessions ({100 * len(looping) / len(g):.0f}%), "
              f"{len(inside)} of them inside one turn; "
              f"{len(stalled)} answering identically every time, which is the loop")
        print(f"  streaks: median {statistics.median([x['max_streak'] for x in g])}, "
              f"max {max(x['max_streak'] for x in g)}; "
              f"{sum(x['repeat_adjacent'] for x in g)} adjacent repeats in all")
    worst = sorted(rows, key=lambda r: (-r["max_streak"], r["calls"]))[:args.top]
    if worst and args.top:
        print(f"\nthe {len(worst)} longest streaks, to read rather than to count:")
        for r in worst:
            e = r["example"] or {}
            stamp = f"{r['last']:%Y-%m-%d %H:%M}Z" if r["last"] else "undated"
            print(f"  {r['max_streak']}x {e.get('name', '?')} "
                  f"{(e.get('arguments') or '')[:70]}\n"
                  f"      {r['session']}, {stamp}, {r['calls']} calls")
    return 0


if __name__ == "__main__":
    sys.exit(main())

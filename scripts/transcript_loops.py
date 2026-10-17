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
import shutil
import statistics
import subprocess
import sys

ROOT = os.path.expanduser("~/deployments/deepseek-harness/data/dsh/sessions")
FILE = "session.v4.jsonl.zstd"
MAGIC = b"\x28\xb5\x2f\xfd"
LOOP_STREAK = 4

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
    args = ap.parse_args()

    flip = parse_instant(args.flip)
    since = parse_instant(args.since)

    rows = []
    for path in sessions(args.root):
        if args.workspace and args.workspace not in path:
            continue
        rs = records(path)
        if not rs:
            continue
        model = served_model(rs, args.default_model)
        if args.model and model != args.model:
            continue
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

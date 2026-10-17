#!/usr/bin/env python3
"""A fixed greedy replay, so that a decode claim can be made at all.

`bench.sh` (`tools/strata_bench.py --long`) is the instrument for prefill: it repeats to
about +-0.6% on prompt throughput and on time to first token. It is not the instrument for
decode. Its agent session feeds the model's own answers back, so every run generates
different text and therefore different draft acceptance; pooled over the 108 agent-decode
samples it produces, the bootstrap 95% CI on the median is about +-6%. The decode levers in
play on this box -- route tail skip, pipeline windows, adapt-async, the quant itself --
trade 4-15% of decode, which is inside the noise of that instrument.

This harness holds the one variable that makes decode irreproducible: the text. Every
request is sent at temperature 0 against a corpus that is a pure function of this file, so
greedy decoding produces the same continuation, and the same draft acceptance, on every run
and on every arm. What differs between runs is the configuration, not the workload.

It is also the half of the question `bench.sh` cannot answer at all. Each prompt's greedy
answer is hashed, so two arms that hash the same answer made the same tokens: a lever
advertised as bit-identical can be shown to be, and a lever that quietly changes what the
model says cannot pass as free.

`--sampling thinking` (or `instruct`) is the arm that prices the run config's sampling block, and
it is the one place this harness deliberately gives up its own guarantee: the same corpus sent at
the model card's set instead of at temperature 0. A sampled continuation differs every run by
construction, so on that arm the answer hash is reported and not gated -- the verdict says how
many distinct answers a cell produced instead of refusing the run over them -- and the number the
arm exists for is decode, which is where the draft acceptance the sampling costs shows up. The
greedy arm stays the default, and stays what every recorded baseline was measured with; a sampled
run is only ever comparable with another sampled run.

Two things this found about the box before it was used for anything. The first pass over a
prompt decodes about 25-45% slower than every pass after it, with time to first token
unchanged -- so it is not the prompt cache, it is the expert caches filling, which is
exactly the memory hierarchy the levers act on. Round 1 is therefore reported apart and
never pooled with the rest. And the absolute decode this reports, about 153 tok/s warm on a
4K prompt, is not the same quantity as the 104-109 tok/s in the recorded `bench.sh`
baselines, which are cold. Only replay-to-replay differences in the same regime mean
anything; do not read this against a `bench.sh` number.

Stdlib only, because it runs inside the engine container:

    docker exec -i qwen38-daytime python3 - --url http://127.0.0.1:8080 \\
        --out /data/replay-results/arm-a-$(date +%F) < scripts/strata_replay.py

Run it at idle. The box serves live traffic, and the session driving this harness is itself
served by the engine under measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

# --------------------------------------------------------------------------------- corpus
# Synthetic and deterministic. Nothing here is a conversation that happened and nothing
# here is a weight, so a published result carries no private text and a replay is
# reproducible by anyone holding this file. The shapes follow what this host actually
# serves: code with long identifiers, tool output that is mostly JSON and paths, and prose
# explaining the code. Draft acceptance is what decode measures, and acceptance is a
# property of the text, so the text is the thing that has to be fixed.

_IDENT = ("render", "collect", "resolve", "materialize", "truncate", "coalesce", "dispatch",
          "normalize", "acquire", "settle", "provision", "reconcile", "publish", "observe")
_TYPE = ("Session", "Lease", "Manifest", "Catalog", "Slot", "Receipt", "Bundle", "Attempt",
         "Window", "Shard", "Cursor", "Digest")
_FIELD = ("id", "revision", "sequence", "offset", "length", "owner", "state", "reason",
          "attempt", "deadline", "signature", "parent")
_STATE = ("pending", "serving", "draining", "restoring")


def _lcg(seed: int):
    """Deterministic across platforms and Python versions, unlike random.Random's stream."""
    state = seed & 0xFFFFFFFF

    def nxt(bound: int) -> int:
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state % bound

    return nxt


def _code(rnd, i: int) -> str:
    name = _IDENT[rnd(len(_IDENT))]
    kind = _TYPE[rnd(len(_TYPE))]
    field = _FIELD[rnd(len(_FIELD))]
    return (
        f"\n\nclass {kind}{i:02d}:\n"
        f'    """Track the {name}ing of one {kind.lower()} through its {field}."""\n'
        f"\n    def __init__(self, {field}, owner, attempt=0):\n"
        f"        self.{field} = {field}\n"
        f"        self.owner = owner\n"
        f"        self.attempt = attempt\n"
        f"        self.state = 'pending'\n"
        f"        self._log = []\n"
        f"\n    def {name}(self, batch):\n"
        f"        if not batch:\n"
        f"            raise ValueError('refusing to {name} an empty batch')\n"
        f"        settled = [item for item in batch if item.get('ready')]\n"
        f"        if len(settled) != len(batch):\n"
        f"            self.state = 'partial'\n"
        f"        self._log.append((time.time(), len(settled), self.{field}))\n"
        f"        return settled\n"
    )


def _tool(rnd, i: int) -> str:
    record = {"path": f"runtime/{_IDENT[rnd(len(_IDENT))]}_{i}.py",
              "line": 40 + rnd(900),
              "revision": f"{i:04x}" + "abcdef1234567890"[: rnd(6) + 2],
              "tokens": 120 + rnd(4000),
              "state": _STATE[rnd(len(_STATE))],
              _FIELD[rnd(len(_FIELD))]: rnd(10000)}
    return json.dumps(record, sort_keys=True) + "\n"


def _prose(rnd, i: int) -> str:
    a = _IDENT[rnd(len(_IDENT))]
    b = _IDENT[rnd(len(_IDENT))]
    c = _TYPE[rnd(len(_TYPE))]
    return (f" When the runtime cannot {a} the {c.lower()} it still has to {b} the attempt, "
            f"because a {c.lower()} that is neither serving nor stopped leaves the router "
            f"unable to say who owns the drain. The rule is that the last healthy release "
            f"is the one that answers, and that a switch which fails before the drain has "
            f"changed nothing at all. ")


_BUILDER = {"code": _code, "tool": _tool, "prose": _prose}
_HEAD = {"code": "from __future__ import annotations\n\nimport json\nimport time\n"}


def _fill(kind: str, seed: int, target: int) -> str:
    """Append blocks until the target length, so a cell is the size its name says."""
    rnd = _lcg(seed)
    out = [_HEAD.get(kind, "")]
    total = len(out[0])
    i = 0
    while total < target:
        piece = _BUILDER[kind](rnd, i)
        out.append(piece)
        total += len(piece)
        i += 1
    return "".join(out)


def _ask(kind: str, body: str) -> list:
    """The same question every time, so the greedy continuation is the same every time.

    Each task asks for an enumeration copied out of the prompt, for two reasons. Decode is
    only measurable over a few hundred tokens, and a summary stops at twenty; and copying
    from the context is what a coding session actually does, so the draft acceptance this
    measures is the acceptance the workload sees.
    """
    if kind == "code":
        task = ("List every class defined in the module above, one per line, in the order they "
                "appear, each line written as `class_name: field`, where field is the one it "
                "assigns to itself in __init__. No preamble and no commentary.")
    elif kind == "tool":
        task = ("List every `path` value that appears in the records above, one per line, in "
                "the order they first appear, with no count and no commentary.")
    else:
        task = ("Quote the opening of the passage above verbatim, one sentence per line, "
                "starting from its first sentence and continuing in order. Do not summarise, "
                "do not reorder, and add no commentary.")
    return [{"role": "system",
             "content": "You are a careful senior engineer. Answer exactly as asked."},
            {"role": "user", "content": body + "\n\n" + task}]


# Characters per token, measured against the served model's own tokenizer on 2026-10-14 by
# asking the engine for the `prompt_tokens` of a known string. The guess this replaces --
# 3.6 for every shape -- was wrong by a factor of two on JSON, and the first full run built
# a cell named 128k that was 208,561 tokens and the engine refused it as over context.
# Shape matters because a JSON record is mostly punctuation.
CPT = {"code": 3.836, "tool": 2.183, "prose": 4.726}

# (cell, shape, seed, prompt tokens). The names are addresses: what gets recorded is the
# engine's own `prompt_tokens`, and `main` refuses a run whose cells drifted off them.
CELLS = (("code-4k", "code", 11, 4_000),
         ("tool-4k", "tool", 23, 4_000),
         ("code-32k", "code", 37, 32_000),
         ("prose-32k", "prose", 53, 32_000),
         ("tool-128k", "tool", 71, 128_000))
ALL_CELLS = tuple(cell[0] for cell in CELLS)
_TARGET = {cell[0]: cell[3] for cell in CELLS}


def corpus(keep=ALL_CELLS) -> list:
    """The prompts, each a pure function of the constants in this file."""
    prompts = []
    for label, kind, seed, tokens in CELLS:
        if label not in keep:
            continue
        body = _fill(kind, seed, int(tokens * CPT[kind]))
        prompts.append({"label": label, "kind": kind, "target_tokens": tokens,
                        "messages": _ask(kind, body),
                        "sha256": hashlib.sha256(body.encode()).hexdigest()[:16]})
    return prompts


# -------------------------------------------------------------------------------- requests
# The arms this harness can send. "greedy" is what every recorded baseline was measured with and
# what the engine does with no sampling block; the other two are the sets the model card states,
# spelled out here rather than read off the box so an arm measures the set and not whatever the
# run config happens to have been told (the same table lives in
# runtime/overlay/strata-vision-entrypoint.sh, which is what writes it in production).
SAMPLING_ARMS = {
    "greedy": {"temperature": 0},
    "thinking": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
    "instruct": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                 "presence_penalty": 1.5},
}


def chat(base: str, key: str, messages: list, max_tokens: int, think: bool,
         sampling: dict, timeout: float) -> dict:
    """One streamed request at the arm's sampling. The request shape matches
    tools/strata_bench.py, so the greedy arm stays comparable with recorded baselines."""
    body = {"model": "strata", "stream": True, "max_tokens": max_tokens, "messages": messages}
    body.update(sampling)
    if not think:
        body["reasoning_effort"] = "none"
        body["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    t0 = time.perf_counter()
    first, text, usage, timings, err, pieces = None, [], None, None, None, 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    j = json.loads(d)
                except ValueError:
                    continue
                if j.get("error"):
                    err = str(j["error"])[:300]
                for ch in j.get("choices") or []:
                    dl = ch.get("delta") or {}
                    if (dl.get("content") or "") + (dl.get("reasoning_content") or ""):
                        if first is None:
                            first = time.perf_counter() - t0
                        pieces += 1
                        text.append(dl.get("content") or "")
                if j.get("usage"):
                    usage = j["usage"]
                if j.get("timings"):
                    timings = j["timings"]
    except urllib.error.HTTPError as e:
        err = f"HTTP {e.code}: {e.read()[:200]!r}"
    except Exception as e:                              # noqa: BLE001
        err = repr(e)[:300]
    total = time.perf_counter() - t0
    return {"first": first, "total": total, "text": "".join(text), "usage": usage,
            "timings": timings, "err": err, "pieces": pieces}


def _get(base: str, key: str, path: str, timeout: float = 10):
    headers = {"Authorization": "Bearer " + key} if key else {}
    with urllib.request.urlopen(urllib.request.Request(base + path, headers=headers),
                                timeout=timeout) as r:
        return json.loads(r.read())


def busy_slots(base: str, key: str):
    """How many slots were already generating when this sample was about to go out.

    "Run it at idle" is the right advice and is not something the harness can assume,
    because this box serves the very session that runs the harness: every tool call made
    while a run is in progress is inference on the engine under measurement. The first
    baseline taken here had a 4K prompt waiting 45 s for its first token behind an agent
    turn, which is four times the real number and larger than any effect in the queue. So
    the harness records the engine's own answer instead of trusting the operator, and a run
    with foreign work in it says so rather than passing as a baseline.
    """
    try:
        return sum(1 for s in _get(base, key, "/slots") if s.get("is_processing"))
    except Exception:                                     # noqa: BLE001
        return None


def wait_for_quiet(base: str, key: str, hold: float, bound: float, settle: float = 2.0):
    """Wait until the engine has been idle for `hold` straight seconds. `None` on success,
    otherwise the reason it gave up.

    This is what makes an unattended baseline possible. The only window in which this box
    is reliably quiet is one nobody is driving it in, so the run has to be started and then
    left alone -- and a tool that starts work the moment it is invoked will measure whatever
    happens to be running. Waiting for the engine's own report turns "run it at idle" from
    advice into a precondition, and a cron entry at four in the morning either gets its
    baseline or says why it did not.
    """
    t0 = time.perf_counter()
    idle_for = 0.0
    while time.perf_counter() - t0 < bound:
        busy = busy_slots(base, key)
        if busy == 0:
            idle_for += settle
            if idle_for >= hold:
                return None
        else:
            idle_for = 0.0
        time.sleep(settle)
    return f"the engine was not idle for {hold:.0f} s within {bound:.0f} s"


def loaded_context(base: str, key: str):
    """The window the engine actually loaded with, which is not necessarily the one the
    profile asked for."""
    try:
        return _get(base, key, "/health").get("max_context")
    except Exception:                                     # noqa: BLE001
        return None


def digest(r: dict) -> dict:
    """The engine's own numbers for one request, read the way the bench reads them."""
    u, t = r.get("usage") or {}, r.get("timings") or {}
    dec = t.get("predicted_per_second")
    if dec is None and r.get("first") is not None and r.get("pieces", 0) > 1 \
            and r["total"] > r["first"]:
        dec = round((r["pieces"] - 1) / (r["total"] - r["first"]), 1)
    return {"ok": r.get("err") is None and r.get("first") is not None, "err": r.get("err"),
            "ttft_s": None if r.get("first") is None else round(r["first"], 3),
            "prompt_tokens": (u or {}).get("prompt_tokens"),
            "generated_tokens": (u or {}).get("completion_tokens") or r.get("pieces"),
            "decode_tps": dec, "prompt_tps": t.get("prompt_per_second"),
            "drafts_offered": t.get("draft_n"), "drafts_accepted": t.get("draft_n_accepted"),
            "answer_sha256": hashlib.sha256(r.get("text", "").encode()).hexdigest()[:16]}


# -------------------------------------------------------------------------------- statistics
def boot_ci(values, resamples: int = 2000, seed: int = 20261015):
    """Percentile bootstrap 95% CI on the median, over a fixed resampling stream so the
    interval itself is reproducible."""
    v = sorted(values)
    if len(v) < 2:
        return (round(v[0], 1), round(v[0], 1)) if v else (None, None)
    rnd = _lcg(seed)
    n = len(v)
    medians = sorted(statistics.median([v[rnd(n)] for _ in range(n)]) for _ in range(resamples))
    return round(medians[int(0.025 * resamples)], 1), round(medians[int(0.975 * resamples)], 1)


def verdict(rows: list, summary: dict, min_tokens: float = None, gate_answers: bool = True):
    """Whether this run may be used, and what stops it if it may not.

    Ordered by how much damage each does to a claim. A run with the engine busy is not a
    slow run, it is a different run, and it is the failure most likely to go unnoticed
    because every number in it looks plausible.

    `gate_answers` is False on a sampled arm, where a cell that produces two answers is the
    arm working as designed and not a broken harness; the caller reports the spread instead.
    """
    failed = [r for r in rows if not r["ok"]]
    if failed:
        return 1, [f"{len(failed)} of {len(rows)} requests failed"]
    busy = [r for r in rows if r.get("busy_before")]
    if busy:
        return 3, [f"{len(busy)} of {len(rows)} samples had the engine already generating when "
                   f"they went out, so something else was using the box. This is not a "
                   f"baseline: nothing here compares against anything until it is run with "
                   f"the host quiet."]
    unstable = sorted(k for k, s in summary.items() if len(s["answers"]) > 1)
    if unstable and gate_answers:
        return 1, [f"greedy answer not stable in: {', '.join(unstable)} -- the replay does not "
                   f"hold decoding fixed in those cells, so no claim may rest on them"]
    drifted = sorted({r["label"] for r in rows if r.get("prompt_tokens")
                      and abs(r["prompt_tokens"] - r["target_tokens"])
                      > 0.15 * r["target_tokens"]})
    if drifted:
        return 1, [f"cell size drifted more than 15% off its name: {', '.join(drifted)} -- the "
                   f"CPT table no longer describes this tokenizer, so these cells are not the "
                   f"sizes the recorded run was"]
    if min_tokens:
        short = []
        for label in sorted({r["label"] for r in rows}):
            got = [r["generated_tokens"] for r in rows
                   if r["label"] == label and r["ok"] and r["phase"] == "warm"]
            if got and statistics.median(got) < min_tokens:
                short.append(f"{label} ({int(statistics.median(got))} tokens)")
        if short:
            return 1, [f"answer too brief to measure decode: {', '.join(short)} -- a rate "
                       f"over a handful of tokens is dominated by the first tokens, where "
                       f"acceptance is lowest, so make the task enumerate until it hits the "
                       f"cap"]
    return 0, []


def summarise(rows: list) -> dict:
    out = {}
    for label in dict.fromkeys(r["label"] for r in rows):
        cell = [r for r in rows if r["label"] == label and r["ok"]]
        if not cell:
            continue
        dec = [r["decode_tps"] for r in cell if r["decode_tps"]]
        pre = [r["prompt_tps"] for r in cell if r["prompt_tps"]]
        ttft = [r["ttft_s"] for r in cell if r["ttft_s"]]
        offered = sum(r["drafts_offered"] or 0 for r in cell)
        accepted = sum(r["drafts_accepted"] or 0 for r in cell)
        lo, hi = boot_ci(dec)
        out[label] = {"n": len(cell),
                      "decode_median": round(statistics.median(dec), 1) if dec else None,
                      "decode_ci95": [lo, hi],
                      "decode_spread": round(max(dec) - min(dec), 1) if len(dec) > 1 else None,
                      "prompt_median": round(statistics.median(pre), 1) if pre else None,
                      "ttft_median": round(statistics.median(ttft), 3) if ttft else None,
                      "accept_pct": round(100 * accepted / offered, 1) if offered else None,
                      "answers": sorted({r["answer_sha256"] for r in cell})}
    return out


def table(summary: dict, title: str) -> None:
    print(f"\n{title}")
    print(f"{'cell':12s} {'n':>2s} {'decode':>7s} {'95% CI':>16s} {'spread':>7s} "
          f"{'prefill':>8s} {'first':>7s} {'accept':>7s}  answers")
    for label, s in summary.items():
        lo, hi = s["decode_ci95"]
        ci = f"[{lo}, {hi}]" if s["n"] > 1 and lo else "-"
        print(f"{label:12s} {s['n']:2d} {str(s['decode_median']):>7s} {ci:>16s} "
              f"{str(s['decode_spread']):>7s} {str(s['prompt_median']):>8s} "
              f"{str(s['ttft_median']):>7s} {str(s['accept_pct']) + '%':>7s}  "
              f"{len(s['answers'])} distinct")


# ------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description="Fixed greedy replay for decode measurements.")
    ap.add_argument("--url", required=True, help="Engine base URL, e.g. http://127.0.0.1:8080")
    ap.add_argument("--key", default="", help="Bearer key, if the engine has one")
    ap.add_argument("--out", required=True, help="Output directory; put it on a durable mount")
    ap.add_argument("--cells", default=",".join(ALL_CELLS),
                    help="Comma-separated cells; drop tool-128k to keep a run short")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=1,
                    help="Rounds whose prompt cache is cold, reported apart and never pooled "
                         "with the rest: the first pass pays prefill inside the measurement "
                         "and every later pass is served from cache, and the two are about "
                         "25%% apart on decode")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--think", action="store_true", help="Leave thinking on (the suite does not)")
    ap.add_argument("--sampling", default="greedy", choices=sorted(SAMPLING_ARMS),
                    help="What the requests send for sampling. greedy, the default, is what every "
                         "recorded baseline was measured with and what the engine does with no "
                         "sampling block; thinking and instruct are the model card's sets, and on "
                         "those arms the answer hash is reported and not gated, because a sampled "
                         "continuation differs every run by construction")
    ap.add_argument("--settle", type=float, default=2.0, help="Seconds to rest between requests")
    ap.add_argument("--require-quiet", type=float, default=20.0,
                    help="Start only after the engine has been idle this many seconds running")
    ap.add_argument("--quiet-bound", type=float, default=3600.0,
                    help="Give up rather than measure a busy box")
    ap.add_argument("--retries", type=int, default=2,
                    help="Re-take a sample that came back while the engine was busy, instead "
                         "of voiding the whole run over one interruption")
    ap.add_argument("--compare", metavar="DIR", help="A previous run's directory to diff against")
    args = ap.parse_args()

    if args.rounds <= args.warmup:
        print("--rounds must exceed --warmup, or nothing is measured warm", file=sys.stderr)
        return 2
    keep = tuple(c.strip() for c in args.cells.split(",") if c.strip())
    unknown = [c for c in keep if c not in ALL_CELLS]
    if unknown:
        print(f"unknown cells: {', '.join(unknown)}; known: {', '.join(ALL_CELLS)}",
              file=sys.stderr)
        return 2
    prompts = corpus(keep)

    window = loaded_context(args.url, args.key)
    if window:
        over = [p for p in prompts if p["target_tokens"] + args.max_tokens > window]
        if over:
            print("skipping cells that do not fit the loaded window of "
                  f"{window} tokens: {', '.join(p['label'] for p in over)}", file=sys.stderr)
            prompts = [p for p in prompts if p not in over]
    if not prompts:
        print("nothing left to run", file=sys.stderr)
        return 2

    if args.require_quiet > 0:
        print(f"waiting for the engine to be idle {args.require_quiet:.0f} s running "
              f"(bound {args.quiet_bound:.0f} s)...", flush=True)
        reason = wait_for_quiet(args.url, args.key, args.require_quiet, args.quiet_bound)
        if reason:
            print(f"{reason}: refusing to measure a busy box", file=sys.stderr)
            return 4
        print("idle; starting", flush=True)

    print("corpus: " + ", ".join(p["label"] + "=" + p["sha256"] for p in prompts))
    print("sampling arm: " + args.sampling + " ("
          + ", ".join(f"{k}={v}" for k, v in sorted(SAMPLING_ARMS[args.sampling].items())) + ")")
    rows = []
    discarded = []          # kept for the record, never summarised
    for rnd in range(args.rounds):
        for p in prompts:                               # fixed order: drift shows as drift
            for attempt in range(args.retries + 1):
                busy = busy_slots(args.url, args.key)
                r = chat(args.url, args.key, p["messages"], args.max_tokens, args.think,
                         SAMPLING_ARMS[args.sampling], timeout=3600)
                d = digest(r)
                d.update(label=p["label"], kind=p["kind"], round=rnd, corpus_sha256=p["sha256"],
                         target_tokens=p["target_tokens"], busy_before=busy, attempt=attempt,
                         phase="cold" if rnd < args.warmup else "warm")
                if d["ok"] and not busy:
                    break
                discarded.append(d)
                if attempt < args.retries:
                    print(f"  {p['label']}: discarded "
                          f"({'engine was busy' if busy else d['err']}), taking it again",
                          file=sys.stderr, flush=True)
                    time.sleep(args.settle)
            rows.append(d)
            note = "  CONTAMINATED: engine was busy" if busy else ""
            if not d["ok"]:
                print(f"  run {rnd + 1} {p['label']}: FAILED {d['err']}", file=sys.stderr)
            else:
                print(f"  run {rnd + 1} [{d['phase']:4s}] {p['label']}: decode "
                      f"{d['decode_tps']} tok/s, first token {d['ttft_s']} s, "
                      f"{d['prompt_tokens']} prompt tokens, {d['generated_tokens']} tokens, "
                      f"answer {d['answer_sha256']}{note}", flush=True)
            time.sleep(args.settle)

    warm = [r for r in rows if r["phase"] == "warm"]
    cold = [r for r in rows if r["phase"] == "cold"]
    summary = summarise(warm)
    table(summary, f"warm (rounds {args.warmup + 1}-{args.rounds}) -- this is the measurement")
    table(summarise(cold), f"cold (rounds 1-{args.warmup}) -- reference only")

    if args.compare:
        path = os.path.join(args.compare.rstrip("/"), "results.json")
        try:
            with open(path) as f:
                prior = json.load(f)
        except (OSError, ValueError) as e:
            print(f"\ncannot read {path}: {e}", file=sys.stderr)
            return 2
        print("\nversus the recorded run:")
        for label, s in summary.items():
            q = (prior.get("summary") or {}).get(label) or {}
            before, ci = q.get("decode_median"), q.get("decode_ci95") or [None, None]
            if not (before and s["decode_median"]):
                continue
            delta = 100 * (s["decode_median"] - before) / before
            # Real only if the new median falls outside the interval the recorded run was
            # uncertain about. Inside it, these are the same speed and that is the finding.
            outside = None not in ci and not (ci[0] <= s["decode_median"] <= ci[1])
            same = q.get("answers") == s["answers"]
            print(f"  {label:12s} decode {before} -> {s['decode_median']} ({delta:+.1f}%), "
                  f"{'outside' if outside else 'inside'} the recorded interval; "
                  f"{args.sampling} answer {'unchanged' if same else 'CHANGED'}"
                  + ("" if same or args.sampling == "greedy" else " (expected on a sampled arm)"))
        prior_corpus = {r["label"]: r.get("corpus_sha256") for r in prior.get("rows", [])}
        for p in prompts:
            was = prior_corpus.get(p["label"])
            if was is not None and was != p["sha256"]:
                print(f"  WARNING {p['label']}: the corpus changed between runs, so these are "
                      f"different workloads and the numbers do not compare", file=sys.stderr)
                return 2

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"tool": "strata_replay",
                   "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "params": {"rounds": args.rounds, "warmup": args.warmup,
                              "max_tokens": args.max_tokens, "think": args.think,
                              "sampling": args.sampling,
                              **SAMPLING_ARMS[args.sampling], "cells": list(keep)},
                   "summary": summary, "cold": summarise(cold), "rows": rows,
                   "discarded": discarded}, f, indent=1)
    print(f"\nwrote {os.path.join(args.out, 'results.json')}")

    code, reasons = verdict(rows, summary, min_tokens=0.6 * args.max_tokens,
                            gate_answers=args.sampling == "greedy")
    if args.sampling != "greedy":
        spread = {k: len(s["answers"]) for k, s in summary.items() if len(s["answers"]) > 1}
        print(f"sampled arm: the answer is not held fixed, so it is reported and not gated: "
              f"{len(spread)} of {len(summary)} cells produced more than one answer"
              + (f" ({', '.join(f'{k} x{v}' for k, v in sorted(spread.items()))})" if spread else ""))
    for reason in reasons:
        print(reason, file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())

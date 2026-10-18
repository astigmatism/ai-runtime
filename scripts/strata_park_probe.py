#!/usr/bin/env python3
"""Interrupt long conversations on purpose and read what the engine gives back.

    python3 scripts/strata_park_probe.py --url http://127.0.0.1:18080 \
        --log /home/astigmatism/ai/models/strata-trial/data/strata-engine.log

The benchmark cannot find the conversation cache: a benchmark run keeps every
conversation it is talking about hot in a RAM slot, and the cache only pays when
the slot was lost.  This drives the shape the cache exists for -- several long
conversations, each resumed after a gap, with the others in between taking the
slots -- and reads the answer off the engine's own two statements:

    prompt 87302 tokens = 81939 reused + 5363 read in 6359.4 ms
    conversation cache: restored 81939 tokens (checkpoint) in 135.2 ms

A resumed turn that reads almost none of its own prefix came back from RAM; one
that reads all of it recomputed it.  ``--conversations`` above the engine's park
slot count is the point of the exercise: the conversations that do not fit are the
measurement of what a slot is worth.  ``/metrics`` is sampled before and after, so
``parks`` and ``restores`` in this script's output are the engine agreeing with it.
"""

import argparse
import json
import random
import re
import subprocess
import time
import urllib.request

PROMPT_LINE = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in ([\d.]+) ms")
CACHE_LINE = re.compile(r"conversation cache: [^\n]*")


def words(count, seed):
    pick = random.Random(seed)
    return " ".join("w%d" % pick.randrange(99999) for _ in range(count))


def metrics(url):
    with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=10) as response:
        state = json.load(response)["conversation_cache"]
    return {k: state[k] for k in ("parks", "restores", "parked", "evictions", "bytes") if k in state}


def request(url, conversation, timeout):
    body = json.dumps({"model": "qwen3.8", "messages": conversation, "max_tokens": 16, "temperature": 0}).encode()
    call = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(call, timeout=timeout) as response:
        return json.load(response).get("usage", {})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:18080")
    ap.add_argument("--log", required=True, help="the engine's durable log, for its own per-request lines")
    ap.add_argument("--conversations", type=int, default=3)
    ap.add_argument("--words", type=int, default=13000, help="opening document size in words (~2 tokens each)")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--gap", type=float, default=18.0, help="seconds another conversation holds the slot")
    ap.add_argument("--timeout", type=float, default=600.0)
    opts = ap.parse_args()

    names = [chr(ord("A") + i) for i in range(opts.conversations)]
    documents = {n: words(opts.words, i) for i, n in enumerate(names)}
    conversations = {n: [{"role": "system", "content": "You are a careful assistant. Answer in one short sentence."}]
                     for n in names}
    print("engine says:", json.dumps(metrics(opts.url)), flush=True)
    for round_number in range(opts.rounds):
        for name in names:
            conversation = conversations[name]
            conversation.append({"role": "user", "content": (
                documents[name] if round_number == 0 else "Now note item %d: %s " % (round_number, words(900, 100 + round_number)))
                + "\nHow many distinct items are listed? One word."})
            started = time.time()
            usage = request(opts.url, conversation, opts.timeout)
            waited = time.time() - started
            tail = subprocess.run(["tail", "-c", "200000", opts.log], capture_output=True, text=True).stdout
            prompt = PROMPT_LINE.findall(tail)
            cached = CACHE_LINE.findall(tail)
            print("round %d %s  ctx %6d in %5.1f s | %s" % (round_number, name, usage.get("prompt_tokens", 0),
                  waited, prompt[-1] if prompt else "no engine line found"), flush=True)
            for line in [c for c in cached if "restored" in c][-1:]:
                print("   ", line[:160], flush=True)
            conversation.append({"role": "assistant", "content": "Three."})
            time.sleep(opts.gap)
    print("engine says:", json.dumps(metrics(opts.url)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare two Strata bench result folders and say which differences the instrument can see.

    python3 scripts/strata_bench_compare.py BASE_DIR CANDIDATE_DIR

Each argument is a bench output folder (the one holding ``results.json``, i.e.
``bench-base/2026-...-<machine>-<model>/``) or the ``results.json`` itself.  The bench prints medians and ranges
and leaves the pairing to the reader; this pairs the per-request samples itself and
reports a bootstrap 95% CI for the ratio, because on a one-slot server the bench's
own range over three runs is wider than most effects worth switching for.

Two of the metrics are instruments and the rest are weather.  ``prompt tok/s`` is
the engine's own throughput for its own prefill work and repeats to about 0.5%
across runs half an hour apart, and the agent-session cache hit repeats to about
0.5% too.  Everything that measures *time the request waited* -- TTFT, per-request
decode, the 4-client aggregate -- moves by multiples when anything else is talking
to the model, in one direction only, and two quiet runs of the same configuration
still disagree by 10-30% at the bench's sample size.  A CI that spans 1.0 is not a
tie and not a win: it is this instrument saying it cannot see the question.
"""

import json
import random
import statistics
import sys

METRICS = (
    ("chat decode tok/s", ("chat",), "decode_tps"),
    ("chat ttft s", ("chat",), "ttft_s"),
    ("4K prompt tok/s", ("prompts", "4K"), "prompt_tps"),
    ("4K ttft s", ("prompts", "4K"), "ttft_s"),
    ("16K prompt tok/s", ("prompts", "16K"), "prompt_tps"),
    ("16K ttft s", ("prompts", "16K"), "ttft_s"),
    ("32K prompt tok/s", ("prompts", "32K"), "prompt_tps"),
    ("32K ttft s", ("prompts", "32K"), "ttft_s"),
    ("agent turn decode tok/s", ("agent",), "decode_tps"),
    ("agent turn ttft s", ("agent",), "ttft_s"),
)
# The prefill family and the session cache hit are the load-insensitive ones; the
# table says so out loud instead of leaving it to the reader's memory.
SHARP = ("prompt tok/s", "agent cache hit %")


def rows(results, path, key):
    node = results
    for part in path:
        node = node[part]
    if path[0] == "agent":
        node = [turn for session in node["sessions"] for turn in session]
    else:
        node = node["requests"]
    return [r[key] for r in node if r.get("ok") and isinstance(r.get(key), (int, float))]


def metrics(results):
    out = {name: rows(results, path, key) for name, path, key in METRICS}
    out["agent cache hit %"] = [
        100.0 * sum(t["reused_tokens"] for t in session) / max(1, sum(t["prompt_tokens"] for t in session))
        for session in results["agent"]["sessions"]]
    out["4 clients total tok/s"] = [r["total_tps"] for r in results["clients"]["rounds"] if r.get("total_tps")]
    return {k: v for k, v in out.items() if v}


def interval(a, b, draws=4000):
    medians = lambda xs: [statistics.median(random.choices(xs, k=len(xs))) for _ in range(draws)]
    ratios = sorted(y / x for x, y in zip(medians(a), medians(b)) if x > 0)
    return ratios[int(0.025 * draws)], ratios[int(0.975 * draws)]


def main():
    if len(sys.argv) != 3:
        print(__doc__.strip())
        return 2
    def load(path):
        path = path.rstrip("/")
        return json.load(open(path if path.endswith(".json") else path + "/results.json"))["results"]
    base, want = load(sys.argv[1]), load(sys.argv[2])
    a, b = metrics(base), metrics(want)
    random.seed(7)
    print("%-24s %10s %10s %7s %-16s %s" % ("metric", "baseline", "candidate", "ratio", "95% CI", "n"))
    for name in a:
        x, y = a[name], b.get(name, [])
        if len(x) < 2 or len(y) < 2:
            print("%-24s %10d %10d   (too few samples to pair)" % (name, len(x), len(y)))
            continue
        low, high = interval(x, y)
        ratio = statistics.median(y) / statistics.median(x)
        lower_is_better = "ttft" in name
        better = (ratio < 1) == lower_is_better
        decisive = (high < 1) if lower_is_better else (low > 1)
        if high - low < 0.03 and abs(ratio - 1) <= 0.03:
            seen = "no difference; this instrument resolves %.1f%%" % ((high - low) / 2 * 100)
        elif decisive:
            seen = ("better" if better else "worse") + " beyond the interval"
        else:
            seen = ("better" if better else "worse") + " -- invisible, the CI spans 1.0"
        note = "  <- load-insensitive" if any(s in name for s in SHARP) else ""
        print("%-24s %10.2f %10.2f %7.3f %.3f - %.3f  %s  n=%d/%d%s"
              % (name, statistics.median(x), statistics.median(y), ratio, low, high, seen, len(x), len(y), note))
    print("\nRead the CI, not the ratio: on this instrument a decode difference under ~13% and a prefill"
          "\ndifference under ~0.5% are the same story, and that story is 'this box cannot tell'."
          "\nNothing here measures answer quality; a speed instrument that cannot see an axis is not"
          "\nevidence that the axis does not matter.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

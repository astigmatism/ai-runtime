#!/bin/sh
# This house's overlay on Strata's own entry point, not an upstream file.
#
# Three facts about this container have to reach the engine's own run config, and none of them is
# an argument the engine takes.
#
# 1. That it runs an image encoder at all (with which mmproj). The "vision" section of the run
#    config is written by setup.py's setup pass, and the upstream entry point runs that pass only
#    for a workspace that has no config yet. A workspace set up before vision was asked for has a
#    config and no section, so the upstream entry point skips the pass and serves text-only with
#    the encoder never started and nothing in any log: that is the 2026-10-16 boot, which sat
#    dead on the vision card and had to be switched off by hand.
# 2. Which card the encoder runs on. The profile states it as VISION_DEVICE, and serve/server.py
#    reads it from the "cuda_device" key of that same section (vision_env); setup.py has no flag
#    for it, so without this step the placement would live in a hand edit inside the workspace:
#    invisible to the controller, unrenderable, and gone with the workspace.
# 3. What sampling a request that sends none of its own gets. The profile states a preset as
#    SAMPLING. With no "sampling" block the engine decodes greedy - serve/server.py says so itself
#    ("An absent temperature keeps the engine's default, which is greedy") - and setup.py never
#    writes the block, so every routed client that omits sampling (the router forwards an omitted
#    temperature as omitted, and this house's coding agent sends none) decodes at the one setting
#    the model card does not recommend and the most repetition-prone of them all. The numbers
#    belong to a named preset and are written down here and nowhere else, because a free-form value
#    could be one the engine refuses to start on: a bad sampling value is a SystemExit, and for an
#    exclusive resident that is nothing serving. "thinking" is the Qwen3.8-Flash-Next card's
#    thinking-mode set, which is also the set this house already runs its llama.cpp residents with;
#    "instruct" is its non-thinking set (whose presence_penalty the engine does not apply inside
#    batch windows); "off" takes the block back out, which is the engine's own default, and a profile
#    stating nothing leaves the file alone. A preset this table does not know fails the boot before it
#    writes anything, and the controller refuses such a profile when it renders, before any drain.
# 4. Whether a declared tool's call written in a form beside the template's still counts as a call.
#    The profile states it as TOOL_CALL_RECOVERY, "on" or "off". serve/server.py reads the run config's
#    top-level "tool_call_recovery" and defaults it to false, and setup.py never writes it, so the
#    engine's answer to this house's agent is the off answer: upstream measures 92.9% of agent turns'
#    tool calls read as intended without it and 98.3% with it (1,462 turns, its own drift corpus). What
#    it changes is only what a request that *declared* tools gets back, and only in four documented
#    forms; anything else in those forms stays the text it was. It is a switch and not a free-form
#    value for the same reason sampling is one: serve/server.py raises SystemExit unless the key is a
#    JSON true or false, and for an exclusive resident a config it refuses is nothing serving. "off"
#    is a statement, so it takes the key back out; a profile stating nothing leaves the file alone.
#
# So: when the container is told the encoder's card, make sure the config it is about to start
# carries a vision section, and write the card into it. When it is told a sampling preset, make sure
# that config exists at all, and write the block into it. Where the section or the file is missing,
# the setup pass that would create it is the one the upstream entry point would have run for this
# exact environment, so this runs it itself (the function keeps its argument list out of this entry
# point's own); a fresh workspace pays that pass once here instead of twice later, and a workspace
# already set up skips straight to the patch. A pass that still leaves no section (VISION says no)
# is a misconfigured card, and a misconfigured card fails the boot instead of serving text-only with
# a log gap. Then hand over to the upstream entry point unchanged.
set -e

STRATA_DATA="${STRATA_DATA:-/data}"
FAMILY="${FAMILY:-qwen}"
MODEL="${MODEL:-IQ2_XS}"
CONTEXT="${CONTEXT:-32768}"
VISION="${VISION:-no}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-}"
KV="${KV:-}"
GPUS="${GPUS:-}"
GPU="${GPU:-}"
LAYER_SPLIT="${LAYER_SPLIT:-}"
LOW_RAM="${LOW_RAM:-auto}"
GGUF_DIR="${GGUF_DIR:-}"
RESIDENT_BUDGET_GIB="${RESIDENT_BUDGET_GIB:-}"
KV_STREAMING="${KV_STREAMING:-}"
SAMPLING="${SAMPLING:-}"
TOOL_CALL_RECOVERY="${TOOL_CALL_RECOVERY:-}"

# The same tag the upstream entry point derives, so what follows writes the file it is about to read.
case "$FAMILY" in
    qwen) prefix='' ;;
    *) prefix="${FAMILY}-" ;;
esac
tag=$(printf '%s' "$MODEL" | tr 'A-Z' 'a-z')
cfg="$STRATA_DATA/config/strata-${prefix}${tag}.json"

has_vision_section() {
    python3 - "$1" <<'PY'
import json, sys

try:
    with open(sys.argv[1], encoding='utf-8-sig') as handle:
        cfg = json.load(handle)
except (OSError, ValueError):
    sys.exit(1)
sys.exit(0 if isinstance(cfg.get('vision'), dict) else 1)
PY
}

run_setup_pass() {
    echo "strata overlay: $1; running the setup pass the upstream entry point would run"
    mkdir -p "$STRATA_DATA/config"
    # The upstream entry point's setup arguments, verbatim: the same defaults, the same conditional
    # flags, and the same --no-start, so the config this writes is the config that boot would have
    # written. A function has its own positional parameters, so this cannot reach the argument list
    # this entry point execs with.
    set -- --family "$FAMILY" --model "$MODEL" --context "$CONTEXT" --vision "$VISION" \
        --data-dir "$STRATA_DATA" --host "$HOST" --api-key "$API_KEY" \
        --port "$PORT" --no-start --low-ram "$LOW_RAM"
    if [ -n "$KV" ]; then set -- "$@" --kv "$KV"; fi
    if [ -n "$GPUS" ]; then set -- "$@" --gpus "$GPUS"; fi
    if [ -n "$GPU" ]; then set -- "$@" --gpu "$GPU"; fi
    if [ -n "$LAYER_SPLIT" ]; then set -- "$@" --layer-split "$LAYER_SPLIT"; fi
    if [ -n "$GGUF_DIR" ]; then set -- "$@" --gguf-dir "$GGUF_DIR"; fi
    if [ -n "$RESIDENT_BUDGET_GIB" ]; then set -- "$@" --resident-budget-gib "$RESIDENT_BUDGET_GIB"; fi
    if [ -n "$KV_STREAMING" ]; then set -- "$@" --kv-streaming "$KV_STREAMING"; fi
    .venv/bin/python setup.py --setup --yes "$@"
    # setup.py writes the config next to itself; the copy on the volume is the one that counts (the
    # upstream entry point does the same after its own pass). The relative path is the container's
    # working directory, the one this entry point's exec relies on.
    if [ -e "./strata-${prefix}${tag}.json" ]; then
        cmp -s "./strata-${prefix}${tag}.json" "$cfg" || cp -f "./strata-${prefix}${tag}.json" "$cfg"
    fi
}

if [ -n "${VISION_DEVICE:-}" ]; then
    if [ ! -f "$cfg" ] || ! has_vision_section "$cfg"; then
        run_setup_pass "$cfg has no vision section"
        if ! has_vision_section "$cfg"; then
            echo "strata overlay: $cfg still has no vision section after the setup pass (VISION says $VISION);" >&2
            echo "  the container states a card for an encoder vision is told not to run" >&2
            exit 1
        fi
    fi

    python3 - "$cfg" "$VISION_DEVICE" <<'PY'
import json, os, sys

path, device = sys.argv[1], int(sys.argv[2])
with open(path, encoding='utf-8-sig') as handle:
    cfg = json.load(handle)
vision = cfg.get('vision')
if not isinstance(vision, dict):
    print('strata overlay: %s has no vision section; VISION_DEVICE=%d is nothing to place'
          % (path, device), flush=True)
elif vision.get('cuda_device') == device:
    print('strata overlay: image encoder already on CUDA device %d' % device, flush=True)
else:
    vision['cuda_device'] = device
    # setup.py carries over every key it is not asked about, so this survives a reinstall; it is
    # written on every start regardless, because the profile is the authority and not the file.
    tmp = path + '.placement.tmp'
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(cfg, handle, indent=1)
        handle.write('\n')
    os.replace(tmp, path)
    print('strata overlay: image encoder on CUDA device %d (%s)' % (device, path), flush=True)
PY
fi

# An unset SAMPLING is no statement at all: nothing is written and nothing is removed, so a profile
# that has never heard of this key boots exactly as it did before the key existed. "off" is a
# statement, and honouring it means removing a block the profile no longer states -- that is what
# makes going back to greedy a profile change and not a hand edit inside the workspace, which the
# workspace would otherwise carry into a text-only image that cannot take it back out.
if [ "$SAMPLING" = off ]; then
    if [ -f "$cfg" ]; then
        python3 - "$cfg" <<'PY'
import json, os, sys

path = sys.argv[1]
with open(path, encoding='utf-8-sig') as handle:
    cfg = json.load(handle)
if 'sampling' not in cfg:
    print('strata overlay: no sampling block to remove; the engine keeps its own default', flush=True)
else:
    block = cfg.pop('sampling')
    tmp = path + '.sampling.tmp'
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(cfg, handle, indent=1)
        handle.write('\n')
    os.replace(tmp, path)
    print('strata overlay: removed the sampling block the profile does not state (%s); requests that '
          'send none decode greedily again' % json.dumps(block, sort_keys=True), flush=True)
PY
    fi
elif [ -n "$SAMPLING" ]; then
    if [ ! -f "$cfg" ]; then
        # A fresh workspace: the upstream entry point is about to run its own setup pass, and that
        # pass writes a config out of nothing, so a block written here first would be overwritten.
        run_setup_pass "$cfg does not exist and a sampling preset has to go in it"
        if [ ! -f "$cfg" ]; then
            echo "strata overlay: the setup pass left no $cfg to write a sampling preset into" >&2
            exit 1
        fi
    fi

    python3 - "$cfg" "$SAMPLING" <<'PY'
import json, os, sys

# The two sets the Qwen3.8-Flash-Next card states, by name. A request that brings its own sampling
# still wins over this: the block is only what a request that sends none gets.
PRESETS = {
    'thinking': {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0},
    'instruct': {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 1.5},
}

path, preset = sys.argv[1], sys.argv[2]
want = PRESETS.get(preset)
if want is None:
    print('strata overlay: SAMPLING=%s is not a preset this entry point knows (%s)'
          % (preset, ' or '.join(sorted(PRESETS))), file=sys.stderr, flush=True)
    sys.exit(1)
with open(path, encoding='utf-8-sig') as handle:
    cfg = json.load(handle)
have = cfg.get('sampling')
if have == want:
    print('strata overlay: sampling already the %s preset' % preset, flush=True)
else:
    if isinstance(have, dict) and have:
        print('strata overlay: replacing a sampling block the profile does not state: %s'
              % json.dumps(have, sort_keys=True), flush=True)
    cfg['sampling'] = dict(want)
    # Written on every start, because the profile is the authority and not the file; every other key
    # goes back exactly as it was read, and setup.py carries this one over on a later install.
    tmp = path + '.sampling.tmp'
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(cfg, handle, indent=1)
        handle.write('\n')
    os.replace(tmp, path)
    print('strata overlay: sampling preset %s (%s) for requests that send none (%s)'
          % (preset, ', '.join('%s=%s' % item for item in sorted(want.items())), path), flush=True)
PY
fi

# The same rule as the sampling block above, and for the same reason: the run config lives in the
# workspace, which outlives every configuration, so going back has to be a profile change rather than
# a hand edit inside a directory a later image cannot take back out of. `off` therefore removes the
# key instead of writing false, which leaves the file exactly as setup.py wrote it.
if [ "$TOOL_CALL_RECOVERY" = off ]; then
    if [ -f "$cfg" ]; then
        python3 - "$cfg" <<'PY'
import json, os, sys

path = sys.argv[1]
with open(path, encoding='utf-8-sig') as handle:
    cfg = json.load(handle)
if 'tool_call_recovery' not in cfg:
    print('strata overlay: no tool_call_recovery key to remove; the engine keeps its own default',
          flush=True)
else:
    value = cfg.pop('tool_call_recovery')
    tmp = path + '.recovery.tmp'
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(cfg, handle, indent=1)
        handle.write('\n')
    os.replace(tmp, path)
    print('strata overlay: removed tool_call_recovery=%s the profile does not state; only the '
          'template\'s own tool-call form is a call again' % json.dumps(value), flush=True)
PY
    fi
elif [ -n "$TOOL_CALL_RECOVERY" ]; then
    if [ ! -f "$cfg" ]; then
        run_setup_pass "$cfg does not exist and tool_call_recovery has to go in it"
        if [ ! -f "$cfg" ]; then
            echo "strata overlay: the setup pass left no $cfg to write tool_call_recovery into" >&2
            exit 1
        fi
    fi

    python3 - "$cfg" "$TOOL_CALL_RECOVERY" <<'PY'
import json, os, sys

# Two words and one fact behind them: serve/server.py accepts a JSON boolean for this key and raises
# SystemExit on anything else, so the value this writes is never a string, never a number, and never
# a word the engine has not agreed to read.
VALUES = {'on': True, 'off': False}

path, stated = sys.argv[1], sys.argv[2]
if stated not in VALUES:
    print('strata overlay: TOOL_CALL_RECOVERY=%s is not a value this entry point knows (%s)'
          % (stated, ' or '.join(sorted(VALUES))), file=sys.stderr, flush=True)
    sys.exit(1)
want = VALUES[stated]
with open(path, encoding='utf-8-sig') as handle:
    cfg = json.load(handle)
have = cfg.get('tool_call_recovery')
# Identity and not equality: a hand-written 1 in the workspace equals True in this language and would
# be left in place, and the engine starts on a JSON boolean or raises SystemExit.
if have is want:
    print('strata overlay: tool_call_recovery already %s' % stated, flush=True)
else:
    if have is not None:
        print('strata overlay: replacing a tool_call_recovery the profile does not state: %s'
              % json.dumps(have), flush=True)
    cfg['tool_call_recovery'] = want
    tmp = path + '.recovery.tmp'
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(cfg, handle, indent=1)
        handle.write('\n')
    os.replace(tmp, path)
    print('strata overlay: tool_call_recovery %s (%s): a declared tool\'s call written beside the '
          'template\'s form is read as the call (%s)' % (stated, json.dumps(want), path), flush=True)
PY
fi

exec ./docker-entrypoint.upstream.sh "$@"

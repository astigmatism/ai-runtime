#!/bin/sh
# This house's overlay on Strata's own entry point, not an upstream file.
#
# Twelve facts about this container have to reach the engine's own run config. Six of them are not
# arguments the engine takes; the other six are, but they go in the run config's "args" list, which
# setup.py writes and offers no flag to extend.
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
# 5. Whether a conversation that came back is read again from the start. CONVERSATION_CACHE_MIB is a
#    RAM budget for parked conversations; the engine's own default is 0, which means every returning
#    conversation is re-prefilled. Unlike facts 1-4 this IS an engine argument, but it goes in the run
#    config's "args" list, and setup.py only ever prints a tip about it (its gate, `ram - model_ram_gb
#    >= 24` GiB, this box passes by 44 GiB) without ever writing it. A profile therefore states the
#    budget and the RAM floor it will not park beneath -- CONVERSATION_CACHE_MIN_FREE_MIB, required,
#    because this workspace keeps a 32.93 GiB page-locked expert complement and a cache that outbids
#    it pays in page faults on the serving path. CONVERSATION_CACHE_SLOTS and CONVERSATION_CACHE_MIN_TOKENS
#    may go unstated and are then written at the engine's own defaults (4 and 0), because a stated
#    budget writes all four arguments and a number an older profile wrote must not survive the switch
#    in silence; CONVERSATION_CACHE_MIN_TOKENS also keeps a short side request from taking the slot a
#    127,000-token conversation needs. "off" takes all four flags back out, which is the engine's
#    default.
#    Only exact token prefixes are reused, so this buys the conversation that was evicted and restored,
#    and nothing at all for one whose prompt stops matching partway in; the first case is measured
#    here at 6.3% of prompt time and the second at 25.5%, and the two are told apart by /metrics'
#    "conversation_cache" counters, not by hope.
# 6. Which file the n-gram/PLE table is read from. PLE_GGUF is a path, and like fact 5 it is an engine
#    argument with no setup.py flag: setup.py passes --ple-gguf by itself only at two or fewer shards
#    (its own `len(shards) <= 2` test), so this three-shard file has the table read out of shard 2, where
#    it sits at the quantization Unsloth chose for it (IQ4_NL here, and the engine's PLE row reader also
#    takes Q8_0 and FP8 rows). Pointing the key at a wider table is the one accuracy purchase on this box
#    that does not go through the RAM-to-GPU expert streaming that cost UD-Q4_K_XL 7-15% of decode and
#    STRATA_HC_Q8 15% of prefill, because the table is read off the SSD by the CPU path at ~94,000 rows
#    per second and never streamed to a card. The file is checked to exist before it is written: the
#    engine's own help says that without this key "LAYER 1's PLE IS SILENTLY SKIPPED", and a silently
#    missing layer is the failure mode this overlay exists to refuse.
# 7. How long a session it holds. The profile states it as CONTEXT, and it is an engine argument --
#    --max-context, in that same args list -- but setup.py only ever uses it during the setup pass,
#    which the upstream entry point runs once per workspace (fact 1). So raising a profile from 160K
#    to a wider window skipped the pass, left the --max-context the workspace was born with sitting in
#    the args list, and booted an engine on the old window; the switch then verified /slots against the
#    window the profile had promised and refused it with reason 'slot context' after the drain, and the
#    restoration brought the old window back. That is why the window is written into the args list on
#    every start from the stated CONTEXT rather than being left to the file: for the other facts "the
#    profile is the authority and not the file" is a matter of taste, and here it is the only thing that
#    makes a session size a deployable setting at all. A CONTEXT that is not a positive number of tokens
#    fails the boot before anything is written. No ceiling is repeated here: the runtime refuses a
#    window the router cannot carry when it renders, before anything is drained, and the engine reports
#    the window it starts on or does not start.
# 8. Which position ring the window is read through. The profile states ROPE_SCALING (yarn or linear),
#    ROPE_SCALE and, for yarn, YARN_ORIG_CTX, and they are engine arguments -- --rope-scaling,
#    --rope-scale, --yarn-orig-ctx -- fixed at startup because cached K is post-RoPE. The pack is
#    trained to 262,144 with no scaling keys of its own, so a window wider than that is interpolation
#    and a window at or under it is native. Unusually, silence here is a statement: a profile that
#    names no ring promises the native one, so this fact DROPS any rope block the file carries when
#    the profile states nothing -- the one fact where leaving the file alone would be a lie, because
#    a 262,144 window left reading through a stranger's factor 2.0 boots, serves, and answers through
#    positions the profile never promised, and the switch has no /slots number to catch it on. yarn
#    demands a --yarn-orig-ctx and a window above it (the engine refuses the pair otherwise); a scale
#    that is not a positive float fails the boot before anything is written. The measurements behind
#    shipping any factor at all are in docs/deployment.md, "Beyond the ladder" and the factor table.
#  9. How much of each text card the expert cache may not size itself into. VRAM_RESERVE_MIB is MiB, and
#    like facts 5-8 it is an engine argument (--vram-reserve-mib) with no setup.py flag, so it lands in
#    that args list. It reaches this entry point for the first time on 2026-10-20 because it has been a
#    hand edit inside the workspace since 2026-10-12 -- one of the two docs/deployment.md names as
#    undeclared, surviving only because nothing has reinstalled the workspace. What it costs is slots and
#    not bytes: the reserve is not held memory, it is room the engine declines to take, and at the 2.31
#    MiB each cached expert measured 2026-10-19 that is ~885 slots a card. 2,048 MiB was written for four
#    cards (5.0% of their VRAM); the same number on the two-card pair the trial runs on is 10.0% of that
#    pair, which is why a reserve calibrated for one layout is not carried unchanged into another. How
#    many slots it is actually worth is printed by the boot ("expert cache auto: <free> free, <n> MiB
#    reserved ... -> N slots") and read from there, never computed here. Unstated leaves the file alone.
#    0 is a value (reserve nothing), so it is accepted; the word that takes the flag out is "off".
# 10. Which direction is subtracted from the residual stream, how hard, and between which layers. Four
#    arguments that are one fact: --control-vector-scaled <file>:<scale>, --control-vector-layer-range
#    <low> <high>, --cvec-mode, --cvec-dir. CONTROL_VECTOR is a path; CONTROL_VECTOR_SCALE defaults to
#    1.0, the strength this box has served since 2026-10-15; CONTROL_VECTOR_LAYERS is "low:high". The
#    mode and direction are deliberately NOT configurable here: project/per-layer are properties of the
#    vector file this house fitted, not of the deployment, and a profile free to get them wrong is a
#    profile that can subtract a fitted direction in a mode the file was never fitted for. This block is
#    written by its own function rather than through edit_args for one mechanical reason worth knowing
#    before editing either: --control-vector-layer-range takes TWO values and edit_args' strip walks
#    one, so a stale range would leave its second number behind as an orphaned token that the engine
#    then reads as the following flag's value. Like fact 9 this has been a hand edit since 2026-10-15,
#    and unlike fact 9 it is on the two-card trial's critical path for a reason that is not
#    tidiness: the probe is meant to measure the configuration in service minus the card count, and a
#    vector that rides along in a workspace instead of a profile makes every number that comparison
#    prints a different configuration's. "off" takes all four out.
# 11. Where the engine's own log goes. STRATA_LOG is a path, and it is neither of the two kinds above: it
#    is a top-level key of the run config, which is how the workspace has carried it since 2026-10-12
#    (the log file's own first line records the preflight that made it). It belongs in this list because
#    nearly every instrument this house reads -- the expert-cache hit rate, the "resident RAM" line, the
#    slots the boot sized the cache to, the placement search's predicted window cost -- is read out of
#    that file, and a durable log whose path lives only inside a directory the controller cannot see is
#    an instrument that goes missing on the day it is needed. Unstated leaves the file alone, so what is
#    serving keeps what it was set up with; "off" takes the key out, which puts the log where the engine
#    puts it.
#
# So: when the container is told the encoder's card, make sure the config it is about to start
# carries a vision section, and write the card into it. When it is told a sampling preset, make sure
# that config exists at all, and write the block into it. When it is told a session window, write that
# window into the args list the engine reads its context from. Where the section or the file is missing,
# the setup pass that would create it is the one the upstream entry point would have run for this
# exact environment, so this runs it itself (the function keeps its argument list out of this entry
# point's own); a fresh workspace pays that pass once here instead of twice later, and a workspace
# already set up skips straight to the patch. A pass that still leaves no section (VISION says no)
# is a misconfigured card, and a misconfigured card fails the boot instead of serving text-only with
# a log gap. Then hand over to the upstream entry point unchanged.
set -e

# Validate before setup or any workspace edits, including on a fresh workspace.
case "${REASONING_CLOSE_RETRY:-off}" in
    on|off) ;;
    *) echo "strata overlay: REASONING_CLOSE_RETRY must be on or off" >&2; exit 1 ;;
esac

STRATA_DATA="${STRATA_DATA:-/data}"
FAMILY="${FAMILY:-qwen}"
MODEL="${MODEL:-IQ2_XS}"
# Kept twice on purpose: what the environment actually stated, and what the setup pass is handed when
# nothing did. Fact 7 is written only for the first of those, so a container started by hand over an
# existing workspace keeps the window it was set up with instead of being quietly resized to 32768.
CONTEXT_STATED="${CONTEXT:-}"
CONTEXT="${CONTEXT_STATED:-32768}"
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
CONVERSATION_CACHE_MIB="${CONVERSATION_CACHE_MIB:-}"
CONVERSATION_CACHE_MIN_FREE_MIB="${CONVERSATION_CACHE_MIN_FREE_MIB:-}"
CONVERSATION_CACHE_SLOTS="${CONVERSATION_CACHE_SLOTS:-}"
CONVERSATION_CACHE_MIN_TOKENS="${CONVERSATION_CACHE_MIN_TOKENS:-}"
ROPE_SCALING="${ROPE_SCALING:-}"
ROPE_SCALE="${ROPE_SCALE:-}"
YARN_ORIG_CTX="${YARN_ORIG_CTX:-}"
PLE_GGUF="${PLE_GGUF:-}"
# Facts 9-11, the three settings that reached this container as hand edits inside the workspace.
VRAM_RESERVE_MIB="${VRAM_RESERVE_MIB:-}"
CONTROL_VECTOR="${CONTROL_VECTOR:-}"
CONTROL_VECTOR_SCALE="${CONTROL_VECTOR_SCALE:-}"
CONTROL_VECTOR_LAYERS="${CONTROL_VECTOR_LAYERS:-}"
STRATA_LOG="${STRATA_LOG:-}"

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

# 12. One continuation when a stop leaves only reasoning. Never promote reasoning to an answer.
# Unlike tool-call recovery, absence removes the key: switching to an unstated profile must disable
# retry even when its persistent workspace was last used by an opted-in profile.
if [ "${REASONING_CLOSE_RETRY:-off}" = on ] && [ ! -f "$cfg" ]; then
    run_setup_pass "$cfg does not exist and reasoning_close_retry has to go in it"
    if [ ! -f "$cfg" ]; then
        echo "strata overlay: the setup pass left no $cfg to write reasoning_close_retry into" >&2
        exit 1
    fi
fi
if [ -f "$cfg" ]; then
    python3 - "$cfg" "${REASONING_CLOSE_RETRY:-off}" <<'PY'
import json, os, sys

path, stated = sys.argv[1:]
with open(path, encoding='utf-8-sig') as handle:
    cfg = json.load(handle)
if stated == 'on':
    changed = cfg.get('reasoning_close_retry') is not True
    cfg['reasoning_close_retry'] = True
else:
    changed = 'reasoning_close_retry' in cfg
    cfg.pop('reasoning_close_retry', None)
if changed:
    tmp = path + '.reasoning-close-retry.tmp'
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(cfg, handle, indent=1)
        handle.write('\n')
    os.replace(tmp, path)
print('strata overlay: reasoning_close_retry %s (%s)' % (stated, path), flush=True)
PY
fi

# The argv analogue of the blocks above. The two facts that follow are engine arguments, and the
# engine reads them out of the run config's "args" list, which is a flat argv array and not a key:
# setup.py writes it, offers no flag to extend it, and (for the cache) only prints a tip that the
# operator has to act on by hand. So this is where a profile's statement has to land. It goes through
# the same ceremony as the blocks above -- read the file, change one thing, write it back through a
# temporary file, say out loud what was done -- because an argument the engine refuses is a boot that
# does not happen, and for an exclusive resident that is nothing serving.
edit_args() {
    # $1 = the temporary file's suffix, $2 = "set" or "drop", then the flags to take out or the flag
    # and value pairs to put in. $cfg comes from the enclosing scope, and the shift below reaches only
    # this function's own positional parameters, not the argument list this entry point execs with.
    ea_suffix="$1" ea_op="$2"
    shift 2
    python3 - "$cfg" "$ea_suffix" "$ea_op" "$@" <<'PY'
import json, os, sys

path, suffix, op = sys.argv[1], sys.argv[2], sys.argv[3]
rest = sys.argv[4:]
with open(path, encoding='utf-8-sig') as handle:
    cfg = json.load(handle)
args = cfg.get('args')
if op == 'drop' and not isinstance(args, list):
    # A file that carries no argv carries none of these flags either: dropping from it is a
    # no-op, not a failed boot. (Writing INTO such a file stays a refusal below -- inventing
    # the engine's argv list here would say more about the engine's defaults than we know.)
    print('strata overlay: %s carries no args list; none of %s is in force'
          % (path, ', '.join(rest)), flush=True)
    sys.exit(0)
if not isinstance(args, list):
    print('strata overlay: %s has no args list to write %s into' % (path, suffix), file=sys.stderr,
          flush=True)
    sys.exit(1)


def strip(current, flags):
    out, i = [], 0
    while i < len(current):
        if current[i] in flags:
            i += 2
            continue
        out.append(current[i])
        i += 1
    return out


if op == 'drop':
    present = [flag for flag in rest if flag in args]
    if not present:
        print('strata overlay: none of %s is in the args list; the engine keeps its own default'
              % ', '.join(rest), flush=True)
        sys.exit(0)
    want = strip(args, set(rest))
    action = 'removed %s the profile does not state' % ', '.join(present)
else:
    pairs = [(rest[i], rest[i + 1]) for i in range(0, len(rest), 2)]
    flags = {flag for flag, _ in pairs}
    have = [(args[i], args[i + 1]) for i in range(len(args) - 1) if args[i] in flags]
    if sorted(have) == sorted(pairs):
        print('strata overlay: %s already in the args list (%s)'
              % (', '.join(sorted(flags)), ', '.join('%s %s' % pair for pair in pairs)), flush=True)
        sys.exit(0)
    want = strip(args, flags) + [value for pair in pairs for value in pair]
    action = 'wrote %s' % ', '.join('%s %s' % pair for pair in pairs)

cfg['args'] = want
# Written on every change and never in place: the profile is the authority and not the file, and every
# other entry in the list goes back exactly as it was read.
tmp = path + '.' + suffix + '.tmp'
with open(tmp, 'w', encoding='utf-8') as handle:
    json.dump(cfg, handle, indent=1)
    handle.write('\n')
os.replace(tmp, path)
print('strata overlay: %s (%s)' % (action, path), flush=True)
PY
}

positive() {
    # $1 = the name to blame in the message, $2 = the value. Digits and more than zero, because every
    # one of these is an argument the engine parses as an int and refuses to start on otherwise.
    case "$2" in
        '' | *[!0-9]*) return 1 ;;
    esac
    [ "$2" -gt 0 ]
}

digits() {
    case "$1" in
        '' | *[!0-9]*) return 1 ;;
    esac
    return 0
}

apply_max_context() {
    # Fact 7. Unstated is not a statement: a container started by hand, over a workspace somebody set
    # up, keeps the window that workspace carries. There is no "off" for this one either -- the window
    # is the one setting the engine has to have an opinion about, and the engine's own default (4096)
    # is not a session this house would notice.
    if [ -z "$CONTEXT_STATED" ]; then
        return 0
    fi
    if ! positive CONTEXT "$CONTEXT_STATED"; then
        echo "strata overlay: CONTEXT=$CONTEXT_STATED is not a number of tokens, and it is not left to" >&2
        echo "  the setup pass to discover. A config that quietly keeps the window the workspace was" >&2
        echo "  born with boots fine, serves, and is refused by the switch for reporting a context other" >&2
        echo "  than the one the profile promised -- after the drain. Better a boot that does not happen." >&2
        return 1
    fi
    if [ ! -f "$cfg" ]; then
        run_setup_pass "$cfg does not exist and the session window has to be written into it"
        if [ ! -f "$cfg" ]; then
            echo "strata overlay: the setup pass left no $cfg to write the session window into" >&2
            return 1
        fi
    fi
    # Written on every start, and not only when a setup pass ran. This is the one fact where the file
    # being authoritative is not a preference: the controller verifies the window the engine reports
    # against the number it rendered into the catalog, so a window that is only right on the first boot
    # of a workspace is a profile that cannot be resized at all.
    edit_args ctx set --max-context "$CONTEXT_STATED"
}

apply_conversation_cache() {
    if [ "$CONVERSATION_CACHE_MIB" = off ]; then
        if [ -f "$cfg" ]; then
            edit_args convcache drop --conversation-cache-mib --conversation-cache-slots \
                --conversation-cache-min-free-mib --conversation-cache-min-tokens
        fi
        return 0
    fi
    if [ -z "$CONVERSATION_CACHE_MIB" ]; then
        # No statement at all: a profile that has never heard of this key boots as it did before it.
        return 0
    fi
    if ! positive CONVERSATION_CACHE_MIB "$CONVERSATION_CACHE_MIB"; then
        echo "strata overlay: CONVERSATION_CACHE_MIB=$CONVERSATION_CACHE_MIB is not an amount of MiB" >&2
        return 1
    fi
    if ! positive CONVERSATION_CACHE_MIN_FREE_MIB "$CONVERSATION_CACHE_MIN_FREE_MIB"; then
        echo "strata overlay: CONVERSATION_CACHE_MIB=$CONVERSATION_CACHE_MIB states a budget and no" >&2
        echo "  CONVERSATION_CACHE_MIN_FREE_MIB to park nothing beneath. It is not left to the engine's" >&2
        echo "  default: this workspace holds a 32.93 GiB page-locked expert complement, and a parked" >&2
        echo "  conversation that outbids it pays for it on the serving path." >&2
        return 1
    fi
    if [ ! -f "$cfg" ]; then
        run_setup_pass "$cfg does not exist and a conversation cache has to go in it"
        if [ ! -f "$cfg" ]; then
            echo "strata overlay: the setup pass left no $cfg to write a conversation cache into" >&2
            return 1
        fi
    fi
    set -- --conversation-cache-mib "$CONVERSATION_CACHE_MIB" \
        --conversation-cache-min-free-mib "$CONVERSATION_CACHE_MIN_FREE_MIB"
    # A stated budget writes all four arguments, and the two that may go unstated are written at the
    # engine's own defaults rather than left out. Left out, a number an older profile wrote would
    # survive the switch in silence -- and the profile is the authority and not the file.
    if [ -n "$CONVERSATION_CACHE_SLOTS" ]; then
        if ! positive CONVERSATION_CACHE_SLOTS "$CONVERSATION_CACHE_SLOTS"; then
            echo "strata overlay: CONVERSATION_CACHE_SLOTS=$CONVERSATION_CACHE_SLOTS is not a number" >&2
            return 1
        fi
    else
        CONVERSATION_CACHE_SLOTS=4
    fi
    set -- "$@" --conversation-cache-slots "$CONVERSATION_CACHE_SLOTS"
    if [ -n "$CONVERSATION_CACHE_MIN_TOKENS" ]; then
        # Zero is a statement here (the engine's own default: park any conversation), so digits and not
        # positive -- unlike the budget, where zero means the cache is off and the profile would want
        # "off" to say so.
        if ! digits "$CONVERSATION_CACHE_MIN_TOKENS"; then
            echo "strata overlay: CONVERSATION_CACHE_MIN_TOKENS=$CONVERSATION_CACHE_MIN_TOKENS is not a" >&2
            echo "  number of tokens (0 parks any conversation, which is the engine's default)" >&2
            return 1
        fi
    else
        CONVERSATION_CACHE_MIN_TOKENS=0
    fi
    set -- "$@" --conversation-cache-min-tokens "$CONVERSATION_CACHE_MIN_TOKENS"
    edit_args convcache set "$@"
}

apply_ple_table() {
    if [ "$PLE_GGUF" = off ]; then
        if [ -f "$cfg" ]; then
            edit_args ple drop --ple-gguf
        fi
        return 0
    fi
    if [ -z "$PLE_GGUF" ]; then
        # Nothing stated: at three shards setup.py reads the table out of the shard Unsloth put it in,
        # which is what every boot of this workspace has done.
        return 0
    fi
    if [ ! -f "$PLE_GGUF" ]; then
        echo "strata overlay: PLE_GGUF=$PLE_GGUF is not a file this container can read." >&2
        echo "  Refusing rather than booting without it: the engine's own help says that a PLE table it" >&2
        echo "  was not given leaves layer 1's PLE silently skipped, and a silent missing layer is worth" >&2
        echo "  more than a boot that looks fine." >&2
        return 1
    fi
    if [ ! -f "$cfg" ]; then
        run_setup_pass "$cfg does not exist and a PLE table has to be named in it"
        if [ ! -f "$cfg" ]; then
            echo "strata overlay: the setup pass left no $cfg to name a PLE table in" >&2
            return 1
        fi
    fi
    edit_args ple set --ple-gguf "$PLE_GGUF"
}

positive_float() {
    # $1 = the name to blame in the message, $2 = the value. Digits with at most one point and more
    # than zero: the engine parses this as a float, and a rope scale it cannot parse is a boot that
    # does not happen -- which is the whole house style for arguments the engine refuses.
    case "$2" in
        '' | *[!0-9.]* | .* | *. | *..*) return 1 ;;
    esac
    python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) > 0 else 1)' "$2" 2>/dev/null
}

apply_rope_scaling() {
    # Fact 8. What positions the engine reads its window with. The pack is trained to 262,144 and
    # carries no rope scaling keys, so any window wider than that is YaRN interpolation and every
    # narrower promise is the native geometry -- which is why silence here STRIPS instead of leaving
    # the file alone: a rope block a wider profile wrote must not survive the switch back in silence,
    # because a 262,144 window read through factor 2.0 is a session that boots, serves, and answers
    # through the wrong position ring. The profile is the authority and not the file; for geometry
    # that has always been true, this fact just says so out loud.
    #
    # Measured on this box on 2026-10-18, same pack, same engine: factor 2.0 to 524,288 boots in
    # 100 s like the trained window does, decodes within noise of it, and answers a 493K-token tail
    # question identically in five rounds where the trained edge answers its own edge half the time.
    # The factor ladder (docs/deployment.md) drew the shipping line at 2.0; a profile that states a
    # steeper one is borrowing against that ladder's tail, and should say so in its variant_label.
    case "$ROPE_SCALING" in
        '' | none | off)
            if [ -f "$cfg" ]; then
                edit_args rope drop --rope-scaling --rope-scale --yarn-orig-ctx
            fi
            return 0
            ;;
        yarn | linear)
            ;;
        *)
            echo "strata overlay: ROPE_SCALING=$ROPE_SCALING is not one of yarn, linear, none. A" >&2
            echo "  geometry the engine has never heard of is a crash loop for an exclusive resident." >&2
            return 1
            ;;
    esac
    if ! positive_float ROPE_SCALE "$ROPE_SCALE"; then
        echo "strata overlay: ROPE_SCALING=$ROPE_SCALING states no positive ROPE_SCALE to go with it." >&2
        echo "  The rope scale is the statement; the scaling name only says which interpolation reads" >&2
        echo "  it, and leaving it to the engine's default is how a window gets a ring nobody named." >&2
        return 1
    fi
    set -- --rope-scaling "$ROPE_SCALING" --rope-scale "$ROPE_SCALE"
    if [ "$ROPE_SCALING" = yarn ]; then
        if ! positive YARN_ORIG_CTX "$YARN_ORIG_CTX"; then
            echo "strata overlay: ROPE_SCALING=yarn states no positive YARN_ORIG_CTX: YaRN needs to" >&2
            echo "  know which trained window it is stretching, and --yarn-orig-ctx's own default is" >&2
            echo "  262144 only for this generation's packs -- a number to be stated, not assumed." >&2
            return 1
        fi
        if positive CONTEXT "$CONTEXT_STATED" && [ "$CONTEXT_STATED" -le "$YARN_ORIG_CTX" ]; then
            echo "strata overlay: CONTEXT_STATED=$CONTEXT_STATED with YARN_ORIG_CTX=$YARN_ORIG_CTX:" >&2
            echo "  the engine refuses a YaRN window that does not exceed the trained one it scales." >&2
            return 1
        fi
        set -- "$@" --yarn-orig-ctx "$YARN_ORIG_CTX"
    fi
    if [ ! -f "$cfg" ]; then
        run_setup_pass "$cfg does not exist and a position ring has to be written into it"
        if [ ! -f "$cfg" ]; then
            echo "strata overlay: the setup pass left no $cfg to write the position ring into" >&2
            return 1
        fi
    fi
    edit_args rope set "$@"
}

apply_vram_reserve() {
    # Fact 9. The reserve is room the engine declines to size the expert cache into, so what it buys is
    # latency headroom and what it costs is cached experts. Both sides of that are printed by the boot
    # and neither is computed here, which is why this fact writes a number and does not have an opinion
    # about it. 0 is accepted because it is a value (reserve nothing); "off" is the word for the flag
    # not being in force at all, which is the engine's own default.
    if [ "$VRAM_RESERVE_MIB" = off ]; then
        if [ -f "$cfg" ]; then
            edit_args reserve drop --vram-reserve-mib
        fi
        return 0
    fi
    if [ -z "$VRAM_RESERVE_MIB" ]; then
        return 0
    fi
    if ! digits "$VRAM_RESERVE_MIB"; then
        echo "strata overlay: VRAM_RESERVE_MIB=$VRAM_RESERVE_MIB is not a number of MiB. The engine" >&2
        echo "  parses this as an int and does not start on one it cannot read, and for an exclusive" >&2
        echo "  resident a boot that does not happen is a cheaper outcome than a crash loop." >&2
        return 1
    fi
    if [ ! -f "$cfg" ]; then
        run_setup_pass "$cfg does not exist and a VRAM reserve has to be written into it"
        if [ ! -f "$cfg" ]; then
            echo "strata overlay: the setup pass left no $cfg to write the VRAM reserve into" >&2
            return 1
        fi
    fi
    edit_args reserve set --vram-reserve-mib "$VRAM_RESERVE_MIB"
}

edit_cvec() {
    # Fact 10's writer. It is not edit_args for one reason: --control-vector-layer-range takes TWO
    # values, and edit_args' strip walks one -- a stale range would leave its second number behind as
    # an orphaned token, which the engine reads as the following flag's value. The mode and direction
    # are in the table below and nowhere else, because they are properties of the vector file.
    ec_op="$1"
    shift
    python3 - "$cfg" "$ec_op" "$@" <<'PY'
import json, os, sys

ARITY = {'--control-vector-scaled': 1, '--control-vector-layer-range': 2,
         '--cvec-mode': 1, '--cvec-dir': 1}
MODE = ('--cvec-mode', ['project'])
DIRECTION = ('--cvec-dir', ['per-layer'])

path, op = sys.argv[1], sys.argv[2]
rest = sys.argv[3:]
with open(path, encoding='utf-8-sig') as handle:
    cfg = json.load(handle)
args = cfg.get('args')
if not isinstance(args, list):
    if op == 'drop':
        print('strata overlay: %s carries no args list; no control vector is in force' % path, flush=True)
        sys.exit(0)
    print('strata overlay: %s has no args list to write a control vector into' % path, file=sys.stderr,
          flush=True)
    sys.exit(1)


def split(current):
    # Walk the argv with the arity table, so a two-value flag is taken apart as two values.
    plain, found = [], {}
    i = 0
    while i < len(current):
        token = current[i]
        if token in ARITY:
            found[token] = current[i + 1:i + 1 + ARITY[token]]
            i += 1 + ARITY[token]
            continue
        plain.append(token)
        i += 1
    return plain, found


plain, have = split(args)
if op == 'drop':
    if not have:
        print('strata overlay: no control vector is in the args list; the engine subtracts nothing',
              flush=True)
        sys.exit(0)
    want = plain
    action = 'removed the control vector the profile does not state: %s' % ', '.join(
        '%s %s' % (flag, ' '.join(values)) for flag, values in sorted(have.items()))
else:
    scaled, low, high = rest[0], rest[1], rest[2]
    want_block = [('--control-vector-scaled', [scaled]), ('--control-vector-layer-range', [low, high]),
                  MODE, DIRECTION]
    if all(have.get(flag) == values for flag, values in want_block):
        print('strata overlay: control vector already in the args list (%s %s, layers %s-%s, %s %s)'
              % (scaled, ''.join(have.get('--control-vector-scaled', [''])[1:]), low, high,
                 MODE[1][0], DIRECTION[1][0]), flush=True)
        sys.exit(0)
    want = plain + [value for pair in want_block for value in (pair[0],) + tuple(pair[1])]
    action = ('wrote the control vector %s over layers %s-%s in %s mode, %s'
              % (scaled, low, high, MODE[1][0], DIRECTION[1][0]))

cfg['args'] = want
tmp = path + '.cvec.tmp'
with open(tmp, 'w', encoding='utf-8') as handle:
    json.dump(cfg, handle, indent=1)
    handle.write('\n')
os.replace(tmp, path)
print('strata overlay: %s (%s)' % (action, path), flush=True)
PY
}

apply_control_vector() {
    # Fact 10. One fact in four arguments, and it is all-or-nothing on purpose: a vector named with no
    # layer range is a direction subtracted from whichever layers the engine feels like, and the range
    # this box serves (4-44) is where the fitted direction lives.
    case "$CONTROL_VECTOR" in
        off)
            if [ -f "$cfg" ]; then
                edit_cvec drop
            fi
            return 0
            ;;
        '')
            return 0
            ;;
    esac
    if [ ! -f "$CONTROL_VECTOR" ]; then
        echo "strata overlay: CONTROL_VECTOR=$CONTROL_VECTOR is not a file this container can read." >&2
        echo "  Refusing rather than booting without it: a profile that names a direction and serves" >&2
        echo "  without it is a configuration whose measurements are not its own, and the absence is" >&2
        echo "  invisible in every log the engine writes." >&2
        return 1
    fi
    if [ -z "$CONTROL_VECTOR_SCALE" ]; then
        # 1.0 is the strength this box has served with since 2026-10-15, and the fitted length of the
        # vector is already that scale's unit -- so the default is the number, not an assumption about
        # a strength somebody meant to state.
        CONTROL_VECTOR_SCALE=1.0
    fi
    if ! positive_float CONTROL_VECTOR_SCALE "$CONTROL_VECTOR_SCALE"; then
        echo "strata overlay: CONTROL_VECTOR_SCALE=$CONTROL_VECTOR_SCALE is not a positive number. The" >&2
        echo "  engine parses it as a float on the same argv entry as the path, so a bad one is a boot" >&2
        echo "  that does not happen." >&2
        return 1
    fi
    case "$CONTROL_VECTOR_LAYERS" in
        *:*:*)
            echo "strata overlay: CONTROL_VECTOR_LAYERS=$CONTROL_VECTOR_LAYERS is not \"low:high\"." >&2
            return 1
            ;;
        *:*)
            low="${CONTROL_VECTOR_LAYERS%%:*}" high="${CONTROL_VECTOR_LAYERS##*:}"
            ;;
        *)
            low='' high=''
            ;;
    esac
    if ! digits "$low" || ! digits "$high" || [ "$low" -gt "$high" ]; then
        echo "strata overlay: CONTROL_VECTOR_LAYERS=$CONTROL_VECTOR_LAYERS is not two non-negative" >&2
        echo "  integers with the first no higher than the second. The range is which layers the" >&2
        echo "  direction is subtracted from; it is stated, not assumed, exactly like --yarn-orig-ctx." >&2
        return 1
    fi
    if [ ! -f "$cfg" ]; then
        run_setup_pass "$cfg does not exist and a control vector has to be written into it"
        if [ ! -f "$cfg" ]; then
            echo "strata overlay: the setup pass left no $cfg to write the control vector into" >&2
            return 1
        fi
    fi
    edit_cvec set "$CONTROL_VECTOR:$CONTROL_VECTOR_SCALE" "$low" "$high"
}

apply_engine_log() {
    # Fact 11. A top-level key, not an argv entry: the engine reads `log` from the run config, which is
    # how the workspace has pointed its log at /data since 2026-10-12. Written here so the path is a
    # profile statement and not a memory held by a directory.
    if [ "$STRATA_LOG" = off ]; then
        if [ -f "$cfg" ]; then
            python3 - "$cfg" <<'PY'
import json, os, sys

path = sys.argv[1]
with open(path, encoding='utf-8-sig') as handle:
    cfg = json.load(handle)
if 'log' not in cfg:
    print('strata overlay: no log key to remove; the engine writes where it writes', flush=True)
else:
    value = cfg.pop('log')
    tmp = path + '.log.tmp'
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(cfg, handle, indent=1)
        handle.write('\n')
    os.replace(tmp, path)
    print('strata overlay: removed log=%s the profile does not state; the engine keeps its own log'
          ' location (%s)' % (json.dumps(value), path), flush=True)
PY
        fi
        return 0
    fi
    if [ -z "$STRATA_LOG" ]; then
        return 0
    fi
    case "$STRATA_LOG" in
        /*) ;;
        *)
            echo "strata overlay: STRATA_LOG=$STRATA_LOG is not an absolute path, and a relative one is" >&2
            echo "  written relative to whatever the engine's working directory happens to be -- which" >&2
            echo "  is how an instrument ends up somewhere nobody looks." >&2
            return 1
            ;;
    esac
    if [ ! -d "$(dirname "$STRATA_LOG")" ]; then
        echo "strata overlay: STRATA_LOG=$STRATA_LOG has no directory to write it in. Every instrument" >&2
        echo "  this house reads (expert-cache hit rate, resident RAM, the slots the boot sized) comes" >&2
        echo "  out of that file, so a path that cannot be opened is worth a refused boot." >&2
        return 1
    fi
    if [ ! -f "$cfg" ]; then
        run_setup_pass "$cfg does not exist and the engine log path has to be written into it"
        if [ ! -f "$cfg" ]; then
            echo "strata overlay: the setup pass left no $cfg to write the engine log path into" >&2
            return 1
        fi
    fi
    python3 - "$cfg" "$STRATA_LOG" <<'PY'
import json, os, sys

path, want = sys.argv[1], sys.argv[2]
with open(path, encoding='utf-8-sig') as handle:
    cfg = json.load(handle)
have = cfg.get('log')
if have == want:
    print('strata overlay: engine log already %s' % want, flush=True)
    sys.exit(0)
if have is not None:
    print('strata overlay: replacing log=%s the profile does not state' % json.dumps(have), flush=True)
cfg['log'] = want
tmp = path + '.engine-log.tmp'
with open(tmp, 'w', encoding='utf-8') as handle:
    json.dump(cfg, handle, indent=1)
    handle.write('\n')
os.replace(tmp, path)
print('strata overlay: engine log %s (%s)' % (want, path), flush=True)
PY
}

apply_max_context
apply_conversation_cache
apply_ple_table
apply_rope_scaling
apply_vram_reserve
apply_control_vector
apply_engine_log

if [ "${STRATA_SOURCE_SERVER:-0}" = 1 ]; then
    [ -f "$cfg" ] || { echo "strata overlay: missing prepared run config $cfg" >&2; exit 1; }
    exec ./strata-source-server.sh "$cfg"
fi
exec ./docker-entrypoint.upstream.sh "$@"

#!/bin/sh
# This house's overlay on Strata's own entry point, not an upstream file.
#
# Two facts about this container have to reach the engine's own run config, and neither is an
# argument the engine takes.
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
#
# So: when the container is told the encoder's card, make sure the config it is about to start
# carries a vision section, and write the card into it, then hand over to the upstream entry
# point unchanged. Where the section is missing, the setup pass that would create it is the one
# the upstream entry point would have run for this exact environment, so this runs it itself
# (the subshell keeps its argument list out of this entry point's own); a fresh workspace pays
# that pass once here instead of twice later, and a workspace already set up with vision skips
# straight to the patch. A pass that still leaves no section (VISION says no) is a misconfigured
# card, and a misconfigured card fails the boot instead of serving text-only with a log gap.
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

if [ -n "${VISION_DEVICE:-}" ]; then
    # The same tag the upstream entry point derives, so this writes the file it is about to read.
    case "$FAMILY" in
        qwen) prefix='' ;;
        *) prefix="${FAMILY}-" ;;
    esac
    tag=$(printf '%s' "$MODEL" | tr 'A-Z' 'a-z')
    cfg="$STRATA_DATA/config/strata-${prefix}${tag}.json"

    if [ ! -f "$cfg" ] || ! has_vision_section "$cfg"; then
        echo "strata overlay: $cfg has no vision section; running the setup pass the upstream entry point would run"
        mkdir -p "$STRATA_DATA/config"
        (
            # The upstream entry point's setup arguments, verbatim: the same defaults, the same
            # conditional flags, and the same --no-start, so the config it writes is the config
            # that boot would have written.
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
            # setup.py writes the config next to itself; the copy on the volume is the one that
            # counts (the upstream entry point does the same after its own pass). The relative
            # path is the container's working directory, the one this entry point's exec relies on.
            if [ -e "./strata-${prefix}${tag}.json" ]; then
                cmp -s "./strata-${prefix}${tag}.json" "$cfg" || cp -f "./strata-${prefix}${tag}.json" "$cfg"
            fi
        )
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

exec ./docker-entrypoint.upstream.sh "$@"

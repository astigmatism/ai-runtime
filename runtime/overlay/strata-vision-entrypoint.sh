#!/bin/sh
# This house's overlay on Strata's own entry point, not an upstream file.
#
# The card the image encoder runs on is a fact about the machine, and the profile that starts this
# container states it as VISION_DEVICE. Strata reads that fact from exactly one place - the
# "cuda_device" key of the "vision" section of its own run config (serve/server.py: vision_env) -
# and setup.py has no flag for it. Without this step the placement would live in a hand edit inside
# the workspace: invisible to the controller, unrenderable, and gone with the workspace.
#
# So: apply the declared placement to the config this entry point is about to start, then hand over
# to the upstream entry point unchanged. Two states have nothing to apply to, and neither is an
# error. A config that does not exist yet belongs to setup.py; the placement lands on the next
# start, which is the boot the provisioning flow serves from. A config with no "vision" section is
# a model started without images, and there is no encoder to place.
set -e

STRATA_DATA="${STRATA_DATA:-/data}"
FAMILY="${FAMILY:-qwen}"
MODEL="${MODEL:-IQ2_XS}"

if [ -n "${VISION_DEVICE:-}" ]; then
    # The same tag the upstream entry point derives, so this writes the file it is about to read.
    case "$FAMILY" in
        qwen) prefix='' ;;
        *) prefix="${FAMILY}-" ;;
    esac
    tag=$(printf '%s' "$MODEL" | tr 'A-Z' 'a-z')
    cfg="$STRATA_DATA/config/strata-${prefix}${tag}.json"
    if [ -f "$cfg" ]; then
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
    else
        echo "strata overlay: $cfg does not exist yet; the encoder placement lands on the next start"
    fi
fi

exec ./docker-entrypoint.upstream.sh "$@"

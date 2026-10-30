#!/bin/sh
# The upstream source-server CLI, with existing deployment paths and policy.
set -eu
cfg=$1
case "${STRATA_ANSWER_ROOM:-off}" in
    on|off) ;;
    *) echo 'STRATA_ANSWER_ROOM must be on or off' >&2; exit 1 ;;
esac
.venv/bin/python - "$cfg" "${STRATA_ANSWER_ROOM:-off}" "${TOOL_CALL_RECOVERY:-}" "${CONVERSATION_CACHE_MIB:-}" <<'PY'
import json, os, sys
path, answer_room, tool_recovery, conversation_cache = sys.argv[1:]
with open(path, encoding='utf-8-sig') as handle:
    config = json.load(handle)
# The primary router contract promises unrestricted output/thinking. Newer
# Strata defaults reserve answer room by shortening thinking; state the policy
# explicitly instead of inheriting that changed upstream default.
want = answer_room == 'on'
changes = {'answer_room': want}
# 0.2.8 changed the upstream default to true. Preserve an explicit profile off
# even though the legacy entrypoint represents off by removing the key.
if tool_recovery == 'off':
    changes['tool_call_recovery'] = False
changed = any(config.get(key) is not value for key, value in changes.items())
# The new engine automatically sizes a conversation cache when this argument is
# omitted. An existing off declaration must continue to mean no cache.
if conversation_cache == 'off':
    argv = list(config.get('args') or [])
    revised, index = [], 0
    while index < len(argv):
        if argv[index] == '--conversation-cache-mib':
            index += 2
        else:
            revised.append(argv[index])
            index += 1
    revised += ['--conversation-cache-mib', '0']
    if revised != argv:
        changes['args'] = revised
        changed = True
if changed:
    config.update(changes)
    temporary = path + '.answer-room.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(config, handle, indent=1)
        handle.write('\n')
    os.replace(temporary, path)
print('strata adapter: answer_room=%s; upstream source server' % want, flush=True)
PY
exec .venv/bin/python serve/server.py --engine strata --config "$cfg" --port "${PORT:-8080}"

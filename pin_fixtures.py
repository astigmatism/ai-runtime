#!/usr/bin/env python3
"""Pin the trial configurations into the two fixture files the way the suite reads them.

The digests are produced by the same helpers the assertions use, so a pin that disagrees with
the render is this script refusing, not a test failing later.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, '.')
sys.path.insert(0, 'tests')
import test_config as T                                          # noqa: E402
from runtime.config import render                                 # noqa: E402

NEW = {
    'flash-next-strata-iq3-xxs-daisy':
        'New on 2026-10-20, not a rename: the first configuration of the second machine, and the first '
        'entry in this file pinned on a host that is not Rosalina.',
    'flash-next-strata-iq3-xxs-daisy-tc':
        'New on 2026-10-25, not a rename: the tensor-core prompt path measured on the second '
        'machine on that date (+11 % prompt, 10 % less time to the first token of a 58k prompt, '
        'decode inside the A/A band, answers identical). Pinned on the second machine for the same '
        'reason as the configuration it is one line apart from.',
    'flash-next-strata-iq3-xxs-daisy-128k':
        'New on 2026-10-25, not a rename: the 131,072-token window, measured serving a 118,514-token '
        'prompt on the 64 GB box with the tensor-core prompt path under it. Pinned because a window '
        'this size is exactly what a stray argument edit moves without saying so.',
    'flash-next-strata-iq3-s-daisy':
        'New on 2026-10-25, not a rename: the IQ3_S tier on the second machine, measured at 6.6 % '
        'below IQ3_XXS for 87.69 % top-1 against 84.46 %, and the first configuration here whose '
        'second artifact is a hardlink to another tier\'s PLE shard rather than a second copy.',
}

# Said once, on every entry the day moves: a pin is the only place that says which launch was
# agreed, so a rebuild that changes what a profile means has to be legible here as well as there.
REASON = (" Re-pinned on 2026-10-27: the tag that machine runs was rebuilt to carry this house's entry "
          "point, so its configurations finally state a refusal direction, a sampling preset, a "
          "tool-call recovery and a PLE table that nothing on that machine had been reading, and the "
          "launch moved with them.")

renamed_path = Path('tests/fixtures/renamed-profiles.json')
active_path = Path('tests/fixtures/active-bundles.json')
renamed = json.loads(renamed_path.read_text())
active = json.loads(active_path.read_text())

for name, note in NEW.items():
    host = T.fixture_host(name)
    bundle = render(T.ROOT / 'config', host, name)
    service = bundle['compose']['services'][T.OWN[name]]
    note = (renamed.get(name) or {}).get('note') or note
    entry = {'previous_id': (renamed.get(name) or {}).get('previous_id'),
             'compose_sha256': T.digest(T.filed_by_role(bundle)['compose']),
             'coding_service_sha256': T.digest(T.stamped(T.OWN[name], service)),
             'note': note if REASON in note else note + REASON}
    renamed[name] = entry
    active['bundles'][name] = T.digest(T.filed_by_role(bundle))
    print('pinned', name, 'alone' if name in T.ALONE else 'shares-a-model',
          entry['coding_service_sha256'][:12])

renamed_path.write_text(json.dumps(renamed, indent=2) + '\n')
active_path.write_text(json.dumps(active, indent=2) + '\n')
print('wrote', renamed_path, 'and', active_path)

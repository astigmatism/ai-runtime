import copy
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from runtime.config import read, render
from runtime.migration import Migration, WRAPPERS
from runtime.system import atomic_json

ROOT = Path(__file__).resolve().parents[1]
HOST = read(ROOT / 'tests/fixtures/legacy-fingerprints.json')['host']


class MigrationTests(unittest.TestCase):
    def test_changed_saved_profile_blocks_preparation_before_private_state_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp).resolve() / 'home'; root = home / 'apps/local-ai-runtime'
            root.mkdir(parents=True); shutil.copytree(ROOT / 'config', root / 'config')
            primary = home / 'apps/local-ai-primary'
            host = copy.deepcopy(HOST)
            host.update(model_root=str(home / 'ai/models'), uid=os.getuid(), gid=os.getgid())
            current = render(root / 'config', host, 'daytime')['compose']
            saved = render(root / 'config', host, 'daytime-27b')['compose']
            atomic_json(primary / 'compose.json', current)
            atomic_json(primary / 'profiles/selected.json', {'selected': 'daytime'})
            atomic_json(primary / 'profiles/daytime/compose.json', current)
            saved['services']['coding']['command'].append('--changed-after-import')
            atomic_json(primary / 'profiles/daytime-27b/compose.json', saved)
            stack = home / 'apps/local-ai-ollama-stack'; stack.mkdir()
            (stack / '.env').write_text('ADMIN_TOKEN=synthetic-token\n')
            migration = Migration(root, home)
            with patch.object(migration.updater, 'source_preflight', return_value='a' * 40):
                with self.assertRaisesRegex(RuntimeError, 'daytime-27b: saved profile differs'):
                    migration.prepare()
            self.assertFalse((root / '.env').exists())
            self.assertFalse((root / '.state/host.json').exists())
            self.assertFalse((root / '.state/router-token').exists())

    def test_migration_baseline_covers_both_profiles_and_recovery_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            migration = Migration(Path(tmp) / 'repo', Path(tmp) / 'home')
            sources = migration.tracked_sources()
            self.assertIn('daytime-27b', WRAPPERS)
            self.assertIn(migration.home / 'daytime-27b', sources)
            for profile in ('daytime', 'daytime-27b'):
                self.assertIn(migration.primary / 'profiles' / profile / 'compose.json', sources)
                self.assertIn(migration.primary / 'profiles' / profile / 'qualified.json', sources)


if __name__ == '__main__':
    unittest.main()

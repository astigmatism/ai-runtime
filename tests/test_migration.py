import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from runtime.config import read, render
from runtime.migration import Migration, WRAPPERS, legacy_deploy_guard
from runtime.system import atomic_json

ROOT = Path(__file__).resolve().parents[1]
HOST = read(ROOT / 'tests/fixtures/legacy-fingerprints.json')['host']


class MigrationTests(unittest.TestCase):
    def test_inspection_reuses_only_verified_unchanged_checksums(self):
        for cached in (True, False):
            with self.subTest(cached=cached), tempfile.TemporaryDirectory() as tmp:
                migration = Migration(Path(tmp) / 'repo', Path(tmp) / 'home')
                result = {'mode': 'inspection', 'revision': 'a' * 40, 'profile': 'daytime-27b',
                    'config_sha256': 'b' * 64, 'matches_live': True, 'router_draining': False,
                    'active_requests': 0, 'queued_requests': 0,
                    'validation': {'all_checksums_verified': True, 'receipts': {'model': {'verified': True}}}}
                first = copy.deepcopy(result); first['validation']['all_checksums_verified'] = cached
                replies = [json.dumps(first)] + ([] if cached else [json.dumps(result)])
                metadata = [{'Config': {'Labels': {'org.opencontainers.image.revision': 'a' * 40}}}]
                with patch.object(migration.updater, 'source_preflight', return_value='a' * 40), \
                        patch.object(migration.updater, 'run', return_value=json.dumps(metadata)), \
                        patch.object(migration.updater, 'candidate', side_effect=replies) as candidate, \
                        patch('builtins.print'):
                    migration.inspection()
                self.assertEqual(candidate.call_count, 1 if cached else 2)
                self.assertNotIn('full_hash', candidate.call_args_list[0].kwargs)
                if not cached: self.assertTrue(candidate.call_args_list[1].kwargs['full_hash'])
                self.assertEqual(read(migration.state / 'inspection.json'), result)

    def test_inspection_drift_fails_before_hashing_or_recording_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            migration = Migration(Path(tmp) / 'repo', Path(tmp) / 'home')
            metadata = [{'Config': {'Labels': {'org.opencontainers.image.revision': 'a' * 40}}}]
            with patch.object(migration.updater, 'source_preflight', return_value='a' * 40), \
                    patch.object(migration.updater, 'run', return_value=json.dumps(metadata)), \
                    patch.object(migration.updater, 'candidate', return_value='{"matches_live": false}') as candidate:
                with self.assertRaisesRegex(RuntimeError, 'resolve drift'): migration.inspection()
            self.assertEqual(candidate.call_count, 1)
            self.assertFalse((migration.state / 'inspection.json').exists())

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
            self.assertIn(migration.home / 'apps/local-ai-ollama-stack/deploy-runtime.sh', sources)
            for profile in ('daytime', 'daytime-27b'):
                self.assertIn(migration.primary / 'profiles' / profile / 'compose.json', sources)
                self.assertIn(migration.primary / 'profiles' / profile / 'qualified.json', sources)
            self.assertFalse(any('daytime-flash-f16' in str(path) for path in sources))

    def test_legacy_launcher_stays_blocked_without_systemd_or_docker(self):
        result = subprocess.run(['/bin/sh'], input=legacy_deploy_guard(), text=True,
            capture_output=True, env={'PATH': '/nonexistent'})
        self.assertEqual(result.returncode, 2)
        self.assertIn('legacy all-GPU launcher is retired', result.stderr)


if __name__ == '__main__':
    unittest.main()

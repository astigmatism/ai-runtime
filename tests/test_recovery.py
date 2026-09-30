import json
import os
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest.mock import patch

from runtime.config import read
from runtime.migration import WRAPPERS
from runtime.recovery import export, install_wrappers, model_inventory, stable_state
from runtime.system import atomic_json, lock

ROOT = Path(__file__).resolve().parents[1]


class RecoveryTests(unittest.TestCase):
    def test_export_preserves_private_state_without_docker_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'runtime'; state = root / '.state'
            active = {'revision': 'a' * 40, 'image': 'local/controller:current',
                      'bundle': {'profile': 'daytime-27b'}}
            atomic_json(state / 'active.json', active)
            atomic_json(state / 'host.json', {'model_root': '/models'})
            (state / 'router-token').write_text('synthetic-private-value')
            (root / '.env').write_text('PRIVATE=config')
            atomic_json(root / 'config/shared.json', {'engines': {}})
            (root / 'config/profiles').mkdir()
            destination = Path(tmp) / 'backup'
            calls = []
            def run(*args, **kwargs):
                calls.append(args)
                if args[:2] == ('docker', 'exec'):
                    return json.dumps({'ready': True, 'deployed_revision': active['revision']})
                if args[:3] == ('docker', 'image', 'inspect'):
                    return json.dumps([{'Id': 'sha256:test', 'Size': 1,
                        'Config': {'Labels': {'org.opencontainers.image.revision': active['revision']}}}])
                raise AssertionError('Unexpected Docker operation: ' + str(args))
            def git(*args):
                if args[0] == 'bundle':
                    Path(args[2]).write_bytes(b'synthetic git bundle'); return ''
                return active['revision']
            with patch('runtime.recovery.Updater.source_preflight'), \
                    patch('runtime.recovery.Updater.run', side_effect=run), \
                    patch('runtime.recovery.Updater.git', side_effect=git), patch('builtins.print'):
                with lock(state / 'runtime.lock'):
                    with self.assertRaisesRegex(RuntimeError, 'maintenance lock'):
                        export(root, destination, include_images=False)
                self.assertFalse(destination.exists())
                export(root, destination, include_images=False)
            inventory = read(destination / 'inventory.json')
            self.assertFalse(inventory['images_included'])
            self.assertFalse(inventory['models_included'])
            self.assertNotIn('synthetic-private-value', (destination / 'inventory.json').read_text())
            with tarfile.open(destination / 'private-state.tar.gz') as archive:
                self.assertEqual(archive.extractfile('.state/router-token').read(), b'synthetic-private-value')
            self.assertIn('private-state.tar.gz', read(destination / 'checksums.json'))
            self.assertEqual(destination.stat().st_mode & 0o777, 0o700)
            self.assertEqual((destination / 'private-state.tar.gz').stat().st_mode & 0o777, 0o600)

    def test_no_legacy_files_needed_to_install_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            install_wrappers(home)
            install_wrappers(home)
            self.assertEqual(set(p.name for p in home.iterdir()), set(WRAPPERS))
            for path in home.iterdir():
                self.assertIn('docker exec local-ai-runtime', path.read_text())
                self.assertNotIn('apps/local-ai-primary', path.read_text())
                self.assertTrue(os.access(path, os.X_OK))

    def test_wrapper_install_preserves_existing_files_and_symlinks(self):
        for symlink in (False, True):
            with self.subTest(symlink=symlink), tempfile.TemporaryDirectory() as tmp:
                home = Path(tmp)
                target = home / 'original'; target.write_text('keep this')
                if symlink: (home / 'daytime').symlink_to(target)
                else: (home / 'daytime').write_text('keep this')
                with self.assertRaises(RuntimeError): install_wrappers(home)
                self.assertEqual(target.read_text(), 'keep this')
                self.assertFalse((home / 'primary').exists())

    def test_pending_runtime_or_router_work_blocks_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            for phase in ['prepared', 'draining', 'applying', 'verifying', 'needs-attention']:
                atomic_json(state / 'transaction.json', {'phase': phase})
                with self.assertRaises(RuntimeError): stable_state(state)
            atomic_json(state / 'transaction.json', {'phase': 'succeeded'})
            atomic_json(state / 'router-maintenance.json', {'reserved': True})
            with self.assertRaises(RuntimeError): stable_state(state)
            atomic_json(state / 'router-maintenance.json', {'reserved': False})
            atomic_json(state / 'update-job.json', {'phase': 'controller-replacement'})
            with self.assertRaises(RuntimeError): stable_state(state)
            atomic_json(state / 'update-job.json', {'phase': 'succeeded'})
            stable_state(state)

    def test_all_profiles_have_unique_artifact_inventory(self):
        artifacts = model_inventory(ROOT)
        self.assertEqual(len(artifacts), 40)
        self.assertEqual(sum(a['bytes'] for a in artifacts), 152039442720)
        # Download documentation is not included in the production image.
        sources = ROOT / 'docs/model-downloads.json'
        if sources.exists():
            downloads = {x['path']: x for x in read(sources)['artifacts']}
            for artifact in artifacts:
                entry = downloads[artifact['path']]
                self.assertEqual(entry['sha256'], artifact['sha256'])
                self.assertEqual(entry['bytes'], artifact['bytes'])
                self.assertTrue(entry['download_url'].startswith('https://huggingface.co/'))


if __name__ == '__main__':
    unittest.main()

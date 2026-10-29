import json
import os
from pathlib import Path
import shutil
import tempfile
import tarfile
import unittest
from unittest.mock import patch

from runtime.config import PROFILES_VERSION, REGISTRY_FILE, read
from runtime.migration import WRAPPERS
from runtime.recovery import export, install_wrappers, model_inventory, stable_state
from runtime.system import atomic_json, lock
import machine

ROOT = Path(__file__).resolve().parents[1]


class RecoveryTests(unittest.TestCase):
    def test_export_preserves_private_state_without_docker_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'runtime'; state = root / '.state'
            active = {'revision': 'a' * 40, 'image': 'local/controller:current',
                      'bundle': {'profile': 'qwen27b-q6k-pair'}}
            atomic_json(state / 'active.json', active)
            home = state / 'profiles'  # Where this box says it keeps its own configurations.
            home.mkdir()
            atomic_json(state / 'host.json', {'model_root': '/models', 'profiles_dir': str(home)})
            (state / 'router-token').write_text('synthetic-private-value')
            (root / '.env').write_text('PRIVATE=config')
            atomic_json(root / 'config/shared.json', {'engines': {}})
            # The receipt names the models of the configuration this machine is running, and it reads them
            # where the machine keeps them. The checkout beside it ships no bodies to inventory.
            for name in ('qwen27b-q6k-pair', 'everyday'):
                shutil.copyfile(machine.HOME / (name + '.json'), home / (name + '.json'))
            atomic_json(home / REGISTRY_FILE, {'schema_version': PROFILES_VERSION, 'profiles': ['qwen27b-q6k-pair']})
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
            # A service restoration that has not finished is not a moment to take a receipt.
            atomic_json(state / 'restoration.json', {'attempts': 2, 'next_attempt_at': '2026-10-12T04:26:00+00:00'})
            with self.assertRaises(RuntimeError): stable_state(state)
            atomic_json(state / 'restoration.json', {'attempts': 2,
                'finished_at': '2026-10-12T04:56:00+00:00'})
            stable_state(state)

    def test_all_profiles_have_unique_artifact_inventory(self):
        # Asked of the machine that keeps these configurations -- the one whose bodies and record these
        # tests are built from -- since the source ships no bodies of its own to inventory.
        artifacts = model_inventory(ROOT, machine.HOST)
        # The shared-Q4_K_M Flash-Next MTP draft (1,907,151,936 bytes) retired with the paired Flash-Next profiles.
        # The Strata model adds its UD-IQ4_XS revision: three shards, 93,682,584,224 bytes in all,
        # and the two-card 3-bit arm adds ISTA-DASLab's GSQ-RCO IQ3_S: two shards, 83,617,662,656 bytes.
        # Both quants are mounted by exactly one configuration each, which is the only reason both are
        # inventory items at all; a pack no configuration runs is a disk problem, not an artifact.
        # and ISTA-DASLab's mmproj (907,543,008 bytes) since the encoder on the shared card has read
        # pictures for it, from 2026-10-16.
        # The second machine's IQ3_XXS tier adds two paths of its own, 75,839,998,528 bytes. It is the
        # first inventory entry that is not on the machine the rest of them are: the inventory says what
        # a configuration mounts wherever that configuration runs, and its shard 2 is byte-for-byte the
        # IQ3_S shard this host already carries — a second path, the same 28,800,138,432 bytes, held
        # once on the disk by hardlinks and twice here because an inventory that described filesystem
        # plumbing could not say which configuration would lose its model when a path went away.
        # The XL trial's UD-Q4_K_XL -- four shards, 111,334,654,784 bytes, four rather than three, which
        # is the difference between that quant and this one -- left with the profile on 2026-10-15. It
        # lost to UD-IQ4_XS on every measured cell, so its shards are not an inventory item.
        # The two llama.cpp Flash-Next solo configurations shared one 35-artifact set -- 33 shards of
        # AD-4.27bpw-Q4_K_M, the F16 multimodal projector, and the Q4_0 MTP draft, 97,629,051,456 bytes
        # -- and it goes with them on 2026-10-15. An artifact no configuration mounts is still verified
        # on every validation pass and still occupies the model root, so it belongs to the inventory
        # only while something can run it.
        self.assertEqual(len(artifacts), 15)
        self.assertEqual(sum(a['bytes'] for a in artifacts), 334050225568)
        # Download documentation is not included in the production image.
        sources = ROOT / 'docs/model-downloads.json'
        if sources.exists():
            downloads = {x['path']: x for x in read(sources)['artifacts']}
            for artifact in artifacts:
                with self.subTest(path=artifact['path']):
                    entry = downloads[artifact['path']]
                    self.assertEqual(entry['sha256'], artifact['sha256'])
                    self.assertEqual(entry['bytes'], artifact['bytes'])
                    # A locally built artifact has no public download. It must instead name
                    # checksum-pinned public inputs so it can be rebuilt and re-verified.
                    self.assertEqual(('download_url' in entry) + ('derivation' in entry), 1)
                    if 'derivation' in entry:
                        self.assertTrue(entry['derivation']['method'])
                        inputs = entry['derivation']['inputs']
                        self.assertTrue(inputs)
                    else:
                        inputs = [entry]
                    for source in inputs:
                        self.assertTrue(source['download_url'].startswith('https://huggingface.co/'))
                        self.assertRegex(source['sha256'], '^[0-9a-f]{64}$')
                        self.assertIs(type(source['bytes']), int)
                        if source.get('path') in downloads:
                            self.assertEqual(downloads[source['path']]['sha256'], source['sha256'])
                            self.assertEqual(downloads[source['path']]['bytes'], source['bytes'])
            # One locally built artifact is inventoried, and only one: the fitted control vector the
            # two-card configuration declares. Listing the paths that are allowed to be local is the
            # point -- a second hand-built file appearing in a profile has to be added here on purpose,
            # because an artifact with no public source cannot be re-fetched by the box that lost it.
            self.assertEqual([path for path, entry in downloads.items() if 'derivation' in entry],
                ['llm/host-provided/refusal-direction-r.gguf'])

    def test_the_inventory_reads_the_definitions_the_machine_keeps(self):
        # A recovery receipt says which models this machine carries, so it reads the definitions this
        # machine keeps. A checkout that ships a copy of them is a copy of some other machine's answer,
        # and a machine that keeps its own bodies in the directory its settings name has nothing to read
        # in the checkout at all.
        with tempfile.TemporaryDirectory() as tmp:
            root, home = Path(tmp) / 'runtime', Path(tmp) / 'machine-profiles'
            (root / 'config/profiles').mkdir(parents=True); home.mkdir()
            atomic_json(root / 'config/profiles' / 'a-configuration-of-another-machine.json',
                {'id': 'a-configuration-of-another-machine',
                    'artifacts': [{'path': 'llm/somewhere-else.gguf', 'bytes': 7, 'sha256': 'a' * 64}]})
            # The machine whose receipt this is: one configuration it offers, and the model body it names
            # beside it. Nothing else sits in its directory, because nothing else there would be explained.
            for name in ('qwen27b-q8-pair', 'everyday'):
                shutil.copyfile(machine.HOME / (name + '.json'), home / (name + '.json'))
            atomic_json(home / REGISTRY_FILE, {'schema_version': PROFILES_VERSION, 'profiles': ['qwen27b-q8-pair']})
            asked = model_inventory(root, {'profiles_dir': str(home)})
            # An inventory is a set of files this machine carries, not a list of mountings: the projector
            # both of these bodies mount is one item, and the same item twice would say two files can go
            # away before the configuration that needs it stops working.
            declared = {}
            for name in ('qwen27b-q8-pair', 'everyday'):
                for item in read(machine.HOME / (name + '.json'))['artifacts']:
                    declared[item['path']] = {k: item[k] for k in ('path', 'bytes', 'sha256')}
            self.assertEqual({a['path']: a for a in asked}, declared,
                'the artifacts of the configuration this machine offers and of the models it names, read '
                'from the machine instead of from the checkout')
            # A checkout is not a machine. The body beside its config directory is somebody else's, and a
            # checkout that has written down no answer cannot be asked what it carries: an export that
            # filed an empty inventory as a receipt would promise a restore with no models in it.
            with self.assertRaisesRegex(RuntimeError, 'keeps no list of configurations'):
                model_inventory(root)
            # The record says which configurations this machine offers, in the fields this revision
            # understands about it. A record that tried to declare artifacts of its own -- as if a list of
            # names mounted a file -- is refused where it is read, so an inventory can never be padded with
            # a model nothing can run.
            atomic_json(home / REGISTRY_FILE, {'schema_version': PROFILES_VERSION, 'profiles': ['qwen27b-q8-pair'],
                'artifacts': [{'path': 'llm/a-record-mounts-nothing.gguf', 'bytes': 1, 'sha256': 'b' * 64}]})
            with self.assertRaisesRegex(RuntimeError, 'nothing this revision understands'):
                model_inventory(root, {'profiles_dir': str(home)})


if __name__ == '__main__':
    unittest.main()

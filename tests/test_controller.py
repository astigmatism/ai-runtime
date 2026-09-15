import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from runtime.config import read, render
from runtime.controller import Controller
from runtime.system import atomic_json, lock, System

ROOT = Path(__file__).resolve().parents[1]
HOST = read(ROOT / 'tests/fixtures/legacy-fingerprints.json')['host']


class FakeSystem:
    readonly = False

    def __init__(self, bundle):
        self.bundle = bundle; self.events = []; self.draining = False; self.reason = None
        self.active_count = 0; self.queued_count = 0; self.direct_busy = False
        self.generation_fails = False; self.fail_reconcile = False; self.serial = 0
        self.containers = {}
        for role, cfg in bundle['compose']['services'].items(): self.install(role, cfg)

    def install(self, role, cfg):
        self.serial += 1
        self.containers[cfg['container_name']] = {'Id': role + str(self.serial), 'Image': self.bundle['manifest']['engine']['image_id'],
            'State': {'Running': True, 'StartedAt': '2026-01-01T00:00:00Z'}, 'RestartCount': 0,
            'Config': {'Cmd': cfg['command'], 'Entrypoint': cfg['entrypoint'], 'User': cfg['user']},
            'HostConfig': {'ReadonlyRootfs': True, 'RestartPolicy': {'Name': 'unless-stopped'},
                'Init': True, 'DeviceRequests': [{'DeviceIDs': cfg['deploy']['resources']['reservations']['devices'][0]['device_ids']}],
                'PortBindings': {'8080/tcp': [{'HostIp': '127.0.0.1', 'HostPort': cfg['ports'][0].split(':')[1]}]}},
            'Mounts': [{'Type': 'bind', 'Source': v['source'], 'Destination': v['target'], 'RW': False} for v in cfg['volumes']],
            'NetworkSettings': {'Networks': {'local-ai-ollama_default': {'Aliases': cfg['networks']['router']['aliases']}}}}

    def inspect(self, name): return copy.deepcopy(self.containers.get(name))

    def docker(self, *args, **kwargs):
        self.events.append(('docker', args))
        if args[:2] == ('image', 'inspect'):
            return json.dumps([{'Id': self.bundle['manifest']['engine']['image_id'],
                'Config': {'Labels': {'org.opencontainers.image.revision': self.bundle['manifest']['engine']['revision']}}}])
        if args[0] == 'compose' and 'up' in args:
            if self.fail_reconcile: raise RuntimeError('Simulated load failure')
            compose = read(args[args.index('-f') + 1])
            roles = args[args.index('450') + 1:]
            for role in roles: self.install(role, compose['services'][role])
        return ''

    def http(self, url, body=None, headers=None, timeout=None):
        if url.endswith('/runtime-state'):
            return {'runtime': {'draining': self.draining, 'drain_reason': self.reason,
                    'active_count': self.active_count, 'queued_count': self.queued_count},
                'models': [{'id': ci['Config']['Cmd'][ci['Config']['Cmd'].index('--alias') + 1],
                    'x_ollama_router': {'health': {'available': True}, 'output_policy': 'unrestricted',
                        'context_window': int(ci['Config']['Cmd'][ci['Config']['Cmd'].index('--ctx-size') + 1]),
                        'default_output_tokens': None, 'max_output_tokens': None}} for ci in self.containers.values()]}
        if url.endswith('/runtime-drain'):
            self.events.append(('drain', body['enabled'])); self.draining = body['enabled']; self.reason = body['reason']; return {}
        if url.endswith('/reload-config'):
            self.events.append(('publish',)); return {}
        name = url.split('/')[2].split(':')[0]
        if url.endswith('/health'): return {'status': 'ok'}
        if url.endswith('/slots'):
            ci = self.containers[name]; argv = ci['Config']['Cmd']
            return [{'n_ctx': int(argv[argv.index('--ctx-size') + 1]), 'is_processing': self.direct_busy}]
        if url.endswith('/completion'):
            self.events.append(('generation', name))
            if self.generation_fails: raise RuntimeError('Simulated acceptance failure')
            return {'content': 'K'}
        raise AssertionError(url)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name); host = copy.deepcopy(HOST)
        host['router_state_dir'] = str(self.state / 'router')
        atomic_json(self.state / 'host.json', host); (self.state / 'router-token').write_text('synthetic-token')
        self.bundle = render(ROOT / 'config', host, 'daytime-swift')
        self.system = FakeSystem(self.bundle)
        self.c = Controller(ROOT / 'config', self.state, 'a' * 40, self.system)
        self.active = {'revision': 'a' * 40, 'image': 'old-image', 'bundle': self.bundle}
        atomic_json(self.state / 'active.json', self.active)
        self.prepare = patch.object(self.c, 'prepare', side_effect=lambda b: self.system.events.append(('prepare',)))
        self.prepare.start(); self.addCleanup(self.prepare.stop)
        self.sleep = patch('runtime.controller.time.sleep'); self.sleep.start(); self.addCleanup(self.sleep.stop)

    def test_healthy_startup_does_not_drain_recreate_or_publish(self):
        result = self.c.transition()
        self.assertTrue(result['already_active'])
        self.assertEqual(self.system.events, [('prepare',)])

    def test_stale_controller_cannot_apply_old_source(self):
        self.c.revision = 'b' * 40
        with self.assertRaisesRegex(RuntimeError, 'superseded'): self.c.transition()
        self.assertEqual(self.system.events, [])

    def test_profile_change_recreates_only_daytime_after_drain(self):
        night = self.system.inspect('qwen38-nighttime')['Id']
        result = self.c.transition('daytime')
        self.assertEqual(result['changed_roles'], ['coding'])
        self.assertEqual(self.system.inspect('qwen38-nighttime')['Id'], night)
        self.assertFalse(self.system.draining)
        events = self.system.events
        self.assertLess(events.index(('drain', True)), next(i for i, x in enumerate(events) if x[0] == 'docker'))
        self.assertLess(next(i for i, x in enumerate(events) if x[0] == 'generation'), events.index(('publish',)))
        self.assertEqual(events[-1], ('drain', False))
        self.assertEqual(read(self.state / 'active.json')['bundle']['profile'], 'daytime')

    def test_busy_router_queue_or_direct_slot_aborts_before_recreation(self):
        for attribute in ['active_count', 'queued_count', 'direct_busy']:
            with self.subTest(attribute=attribute):
                setattr(self.system, attribute, 1)
                with patch('runtime.controller.time.monotonic', side_effect=[0, 0, 301]):
                    with self.assertRaisesRegex(RuntimeError, 'Drain timed out'): self.c.transition('daytime')
                self.assertFalse(any(e[0] == 'docker' for e in self.system.events))
                self.assertFalse(self.system.draining)
                self.assertEqual(read(self.state / 'active.json'), self.active)
                setattr(self.system, attribute, 0); self.system.events.clear()

    def test_failed_load_recovers_without_recreating_healthy_previous_pair(self):
        self.system.fail_reconcile = True
        before = {k: v['Id'] for k, v in self.system.containers.items()}
        with self.assertRaisesRegex(RuntimeError, 'Simulated load failure'): self.c.transition('daytime')
        self.assertEqual({k: v['Id'] for k, v in self.system.containers.items()}, before)
        self.assertEqual(read(self.state / 'transaction.json')['phase'], 'recovered')
        self.assertFalse(self.system.draining)

    def test_failed_recovery_keeps_drain_and_blocks_new_operations(self):
        self.system.generation_fails = True
        with self.assertRaisesRegex(RuntimeError, 'recovery needs attention'): self.c.transition('daytime')
        self.assertTrue(self.system.draining)
        self.assertEqual(read(self.state / 'transaction.json')['phase'], 'needs-attention')
        with self.assertRaisesRegex(RuntimeError, 'interrupted'): self.c.transition()
        self.system.generation_fails = False
        self.c.recover()
        self.assertFalse(self.system.draining)

    def test_foreign_drain_is_never_cleared(self):
        self.system.draining = True; self.system.reason = 'another owner'
        with self.assertRaisesRegex(RuntimeError, 'Another operation owns'): self.c.transition('daytime')
        self.assertNotIn(('drain', False), self.system.events)

    def test_shared_lock_blocks_cli_and_startup(self):
        with lock(self.state / 'runtime.lock'):
            with self.assertRaisesRegex(RuntimeError, 'maintenance lock'): self.c.transition()

    def test_adoption_refuses_profile_mismatch(self):
        (self.state / 'active.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'exact match'): self.c.transition('daytime', adopt=True)
        self.assertNotIn(('drain', True), self.system.events)

    def test_readonly_system_refuses_docker_and_http_mutation(self):
        system = System(readonly=True)
        with self.assertRaisesRegex(RuntimeError, 'inspection'): system.docker('stop', 'container')
        with self.assertRaisesRegex(RuntimeError, 'inspection'): system.http('http://localhost', {})

    def test_status_never_exposes_router_token(self):
        status = self.c.status()
        self.assertTrue(status['ready'])
        self.assertNotIn('synthetic-token', json.dumps(status))
        self.assertNotIn('router-token', json.dumps(status))

    def test_router_reservation_blocks_profile_changes_until_verified_completion(self):
        self.c.router_maintenance(True)
        self.assertTrue(self.system.draining)
        with self.assertRaisesRegex(RuntimeError, 'reserves the runtime'): self.c.transition('daytime')
        self.c.republish()
        self.c.router_maintenance(False)
        self.assertFalse(self.system.draining)
        self.assertFalse(read(self.state / 'router-maintenance.json')['reserved'])

    def test_configuration_change_outside_argv_still_reconciles_affected_service(self):
        proposed = copy.deepcopy(self.bundle)
        proposed['compose']['services']['coding']['logging']['options']['max-size'] = '30m'
        with patch.object(self.c, 'desired', return_value=proposed): result = self.c.transition()
        self.assertEqual(result['changed_roles'], ['coding'])


class ArtifactTests(unittest.TestCase):
    def test_same_size_change_invalidates_cached_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model = root / 'model.gguf'; model.write_bytes(b'good')
            host = copy.deepcopy(HOST); host['model_root'] = str(root)
            atomic_json(root / 'host.json', host)
            bundle = render(ROOT / 'config', host, 'daytime-swift')
            bundle['artifacts'] = [{'source': str(model), 'bytes': 4, 'sha256': hashlib.sha256(b'good').hexdigest()}]
            c = Controller(ROOT / 'config', root, 'a' * 40, FakeSystem(bundle))
            self.assertTrue(c.validate(bundle)['all_checksums_verified'])
            model.write_bytes(b'evil')
            with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'): c.validate(bundle)

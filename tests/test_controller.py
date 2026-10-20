import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from runtime.config import CONFIGURATIONS, read, render, service_engine
from runtime.controller import Controller
from runtime.system import atomic_json, lock, System

ROOT = Path(__file__).resolve().parents[1]
HOST = read(ROOT / 'tests/fixtures/legacy-fingerprints.json')['host']
VISION = 'GPU-465a9a1e-fcb9-27f2-8f6a-cca1c30de981'


def vision_host():
    host = copy.deepcopy(HOST)
    host.update(vision_gpu_id=VISION, vision_gpu_name='RTX 3080', vision_cuda_device=4,
        cuda_order={'daytime': host['gpu_ids']['daytime'][::-1], 'nighttime': host['gpu_ids']['nighttime'][:]})
    return host


class FakeSystem:
    readonly = False

    def __init__(self, bundle):
        self.bundle = bundle; self.events = []; self.draining = False; self.reason = None
        self.active_count = 0; self.queued_count = 0; self.direct_busy = False
        self.generation_fails = False; self.fail_reconcile = False; self.serial = 0
        self.failed_ups = 0; self.failed_generations = 0  # fail only the next N loads or acceptances
        # A backend configured by the environment has no argument vector to read its window or its
        # served name from, so the fake reads them the way the controller does: from the published
        # configuration. slot_context simulates a backend that came up on another window.
        self.services = {x['container_name']: x for x in bundle['manifest']['services']}
        self.slot_context = None
        # An engine that answers its health check before the model is resident (Strata's /health
        # says status ok from server start and carries the model state as loaded; /slots stays
        # empty until the read completes). model_loaded plays that state; load_after_slots flips
        # it to true after N slot looks, so a test can play the load completing under a wait.
        self.model_loaded = True
        self.load_after_slots = None
        # An engine that stops itself while loading (the trial's 18:47 attempt died mid-read with
        # the cards free): after N slot looks the container is stopped and its last words are in
        # the log, which is exactly what the record must carry before restoration removes it.
        self.die_after_slots = None
        self.die_words = []
        self.container_logs = {}
        self.engines = {s['engine']['tag']: s['engine'] for s in bundle['manifest']['services']}
        shared = read(ROOT / 'config/shared.json')
        self.engines.update({e['tag']: e for e in shared['engines'].values()})
        self.containers = {}
        for role, cfg in bundle['compose']['services'].items(): self.install(role, cfg)

    def install(self, role, cfg):
        self.serial += 1
        self.containers[cfg['container_name']] = {'Id': role + str(self.serial), 'Image': self.engines[cfg['image']]['image_id'],
            # Model residency is taken from the flag as the container is installed, so a test can
            # play one engine still reading its model beside a backend that is already resident.
            'State': {'Running': True, 'StartedAt': '2026-01-01T00:00:00Z', 'Loaded': self.model_loaded,
                      'Provisioning': bool(self.engines[cfg['image']].get('provisioning'))}, 'RestartCount': 0,
            'Config': {'Cmd': cfg['command'], 'Entrypoint': cfg['entrypoint'],
                # A service rendered without a user runs as the image's own, which the daemon reports
                # as an empty string — the same shape the observer has to accept as matching.
                'User': cfg.get('user') or '', 'Env': [k + '=' + v for k, v in cfg.get('environment', {}).items()]},
            'HostConfig': {'ReadonlyRootfs': cfg.get('read_only', True), 'RestartPolicy': {'Name': 'unless-stopped'},
                'Init': True, 'DeviceRequests': [{'Driver': 'nvidia', 'Count': 0, 'Capabilities': [['gpu']], 'DeviceIDs': cfg['deploy']['resources']['reservations']['devices'][0]['device_ids']}],
                'PortBindings': {'8080/tcp': [{'HostIp': '127.0.0.1', 'HostPort': cfg['ports'][0].split(':')[1]}]}},
            'Mounts': [{'Type': 'bind', 'Source': v['source'], 'Destination': v['target'],
                    'RW': not v.get('read_only', True)} for v in cfg['volumes']],
            'NetworkSettings': {'Networks': {'local-ai-ollama_default': {'Aliases': cfg['networks']['router']['aliases']}}}}

    def window_of(self, name, ci):
        argv = ci['Config']['Cmd']
        return (int(argv[argv.index('--kv-unified-per-slot') + 1]) if '--kv-unified-per-slot' in argv
            else self.services[name]['context_tokens'])

    def slots_of(self, name, ci):
        argv = ci['Config']['Cmd']
        return (int(argv[argv.index('--parallel') + 1]) if '--parallel' in argv
            else self.services[name]['parallel_slots'])

    def alias_of(self, name, ci):
        argv = ci['Config']['Cmd']
        return (argv[argv.index('--alias') + 1] if '--alias' in argv
            else self.services[name]['model_alias'])

    def inspect(self, name): return copy.deepcopy(self.containers.get(name))

    def docker(self, *args, **kwargs):
        self.events.append(('docker', args))
        if args[:2] == ('image', 'inspect'):
            engine = self.engines[args[2]]
            return json.dumps([{'Id': engine['image_id'],
                'Config': {'Labels': {'org.opencontainers.image.revision': engine['revision']}}}])
        if args[0] == 'stop': self.containers[args[-1]]['State']['Running'] = False
        if args[0] == 'rm': self.containers.pop(args[-1])
        if args[0] == 'logs':
            return '\n'.join(self.container_logs.get(args[-1], []))
        if args[0] == 'compose' and 'up' in args:
            if self.fail_reconcile or self.failed_ups:
                self.failed_ups = max(0, self.failed_ups - 1)
                raise RuntimeError('Simulated load failure')
            compose = read(args[args.index('-f') + 1])
            # The roles are the arguments left after the flags: a resident starts under a bounded
            # readiness wait, and an engine that names its own window waits longer than the usual
            # 450 s. A self-provisioning engine's throwaway container is started with no wait at
            # all, because provision.watch() is the thing measuring that window, so the roles are
            # read off the end of the call rather than off a flag only the resident carries.
            tail = list(args[args.index('up') + 1:])
            flags = ('--pull', '--wait-timeout')  # each takes a value, which is not a role
            roles = []
            while tail:
                item = tail.pop(0)
                if item in flags:
                    tail.pop(0)
                elif not item.startswith('-'):
                    roles.append(item)
            for role in roles: self.install(role, compose['services'][role])
        return ''

    def http(self, url, body=None, headers=None, timeout=None):
        if url.endswith('/runtime-state'):
            # The deployed router reports the backend kinds it can serve, which is what the
            # switch gate verifies before it takes the drain.
            return {'runtime': {'draining': self.draining, 'drain_reason': self.reason,
                    'active_count': self.active_count, 'queued_count': self.queued_count},
                'router': {'supported_backend_kinds': ['ollama', 'llama_cpp', 'strata']},
                'models': [{'id': self.alias_of(name, ci),
                    'x_ollama_router': {'health': {'available': True}, 'output_policy': 'unrestricted',
                        'context_window': self.window_of(name, ci),
                        'default_output_tokens': None, 'max_output_tokens': None}}
                    for name, ci in self.containers.items()]}
        if url.endswith('/runtime-drain'):
            self.events.append(('drain', body['enabled'])); self.draining = body['enabled']; self.reason = body['reason']; return {}
        if url.endswith('/reload-config'):
            self.events.append(('publish',)); return {}
        name = url.split('/')[2].split(':')[0]
        ci = self.containers.get(name)
        if url.endswith('/health'):
            return {'status': 'ok', 'loaded': ci['State'].get('Loaded', self.model_loaded)
                if ci else self.model_loaded}
        if url.endswith('/slots'):
            if ci:
                state = ci['State']
                # An engine that stops itself while loading (the trial's 18:47 attempt died
                # mid-read with the cards free): after N of its own slot looks it is gone, its
                # last words are in the log, and it never answers again. Scoped to
                # self-provisioning engines, so the replacement backend a failed switch restores
                # is not played dead the same way.
                if (self.die_after_slots is not None and state.get('Provisioning')
                        and state.get('Running') and not state.get('Dead')):
                    state['probes'] = state.get('probes', 0) + 1
                    if state['probes'] >= self.die_after_slots:
                        state.update(Running=False, Dead=True, ExitCode=1,
                            FinishedAt='2026-10-12T18:49:31.000000000Z')
                        # The dead engine releases the cards: the global flag is the stand-in for
                        # a load, and the replacement backend is installed after this look.
                        self.model_loaded = True
                        self.container_logs.setdefault(name, []).extend(self.die_words)
            if self.load_after_slots is not None:
                self.load_after_slots -= 1
                if self.load_after_slots <= 0:
                    self.model_loaded = True
                    for c in self.containers.values(): c['State']['Loaded'] = True
            loaded = ci['State'].get('Loaded', self.model_loaded) if ci else self.model_loaded
            if not loaded:
                return []
            return [{'id': i, 'n_ctx': self.slot_context or self.window_of(name, ci),
                    'is_processing': self.direct_busy, 'n_prompt_tokens': 0}
                for i in range(self.slots_of(name, ci))]
        if url.endswith('/completion'):
            self.events.append(('generation', name))
            if self.generation_fails or self.failed_generations:
                self.failed_generations = max(0, self.failed_generations - 1)
                raise RuntimeError('Simulated acceptance failure')
            return {'content': 'K'}
        # The OpenAI chat dialect, which is what an engine configured by its environment answers.
        if url.endswith('/v1/chat/completions'):
            self.events.append(('generation', name))
            if self.generation_fails or self.failed_generations:
                self.failed_generations = max(0, self.failed_generations - 1)
                raise RuntimeError('Simulated acceptance failure')
            return {'choices': [{'message': {'content': 'K'}}]}
        raise AssertionError(url)


class ControllerTests(unittest.TestCase):
    HOST = HOST

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name); host = copy.deepcopy(self.HOST)
        host['router_state_dir'] = str(self.state / 'router')
        atomic_json(self.state / 'host.json', host); (self.state / 'router-token').write_text('synthetic-token')
        self.bundle = render(ROOT / 'config', host, 'qwen27b-q8-pair')
        self.system = FakeSystem(self.bundle)
        self.c = Controller(ROOT / 'config', self.state, 'a' * 40, self.system)
        self.active = {'revision': 'a' * 40, 'image': 'old-image', 'bundle': self.bundle}
        atomic_json(self.state / 'active.json', self.active)
        self.prepare = patch.object(self.c, 'prepare', side_effect=lambda b: self.system.events.append(('prepare',)))
        self.prepare.start(); self.addCleanup(self.prepare.stop)
        self.sleep = patch('runtime.controller.time.sleep'); self.sleep.start(); self.addCleanup(self.sleep.stop)

    def test_healthy_startup_does_not_drain_recreate_or_publish(self):
        self.system.active_count = 1; self.system.direct_busy = True
        result = self.c.transition()
        self.assertTrue(result['already_active'])
        self.assertEqual(self.system.events, [('prepare',)])

    def test_switching_engines_preserves_the_everyday_resident_and_persists_selection(self):
        night = self.system.inspect('qwen38-nighttime')['Id']
        result = self.c.transition('qwen27b-q6k-pair')
        self.assertEqual(result['changed_roles'], ['coding'])
        self.assertEqual(self.system.inspect('qwen38-nighttime')['Id'], night)
        self.assertEqual(self.c.desired()['profile'], 'qwen27b-q6k-pair')
        self.assertTrue(self.c.transition()['already_active'])
        self.assertEqual(self.c.transition('qwen27b-q8-pair')['changed_roles'], ['coding'])
        self.assertEqual(self.system.inspect('qwen38-nighttime')['Id'], night)

    def test_wrong_engine_is_detected_only_on_affected_backend(self):
        self.system.containers['qwen38-daytime']['Image'] = 'sha256:' + 'f' * 64  # not the pinned engine
        observed = self.c.observe(self.bundle)
        self.assertIn('image', observed['coding']['differences'])
        self.assertTrue(observed['everyday']['healthy'])

    def test_stale_controller_cannot_apply_old_source(self):
        self.c.revision = 'b' * 40
        with self.assertRaisesRegex(RuntimeError, 'superseded'): self.c.transition()
        self.assertEqual(self.system.events, [])

    def propose_daytime_change(self):
        proposed = copy.deepcopy(self.bundle)
        cfg = proposed['compose']['services']['coding']
        cfg['command'][cfg['command'].index('--threads') + 1] = '17'
        change = patch.object(self.c, 'desired', return_value=proposed)
        change.start(); self.addCleanup(change.stop)

    def test_launch_change_recreates_only_daytime_after_drain(self):
        self.propose_daytime_change()
        night = self.system.inspect('qwen38-nighttime')['Id']
        result = self.c.transition('qwen27b-q8-pair')
        self.assertEqual(result['changed_roles'], ['coding'])
        self.assertEqual(self.system.inspect('qwen38-nighttime')['Id'], night)
        self.assertFalse(self.system.draining)
        events = self.system.events
        self.assertLess(events.index(('drain', True)), next(i for i, x in enumerate(events) if x[0] == 'docker'))
        self.assertLess(next(i for i, x in enumerate(events) if x[0] == 'generation'), events.index(('publish',)))
        self.assertEqual(events[-1], ('drain', False))
        self.assertEqual(read(self.state / 'active.json')['bundle']['profile'], 'qwen27b-q8-pair')

    def test_busy_router_queue_or_direct_slot_aborts_before_recreation(self):
        self.propose_daytime_change()
        for attribute in ['active_count', 'queued_count', 'direct_busy']:
            with self.subTest(attribute=attribute):
                setattr(self.system, attribute, 1)
                with patch('runtime.controller.time.monotonic', side_effect=[0, 0, 301]):
                    with self.assertRaisesRegex(RuntimeError, 'Drain timed out'): self.c.transition('qwen27b-q8-pair')
                self.assertFalse(any(e[0] == 'docker' for e in self.system.events))
                self.assertFalse(self.system.draining)
                self.assertEqual(read(self.state / 'active.json'), self.active)
                setattr(self.system, attribute, 0); self.system.events.clear()

    def test_failed_load_recovers_without_recreating_healthy_previous_pair(self):
        self.propose_daytime_change()
        self.system.fail_reconcile = True
        before = {k: v['Id'] for k, v in self.system.containers.items()}
        with self.assertRaisesRegex(RuntimeError, 'Simulated load failure'): self.c.transition('qwen27b-q8-pair')
        self.assertEqual({k: v['Id'] for k, v in self.system.containers.items()}, before)
        self.assertEqual(read(self.state / 'transaction.json')['phase'], 'recovered')
        self.assertFalse(self.system.draining)

    def test_failed_recovery_keeps_drain_and_blocks_new_operations(self):
        self.propose_daytime_change()
        self.system.generation_fails = True
        with self.assertRaisesRegex(RuntimeError, 'recovery needs attention'): self.c.transition('qwen27b-q8-pair')
        self.assertTrue(self.system.draining)
        self.assertEqual(read(self.state / 'transaction.json')['phase'], 'needs-attention')
        with self.assertRaisesRegex(RuntimeError, 'interrupted'): self.c.transition()
        self.system.generation_fails = False
        self.c.recover()
        self.assertFalse(self.system.draining)

    def test_foreign_drain_is_never_cleared(self):
        self.system.draining = True; self.system.reason = 'another owner'
        with self.assertRaisesRegex(RuntimeError, 'Another operation owns'): self.c.transition('qwen27b-q8-pair')
        self.assertNotIn(('drain', False), self.system.events)

    def test_shared_lock_blocks_cli_and_startup(self):
        with lock(self.state / 'runtime.lock'):
            with self.assertRaisesRegex(RuntimeError, 'maintenance lock'): self.c.transition()

    def test_adoption_refuses_profile_mismatch(self):
        self.propose_daytime_change()
        (self.state / 'active.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'exact match'): self.c.transition('qwen27b-q8-pair', adopt=True)
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

    def test_status_lists_every_configuration_and_marks_the_live_one(self):
        configs = self.c.status()['configurations']
        self.assertEqual(configs['active'], 'qwen27b-q8-pair')
        self.assertEqual([x['profile'] for x in configs['selectable']], list(CONFIGURATIONS))
        self.assertEqual([x['profile'] for x in configs['selectable'] if x['exclusive']],
            ['flash-next-strata-solo-thinking-recovery',
             'flash-next-strata-solo-thinking-recovery-cache',
             'flash-next-strata-solo-thinking-recovery-cache-yarn'])
        self.assertEqual([x['display_name'] for x in configs['selectable']], ['Qwen3.8 27B Q8 (160K)',
            'Qwen3.8 27B Q6_K (160K)', 'Qwen3.8 Flash-Next (256K)', 'Qwen3.8 Flash-Next (256K)',
            'Qwen3.8 Flash-Next (512K)', 'Qwen3.8 Flash-Next Two-Card (256K)'])
        # Since 2026-10-18 the label is computed from context_tokens, so a rung of the session-size
        # ladder edited in the profile shows up on the page rather than only in the engine that boots
        # on it -- and the two 256K configurations move together on every rung: the run config is one
        # file per workspace, and two configurations of one resident that disagreed about the window
        # would make a switch between them a resize. The 512K configuration states that resize as a
        # fact of its own -- window and ring are its one variable from its sibling -- so the label
        # says 512K. The pair keeps 160K, which is why all five numbers are named.
        #
        # The sixth row is where that naming earns its keep: the two-card probe is a 256K configuration
        # too, and it is the only one of the three whose cards are the reason to select it, so it is
        # the first configuration whose display name says something about the hardware. Two rows both
        # reading "Flash-Next (256K)" would be a page where the operator's choice is a coin flip.
        self.assertEqual([x['context_tokens'] for x in configs['selectable']],
            [163840, 163840, 262144, 262144, 524288, 262144])
        # The Flash-Next configurations, named by prefix rather than by a trailing slice: what tells an
        # operator which of them is which is the variant label, and it also says what each one turns
        # on, because that is the only choice left to make here.
        strata = [x for x in configs['selectable'] if x['profile'].startswith('flash-next')]
        self.assertEqual([x.get('variant_label') for x in strata],
            ['card sampling + call recovery',
             'card sampling + call recovery + parked conversations',
             'card sampling + call recovery + parked conversations + YaRN 2.0 window',
             'two cards, 512 MiB a card reserved'])
        self.assertEqual({x['parallel_slots'] for x in configs['selectable']}, {1})
        self.assertEqual(configs['selectable'][2]['gpu_names'], [*HOST['gpu_names']['daytime'], *HOST['gpu_names']['nighttime']])
        self.assertEqual(configs['selectable'][1]['context_tokens'], 163840)
        self.assertEqual(configs['selectable'][2]['engine'], 'strata-vision-296b59d2f73c')
        # What a client can ask each configuration for. The solo one answers all three IDs from the
        # one resident it loads; a paired one answers `nighttime` through the resident it loads beside
        # its own, so the pair's list is the two residents together.
        self.assertEqual(configs['selectable'][2]['service_ids'], ['daytime', 'local-active', 'nighttime'])
        self.assertEqual(configs['selectable'][0]['service_ids'], ['daytime', 'local-active', 'nighttime'])
        self.assertEqual([x['profile'] for x in configs['always_included']], ['everyday'])
        self.assertEqual(configs['always_included'][0]['display_name'], 'Qwen3.8 27B Abliterated Q6_K (96K)')
        self.assertEqual(configs['always_included'][0]['service_ids'], ['nighttime'])
        self.assertEqual(configs['selectable'][0]['gpu_names'], HOST['gpu_names']['daytime'])
        self.assertEqual(configs['always_included'][0]['gpu_names'], HOST['gpu_names']['nighttime'])
        self.assertNotIn('synthetic-token', json.dumps(configs))
        self.assertNotIn(HOST['model_root'], json.dumps(configs))
        self.c.transition('qwen27b-q6k-pair')
        self.assertEqual(self.c.status()['configurations']['active'], 'qwen27b-q6k-pair')
        (self.state / 'active.json').unlink()  # Before adoption nothing is live, but the registry is still listable.
        pre_migration = self.c.status()
        self.assertFalse(pre_migration['ready'])
        self.assertEqual(pre_migration['configurations']['active'], 'qwen27b-q8-pair')
        self.assertEqual(len(pre_migration['configurations']['selectable']), len(CONFIGURATIONS))

    def test_unreadable_registry_never_hides_the_health_of_the_pair(self):
        with patch('runtime.controller.available_profiles', side_effect=RuntimeError('registry exploded')):
            status = self.c.status()
        self.assertTrue(status['ready'])
        self.assertEqual(status['configurations']['active'], 'qwen27b-q8-pair')
        self.assertEqual(status['configurations']['selectable'], [])
        self.assertEqual(status['configurations']['always_included'], [])
        self.assertIn('registry exploded', status['configurations']['error'])
        self.assertTrue(all(s['healthy'] for s in status['services']))

    def test_router_reservation_blocks_profile_changes_until_verified_completion(self):
        self.c.router_maintenance(True)
        self.assertTrue(self.system.draining)
        with self.assertRaisesRegex(RuntimeError, 'reserves the runtime'): self.c.transition('qwen27b-q8-pair')
        self.c.republish()
        self.c.router_maintenance(False)
        self.assertFalse(self.system.draining)
        self.assertFalse(read(self.state / 'router-maintenance.json')['reserved'])

    def test_publication_identifies_the_configuration_even_for_an_older_bundle(self):
        self.c.republish()
        catalog = read(self.state / 'router/active-model.json')
        self.assertEqual({k: catalog['configuration'][k] for k in ('id', 'exclusive', 'runtime_revision')},
            {'id': 'qwen27b-q8-pair', 'exclusive': False, 'runtime_revision': 'a' * 40})
        self.assertEqual(catalog['offline_services'], [])
        # A previous release's bundle predates these fields; recovery still publishes an accurate id.
        legacy = copy.deepcopy(self.bundle)
        del legacy['catalog']['configuration'], legacy['catalog']['offline_services']
        self.c.publish(legacy)
        catalog = read(self.state / 'router/active-model.json')
        self.assertEqual((catalog['configuration']['id'], catalog['configuration']['exclusive']),
            ('qwen27b-q8-pair', False))
        self.assertNotIn('offline_services', catalog)

    def test_configuration_change_outside_argv_still_reconciles_affected_service(self):
        proposed = copy.deepcopy(self.bundle)
        proposed['compose']['services']['coding']['logging']['options']['max-size'] = '30m'
        with patch.object(self.c, 'desired', return_value=proposed): result = self.c.transition()
        self.assertEqual(result['changed_roles'], ['coding'])


class ExclusiveProfileTests(unittest.TestCase):
    """An exclusive configuration removes the Everyday resident before loading; a paired one restores it."""
    HOST = vision_host()
    setUp = ControllerTests.setUp
    NIGHT = 'qwen38-nighttime'
    SOLO = 'flash-next-strata-solo-thinking-recovery'

    def setUp(self):
        ControllerTests.setUp(self)
        # This engine builds its expert pack into a workspace before it answers a health check, and
        # the switch does that build before it takes the drain. The build is the subject of
        # tests/test_provision.py; what is under test here is which GPUs a backend may hold and the
        # order the residents move in, and the workspace the bundle binds does not exist on a test
        # host, so the switch is pointed at the half these tests are about.
        skip = patch.object(self.c, 'preparation_needed', return_value=False)
        skip.start(); self.addCleanup(skip.stop)
        # Docker learns which configuration to start from the compose file it is handed. The fake
        # learns what the backends of that configuration report the same way: an engine configured
        # by its environment has no argument vector to read a served name or a window off, so it
        # answers from the published configuration it was started from. Restoration hands the
        # controller the bundle it came back to through this same path, which is what lets a
        # restored backend answer as the configuration it was restored to.
        written = self.c.compose_path
        def compose_path(bundle):
            for service in bundle['manifest']['services']:
                self.system.services[service['container_name']] = service
            return written(bundle)
        learn = patch.object(self.c, 'compose_path', side_effect=compose_path)
        learn.start(); self.addCleanup(learn.stop)

    def ups(self):
        """The residents that were started, in order.

        Only a resident's `compose up` carries a bounded readiness wait; a self-provisioning engine's
        throwaway container is started without one, and it is not a service the router serves. The
        flag is what tells the two apart, not the length of the window -- an engine that names its
        own waits far longer than the usual 450 s.
        """
        return [e[1][e[1].index('--wait-timeout') + 2:] for e in self.system.events
            if e[0] == 'docker' and e[1][0] == 'compose' and 'up' in e[1] and '--wait-timeout' in e[1]]

    def published(self):
        return [m['model'] for m in read(self.state / 'router/active-model.json')['models']]

    def solo_settings(self):
        """What identifies the solo backend, for a backend that has no argument vector.

        The Strata engine is configured from its environment, so the environment list is the
        fingerprint the llama.cpp backends' argv is for: two containers match only if the same
        configuration is running in both.
        """
        return self.system.inspect('qwen38-daytime')['Config']['Env']

    def test_switch_to_solo_removes_everyday_before_loading_and_publishes_one_model(self):
        day = self.system.inspect('qwen38-daytime')['Id']
        result = self.c.transition(self.SOLO)
        self.assertEqual(result['changed_roles'], ['coding'])
        self.assertIsNone(self.system.inspect(self.NIGHT))
        events = self.system.events
        drain = events.index(('drain', True))
        stop = events.index(('docker', ('stop', '--time', '60', self.NIGHT)))
        remove = events.index(('docker', ('rm', self.NIGHT)))
        load = next(i for i, e in enumerate(events) if e[0] == 'docker' and 'up' in e[1])
        self.assertLess(drain, stop); self.assertLess(stop, remove); self.assertLess(remove, load)
        self.assertEqual(self.ups(), [('coding',)])
        self.assertNotEqual(self.system.inspect('qwen38-daytime')['Id'], day)
        self.assertEqual(self.published(), ['qwen3.8-flash-next-ud-iq4_xs'])
        catalog = read(self.state / 'router/active-model.json')
        self.assertEqual({k: catalog['configuration'][k] for k in ('id', 'exclusive', 'runtime_revision')},
            {'id': self.SOLO, 'exclusive': True, 'runtime_revision': 'a' * 40})
        self.assertTrue(catalog['configuration']['published_at'])
        # Nothing is offline here, and that is the point of this configuration: it answers `nighttime`
        # from the one backend it loads, so a client holding that name is never told the service is off.
        self.assertEqual(catalog['offline_services'], [])
        self.assertEqual([m['aliases'] for m in catalog['models']], [['local-active', 'daytime', 'nighttime']])
        self.assertEqual(events[-1], ('drain', False))
        tx = read(self.state / 'transaction.json')
        self.assertEqual((tx['phase'], tx['removed_backends']), ('succeeded', [self.NIGHT]))
        self.assertNotIn('protected_services', tx)
        status = self.c.status()
        self.assertTrue(status['ready'])
        self.assertEqual(status['offline_roles'], ['everyday'])
        [service] = status['services']
        self.assertTrue(service['exclusive'])
        self.assertEqual(service['gpu_ids'], [*self.HOST['gpu_ids']['daytime'], *self.HOST['gpu_ids']['nighttime']])
        self.assertEqual(service['gpu_names'], [*self.HOST['gpu_names']['daytime'], *self.HOST['gpu_names']['nighttime']])
        # The encoder runs on the shared card, and this configuration loads no second resident, so it
        # has that card to itself: `vision_gpu_shared` false is the difference from a paired
        # configuration, where the projector is a guest on the Everyday resident's host.
        self.assertEqual((service['vision_device'], service['vision_gpu_shared']), ('CUDA4', False))
        self.assertTrue(self.c.transition()['already_active'])
        # A switch that stays exclusive has nothing to remove and never starts the second resident.
        # One configuration owns all four text GPUs, so this arrives as the same topology under a
        # changed backend: the coding resident reloads, the placement does not move.
        self.system.events.clear()
        changed = copy.deepcopy(self.c.desired())
        changed['compose']['services']['coding']['environment']['LAYER_SPLIT'] = '0,0,0,0'
        with patch.object(self.c, 'desired', return_value=changed):
            self.assertEqual(self.c.transition(self.SOLO)['changed_roles'], ['coding'])
        self.assertFalse(any(e[0] == 'docker' and e[1][0] in ('stop', 'rm') for e in self.system.events))
        self.assertEqual(self.ups(), [('coding',)])
        self.assertIsNone(self.system.inspect(self.NIGHT))

    def test_status_keeps_an_offline_resident_and_an_offline_service_apart(self):
        """`offline_roles` is placement; `offline_services` is what a client cannot ask for.

        The solo configuration loads one resident and answers all three service IDs, so the honest
        status says the Everyday resident is not loaded and no service is offline. Read as one fact
        they cancel each other out, which is how a page ends up telling a client its `nighttime` is
        missing while every request to it is being served.
        """
        status = self.c.status()
        self.assertEqual((status['offline_roles'], status['offline_services']), ([], []))
        self.c.transition(self.SOLO)
        status = self.c.status()
        self.assertTrue(status['ready'])
        self.assertEqual(status['offline_roles'], ['everyday'])
        self.assertEqual(status['offline_services'], [])

    def test_switch_back_from_solo_starts_daytime_before_everyday(self):
        original = self.system.inspect('qwen38-daytime')['Config']['Cmd']
        self.c.transition(self.SOLO)
        self.system.events.clear()
        self.assertEqual(self.c.transition('qwen27b-q8-pair')['changed_roles'], ['coding', 'everyday'])
        self.assertEqual(self.ups(), [('coding',), ('everyday',)])
        self.assertEqual(self.system.inspect('qwen38-daytime')['Config']['Cmd'], original)
        self.assertTrue(self.system.inspect(self.NIGHT)['State']['Running'])
        self.assertEqual(self.published(), ['qwen3.8-27b-q8_0-tensor-next', 'qwen3.8-27b-abliterated-q6_k'])
        status = self.c.status()
        self.assertTrue(status['ready']); self.assertEqual(status['offline_roles'], [])
        self.assertEqual(status['services'][0]['gpu_names'], self.HOST['gpu_names']['daytime'])

    def test_failed_solo_load_restores_everyday_and_the_pair(self):
        day = self.system.inspect('qwen38-daytime')['Id']
        self.system.failed_ups = 1
        with self.assertRaisesRegex(RuntimeError, 'Simulated load failure'): self.c.transition(self.SOLO)
        self.assertEqual(self.ups(), [('coding',), ('everyday',)])
        self.assertEqual(self.system.inspect('qwen38-daytime')['Id'], day)
        self.assertTrue(self.system.inspect(self.NIGHT)['State']['Running'])
        self.assertEqual(read(self.state / 'transaction.json')['phase'], 'recovered')
        self.assertEqual(self.c.desired()['profile'], 'qwen27b-q8-pair')
        self.assertEqual(len(self.published()), 2)
        self.assertFalse(self.system.draining)
        self.assertTrue(self.c.status()['ready'])

    def test_rejected_solo_restores_daytime_before_recreating_everyday(self):
        original = self.system.inspect('qwen38-daytime')['Config']['Cmd']
        self.system.failed_generations = 1
        with self.assertRaisesRegex(RuntimeError, 'Simulated acceptance failure'): self.c.transition(self.SOLO)
        self.assertEqual(self.ups(), [('coding',), ('coding',), ('everyday',)])
        self.assertEqual(self.system.inspect('qwen38-daytime')['Config']['Cmd'], original)
        self.assertTrue(self.system.inspect(self.NIGHT)['State']['Running'])
        self.assertEqual(read(self.state / 'transaction.json')['phase'], 'recovered')
        self.assertTrue(self.c.status()['ready'])

    def test_failed_return_to_the_pair_removes_new_everyday_and_restores_solo(self):
        self.c.transition(self.SOLO)
        solo = self.solo_settings()
        self.system.events.clear(); self.system.failed_generations = 1
        with self.assertRaisesRegex(RuntimeError, 'Simulated acceptance failure'): self.c.transition('qwen27b-q8-pair')
        self.assertEqual(self.ups(), [('coding',), ('everyday',), ('coding',)])
        self.assertIsNone(self.system.inspect(self.NIGHT))
        self.assertEqual(self.system.inspect('qwen38-daytime')['Config']['Env'], solo)
        self.assertEqual(read(self.state / 'transaction.json')['phase'], 'recovered')
        self.assertEqual(self.c.desired()['profile'], self.SOLO)
        self.assertEqual(self.published(), ['qwen3.8-flash-next-ud-iq4_xs'])

    def test_two_slot_backend_is_verified_by_slot_count_and_per_slot_context(self):
        # No configuration runs two slots now; the capability is exercised on a copy of the configuration.
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        config = Path(tmp.name) / 'config'; shutil.copytree(ROOT / 'config', config)
        path = config / ('profiles/' + self.SOLO + '.json'); definition = read(path)
        definition['parallel_slots'] = 2; path.write_text(json.dumps(definition))
        self.c.config_dir = config
        self.c.transition(self.SOLO)
        self.assertTrue(self.c.status()['ready'])
        bundle = self.c.desired(); name = 'qwen38-daytime'
        # This engine reports its own slots over /slots; one slot where the configuration contracted
        # for two is a backend that came up smaller than it was asked to.
        self.system.services[name]['parallel_slots'] = 1
        self.assertIn('slot context', self.c.observe(bundle)['coding']['differences'])

    def test_everyday_beside_an_exclusive_profile_is_drift_and_is_removed(self):
        pair = render(ROOT / 'config', self.HOST, 'qwen27b-q8-pair')
        self.c.transition(self.SOLO)
        solo = self.c.desired()
        self.system.install('everyday', pair['compose']['services']['everyday'])
        self.assertIn('competing everyday backend running', self.c.observe(solo)['coding']['differences'])
        self.assertFalse(self.c.status()['ready'])
        result = self.c.transition()
        self.assertEqual(result['changed_roles'], ['coding'])
        self.assertIsNone(self.system.inspect(self.NIGHT))
        self.assertTrue(self.c.status()['ready'])


class ArtifactTests(unittest.TestCase):
    def test_each_pinned_engine_is_validated_before_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); atomic_json(root / 'host.json', HOST)
            bundle = render(ROOT / 'config', HOST, 'qwen27b-q8-pair')
            bundle['artifacts'] = []
            for role in ('coding', 'everyday'):
                with self.subTest(role=role):
                    system = FakeSystem(bundle)
                    system.engines = copy.deepcopy(system.engines)
                    # The pair shares one engine image, so each role gets its own wrong-image copy of it.
                    tag = service_engine(bundle, role)['tag']
                    bundle_role = copy.deepcopy(bundle)
                    other = [s for s in bundle_role['manifest']['services'] if s['role'] == role][0]
                    other['engine'] = {**other['engine'], 'tag': tag + '-' + role}
                    bundle_role['compose']['services'][role]['image'] = tag + '-' + role
                    system.engines[tag + '-' + role] = {**system.engines[tag], 'image_id': 'wrong-image'}
                    c = Controller(ROOT / 'config', root, 'a' * 40, system)
                    with self.assertRaisesRegex(RuntimeError, role + ': pinned inference engine'):
                        c.validate(bundle_role)

    def test_previous_single_engine_manifest_remains_readable_for_recovery(self):
        bundle = render(ROOT / 'config', HOST, 'qwen27b-q6k-pair')
        engine = service_engine(bundle, 'coding')
        for service in bundle['manifest']['services']: service.pop('engine')
        bundle['manifest'].update(schema_version=1, engine=engine)
        self.assertEqual(service_engine(bundle, 'coding'), engine)
        self.assertEqual(service_engine(bundle, 'everyday'), engine)

    def test_same_size_change_invalidates_cached_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model = root / 'model.gguf'; model.write_bytes(b'good')
            host = copy.deepcopy(HOST); host['model_root'] = str(root)
            atomic_json(root / 'host.json', host)
            bundle = render(ROOT / 'config', host, 'qwen27b-q8-pair')
            bundle['artifacts'] = [{'source': str(model), 'bytes': 4, 'sha256': hashlib.sha256(b'good').hexdigest()}]
            c = Controller(ROOT / 'config', root, 'a' * 40, FakeSystem(bundle))
            self.assertTrue(c.validate(bundle)['all_checksums_verified'])
            model.write_bytes(b'evil')
            with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'): c.validate(bundle)


class EnvironmentEngineObservationTests(unittest.TestCase):
    """The trial's first switch attempt died in this pass, so it has to degrade and never throw.

    It failed with KeyError: 'restart' after the artifact checks had hashed all 93 GB and before any
    container was touched: the generated service dictionary for that backend had lost the keys the
    shared container defaults provide, and the observer read every one of them strictly. A bundle that
    cannot satisfy a comparison must produce a drift reason the operator can read.
    """

    PROFILE = 'flash-next-strata-solo-thinking-recovery'

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name)
        host = copy.deepcopy(HOST)
        host['router_state_dir'] = str(self.state / 'router')
        atomic_json(self.state / 'host.json', host); (self.state / 'router-token').write_text('synthetic-token')
        self.bundle = render(ROOT / 'config', host, self.PROFILE)
        self.system = FakeSystem(self.bundle)
        self.c = Controller(ROOT / 'config', self.state, 'a' * 40, self.system)

    def test_the_strata_backend_observes_clean(self):
        observed = self.c.observe(self.bundle)
        self.assertEqual(observed['coding']['differences'], [])
        self.assertTrue(observed['coding']['healthy'])
        self.assertNotIn('slot_window', observed['coding'])
        self.assertNotIn('health_error', observed['coding'])

    def test_a_service_dictionary_missing_a_setting_drifts_instead_of_crashing(self):
        for key, reason in [('restart', 'restart policy'), ('init', 'init'),
                ('entrypoint', 'entrypoint'), ('read_only', 'read-only root'),
                ('deploy', 'GPU assignment'), ('volumes', 'model mounts'),
                ('ports', 'port bindings'), ('networks', 'router network')]:
            # An absent command and an empty one mean the same thing to the comparison, so that key is
            # deliberately not in this list: the Strata engine takes its settings from the environment.
            # Neither is 'user', which this resident legitimately does not render at all; it has its own
            # test below, because there 'nothing' and 'empty string' are the same container.
            damaged = copy.deepcopy(self.bundle)
            damaged['compose']['services']['coding'].pop(key)
            with self.subTest(setting=key):
                observed = self.c.observe(damaged)
                self.assertIn(reason, observed['coding']['differences'])
                self.assertFalse(observed['coding']['healthy'])

    def test_a_backend_running_as_its_image_user_is_not_called_drifted(self):
        """The Strata resident renders no user and the daemon reports that as an empty string.

        Read as literally different, every health check would insist the container that is correctly
        serving is wrong, which is how a working backend gets restarted on a loop and a switch gets
        blamed for it.
        """
        cfg = self.bundle['compose']['services']['coding']
        self.assertNotIn('user', cfg)
        self.assertEqual(self.system.containers[cfg['container_name']]['Config']['User'], '')
        self.assertNotIn('user', self.c.observe(self.bundle)['coding']['differences'])
        damaged = copy.deepcopy(self.bundle)
        damaged['compose']['services']['coding']['user'] = '0:0'
        self.assertIn('user', self.c.observe(damaged)['coding']['differences'])

    def test_a_backend_that_came_up_on_another_window_says_so_with_its_number(self):
        """Strata reports the window the engine loaded with, which need not be the one asked for.

        An empty slot list is a backend still loading and a smaller window is permanent, and both
        would otherwise sit out the whole health timeout looking identical.
        """
        self.system.slot_context = 131072
        observed = self.c.observe(self.bundle)
        self.assertIn('slot context', observed['coding']['differences'])
        self.assertFalse(observed['coding']['healthy'])
        contracted = self.bundle['manifest']['services'][0]['context_tokens']
        self.assertEqual(observed['coding']['slot_window']['contracted'], contracted)
        self.assertEqual(observed['coding']['slot_window']['reported_context'], [131072])

    def test_the_llama_path_observes_exactly_as_before(self):
        # The guarded reads and the merge sit under both engine shapes; the backends running today are
        # observed with no drift and no new reasons. Both are paired, so both take the ordinary host;
        # the configuration that owns all four text GPUs is the environment engine this class observes
        # above, and it is observed on both host shapes there.
        for profile in ('qwen27b-q8-pair', 'qwen27b-q6k-pair'):
            with self.subTest(profile=profile):
                system = FakeSystem(render(ROOT / 'config', copy.deepcopy(HOST), profile))
                controller = Controller(ROOT / 'config', self.state, 'a' * 40, system)
                observed = controller.observe(system.bundle)
                self.assertEqual({role: x['differences'] for role, x in observed.items()},
                    {role: [] for role in observed})

    def test_the_strata_resident_gains_the_vision_card_and_nothing_else_moves(self):
        """The production host pins the RTX 3080 as the vision card; a host here may name none.

        Since 2026-10-16 the two renders are expected to differ, and the difference is the encoder's
        card plus the variables that name it. The engine's own card list, its mounts, its window and
        its launch environment are byte-identical, which is the property this switch has to keep:
        lending the resident a fifth card may not move a byte of the four it is laid out across.
        """
        plain = render(ROOT / 'config', copy.deepcopy(HOST), self.PROFILE)
        lent = render(ROOT / 'config', vision_host(), self.PROFILE)
        before, after = (x['compose']['services']['coding'] for x in (plain, lent))
        self.assertEqual(after['environment']['CUDA_VISIBLE_DEVICES'],
            before['environment']['CUDA_VISIBLE_DEVICES'] + ',4')
        self.assertEqual((after['environment']['VISION'], after['environment']['VISION_DEVICE']),
            ('yes', '4'))
        quiet = lambda e: {k: v for k, v in e.items() if k not in ('VISION', 'VISION_DEVICE', 'CUDA_VISIBLE_DEVICES')}
        self.assertEqual(quiet(after['environment']), quiet(before['environment']))
        reserved = lambda s: s['deploy']['resources']['reservations']['devices'][0]['device_ids']
        self.assertEqual(reserved(after)[:4], reserved(before))
        self.assertEqual(reserved(after)[4], VISION)
        for key in ('volumes', 'command', 'entrypoint', 'ports', 'healthcheck', 'ulimits'):
            with self.subTest(service_key=key):
                self.assertEqual(after[key], before[key])
        model, image = (x['catalog']['models'][0] for x in (plain, lent))
        self.assertIs(model['capability_profile']['vision'], False)
        self.assertIs(image['capability_profile']['vision'], True)
        self.assertEqual(model['input_modalities'], ['text'])
        self.assertEqual(image['input_modalities'], ['text', 'image'])
        self.assertEqual(image['gpu_uuids'], [*model['gpu_uuids'], VISION])
        self.assertEqual(image['text_gpu_uuids'], model['gpu_uuids'])

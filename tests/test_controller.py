import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from models import CODING, CODING_Q6K, EVERYDAY, STRATA_DAISY, STRATA_PAIR, STRATA_SOLO
from runtime.config import CONFIGURATIONS, digest, read, render, service_engine
from runtime.controller import Controller
from runtime.system import atomic_json, lock, System

ROOT = Path(__file__).resolve().parents[1]
HOST = read(ROOT / 'tests/fixtures/legacy-fingerprints.json')['host']
# A name no configuration of this revision is built for: a host that states it is healthy and offered
# nothing, which is a state an operator has to be able to see on the page and then go and fix.
UNCLAIMED = 'no-machine-claims-this-one'
# The cards this fixture's host declares, in the order a configuration reserves them: the map draws
# one tile per entry and colours it by who has it, so the fixture's inventory is asserted as a whole.
INVENTORY = {'text': [
    {'uuid': 'GPU-day-1', 'group': 'daytime', 'order': 0, 'position': 0, 'name': 'RTX 3090'},
    {'uuid': 'GPU-day-2', 'group': 'daytime', 'order': 1, 'position': 1, 'name': 'RTX 4080 SUPER'},
    {'uuid': 'GPU-night-1', 'group': 'nighttime', 'order': 2, 'position': 0, 'name': 'RTX 4080'},
    {'uuid': 'GPU-night-2', 'group': 'nighttime', 'order': 3, 'position': 1, 'name': 'RTX 3080 Ti'}],
    'vision': None}
VISION = 'GPU-465a9a1e-fcb9-27f2-8f6a-cca1c30de981'
# Every configuration of this host puts its first model in `qwen38-daytime` and its second in
# `qwen38-nighttime`, so a container name is shared by launches that never share a model id. That is
# why a competing backend can only ever be named by container: a model id says which launch someone
# wanted, and two of them live in the same place.
DAY = 'qwen38-daytime'
NIGHT = 'qwen38-nighttime'


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
        for service, cfg in bundle['compose']['services'].items(): self.install(service, cfg)

    def install(self, service, cfg):
        self.serial += 1
        self.containers[cfg['container_name']] = self.container(service, cfg)

    def container(self, service, cfg):
        return {'Id': service + str(self.serial), 'Image': self.engines[cfg['image']]['image_id'],
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
                'PortBindings': {'8080/tcp': [{'HostIp': cfg['ports'][0].rsplit(':', 2)[0],
                    'HostPort': cfg['ports'][0].rsplit(':', 2)[1]}]}},
            'Mounts': [{'Type': 'bind', 'Source': v['source'], 'Destination': v['target'],
                    'RW': not v.get('read_only', True)} for v in cfg['volumes']],
            'NetworkSettings': {'Networks': {'local-ai-ollama_default': {'Aliases': cfg['networks']['router']['aliases']}}}}

    def unchanged(self, service, cfg):
        """Whether the container standing here is the one a plain `compose up` would leave alone.

        Compose recreates a service only when what is running is not what the file asks for, and the
        safety of asking for every service of a release rests on that: recovery asks for a launch it
        did not disturb and the daemon must not restart it, because a healthy backend that keeps
        serving is the whole job of a recovery. `--force-recreate` overrides this, and the caller that
        passes it is a switch, not a recovery.
        """
        ci = self.containers.get(cfg['container_name'])
        identity = lambda record: {k: v for k, v in record.items() if k not in ('Id', 'State', 'RestartCount')}
        return bool(ci and ci['State']['Running']
            and identity(ci) == identity(self.container(service, cfg)))

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
            recreate = '--force-recreate' in args
            # The services are the arguments left after the flags: a launch starts under a bounded
            # readiness wait, and an engine that names its own window waits longer than the usual
            # 450 s. A self-provisioning engine's throwaway container is started with no wait at
            # all, because provision.watch() is the thing measuring that window, so the services are
            # read off the end of the call rather than off a flag only a served launch carries.
            tail = list(args[args.index('up') + 1:])
            flags = ('--pull', '--wait-timeout')  # each takes a value, which is not a service
            services = []
            while tail:
                item = tail.pop(0)
                if item in flags:
                    tail.pop(0)
                elif not item.startswith('-'):
                    services.append(item)
            for service in services:
                # `compose up` leaves a container alone when what is running is what the file asks for,
                # unless the call carries `--force-recreate` -- which is how a switch says no process
                # started by another configuration survives on these cards, however well its launch
                # document matches. Recovery asks without the flag, and that difference is the point.
                if recreate or not self.unchanged(service, compose['services'][service]):
                    self.install(service, compose['services'][service])
        return ''

    def http(self, url, body=None, headers=None, timeout=None, method=None):
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

    def published(self):
        return json.loads((self.state / 'router' / 'active-model.json').read_text())

    def ups(self):
        """The services each `compose up` asked for, in the order they were asked and in the order it
        named them.

        This is what a switch actually guarantees: which launches the machine went to get, and one at
        a time or together. Only a served launch carries a bounded readiness wait; a self-provisioning
        engine's throwaway container is started without one, and it is not something the router serves.
        The flag is what tells the two apart, not the length of the window -- an engine that names its
        own waits far longer than the usual 450 s.
        """
        return [e[1][e[1].index('--wait-timeout') + 2:] for e in self.system.events
            if e[0] == 'docker' and e[1][0] == 'compose' and 'up' in e[1] and '--wait-timeout' in e[1]]

    def up_calls(self):
        """Every `docker compose up` the machine made, in order, flags and all."""
        return [e[1] for e in self.system.events
            if e[0] == 'docker' and e[1][0] == 'compose' and 'up' in e[1]]

    def test_a_saved_alias_is_the_name_every_surface_shares(self):
        # One name, said the same way by the page, the CLI listing and the catalog this machine
        # publishes: three surfaces that disagree about what a machine is running is the bug, and an
        # alias that only reached the page would be exactly that bug with an extra step.
        self.assertEqual(self.c.set_alias('qwen3.8-27b-q8_0-tensor-next', 'Workhorse'), {'saved': True, 'published': True})
        self.assertEqual(self.c.set_alias('qwen3.8-27b-abliterated-q6_k', 'Night owl'), {'saved': True, 'published': True})
        self.assertEqual(json.loads((self.state / 'aliases.json').read_text()),
            {'version': 2, 'aliases': {'qwen3.8-27b-q8_0-tensor-next': 'Workhorse', 'qwen3.8-27b-abliterated-q6_k': 'Night owl'}})
        status = self.c.status()
        self.assertEqual([x['name'] for x in status['services']], ['Workhorse', 'Night owl'])
        entry = next(x for x in status['configurations']['selectable'] if x['profile'] == 'qwen27b-q8-pair')
        self.assertEqual((entry['display_name'], entry['default_name'], entry['alias']),
            ('Workhorse', 'Qwen3.8 27B Q8 (160K)', 'Workhorse'))
        self.assertEqual([x['display_name'] for x in self.published()['models']], ['Workhorse', 'Night owl'])
        self.assertEqual(status['aliases'],
            {'assignments': {'qwen3.8-27b-q8_0-tensor-next': 'Workhorse', 'qwen3.8-27b-abliterated-q6_k': 'Night owl'}})
        # Clearing one name takes back the default and leaves the other standing.
        self.c.set_alias('qwen3.8-27b-q8_0-tensor-next', None)
        self.assertEqual([x['display_name'] for x in self.published()['models']],
            ['Qwen3.8 27B Q8 (160K)', 'Night owl'])
        self.assertEqual(json.loads((self.state / 'aliases.json').read_text())['aliases'], {'qwen3.8-27b-abliterated-q6_k': 'Night owl'})

    def test_a_published_name_says_which_machine_the_model_runs_on(self):
        # A client reads one model list for a household of machines, so the name a machine publishes
        # says where the model is: the operator's word for the box when they chose one, and the slug it
        # states otherwise. Nothing is renamed in this machine -- these are the defaults, published by
        # a machine that has no aliases written for it.
        serving = ['Rosalina: Qwen3.8 27B Q8 (160K)', 'Rosalina: Qwen3.8 27B Abliterated Q6_K (96K)']
        self.c.host = {**self.c.host, 'host_id': 'rosalina', 'display_name': 'Rosalina'}
        self.c.publish(self.active['bundle'])
        self.assertEqual([x['display_name'] for x in self.published()['models']], serving)
        self.assertEqual([x['name'] for x in self.c.status()['services']], serving)
        # A rename is the operator naming the model, machine and all, and clearing it brings the
        # machine back as part of the default it was never separate from.
        self.c.set_alias('qwen3.8-27b-q8_0-tensor-next', 'Workhorse')
        self.assertEqual([x['display_name'] for x in self.published()['models']], ['Workhorse', serving[1]])
        self.c.set_alias('qwen3.8-27b-q8_0-tensor-next', None)
        self.assertEqual([x['display_name'] for x in self.published()['models']], serving)
        # A machine that states only its slug is still a machine, and says so in the same shape.
        self.c.host = {k: v for k, v in self.c.host.items() if k != 'display_name'}
        self.c.publish(self.active['bundle'])
        self.assertEqual([x['display_name'] for x in self.published()['models']],
            ['rosalina: Qwen3.8 27B Q8 (160K)', 'rosalina: Qwen3.8 27B Abliterated Q6_K (96K)'])
        # And a machine that has never said which one it is says nothing about itself: the names it
        # published before there was a second box, unchanged, and a router with one machine in it
        # reads exactly the document it read yesterday.
        self.c.host = {k: v for k, v in self.c.host.items() if k != 'host_id'}
        self.c.publish(self.active['bundle'])
        self.assertEqual([x['display_name'] for x in self.published()['models']],
            ['Qwen3.8 27B Q8 (160K)', 'Qwen3.8 27B Abliterated Q6_K (96K)'])
        # The name says where a model runs; what a client sends is untouched by it.
        self.assertEqual([x['model'] for x in self.published()['models']],
            [x['model'] for x in self.bundle['catalog']['models']])

    def test_naming_a_model_touches_no_launch_and_needs_no_drain(self):
        # A name is not a release. The launch this machine is serving with stays the byte-for-byte
        # document it was switched with, and no container is asked to do anything about a rename.
        before = json.loads((self.state / 'active.json').read_text())
        events = len([e for e in self.system.events if e[0] == 'docker' and e[1][:1] == ('compose',)])
        self.c.set_alias('qwen3.8-27b-q8_0-tensor-next', 'Workhorse')
        active = json.loads((self.state / 'active.json').read_text())
        self.assertEqual(active['bundle']['config_sha256'], before['bundle']['config_sha256'])
        self.assertEqual(active['bundle']['compose'], before['bundle']['compose'])
        self.assertEqual(len([e for e in self.system.events if e[0] == 'docker' and e[1][:1] == ('compose',)]), events)
        self.assertEqual(self.published()['configuration']['id'], 'qwen27b-q8-pair')
        self.assertEqual([x['display_name'] for x in self.published()['models']], ['Workhorse', self.published()['models'][1]['display_name']])
        # What a client routes on is not a name: the model ids and service IDs survive it unchanged.
        self.assertEqual([x['model'] for x in self.published()['models']],
            [x['model'] for x in before['bundle']['catalog']['models']])
        self.assertEqual(self.published()['default_model'], before['bundle']['catalog']['default_model'])

    def test_a_name_is_refused_with_the_rule_it_broke(self):
        for value, reason in [('nope/' * 20, 'at most 48'), ('two\nlines', 'one printable line')]:
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, reason):
                self.c.set_alias('qwen3.8-27b-q8_0-tensor-next', value)
        with self.assertRaisesRegex(RuntimeError, 'Unknown model'):
            self.c.set_alias('qwen3.8-27b-retired-q8', 'Retired')
        self.assertFalse((self.state / 'aliases.json').exists())

    def test_a_name_saved_while_a_switch_is_in_flight_is_published_by_that_switch(self):
        # The switch owns the drain and the publication; a rename that barged in mid-flight would be
        # a second hand on the same catalog, so it saves and lets the run say the name instead.
        atomic_json(self.state / 'transaction.json', {'phase': 'draining'})
        self.assertEqual(self.c.set_alias('qwen3.8-27b-q8_0-tensor-next', 'Workhorse'),
            {'saved': True, 'published': False,
                'note': 'A switch or restoration is in flight; it publishes this name with its own catalog.'})
        self.assertEqual(json.loads((self.state / 'aliases.json').read_text())['aliases'],
            {'qwen3.8-27b-q8_0-tensor-next': 'Workhorse'})
        self.assertFalse((self.state / 'router' / 'active-model.json').exists())

    def test_the_page_survives_an_alias_file_it_cannot_read(self):
        # A typo in an operator preference must not turn a serving machine into a page that does not
        # load, and must not be swallowed either: the page says the names are not in use.
        (self.state / 'aliases.json').write_text('{"aliases": {"qwen3.8-27b-q8_0-tensor-next"')
        status = self.c.status()
        self.assertTrue(status['ready'])
        self.assertEqual([x['name'] for x in status['services']],
            [x['display_name'] for x in self.bundle['catalog']['models']])
        self.assertIn('could not be read', status['aliases']['note'])
        # An alias for a model this revision does not serve is ignored, and said.
        atomic_json(self.state / 'aliases.json', {'version': 2, 'aliases': {'gone-today': 'Ghost'}})
        status = self.c.status()
        self.assertIn('gone-today', status['aliases']['note'])
        self.assertEqual([x['name'] for x in status['services']],
            [x['display_name'] for x in self.bundle['catalog']['models']])

    def test_an_unadopted_machine_refuses_a_switch_it_was_not_told_to_install(self):
        # The migration gate stays: a machine that was never adopted does not start loading a model
        # because somebody clicked, and the message names the two doors that are open.
        atomic_json(self.state / 'active.json', None)
        with self.assertRaisesRegex(RuntimeError, 'not adopted'):
            self.c.transition('qwen27b-q8-pair')

    def test_an_install_starts_a_machine_that_has_never_run_this_runtime(self):
        # A second machine has no legacy stack to adopt: an empty Docker host, a model to load, and
        # nothing to fall back to. `install` is the switch that admits it — the same drain, the same
        # verification, the same publication — and `previous.json` stays empty because there is no
        # earlier release behind it, which the page reports as having nothing to restore to.
        atomic_json(self.state / 'active.json', None)
        self.system.containers.clear()
        result = self.c.transition('qwen27b-q8-pair', install=True)
        # Nothing was running, so everything the configuration names has to be started, one at a time,
        # in the order the file names them -- the first launch takes the memory it was sized for before
        # the second one asks for cards that are still free rather than busy.
        self.assertEqual(result['changed_services'], [CODING, EVERYDAY])
        self.assertEqual(self.ups(), [(CODING,), (EVERYDAY,)])
        self.assertTrue(self.system.inspect(DAY)['State']['Running'])
        self.assertTrue(self.system.inspect(NIGHT)['State']['Running'])
        active = json.loads((self.state / 'active.json').read_text())
        self.assertEqual(active['bundle']['profile'], 'qwen27b-q8-pair')
        self.assertIsNone(json.loads((self.state / 'previous.json').read_text()))
        self.assertTrue((self.state / 'router' / 'active-model.json').is_file())

    def test_an_install_refuses_a_machine_that_is_already_serving(self):
        # Healthy backends with no active release is a migrated host, and a migrated host is adopted
        # through the reviewed procedure rather than installed over.
        atomic_json(self.state / 'active.json', None)
        with self.assertRaisesRegex(RuntimeError, 'adopt'):
            self.c.transition('qwen27b-q8-pair', install=True)

    def test_healthy_startup_does_not_drain_recreate_or_publish(self):
        self.system.active_count = 1; self.system.direct_busy = True
        result = self.c.transition()
        self.assertTrue(result['already_active'])
        self.assertEqual(self.system.events, [('prepare',)])

    def test_a_switch_asks_for_every_model_the_target_names_in_the_order_it_names_them(self):
        # A switch is not a patch applied to whatever happens to be loaded. The target names two models,
        # so both are asked for, one at a time, in the order the configuration names them -- including
        # the model that is byte-identical in both bundles, because the cards under it are being handed
        # to a different launch and a process that survived that would be surviving a lie about where
        # it runs; the test just below, which names that promise, pins the part about it really being a
        # new process rather than a surviving one.
        day = self.system.inspect(DAY)['Id']; night = self.system.inspect(NIGHT)['Id']
        result = self.c.transition('qwen27b-q6k-pair')
        self.assertEqual(result['changed_services'], [CODING_Q6K, EVERYDAY])
        self.assertEqual(self.ups(), [(CODING_Q6K,), (EVERYDAY,)])
        self.assertEqual(self.c.desired()['profile'], 'qwen27b-q6k-pair')
        self.assertNotEqual(self.system.inspect(DAY)['Id'], day)
        self.assertNotEqual(self.system.inspect(NIGHT)['Id'], night)
        # The launch the machine is dropping is stopped and removed before the target asks for its
        # container name back: every configuration of this host puts a daytime model in `qwen38-daytime`,
        # and the container standing under that name belongs to a model the target does not name. The
        # second model's container is named by the target, so nothing stops it -- the force flag below
        # is what replaces it.
        events = self.system.events
        self.assertLess(events.index(('docker', ('stop', '--time', '60', DAY))),
            next(i for i, e in enumerate(events) if e[0] == 'docker' and 'up' in e[1]))
        self.assertFalse([e for e in events if e[0] == 'docker' and e[1][0] in ('stop', 'rm')
            and e[1][-1] == NIGHT])
        # The configuration now running is unchanged and healthy, with nothing left standing that it
        # does not name: the machine is left alone, and being left alone means no call at all.
        asked = [e for e in self.system.events if e[0] != 'prepare']
        self.assertTrue(self.c.transition()['already_active'])
        self.assertEqual([e for e in self.system.events if e[0] != 'prepare'], asked)
        self.assertEqual(self.c.transition('qwen27b-q8-pair')['changed_services'], [CODING, EVERYDAY])
        self.assertNotEqual(self.system.inspect(NIGHT)['Id'], night)

    def test_a_switch_recreates_a_model_that_is_identical_in_both_configurations(self):
        # Both pair configurations load the same second model, byte for byte, and the daemon would have
        # kept its container. A switch says `--force-recreate` anyway: that process was started under a
        # different configuration, and a model that survives a switch only because nothing about its
        # launch differs is a model this machine is serving without having decided to.
        night = self.system.inspect(NIGHT)['Id']
        self.c.transition('qwen27b-q6k-pair')
        self.assertTrue(self.up_calls() and all('--force-recreate' in call for call in self.up_calls()))
        self.assertNotEqual(self.system.inspect(NIGHT)['Id'], night)
        # Re-selecting the configuration already running, and naming a model, are not switches: the two
        # containers this machine is serving with are not asked to do anything at all.
        serving = self.system.inspect(NIGHT)['Id']
        self.assertTrue(self.c.transition('qwen27b-q6k-pair')['already_active'])
        self.c.set_alias(EVERYDAY, 'Night owl')
        self.assertEqual(self.system.inspect(NIGHT)['Id'], serving)
        self.assertEqual(len(self.up_calls()), 2)

    def test_wrong_engine_is_detected_only_on_affected_backend(self):
        self.system.containers[DAY]['Image'] = 'sha256:' + 'f' * 64  # not the pinned engine
        observed = self.c.observe(self.bundle)
        self.assertIn('image', observed[CODING]['differences'])
        self.assertTrue(observed[EVERYDAY]['healthy'])

    def test_stale_controller_cannot_apply_old_source(self):
        self.c.revision = 'b' * 40
        with self.assertRaisesRegex(RuntimeError, 'superseded'): self.c.transition()
        self.assertEqual(self.system.events, [])

    def propose_daytime_change(self):
        proposed = copy.deepcopy(self.bundle)
        cfg = proposed['compose']['services'][CODING]
        cfg['command'][cfg['command'].index('--threads') + 1] = '17'
        # A rendered bundle's digest names the directory its generated compose file is written into, so
        # a hand-edited fixture has to recompute it the way `render` does: two different bundles under
        # one digest would otherwise be handed the same file on disk, and the daemon would be asked for
        # whichever bundle was written last.
        proposed['config_sha256'] = digest({k: v for k, v in proposed.items() if k != 'config_sha256'})
        change = patch.object(self.c, 'desired', return_value=proposed)
        change.start(); self.addCleanup(change.stop)

    def test_launch_change_replaces_the_whole_configuration_after_drain(self):
        self.propose_daytime_change()
        day = self.system.inspect(DAY)['Id']; night = self.system.inspect(NIGHT)['Id']
        result = self.c.transition('qwen27b-q8-pair')
        self.assertEqual(result['changed_services'], [CODING, EVERYDAY])
        self.assertEqual(self.ups(), [(CODING,), (EVERYDAY,)])
        self.assertNotEqual(self.system.inspect(DAY)['Id'], day)
        self.assertNotEqual(self.system.inspect(NIGHT)['Id'], night)
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
        self.system.failed_ups = 1
        before = {k: v['Id'] for k, v in self.system.containers.items()}
        with self.assertRaisesRegex(RuntimeError, 'Simulated load failure'): self.c.transition('qwen27b-q8-pair')
        # Recovery goes back to the release that was serving by asking its daemon for every service that
        # release names, not only the ones that look broken: the failed switch may have stopped, freed
        # or half-started any of them, and Compose recreates only what is not already the right
        # container. So the two healthy backends this machine never left are still the same containers,
        # which is the whole reason asking for everything is safe.
        self.assertEqual(self.ups()[-1], (CODING, EVERYDAY))
        self.assertIn('--force-recreate', self.up_calls()[0])  # the switch forced every model it named
        self.assertNotIn('--force-recreate', self.up_calls()[-1])  # the recovery asked, it did not force
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
        with self.assertRaisesRegex(RuntimeError, 'drained for this machine by another operation'):
            self.c.transition('qwen27b-q8-pair')
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
             'flash-next-strata-solo-thinking-recovery-cache-yarn',
             # The second machine's configuration is listed on a host that does not name itself, which
             # is what an unscoped host means: the whole registry, with a reason on the rows it cannot
             # run. It is a configuration, not a candidate, and the page says which one it is.
             'flash-next-strata-iq3-xxs-daisy',
             # The three trial configurations of that machine hold every card for the same reason: each
             # fills both of them with cached experts.
             'flash-next-strata-iq3-xxs-daisy-tc', 'flash-next-strata-iq3-xxs-daisy-128k',
             'flash-next-strata-iq3-s-daisy'])
        # Each name says which pack it is, so the two 256K rows that differ only by whether
        # conversations are parked are the pair the page still separates with its variant label, and the
        # rows that differ by cards say so where a person reads.
        self.assertEqual([x['display_name'] for x in configs['selectable']], ['Qwen3.8 27B Q8 (160K)',
            'Qwen3.8 27B Q6_K (160K)', 'Qwen3.8 Flash-Next UD-IQ4 (256K)',
            'Qwen3.8 Flash-Next UD-IQ4 (256K)', 'Qwen3.8 Flash-Next UD-IQ4 (512K)',
            'Qwen3.8 Flash-Next UD-IQ4, two cards (256K)',
            'Qwen3.8 Flash-Next IQ3_S, two cards (256K)', 'Qwen3.8 Flash-Next IQ3_XXS (64K)',
            'Qwen3.8 Flash-Next IQ3_XXS, faster prompts (64K)',
            'Qwen3.8 Flash-Next IQ3_XXS, long context (128K)',
            'Qwen3.8 Flash-Next IQ3_S, higher quality (64K)'])
        # Since 2026-10-18 the label is computed from context_tokens, so a rung of the session-size
        # ladder edited in the profile shows up on the page rather than only in the engine that boots
        # on it -- and the two 256K configurations move together on every rung: the run config is one
        # file per workspace, and two configurations of one model that disagreed about the window would
        # make a switch between them a resize. The 512K configuration states that resize as a
        # fact of its own -- window and ring are its one variable from its sibling -- so the label
        # says 512K. The pair keeps 160K, which is why all five numbers are named.
        #
        # The sixth row is where that naming earns its keep: the two-card probe is a 256K configuration
        # too, and it is the only one of the three whose cards are the reason to select it, so it is
        # the first configuration whose display name says something about the hardware. Two rows both
        # reading "Flash-Next (256K)" would be a page where the operator's choice is a coin flip.
        self.assertEqual([x['context_tokens'] for x in configs['selectable']],
            [163840, 163840, 262144, 262144, 524288, 262144, 262144, 65536, 65536, 131072, 65536])
        # The Flash-Next configurations, named by prefix rather than by a trailing slice: what tells an
        # operator which of them is which is the variant label, and it also says what each one turns
        # on, because that is the only choice left to make here.
        strata = [x for x in configs['selectable'] if x['profile'].startswith('flash-next')]
        self.assertEqual([x.get('variant_label') for x in strata],
            ['card sampling + call recovery',
             'card sampling + call recovery + parked conversations',
             'card sampling + call recovery + parked conversations + YaRN 2.0 window',
             'two cards, nothing else changed',
             'two cards, the 3.5-bit pack',
             'two Turing cards, the 64K window',
              'two Turing cards, the 64K window, the tensor-core prompt path: +11 % prompt processing, 10 % less time to the first token of a 58k prompt, decode unchanged',
              'two Turing cards, the 128K window, the tensor-core prompt path: measured serving a 118,514-token prompt on these cards, for 483 expert-cache slots and 1.8-2.6 % of short-prompt decode',
              'two Turing cards, the 64K window, the IQ3_S tier: 87.69 % top-1 against 84.46 % on the IQ3_XXS tier, for 6.6 % of decode'])
        self.assertEqual({x['parallel_slots'] for x in configs['selectable']}, {1})
        self.assertEqual(configs['selectable'][2]['gpu_names'], [*HOST['gpu_names']['daytime'], *HOST['gpu_names']['nighttime']])
        self.assertEqual(configs['selectable'][1]['context_tokens'], 163840)
        self.assertEqual(configs['selectable'][2]['engine'], 'strata-vision-296b59d2f73c')
        # What a client can ask each configuration for. The solo one answers all three IDs from the one
        # model it loads; a paired one answers `nighttime` through the second model it names, so its
        # list is the two models together and `model_service_ids` is only its own half.
        self.assertEqual(configs['selectable'][2]['service_ids'], ['daytime', 'local-active', 'nighttime'])
        self.assertEqual(configs['selectable'][0]['service_ids'], ['daytime', 'local-active', 'nighttime'])
        self.assertEqual(configs['selectable'][0]['model_service_ids'], ['daytime', 'local-active'])
        # How many models a configuration loads is a fact about its file, and this is the one place the
        # page says it: one across the machine, two with the cards divided, and the same second model
        # named by three of these eleven files rather than carried by all eleven. Nothing is always
        # included, and a row that stops naming a model is a row that stops loading it.
        self.assertEqual([[m['model'] for m in x['models']] for x in configs['selectable']],
            [[CODING, EVERYDAY], [CODING_Q6K, EVERYDAY], [STRATA_SOLO], [STRATA_SOLO], [STRATA_SOLO],
                [STRATA_SOLO, EVERYDAY], [STRATA_PAIR, EVERYDAY], [STRATA_DAISY], [STRATA_DAISY],
                [STRATA_DAISY], ['qwen3.8-flash-next-iq3_s']])
        everyday = configs['selectable'][0]['models'][1]
        self.assertEqual((everyday['display_name'], everyday['service_ids'], everyday['gpu_names']),
            ('Qwen3.8 27B Abliterated Q6_K (96K)', ['nighttime'], HOST['gpu_names']['nighttime']))
        self.assertEqual(configs['selectable'][0]['gpu_names'], HOST['gpu_names']['daytime'])
        self.assertNotIn('synthetic-token', json.dumps(configs))
        self.assertNotIn(HOST['model_root'], json.dumps(configs))
        self.c.transition('qwen27b-q6k-pair')
        self.assertEqual(self.c.status()['configurations']['active'], 'qwen27b-q6k-pair')
        (self.state / 'active.json').unlink()  # Before adoption nothing is live, but the registry is still listable.
        pre_migration = self.c.status()
        self.assertFalse(pre_migration['ready'])
        self.assertEqual(pre_migration['configurations']['active'], 'qwen27b-q8-pair')
        self.assertEqual(len(pre_migration['configurations']['selectable']), len(CONFIGURATIONS))

    def test_the_page_says_which_machine_is_answering(self):
        # A GPU rail that counts the cards it was written against describes a machine that is not
        # there as soon as a second machine reads the same source. The count is published from the
        # host settings, and so is the name, which is the only way a screenshot of a two-card host
        # can be told apart from a four-card host that lost two cards.
        self.assertEqual(self.c.status()['host'], {'id': None, 'gpu_architectures': None,
            'text_gpu_groups': ['daytime', 'nighttime'], 'text_gpus': 4, 'vision_gpu_id': None,
            'gpu_inventory': INVENTORY})
        self.c.host = {**self.c.host, 'host_id': UNCLAIMED, 'gpu_architectures': ['75']}
        status = self.c.status()
        self.assertEqual(status['host'], {'id': UNCLAIMED, 'gpu_architectures': ['75'],
            'text_gpu_groups': ['daytime', 'nighttime'], 'text_gpus': 4, 'vision_gpu_id': None,
            'gpu_inventory': INVENTORY})
        # A host that names a machine no configuration is built for still reports its health -- and
        # offers nothing, which is the state an operator has to be able to see and then fix.
        self.assertEqual(status['ready'], True)
        self.assertEqual(status['configurations']['selectable'], [])
        self.assertEqual(status['configurations']['active'], 'qwen27b-q8-pair',
            'what is running stays reported even when nothing on this machine may be selected')
        self.c.host = {**self.c.host, 'gpu_architectures': ['75', 'Turing']}
        self.assertIn('error', self.c.status()['host'])

    def test_the_power_block_is_always_there(self):
        # The page draws the panel by reading status['power'] and nothing else, so the block has to
        # arrive on a machine with no card-to-plan mapping and no cache — which is the state every
        # host in the house is in. A missing key is a hole in the page, and a hole next to a GPU tile
        # looks exactly like a card that is fine.
        block = self.c.status()['power']
        self.assertEqual(block['available'], True, 'the rung table ships in source, so it is always '
            'there to read; what a host may lack is the mapping from its cards to it')
        # Every card the host declares is in the block even with nothing cached and nothing mapped.
        # A card that has merely never been read has to arrive looking unread: left out of the block,
        # it would look from the page like a card that is fine, which is the one thing it is not.
        self.assertTrue(block['cards'])
        for card in block['cards'].values():
            self.assertEqual(card['state'], 'unread')
        # And nothing mapped is not the same as nothing offered: this machine has never been told
        # which card is which, and its cards declare their own models, so three of them arrive with
        # the ladder measured for those models and one — the RTX 4080, a model no ladder has run —
        # arrives with nothing to choose.
        by_name = {card['name']: card for card in block['cards'].values()}
        for name, plan in (('RTX 3090', 'rtx3090'), ('RTX 3080 Ti', 'rtx3080-ti'),
                ('RTX 4080 SUPER', 'rtx4080-super')):
            self.assertEqual((by_name[name]['plan'], by_name[name]['plan_source']), (plan, 'model'))
            self.assertEqual([rung['name'] for rung in by_name[name]['rungs']],
                ['safe', 'sensible', 'assertive'])
        self.assertEqual((by_name['RTX 4080']['plan'], by_name['RTX 4080']['rungs']), (None, []))
        self.assertEqual(block['writes'], {'available': True, 'reason': None},
            'whether a write may be attempted at all is published, with the reason it may not')
        self.assertIsInstance(block['cadence_seconds'], int)

    def test_a_host_that_names_itself_loads_what_it_was_built_for(self):
        # The same controller on a host that names itself: the registry it publishes is the share of
        # it that was built for this machine, which is the whole separation, seen from the surface.
        config = self.state / 'config'
        shutil.copytree(ROOT / 'config', config)
        for name in ('flash-next-strata-two-card-iq4-xs', 'flash-next-strata-two-card-iq3-s'):
            path = config / 'profiles' / (name + '.json')  # Scoped to this machine as well as Rosalina's.
            definition = read(path)
            # A model that owns every text GPU this two-card machine declares leaves no card for a second
            # one, so scoping these files to `daisy` also means they stop naming the model they used to
            # load beside themselves; a configuration cannot bring a second machine's model onto cards
            # that are already spoken for.
            definition.update(hosts=['rosalina', 'daisy'], exclusive=True, gpu_group='all', models=[name])
            path.write_text(json.dumps(definition))
        host = copy.deepcopy(HOST); host['host_id'] = 'daisy'
        host['gpu_ids'] = {'daytime': HOST['gpu_ids']['daytime'][:2]}
        host['gpu_names'] = {'daytime': HOST['gpu_names']['daytime'][:2]}
        host.pop('cuda_order', None); host.pop('vision_gpu_id', None)
        atomic_json(self.state / 'host.json', host)
        (self.state / 'active.json').unlink()  # This machine has never deployed anything of its own.
        controller = Controller(config, self.state, 'a' * 40, self.system)
        status = controller.status()
        self.assertEqual([x['profile'] for x in status['configurations']['selectable']],
            ['flash-next-strata-two-card-iq4-xs', 'flash-next-strata-two-card-iq3-s',
                'flash-next-strata-iq3-xxs-daisy', 'flash-next-strata-iq3-xxs-daisy-tc',
                'flash-next-strata-iq3-xxs-daisy-128k', 'flash-next-strata-iq3-s-daisy'],
            'the two configurations just scoped to this machine join the ones written for it, '
            'in registry order; what a host is offered is its share and nothing else')
        self.assertEqual(status['configurations']['active'], 'flash-next-strata-two-card-iq4-xs')
        self.assertEqual(status['host']['text_gpus'], 2)
        # What a client of *this* machine can no longer ask for: the one service ID another of its
        # configurations offers and the loaded model does not publish. There is no second row about
        # placement any more -- the model behind `nighttime` is not this machine's to load at all, and
        # an unscoped host is not consulted for it, because a host that names itself is offered only
        # its share of the registry.
        self.assertEqual([(x['model'], x['aliases'], x['reason']) for x in status['offline_services']],
            [('qwen3.8-flash-next-iq3_s', ['daisy'], 'not_loaded_by_this_configuration'),
                (STRATA_DAISY, ['daisy'], 'not_loaded_by_this_configuration')])
        # With nothing else to load, the first configuration of this host is the one to fall back to.
        self.assertEqual(controller.desired()['profile'], 'flash-next-strata-two-card-iq4-xs')

    def test_unreadable_registry_never_hides_the_health_of_the_pair(self):
        with patch('runtime.controller.available_profiles', side_effect=RuntimeError('registry exploded')):
            status = self.c.status()
        self.assertTrue(status['ready'])
        self.assertEqual(status['configurations']['active'], 'qwen27b-q8-pair')
        self.assertEqual(status['configurations']['selectable'], [])
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
        # This fixture's host.json names no machine, so its registry is the whole list, and the one
        # service ID nothing here answers belongs to the second machine's configuration. That is the
        # fallback stated out loud rather than a bug in the catalog: an unscoped host is offered every
        # configuration there is, so it is responsible for saying which of their IDs are not loaded.
        self.assertEqual([(x['model'], x['aliases'], x['reason']) for x in catalog['offline_services']],
            [('qwen3.8-flash-next-iq3_s', ['daisy'], 'not_loaded_by_this_configuration'),
                (STRATA_DAISY, ['daisy'], 'not_loaded_by_this_configuration')])
        # A previous release's bundle predates these fields; recovery still publishes an accurate id.
        legacy = copy.deepcopy(self.bundle)
        del legacy['catalog']['configuration'], legacy['catalog']['offline_services']
        self.c.publish(legacy)
        catalog = read(self.state / 'router/active-model.json')
        self.assertEqual((catalog['configuration']['id'], catalog['configuration']['exclusive']),
            ('qwen27b-q8-pair', False))
        self.assertNotIn('offline_services', catalog)

    def test_a_configuration_change_outside_argv_still_replaces_the_configuration(self):
        proposed = copy.deepcopy(self.bundle)
        proposed['compose']['services'][CODING]['logging']['options']['max-size'] = '30m'
        proposed['config_sha256'] = digest({k: v for k, v in proposed.items() if k != 'config_sha256'})
        with patch.object(self.c, 'desired', return_value=proposed): result = self.c.transition()
        # A log-rotation size is as much a decision about this machine as a launch argument, so the
        # target is a different configuration and every model it names is stopped and loaded again --
        # the switch does not patch one service of a running machine and leave a process behind that
        # belongs to the configuration it just replaced.
        self.assertEqual(result['changed_services'], [CODING, EVERYDAY])
        self.assertEqual(self.ups(), [(CODING,), (EVERYDAY,)])
        self.assertTrue(self.system.inspect(NIGHT)['State']['Running'])


class ExclusiveProfileTests(unittest.TestCase):
    """A model that owns every text GPU this host declares is loaded alone: the switch to it stops every
    model the target does not name, and the switch away loads the models the target names."""
    HOST = vision_host()
    setUp = ControllerTests.setUp
    ups = ControllerTests.ups
    NIGHT = NIGHT
    SOLO = 'flash-next-strata-solo-thinking-recovery'

    def setUp(self):
        ControllerTests.setUp(self)
        # This engine builds its expert pack into a workspace before it answers a health check, and
        # the switch does that build before it takes the drain. The build is the subject of
        # tests/test_provision.py; what is under test here is which GPUs a backend may hold and the
        # order the launches move in, and the workspace the bundle binds does not exist on a test
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

    def published(self):
        return [m['model'] for m in read(self.state / 'router/active-model.json')['models']]

    def solo_settings(self):
        """What identifies the solo backend, for a backend that has no argument vector.

        The Strata engine is configured from its environment, so the environment list is the
        fingerprint the llama.cpp backends' argv is for: two containers match only if the same
        configuration is running in both.
        """
        return self.system.inspect(DAY)['Config']['Env']

    def test_switch_to_one_model_stops_every_model_the_target_does_not_name(self):
        day = self.system.inspect(DAY)['Id']
        result = self.c.transition(self.SOLO)
        # The target names one model, so one is asked for -- and the model it does not name is not a
        # second half entitled to keep running on cards this model was sized to fill.
        self.assertEqual(result['changed_services'], [STRATA_SOLO])
        self.assertIsNone(self.system.inspect(self.NIGHT))
        events = self.system.events
        drain = events.index(('drain', True))
        stop = events.index(('docker', ('stop', '--time', '60', self.NIGHT)))
        remove = events.index(('docker', ('rm', self.NIGHT)))
        load = next(i for i, e in enumerate(events) if e[0] == 'docker' and 'up' in e[1])
        self.assertLess(drain, stop); self.assertLess(stop, remove); self.assertLess(remove, load)
        self.assertEqual(self.ups(), [(STRATA_SOLO,)])
        self.assertNotEqual(self.system.inspect(DAY)['Id'], day)
        self.assertEqual(self.published(), [STRATA_SOLO])
        catalog = read(self.state / 'router/active-model.json')
        self.assertEqual({k: catalog['configuration'][k] for k in ('id', 'exclusive', 'runtime_revision')},
            {'id': self.SOLO, 'exclusive': True, 'runtime_revision': 'a' * 40})
        self.assertTrue(catalog['configuration']['published_at'])
        # Nothing a client of this machine routes on is offline, and that is the point of this
        # configuration: the one model it loaded answers `nighttime` as well as `daytime`, so nobody
        # holding that name is told the service is off. The second machine's own service ID is in the
        # list, because this fixture's host names no machine and is therefore offered the whole registry.
        self.assertNotIn('nighttime', [a for x in catalog['offline_services'] for a in x['aliases']])
        self.assertEqual([x['model'] for x in catalog['offline_services']],
            ['qwen3.8-flash-next-iq3_s', STRATA_DAISY])
        self.assertEqual([m['aliases'] for m in catalog['models']], [['local-active', 'daytime', 'nighttime']])
        self.assertEqual(events[-1], ('drain', False))
        tx = read(self.state / 'transaction.json')
        # `removed_backends` is derived from every model any configuration of this host could start, not
        # from one container a solo configuration used to have to name by hand, and it is the container,
        # not the model, that has to be free: both launches here belong to models this target does not
        # name, so both are stopped and freed before the target asks for `qwen38-daytime` back.
        self.assertEqual((tx['phase'], tx['removed_backends']), ('succeeded', [DAY, self.NIGHT]))
        self.assertEqual(tx['changed_services'], [STRATA_SOLO])
        status = self.c.status()
        self.assertTrue(status['ready'])
        [service] = status['services']
        self.assertEqual(service['service'], STRATA_SOLO)
        self.assertTrue(service['exclusive'])
        self.assertEqual(service['gpu_ids'], [*self.HOST['gpu_ids']['daytime'], *self.HOST['gpu_ids']['nighttime']])
        self.assertEqual(service['gpu_names'], [*self.HOST['gpu_names']['daytime'], *self.HOST['gpu_names']['nighttime']])
        # The encoder runs on the shared card, and this configuration loads no second model, so it has
        # that card to itself: `vision_gpu_shared` false is the difference from a configuration that
        # names two, where the projector is a guest on the other model's host.
        self.assertEqual((service['vision_device'], service['vision_gpu_shared']), ('CUDA4', False))
        self.assertTrue(self.c.transition()['already_active'])
        # The same configuration with one environment variable moved is not a different configuration, so
        # nothing is stopped and nothing is removed: the drift pass reloads the one launch that differs
        # and leaves the cards where they are.
        self.system.events.clear()
        changed = copy.deepcopy(self.c.desired())
        changed['compose']['services'][STRATA_SOLO]['environment']['LAYER_SPLIT'] = '0,0,0,0'
        with patch.object(self.c, 'desired', return_value=changed):
            self.assertEqual(self.c.transition(self.SOLO)['changed_services'], [STRATA_SOLO])
        self.assertFalse(any(e[0] == 'docker' and e[1][0] in ('stop', 'rm') for e in self.system.events))
        self.assertEqual(self.ups(), [(STRATA_SOLO,)])
        self.assertIsNone(self.system.inspect(self.NIGHT))

    def test_the_one_model_that_answers_every_service_id_leaves_nothing_offline(self):
        """`offline_services` is the one fact left, and it is about clients, not about placement.

        The pair loads the model behind `nighttime`, so nothing a client routes on is offline; the solo
        configuration loads one model and answers all three IDs from it, so the honest status still says
        nothing is offline. That is the whole reason this is one fact and not two: a page that also
        reported the unloaded model as a missing placement would tell a client its `nighttime` was gone
        while every request to it was being served.
        """
        # The second machine's own service ID is in both lists: this fixture's host names no machine, so
        # it is offered the whole registry and answers for all of it.
        self.assertEqual([x['model'] for x in self.c.status()['offline_services']],
            ['qwen3.8-flash-next-iq3_s', STRATA_DAISY])
        self.c.transition(self.SOLO)
        status = self.c.status()
        self.assertTrue(status['ready'])
        self.assertNotIn('nighttime', [a for x in status['offline_services'] for a in x['aliases']])
        [service] = status['services']
        self.assertEqual(service['service'], STRATA_SOLO)

    def test_switch_back_from_solo_loads_the_models_in_the_order_the_configuration_names_them(self):
        original = self.system.inspect(DAY)['Config']['Cmd']
        self.c.transition(self.SOLO)
        self.system.events.clear()
        self.assertEqual(self.c.transition('qwen27b-q8-pair')['changed_services'], [CODING, EVERYDAY])
        # One at a time, in the order the file names them: the model that held all four cards has to
        # release the pair the second model was sized for before that model asks for them.
        self.assertEqual(self.ups(), [(CODING,), (EVERYDAY,)])
        self.assertEqual(self.system.inspect(DAY)['Config']['Cmd'], original)
        self.assertTrue(self.system.inspect(self.NIGHT)['State']['Running'])
        self.assertEqual(self.published(), [CODING, EVERYDAY])
        status = self.c.status()
        self.assertTrue(status['ready'])
        self.assertEqual([x['service'] for x in status['services']], [CODING, EVERYDAY])
        self.assertEqual(status['services'][0]['gpu_names'], self.HOST['gpu_names']['daytime'])

    def test_failed_solo_load_restores_every_model_of_the_previous_configuration(self):
        original = self.system.inspect(DAY)['Config']['Cmd']
        self.system.failed_ups = 1
        with self.assertRaisesRegex(RuntimeError, 'Simulated load failure'): self.c.transition(self.SOLO)
        # Recovery does not decide which of the previous release's models still look good: it asks for
        # every one of them, because the failed switch may have freed any of them, and in the same order
        # the switch would have used. Both containers had been freed to hand their names to the refused
        # target, so "restored" means loaded again from the release that was serving, not left alone.
        self.assertEqual(self.ups(), [(STRATA_SOLO,), (CODING,), (EVERYDAY,)])
        self.assertEqual(self.system.inspect(DAY)['Config']['Cmd'], original)
        self.assertTrue(self.system.inspect(self.NIGHT)['State']['Running'])
        self.assertEqual(read(self.state / 'transaction.json')['phase'], 'recovered')
        self.assertEqual(self.c.desired()['profile'], 'qwen27b-q8-pair')
        self.assertEqual(len(self.published()), 2)
        self.assertFalse(self.system.draining)
        self.assertTrue(self.c.status()['ready'])

    def test_rejected_solo_restores_the_previous_models_in_order(self):
        original = self.system.inspect(DAY)['Config']['Cmd']
        self.system.failed_generations = 1
        with self.assertRaisesRegex(RuntimeError, 'Simulated acceptance failure'): self.c.transition(self.SOLO)
        self.assertEqual(self.ups(), [(STRATA_SOLO,), (CODING,), (EVERYDAY,)])
        self.assertEqual(self.system.inspect(DAY)['Config']['Cmd'], original)
        self.assertTrue(self.system.inspect(self.NIGHT)['State']['Running'])
        self.assertEqual(read(self.state / 'transaction.json')['phase'], 'recovered')
        self.assertTrue(self.c.status()['ready'])

    def test_failed_return_to_the_pair_stops_the_model_the_target_does_not_name(self):
        self.c.transition(self.SOLO)
        solo = self.solo_settings()
        self.system.events.clear(); self.system.failed_generations = 1
        with self.assertRaisesRegex(RuntimeError, 'Simulated acceptance failure'): self.c.transition('qwen27b-q8-pair')
        self.assertEqual(self.ups(), [(CODING,), (EVERYDAY,), (STRATA_SOLO,)])
        # The failed target had started the second model, and the release being restored does not name
        # it: it is stopped and freed before the restored model is asked for, so nothing the failed
        # configuration brought up is left beside what this machine decided to load.
        events = self.system.events
        self.assertLess(max(i for i, e in enumerate(events) if e == ('docker', ('rm', self.NIGHT))),
            next(i for i, e in enumerate(events) if e[0] == 'docker' and 'up' in e[1] and e[1][-1] == STRATA_SOLO))
        self.assertIsNone(self.system.inspect(self.NIGHT))
        self.assertEqual(self.system.inspect(DAY)['Config']['Env'], solo)
        self.assertEqual(read(self.state / 'transaction.json')['phase'], 'recovered')
        self.assertEqual(self.c.desired()['profile'], self.SOLO)
        self.assertEqual(self.published(), [STRATA_SOLO])

    def test_two_slot_backend_is_verified_by_slot_count_and_per_slot_context(self):
        # No configuration runs two slots now; the capability is exercised on a copy of the configuration.
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        config = Path(tmp.name) / 'config'; shutil.copytree(ROOT / 'config', config)
        path = config / ('profiles/' + self.SOLO + '.json'); definition = read(path)
        definition['parallel_slots'] = 2; path.write_text(json.dumps(definition))
        self.c.config_dir = config
        self.c.transition(self.SOLO)
        self.assertTrue(self.c.status()['ready'])
        bundle = self.c.desired(); name = DAY
        # This engine reports its own slots over /slots; one slot where the configuration contracted
        # for two is a backend that came up smaller than it was asked to.
        self.system.services[name]['parallel_slots'] = 1
        self.assertIn('slot context', self.c.observe(bundle)[STRATA_SOLO]['differences'])

    def test_a_model_the_running_configuration_does_not_name_is_drift_and_is_removed(self):
        pair = render(ROOT / 'config', self.HOST, 'qwen27b-q8-pair')
        self.c.transition(self.SOLO)
        solo = self.c.desired()
        # Somebody starts the launch of a model this configuration does not name -- a hand-run compose
        # up, or a switch that died between its two batches. It is not a second half entitled to run:
        # it is holding memory every loaded model was sized for, so it makes every service unhealthy
        # rather than leaving one to look fine, and the next pass stops it instead of working around it.
        self.system.install(EVERYDAY, pair['compose']['services'][EVERYDAY])
        self.assertIn('competing backend running: ' + self.NIGHT,
            self.c.observe(solo)[STRATA_SOLO]['differences'])
        self.assertFalse(self.c.status()['ready'])
        result = self.c.transition()
        self.assertEqual(result['changed_services'], [STRATA_SOLO])
        self.assertIsNone(self.system.inspect(self.NIGHT))
        self.assertTrue(self.c.status()['ready'])


class ArtifactTests(unittest.TestCase):
    def test_each_pinned_engine_is_validated_before_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); atomic_json(root / 'host.json', HOST)
            bundle = render(ROOT / 'config', HOST, 'qwen27b-q8-pair')
            bundle['artifacts'] = []
            for service in (CODING, EVERYDAY):
                with self.subTest(service=service):
                    system = FakeSystem(bundle)
                    system.engines = copy.deepcopy(system.engines)
                    # The pair shares one engine image, so each launch gets its own wrong-image copy of
                    # it: the check has to be able to say which of the two it is refusing.
                    tag = service_engine(bundle, service)['tag']
                    broken = copy.deepcopy(bundle)
                    entry = next(s for s in broken['manifest']['services'] if s['service'] == service)
                    entry['engine'] = {**entry['engine'], 'tag': tag + '-' + service}
                    broken['compose']['services'][service]['image'] = tag + '-' + service
                    system.engines[tag + '-' + service] = {**system.engines[tag], 'image_id': 'wrong-image'}
                    c = Controller(ROOT / 'config', root, 'a' * 40, system)
                    with self.assertRaisesRegex(RuntimeError, service + ': pinned inference engine'):
                        c.validate(broken)

    def test_previous_single_engine_manifest_remains_readable_for_recovery(self):
        bundle = render(ROOT / 'config', HOST, 'qwen27b-q6k-pair')
        engine = service_engine(bundle, CODING_Q6K)
        for service in bundle['manifest']['services']: service.pop('engine')
        bundle['manifest'].update(schema_version=1, engine=engine)
        self.assertEqual(service_engine(bundle, CODING_Q6K), engine)
        self.assertEqual(service_engine(bundle, EVERYDAY), engine)

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
        self.assertEqual(observed[STRATA_SOLO]['differences'], [])
        self.assertTrue(observed[STRATA_SOLO]['healthy'])
        self.assertNotIn('slot_window', observed[STRATA_SOLO])
        self.assertNotIn('health_error', observed[STRATA_SOLO])

    def test_a_service_dictionary_missing_a_setting_drifts_instead_of_crashing(self):
        for key, reason in [('restart', 'restart policy'), ('init', 'init'),
                ('entrypoint', 'entrypoint'), ('read_only', 'read-only root'),
                ('deploy', 'GPU assignment'), ('volumes', 'model mounts'),
                ('ports', 'port bindings'), ('networks', 'router network')]:
            # An absent command and an empty one mean the same thing to the comparison, so that key is
            # deliberately not in this list: the Strata engine takes its settings from the environment.
            # Neither is 'user', which this model legitimately does not render at all; it has its own
            # test below, because there 'nothing' and 'empty string' are the same container.
            damaged = copy.deepcopy(self.bundle)
            damaged['compose']['services'][STRATA_SOLO].pop(key)
            with self.subTest(setting=key):
                observed = self.c.observe(damaged)
                self.assertIn(reason, observed[STRATA_SOLO]['differences'])
                self.assertFalse(observed[STRATA_SOLO]['healthy'])

    def test_a_backend_running_as_its_image_user_is_not_called_drifted(self):
        """The Strata model renders no user and the daemon reports that as an empty string.

        Read as literally different, every health check would insist the container that is correctly
        serving is wrong, which is how a working backend gets restarted on a loop and a switch gets
        blamed for it.
        """
        cfg = self.bundle['compose']['services'][STRATA_SOLO]
        self.assertNotIn('user', cfg)
        self.assertEqual(self.system.containers[cfg['container_name']]['Config']['User'], '')
        self.assertNotIn('user', self.c.observe(self.bundle)[STRATA_SOLO]['differences'])
        damaged = copy.deepcopy(self.bundle)
        damaged['compose']['services'][STRATA_SOLO]['user'] = '0:0'
        self.assertIn('user', self.c.observe(damaged)[STRATA_SOLO]['differences'])

    def test_a_backend_that_came_up_on_another_window_says_so_with_its_number(self):
        """Strata reports the window the engine loaded with, which need not be the one asked for.

        An empty slot list is a backend still loading and a smaller window is permanent, and both
        would otherwise sit out the whole health timeout looking identical.
        """
        self.system.slot_context = 131072
        observed = self.c.observe(self.bundle)
        self.assertIn('slot context', observed[STRATA_SOLO]['differences'])
        self.assertFalse(observed[STRATA_SOLO]['healthy'])
        contracted = self.bundle['manifest']['services'][0]['context_tokens']
        self.assertEqual(observed[STRATA_SOLO]['slot_window']['contracted'], contracted)
        self.assertEqual(observed[STRATA_SOLO]['slot_window']['reported_context'], [131072])

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
                self.assertEqual({service: x['differences'] for service, x in observed.items()},
                    {service: [] for service in observed})

    def test_the_strata_model_gains_the_vision_card_and_nothing_else_moves(self):
        """The production host pins the RTX 3080 as the vision card; a host here may name none.

        Since 2026-10-16 the two renders are expected to differ, and the difference is the encoder's
        card plus the variables that name it. The engine's own card list, its mounts, its window and
        its launch environment are byte-identical, which is the property this switch has to keep:
        lending the model a fifth card may not move a byte of the four it is laid out across.
        """
        plain = render(ROOT / 'config', copy.deepcopy(HOST), self.PROFILE)
        lent = render(ROOT / 'config', vision_host(), self.PROFILE)
        before, after = (x['compose']['services'][STRATA_SOLO] for x in (plain, lent))
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


"""A self-provisioning engine must not make the selection a terminal ritual.

The Strata trial's first attempt spent its whole readiness window building the state it had never
built before and was abandoned with the router drained. The line that grew out of that is now
drawn differently: the switch prepares the workspace itself, doing the half that is safe beside a
loaded model before it drains, and waits for the engine's actual answer; the measurements remain
the evidence the gate and the refusals read, and the only refusals left are ones the machine's
evidence backs.
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import main
from unittest.mock import patch

import test_controller as fixture
from runtime.config import read, render
from runtime.controller import Controller
from runtime.provision import (PROJECT, matching, measurement, ready_files, throwaway, workspace_ready,
    workspace_target)
from runtime.system import atomic_json

PROFILE = 'flash-next-strata-160k'


class ProvisioningSystem(fixture.FakeSystem):
    """The engine as observed from outside: a container that is started, and eventually answers.

    The workspace and the health answer are the only two things the tool can watch, so the fake
    decides when each of them happens, which lets a measurement be tested without an engine.
    """

    def __init__(self, bundle, healthy_after=3, exits_after=None, log_text='Setup stopped.',
                 loaded_after=None):
        super().__init__(bundle)
        self.healthy_after = healthy_after
        self.exits_after = exits_after
        self.log_text = log_text
        self.loaded_after = loaded_after
        self.health_polls = 0
        self.started = []
        self.throwaway_port = None
        self.throwaway_name = None
        self.devices = None

    def docker(self, *args, **kwargs):
        if args[0] == 'logs':
            self.events.append(('docker', args))
            return self.log_text
        # The measurement starts and stops its own compose project; the fake plays the part of
        # compose reading the file it was handed, including the card reservations in it. Only the
        # measurement's project: a switch that builds through this same fake also starts the real
        # backend, and that one goes through the ordinary install path.
        if args[0] == 'compose' and args[1:3] == ('-p', PROJECT):
            self.events.append(('docker', args))
            spec = read(args[args.index('-f') + 1])
            if 'up' in args:
                role = args[-1]
                service = spec['services'][role]
                self.throwaway_port = service['ports'][0].split(':')[1]
                self.throwaway_name = service['container_name']
                self.devices = service['deploy']['resources']['reservations']['devices'][0]['device_ids']
                self.started.append((service['container_name'], self.throwaway_port, list(self.devices)))
                self.containers[service['container_name']] = {'Id': 'throwaway',
                    'Name': service['container_name'], 'State': {'Running': True}, 'Config': {'Cmd': []}}
            else:
                for service in spec['services'].values():
                    self.containers.pop(service['container_name'], None)
            return ''
        return super().docker(*args, **kwargs)

    def inspect(self, name):
        state = super().inspect(name)
        if state is not None:
            state['State'].setdefault('Health', {'Status': 'healthy'})
        return state

    def http(self, url, body=None, headers=None, timeout=None):
        # Only the throwaway's own answer is simulated, and only on the route the controller can
        # really use: its name on the shared network. The host's published loopback port answers to
        # nobody from in here, so a tool that depended on it would never see a healthy boot.
        if url.endswith('/health') and url.startswith('http://' + (self.throwaway_name or '') + ':8080'):
            self.health_polls += 1
            if self.exits_after is not None and self.health_polls >= self.exits_after:
                # An engine whose setup refuses does not keep trying: it prints why and goes away, and
                # from outside that is a stopped container whose output is readable for a moment.
                self.containers[self.throwaway_name] = {'State': {'Running': False, 'Status': 'exited',
                    'ExitCode': 1, 'FinishedAt': '2026-10-12T14:54:20.728631538Z'}}
                raise RuntimeError('connection refused')
            if self.health_polls < self.healthy_after:
                raise RuntimeError('connection refused')
            # The engine's server answers before the model is resident, and its health check
            # says so in loaded. healthy_after is the server coming up; loaded_after, when set,
            # is the read completing, polls after the first answer.
            return {'status': 'ok', 'loaded': self.loaded_after is None
                or self.health_polls >= self.loaded_after}
        if url.startswith('http://127.0.0.1:'):
            raise RuntimeError('no route from the controller to the host loopback')
        return super().http(url, body=body, headers=headers, timeout=timeout)


class ProvisionTests(unittest.TestCase):
    HOST = fixture.HOST

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name)
        host = copy.deepcopy(self.HOST)
        host['router_state_dir'] = str(self.state / 'router')
        # The workspace has to be a real directory: it is the one thing the tool watches appear.
        host['model_root'] = str(self.state / 'models')
        atomic_json(self.state / 'host.json', host); (self.state / 'router-token').write_text('synthetic-token')
        # The box is serving llama.cpp and the trial is the thing being asked for; a measurement is
        # only interesting while something else holds the cards.
        self.serving = render(fixture.ROOT / 'config', host, 'qwen27b-q8-with-nighttime')
        self.system = ProvisioningSystem(self.serving)
        self.c = Controller(fixture.ROOT / 'config', self.state, 'a' * 40, self.system)
        atomic_json(self.state / 'active.json',
            {'revision': 'a' * 40, 'image': 'old-image', 'bundle': self.serving})
        self.bundle = render(fixture.ROOT / 'config', host, PROFILE)
        self.workspace = workspace_target(self.bundle['compose']['services']['coding'],
            self.c.provisioning(self.bundle)[2])
        # The tool's polls and the switch's waits share the time module, so this is one patch:
        # the side effect plays the engine packing the workspace a couple of polls in, and the
        # call record doubles as the switch's wait clock, whose readiness cadence the tests
        # assert from it. Two separate patches here would replace each other's sleep, because
        # both target the same global attribute.
        self.polls = patch('runtime.provision.time.sleep')
        self.controller_sleeps = self.polls.start()
        self.controller_sleeps.side_effect = self.build_workspace
        self.addCleanup(self.polls.stop)

    def build_workspace(self, _seconds):
        """The engine packs the workspace a couple of polls in, then is still loading afterwards."""
        self.poll_number = getattr(self, 'poll_number', 0) + 1
        if self.poll_number >= 2:
            ready = self.workspace / 'config'
            ready.mkdir(parents=True, exist_ok=True)
            # The name the engine actually writes, for the family the profile names.
            (ready / 'strata-unsloth-ud-iq4_xs.json').write_text('{}')

    def receipt(self, **values):
        engine = self.c.provisioning(self.bundle)[1]
        recorded = {'profile': PROFILE, 'engine_tag': engine['tag'], 'image_id': engine['image_id'],
            'workspace': str(self.workspace), 'measured_at': '2026-10-12T05:00:00+00:00', 'runs': []}
        recorded.update(values)
        atomic_json(self.state / 'provisioning' / (PROFILE + '.json'), recorded)
        return recorded

    def test_a_switch_builds_the_workspace_before_it_takes_the_drain(self):
        """The selection is the operation.

        The build is the half that is safe beside a loaded model, so the switch does it before the
        drain; the order in the event log is the guarantee, and the boot it waited for is written
        down where the next gate can read it as evidence.
        """
        prepare = patch.object(self.c, 'prepare', side_effect=lambda b: None)
        prepare.start(); self.addCleanup(prepare.stop)
        # The router's record of what a backend announces is part of the switch; the fake keeps
        # that record in its manifest table, and the switch publishes the new one.
        for service in self.bundle['manifest']['services']:
            self.system.services[service['container_name']] = service
        result = self.c.transition(PROFILE)
        self.assertEqual(result['profile'], PROFILE)
        events = self.system.events
        first_drain = next(i for i, e in enumerate(events) if e[0] == 'drain' and e[1])
        build_up = next(i for i, e in enumerate(events) if e[0] == 'docker' and e[1][:1] == ('compose',)
            and e[1][1:3] == ('-p', PROJECT) and 'up' in e[1])
        switch_up = next(i for i, e in enumerate(events) if e[0] == 'docker' and e[1][:1] == ('compose',)
            and e[1][1:3] != ('-p', PROJECT) and 'up' in e[1])
        self.assertLess(build_up, first_drain)
        self.assertLess(first_drain, switch_up)
        recorded = json.loads((self.state / 'provisioning' / (PROFILE + '.json')).read_text())
        self.assertTrue(recorded['workspace_ready'])
        self.assertEqual(recorded['runs'][-2]['phase'], 'provision')
        self.assertEqual(recorded['runs'][-2]['via'], 'switch')
        self.assertEqual(recorded['runs'][-1]['phase'], 'warm')
        self.assertTrue(recorded['healthy_warm_boot'])
        self.assertGreaterEqual(recorded['warm_boot_seconds'], 0)

    def test_a_switch_that_cannot_build_the_workspace_changes_nothing(self):
        """A build failure stops the switch at its edge: no drain, no touched backend, a clean record."""
        system = ProvisioningSystem(self.serving, exits_after=1,
            log_text='=== Step 2: your choices ===\n  [X] that quant is not in the family named\nSetup stopped.')
        c = Controller(fixture.ROOT / 'config', self.state, 'a' * 40, system)
        # The model artifacts are a precondition checked before the build, and have their own
        # tests; this one is about what a failed build leaves behind.
        prepare = patch.object(c, 'prepare', side_effect=lambda b: None)
        prepare.start(); self.addCleanup(prepare.stop)
        with self.assertRaisesRegex(RuntimeError, 'Setup stopped'):
            c.transition(PROFILE)
        self.assertFalse(any(e[0] == 'drain' for e in system.events))
        self.assertEqual(c.load('active.json')['bundle']['profile'], 'qwen27b-q8-with-nighttime')
        transaction = c.load('transaction.json')
        self.assertEqual(transaction['phase'], 'failed')
        self.assertIn('could not build its workspace', str(transaction.get('error')))

    def test_a_measurement_only_counts_for_the_machine_it_was_taken_on(self):
        engine = self.c.provisioning(self.bundle)[1]
        self.receipt(healthy_warm_boot=True, warm_boot_seconds=100)
        self.assertIsNotNone(self.c.provisioned(self.bundle))
        self.receipt(healthy_warm_boot=True, warm_boot_seconds=100, image_id='sha256:' + '0' * 64)
        self.assertIsNone(self.c.provisioned(self.bundle))
        self.receipt(healthy_warm_boot=True, warm_boot_seconds=100, workspace=str(self.state / 'elsewhere'))
        self.assertIsNone(self.c.provisioned(self.bundle))
        # A stale receipt is not a refusal any more: the switch would build and measure the
        # workspace itself, and the gate says so instead of standing in the way.
        self.assertEqual(self.c.switchable(self.bundle), None)
        self.assertTrue(any('build it first' in note for note in self.c.verdict_notes(self.bundle)))

    def test_a_workspace_that_never_came_up_healthy_ends_the_trial(self):
        """An engine that cannot boot from its own provisioned state is not switchable, full stop:
        the switch would only spend a drain and a restoration to relearn what the measurement knew."""
        self.receipt(healthy_warm_boot=False,
            error='The container never answered its health check from the provisioned workspace',
            runs=[{'phase': 'warm', 'reached': 'timed out', 'seconds': 2900}])
        with self.assertRaisesRegex(RuntimeError, 'never completed a warm boot'):
            self.c.switchable(self.bundle)

    def test_a_container_that_stopped_at_startup_is_a_refusal(self):
        """The setup error is a refusal, not a wait: the switch would fail at the same edge, so it
        says so now, in the words the engine gave."""
        self.receipt(runs=[{'phase': 'provision', 'reached': 'exited (1)', 'seconds': 1.3,
            'container_log': 'Setup stopped.'}])
        with self.assertRaisesRegex(RuntimeError, 'stopped by itself at startup'):
            self.c.switchable(self.bundle)

    def test_a_thin_window_is_a_note_not_a_refusal(self):
        """The switch waits for the engine's actual answer and the recovery path stands behind it,
        so a window thin against the measurement is worth knowing before the selection, not a
        reason to sit down."""
        self.receipt(healthy_warm_boot=True, warm_boot_seconds=1500)
        self.assertEqual(self.c.switchable(self.bundle), None)
        self.assertTrue(any('at least 4501s' in note for note in self.c.verdict_notes(self.bundle)))
        self.receipt(healthy_warm_boot=True, warm_boot_seconds=1000)
        self.assertFalse(any('three times' in note for note in self.c.verdict_notes(self.bundle)))

    def test_an_engine_that_installs_inside_its_image_is_not_run_as_the_deployment_user(self):
        """What is baked into an image decides who can run it, and this trial proved it the hard way.

        The engine installs its own Python packages into its virtualenv on first start, which lives
        in the image and is not writable by the house user; the run died with a permission error two
        seconds in and the measurement spent two hours watching a stopped container. So the engine
        says on its entry that it runs as its image's user, and that grant is stated rather than
        assumed — the other engines keep the unprivileged house user they have always had.
        """
        cfg = self.bundle['compose']['services']['coding']
        self.assertNotIn('user', cfg)
        for service in self.serving['compose']['services'].values():
            self.assertEqual(service['user'], f"{self.HOST['uid']}:{self.HOST['gid']}")

    def test_the_trial_asks_for_the_quant_its_files_are(self):
        """The staged files are Unsloth's UD-IQ4_XS; naming the other family makes the engine refuse.

        This is not a style point: with the wrong family the engine stops at its second step saying
        the quant does not exist, which reads as a slow model and cost a whole idle window.
        """
        environment = self.bundle['compose']['services']['coding']['environment']
        self.assertEqual(environment['FAMILY'], 'unsloth')
        self.assertEqual(environment['MODEL'], 'UD-IQ4_XS')

    def test_the_ready_file_is_named_for_the_family_the_engine_was_given(self):
        """A literal file name rots the day the arguments change, and it rots as a silent wait.

        The engine finished its whole build on the trial host in under ten minutes and wrote
        ``config/strata-unsloth-ud-iq4_xs.json``; the declaration still asked for the name the other
        family would have used, so the run kept watching a workspace that was already finished, and
        the engine went on to load a model its cards had no room for and died. The quant stays
        literal because that is what the profile is about; the family follows the arguments.
        """
        provision = self.c.provisioning(self.bundle)[2]
        self.assertEqual([Path(name).name for name in provision['ready_files']],
            ['strata-*-ud-iq4_xs.json'])
        declared = ready_files(self.workspace, provision)
        self.assertFalse(workspace_ready(declared))
        (self.workspace / 'config').mkdir(parents=True, exist_ok=True)
        (self.workspace / 'config' / 'strata-unsloth-ud-iq4_xs.json').write_text('{}')
        self.assertTrue(workspace_ready(declared))
        self.assertEqual([Path(path).name for path in matching(declared)],
            ['strata-unsloth-ud-iq4_xs.json'])

    def test_a_container_that_refuses_to_start_is_not_waited_out(self):
        """A setup error printed once is a finding, not two hours of ``workspace=False``.

        The first two runs on the trial host were this: the engine refused, exited in a second, and
        the tool polled the corpse to the limit while the person watching saw nothing but a counter.
        """
        system = ProvisioningSystem(self.serving, exits_after=2,
            log_text='=== Step 2: your choices ===\n  [X] that quant is not in the family named\nSetup stopped.')
        c = Controller(fixture.ROOT / 'config', self.state, 'a' * 40, system)
        with patch('runtime.provision.time.sleep'):
            receipt = measurement(c, PROFILE, phase='provision', limit=7200, poll=15, log=lambda *a: None)
        run = receipt['runs'][-1]
        self.assertEqual(run['reached'], 'exited (1)')
        self.assertIn('Setup stopped.', run['container_log'])
        self.assertLess(len(run['samples']), 4)
        self.assertFalse(receipt['workspace_ready'])
        # It still says goodbye properly: the project is torn down rather than left stopped on disk.
        self.assertIn('down', [a for a in system.events[-1][1] if a in ('up', 'down')])
        reasons = c.verdict(self.bundle)
        self.assertTrue(any('stopped by itself' in reason for reason in reasons), reasons)

    def test_the_throwaway_project_asks_for_the_same_machine_the_switch_would(self):
        """Anything less and the measurement is about a different container than the one to come."""
        cfg = self.bundle['compose']['services']['coding']
        spec = throwaway(self.bundle, 'coding', cfg['container_name'] + '-provision', 18081)
        service = spec['services']['coding']
        self.assertEqual(list(spec['services']), ['coding'])
        self.assertEqual(spec['name'], 'ai-runtime-provision')
        self.assertEqual(service['container_name'], 'qwen38-daytime-provision')
        self.assertEqual(service['ports'], ['127.0.0.1:18081:8080'])
        self.assertEqual(service['restart'], 'no')
        self.assertEqual(service['labels']['local-ai.provisioning'], 'throwaway')
        for key in ('image', 'entrypoint', 'command', 'environment', 'volumes', 'ulimits',
                'tmpfs', 'healthcheck', 'read_only', 'init', 'pull_policy', 'logging'):
            self.assertEqual(service[key], cfg[key], key)
        # The image-user grant is part of the machine being measured: a throwaway that quietly ran as
        # somebody else would be measuring a container that cannot be switched to.
        self.assertNotIn('user', cfg)
        self.assertNotIn('user', service)
        devices = cfg['deploy']['resources']['reservations']['devices'][0]['device_ids']
        self.assertEqual(len(devices), 4)
        self.assertEqual(service['deploy'], cfg['deploy'])
        # It shares the router network with whatever is serving, because that network is the only
        # place the controller can reach a backend from, but it is reached by its own name and never
        # by the profile's service aliases, which stay the property of the container that is serving.
        self.assertEqual(spec['networks'], self.bundle['compose']['networks'])
        self.assertEqual(service['networks'], {'router': {}})
        self.assertTrue(cfg['networks']['router']['aliases'])
        self.assertEqual(cfg['container_name'], 'qwen38-daytime')
        self.assertEqual(cfg['restart'], 'unless-stopped')
        self.assertEqual(cfg['ports'], ['127.0.0.1:18080:8080'])

    def test_the_build_is_measured_while_the_serving_backend_is_watched(self):
        """Packing competes with a loaded model; this run is also the answer to whether that matters."""
        result = measurement(self.c, PROFILE, limit=60, poll=0, log=lambda line: None)
        self.assertTrue(result['workspace_ready'])
        self.assertGreaterEqual(result['provision_seconds'], 0)
        self.assertTrue(result['neighbour_healthy'])
        self.assertTrue(self.system.started)
        self.assertEqual([name for name, _port, _devices in self.system.started],
            ['qwen38-daytime-provision'])
        # Nothing about the serving pair was touched, and it is still serving.
        self.assertTrue(self.c.serving())
        self.assertFalse(self.system.draining)
        self.assertNotIn('qwen38-daytime-provision', self.system.containers)
        # The size of the pack and the space it left are part of the finding, not something a human
        # has to remember to write down beside the run.
        self.assertGreater(result['pack_bytes'], 0)
        for key in ('before', 'after'):
            self.assertIsInstance(result['disk_free_bytes'][key], int)
        self.assertEqual(result['runs'][-1]['free_bytes_after'], result['disk_free_bytes']['after'])

    def test_the_gate_says_what_the_selection_will_do_without_doing_it(self):
        """A selection nobody can understand before clicking is a selection made in the dark.

        The gate is read-only: what a switch to the profile would do, and what would refuse it,
        given without switching, without locking, without draining.
        """
        # Nothing measured yet: no refusal, and a note that says the build will happen first.
        self.assertEqual(self.c.verdict(self.bundle), [])
        self.c.switchable(self.bundle)
        notes = self.c.verdict_notes(self.bundle)
        self.assertTrue(any('build it first' in note for note in notes), notes)
        self.assertTrue(any('beside the model that is serving' in note for note in notes), notes)
        # A warm boot that was measured and failed: the one refusal that is evidence.
        self.receipt(healthy_warm_boot=False, error='never answered',
            runs=[{'phase': 'warm', 'reached': 'timed out', 'seconds': 2900}])
        reasons = self.c.verdict(self.bundle)
        self.assertEqual(len(reasons), 1)
        self.assertIn('never completed a warm boot', reasons[0])
        with self.assertRaisesRegex(RuntimeError, 'never completed a warm boot'):
            self.c.switchable(self.bundle)
        # Measured and healthy: no refusal at any window, and a thin window is a note.
        self.receipt(healthy_warm_boot=True, warm_boot_seconds=1500)
        self.assertEqual(self.c.verdict(self.bundle), [])
        self.c.switchable(self.bundle)
        self.assertTrue(any('at least 4501s' in note for note in self.c.verdict_notes(self.bundle)))
        self.receipt(healthy_warm_boot=True, warm_boot_seconds=1000)
        self.assertEqual(self.c.verdict(self.bundle), [])
        self.c.switchable(self.bundle)
        self.assertFalse(any('three times' in note for note in self.c.verdict_notes(self.bundle)))

    def test_the_page_sees_what_a_selection_will_do_before_the_click(self):
        """The same verdict the gate reports must sit on the profile itself in the status the page
        reads: a selection nobody can understand before clicking is a selection made in the dark."""
        entry = next(e for e in self.c.status()['configurations']['selectable'] if e['profile'] == PROFILE)
        self.assertEqual(entry['reasons'], [])
        self.assertTrue(any('build it first' in note for note in entry['notes']))
        self.receipt(healthy_warm_boot=False, error='never answered',
            runs=[{'phase': 'warm', 'reached': 'timed out', 'seconds': 2900}])
        entry = next(e for e in self.c.status()['configurations']['selectable'] if e['profile'] == PROFILE)
        self.assertTrue(any('never completed a warm boot' in reason for reason in entry['reasons']))

    def test_a_warm_boot_is_not_measured_while_the_cards_are_busy(self):
        """The tool never stops a serving model to time its replacement; that is a decision, not a side effect."""
        with self.assertRaisesRegex(RuntimeError, 'between turns'):
            measurement(self.c, PROFILE, phase='warm', limit=60, poll=0, log=lambda line: None)
        self.assertEqual(self.system.started, [])

    def test_a_measured_warm_boot_is_what_opens_the_switch(self):
        """The whole point: once the boot is timed here, the switch has a window that is not a guess."""
        self.system.containers['qwen38-daytime']['State']['Running'] = False
        # A clock that advances a minute per look, so the recorded boot is a number and not an
        # artefact of how fast the test happens to spin.
        clock = iter(range(0, 1000, 60))
        with patch('runtime.provision.time.monotonic', side_effect=lambda: next(clock)):
            result = measurement(self.c, PROFILE, phase='warm', limit=600, poll=0, log=lambda line: None)
        self.assertTrue(result['healthy_warm_boot'])
        self.assertEqual(result['warm_boot_seconds'], 240)
        self.assertEqual(result['runs'][-1]['reached'], 'healthy')
        self.assertEqual(self.system.health_polls, self.system.healthy_after)
        # Read over the network the controller actually shares with a backend: a published loopback
        # port is the host's route, not this one's, and a measurement taken through it is no measurement.
        self.assertEqual(result['runs'][-1]['samples'][-1]['health_via'], 'container')
        self.c.switchable(self.bundle)
        recorded = json.loads((self.state / 'provisioning' / (PROFILE + '.json')).read_text())
        self.assertEqual([run['phase'] for run in recorded['runs']], ['warm'])
        self.assertEqual(recorded['warm_boot_seconds'], 240)
        self.assertEqual(recorded['profile'], PROFILE)

    def test_a_warm_boot_is_not_measured_until_the_model_is_resident(self):
        """The engine's server answers before the model is resident, and its health check says
        so in loaded: a measurement that took the first answer as the boot would record a warm
        boot the length of the web page instead of the model's read, and the receipt would open
        the switch on a number the model never took."""
        self.system.containers['qwen38-daytime']['State']['Running'] = False
        self.system.healthy_after = 1
        self.system.loaded_after = 3
        clock = iter(range(0, 1000, 60))
        with patch('runtime.provision.time.monotonic', side_effect=lambda: next(clock)):
            result = measurement(self.c, PROFILE, phase='warm', limit=600, poll=0, log=lambda line: None)
        self.assertTrue(result['healthy_warm_boot'])
        self.assertEqual(self.system.health_polls, self.system.loaded_after)
        self.assertFalse(result['runs'][-1]['samples'][0]['health'])
        self.assertTrue(result['runs'][-1]['samples'][-1]['health'])

    def test_a_switch_waits_out_a_load_that_begins_before_the_model_is_resident(self):
        """Docker is healthy before the model is.

        Strata's /health says status ok from server start and carries the model state as loaded,
        so up --wait came back with the read still in flight, and the switch that took the first
        observe as the boot aborted the trial with the model still reading. The declared
        readiness window is the bound on the controller's own view: it re-observes until the
        engine reports the contracted window, and only then accepts.
        """
        prepare = patch.object(self.c, 'prepare', side_effect=lambda b: None)
        prepare.start(); self.addCleanup(prepare.stop)
        for service in self.bundle['manifest']['services']:
            self.system.services[service['container_name']] = service
        self.system.model_loaded = False
        self.system.load_after_slots = 2
        result = self.c.transition(PROFILE)
        self.assertEqual(result['profile'], PROFILE)
        self.assertEqual(self.c.load('active.json')['bundle']['profile'], PROFILE)
        self.assertTrue(self.system.model_loaded)
        # The wait ran against the controller's own clock, on its own cadence.
        self.controller_sleeps.assert_any_call(15)
        recorded = json.loads((self.state / 'provisioning' / (PROFILE + '.json')).read_text())
        self.assertTrue(recorded['healthy_warm_boot'])

    def test_a_loaded_engine_on_the_wrong_window_fails_without_spending_the_window(self):
        """loaded=true with the wrong slot window is a configuration fault, not a slow start.

        The failure has to come now, with the engine's own number in the record, and no readiness
        wait spent on it: waiting on a fault only turns it into a delay.
        """
        for service in self.bundle['manifest']['services']:
            self.system.services[service['container_name']] = service
        self.system.model_loaded = True
        self.system.slot_context = 131072
        with self.assertRaisesRegex(RuntimeError,
                r'slot window.*"reported_context": \[131072\].*"loaded": true'):
            self.c.reconcile(self.bundle, ['coding'])
        # No readiness wait on a fault: the loop never ran on its own cadence.
        self.assertFalse(any(call.args == (15,) for call in self.controller_sleeps.call_args_list))


if __name__ == '__main__':
    main()

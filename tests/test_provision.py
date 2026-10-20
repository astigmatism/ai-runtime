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
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import main
from unittest.mock import patch

try:
    import test_controller as fixture
except ModuleNotFoundError:
    # Run as `python3 -m unittest tests.test_provision` from the repository root, the fixtures are
    # reached through the package instead of the tests directory, which is not on the import path.
    from tests import test_controller as fixture
from runtime.config import read, render
from runtime.controller import Controller
from runtime.provision import (PROJECT, matching, measurement, ready_files, throwaway, workspace_ready,
    workspace_target)
from runtime.system import atomic_json

# The state file of a run is written under the resolved configuration id, so the exemplar is named
# by that id and not by the alias a client may still send for it.
PROFILE = 'flash-next-strata-solo-thinking-recovery'


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
            # A container whose log the base fake recorded answers with it; the throwaway keeps
            # the single text it is given, because its whole log is the setup output.
            return '\n'.join(self.container_logs.get(args[-1], [])) or self.log_text
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
                # The daemon's own rule, learned from the failure it produced on 2026-10-20: a bind
                # whose source is missing refuses the container, and the message names the path without
                # saying who was expected to create it. Only the workspace bind is checked, because the
                # artifact binds of a fixture point at a model tree no test stages on disk.
                for volume in service['volumes']:
                    if volume['target'] == '/data' and not Path(volume['source']).exists():
                        raise RuntimeError('invalid mount config for type "bind": bind source path '
                            'does not exist: ' + volume['source'])
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
        # The model tree exists; whether this profile's workspace inside it does is the thing under
        # test. provisioning is allowed to make the second and not the first.
        (self.state / 'models').mkdir()
        atomic_json(self.state / 'host.json', host); (self.state / 'router-token').write_text('synthetic-token')
        # The box is serving llama.cpp and the trial is the thing being asked for; a measurement is
        # only interesting while something else holds the cards.
        self.serving = render(fixture.ROOT / 'config', host, 'qwen27b-q8-pair')
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
        self.assertEqual(c.load('active.json')['bundle']['profile'], 'qwen27b-q8-pair')
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

    def test_a_workspace_that_never_came_up_healthy_ends_the_switch(self):
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

    def test_the_resident_asks_for_the_quant_its_files_are(self):
        """The staged files are Unsloth's UD-IQ4_XS; naming the other family makes the engine refuse.

        This is not a style point: with the wrong family the engine stops at its second step saying
        the quant does not exist, which reads as a slow model and cost a whole idle window.
        """
        environment = self.bundle['compose']['services']['coding']['environment']
        self.assertEqual(environment['FAMILY'], 'unsloth')
        self.assertEqual(environment['MODEL'], 'UD-IQ4_XS')

    def test_the_ready_file_finishes_a_workspace_without_naming_a_quant(self):
        """A literal file name rots the day the arguments change, and it rots as a silent wait.

        The engine finished its whole build on the trial host in under ten minutes and wrote
        ``config/strata-unsloth-ud-iq4_xs.json``; the declaration still asked for the name the other
        family would have used, so the run kept watching a workspace that was already finished, and
        the engine went on to load a model its cards had no room for and died.

        The quant used to stay literal for that same reason, and since 2026-10-15 it cannot: one
        Strata engine serves two quants and provisioning is read off the engine, so no quant name
        describes the workspace. Nor is a marker per quant the answer -- workspace_ready() requires
        every declared pattern to match, so adding the second would leave the first profile watching
        for a file it will never write, which is the identical silent wait. The marker is therefore
        workspace-relative, and the per-profile workspace is what makes it mean anything; that is
        asserted for every profile by ProvisioningWorkspaceTests in tests/test_config.py.
        """
        provision = self.c.provisioning(self.bundle)[2]
        self.assertEqual([Path(name).name for name in provision['ready_files']], ['strata-*.json'])
        declared = ready_files(self.workspace, provision)
        self.assertFalse(workspace_ready(declared))
        (self.workspace / 'config').mkdir(parents=True, exist_ok=True)
        (self.workspace / 'config' / 'strata-unsloth-ud-iq4_xs.json').write_text('{}')
        self.assertTrue(workspace_ready(declared))
        self.assertEqual([Path(path).name for path in matching(declared)],
            ['strata-unsloth-ud-iq4_xs.json'])
        # And the other quant on the same engine satisfies it exactly the same way. That is the
        # hazard the broadened glob accepts, stated where it can be seen: this marker cannot tell the
        # two quants apart, so it only means what the literal one meant because the workspace holds
        # one profile's engine and nothing else.
        (self.workspace / 'config' / 'strata-unsloth-ud-iq4_xs.json').unlink()
        (self.workspace / 'config' / 'strata-unsloth-ud-q4_k_xl.json').write_text('{}')
        self.assertTrue(workspace_ready(declared))
        (self.workspace / 'config' / 'strata-unsloth-ud-q4_k_xl.json').unlink()
        self.assertFalse(workspace_ready(declared))
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

    def test_a_workspace_nobody_created_yet_is_made_before_the_daemon_is_asked(self):
        """A new profile has no workspace, and that is not a reason a human has to be awake.

        The switch is advertised as building a workspace from nothing, and the plan this release is
        written against is one where nobody is at the keyboard between the drain and the answer. What
        it actually did on 2026-10-20, on the first provisioning of the two-card profile, was die in
        the daemon's mount validation three lines in: the directory the profile owns did not exist, so
        the bind was refused, and the message named the path without saying who was supposed to have
        made it. The fixture now refuses that way too, so this test would fail as a mount error rather
        than as a missing key.
        """
        self.assertFalse(self.workspace.exists())
        result = measurement(self.c, PROFILE, limit=60, poll=0, log=lambda line: None)
        self.assertTrue(self.workspace.is_dir(), 'the run started without making its own workspace')
        self.assertTrue(result['workspace_ready'])
        run = (self.c.load('provisioning/' + PROFILE + '.json') or {'runs': []})['runs'][-1]
        self.assertTrue(run['workspace_created'], 'the receipt does not say it made the directory')
        # Group-writable, because the engine writes into it as its image's user: the workspaces this
        # host already holds are 775 and a 755 directory is a boot that cannot write its pack.
        self.assertEqual(self.workspace.stat().st_mode & 0o777, 0o775)

    def test_a_model_root_that_is_not_there_is_a_refusal_rather_than_a_new_tree(self):
        """Creating a workspace is bounded by the model root, which is never invented."""
        shutil.rmtree(self.state / 'models')
        with self.assertRaisesRegex(RuntimeError, 'model root does not exist'):
            measurement(self.c, PROFILE, limit=60, poll=0, log=lambda line: None)
        self.assertFalse((self.state / 'models').exists(),
            'provisioning invented a model tree under a root that was not there')
        self.assertFalse(self.workspace.exists())

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

    def test_a_switch_carries_the_last_words_of_an_engine_that_dies_mid_load(self):
        """The record must not let the witness be deleted.

        The trial's 18:47 attempt: the engine answered its health check, began the read with the
        cards free, and stopped itself a couple of minutes in. The restoration that followed
        removed the container, and the log inside it was the engine's only account of why - so
        the failure record has to hold the state, the exit, and the last words itself.
        """
        prepare = patch.object(self.c, 'prepare', side_effect=lambda b: None)
        prepare.start(); self.addCleanup(prepare.stop)
        for service in self.bundle['manifest']['services']:
            self.system.services[service['container_name']] = service
        self.system.model_loaded = False
        self.system.die_after_slots = 2
        self.system.die_words = ['reading the model weights ...',
            'the engine exited code 1: CUDA error out of memory']
        with self.assertRaisesRegex(RuntimeError,
                r'container not running, exit 1, last words: reading the model weights \.\.\. \| '
                r'the engine exited code 1: CUDA error out of memory'):
            self.c.transition(PROFILE)
        # It waited while the engine was still loading, and failed the moment it was not.
        self.assertTrue(any(call.args == (15,) for call in self.controller_sleeps.call_args_list))
        # The house is back to what it was before the attempt.
        self.assertEqual(self.c.load('active.json')['bundle']['profile'], 'qwen27b-q8-pair')


class VisionEntrypointOverlayTests(unittest.TestCase):
    """This house's entry point inside the vision image, run the way the container runs it.

    The run config this entry point is about to start has to carry a ``vision`` section, and that
    section has to name the card the profile states as ``VISION_DEVICE``. The section is written by
    setup.py's setup pass, which the upstream entry point runs only for a workspace that has no
    config yet; a workspace set up before vision was asked for would boot text-only with the
    encoder never started and nothing in any log (2026-10-16). So this entry point runs that pass
    itself where the section is missing, and it is tested -- against the real script, in a directory
    laid out like ``/opt/strata``, with a stub standing in for the upstream entry point it has to
    reach and one for the engine's python the pass calls. The same entry point also writes the run
    config's optional ``sampling`` block from ``SAMPLING``, because the engine's own default is greedy
    and its setup has no flag for that block either; both statements are tested here against the same
    stubs. And it writes the session window ``CONTEXT`` states into the ``--max-context`` of the
    config's ``args`` list, because the setup pass that would otherwise have written it runs once per
    workspace and a profile that widens the window is not a fresh workspace -- which is tested here
    too, including the boot that must not happen when the stated window is not a token count.
    """
    OVERLAY = 'runtime/overlay/strata-vision-entrypoint.sh'
    CONFIG = 'config/strata-unsloth-ud-iq4_xs.json'

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.opt = root / 'opt'; self.opt.mkdir()
        self.workspace = root / 'data'
        shutil.copyfile(fixture.ROOT / self.OVERLAY, self.opt / 'docker-entrypoint.sh')
        upstream = self.opt / 'docker-entrypoint.upstream.sh'
        upstream.write_text('#!/bin/sh\necho "upstream $*"\n')
        upstream.chmod(0o755)  # The Dockerfile does the same to the file it moved aside.
        self.config = self.workspace / self.CONFIG
        self.calls = root / 'setup-calls.log'
        self.install_stub_python()

    def install_stub_python(self):
        # setup.py as this entry point sees it: it records the call, and where a setup pass is asked
        # for it writes the config the way the real one does -- next to itself (the working
        # directory, standing in for the container's /opt/strata) -- with a "vision" section only
        # when --vision says one runs, carrying over the top-level keys of a config already there.
        venv = self.opt / '.venv' / 'bin'
        venv.mkdir(parents=True)
        stub = venv / 'python'
        stub.write_text(
            '#!/usr/bin/env python3\n'
            'import json, os, sys\n'
            'args = sys.argv[1:]\n'
            'with open(os.environ["SETUP_CALL_LOG"], "a") as f:\n'
            '    f.write(" ".join(args) + chr(10))\n'
            'if "--setup" in args:\n'
            '    family, model = os.environ.get("FAMILY", "qwen"), os.environ.get("MODEL", "")\n'
            '    tag = ("" if family == "qwen" else family + "-") + model.lower()\n'
            '    cfg_path = os.path.join(os.environ["STRATA_DATA"], "config", "strata-%s.json" % tag)\n'
            '    try:\n'
            '        old = json.load(open(cfg_path, encoding="utf-8-sig"))\n'
            '    except (OSError, ValueError):\n'
            '        old = {}\n'
            '    cfg = dict(old)\n'
            '    cfg.setdefault("model", model)\n'
            '    value = args[args.index("--vision") + 1] if "--vision" in args else "no"\n'
            '    if value in ("yes", "gpu", "cpu"):\n'
            '        kept = {k: v for k, v in (old.get("vision") or {}).items()\n'
            '                if k not in ("exe", "mmproj", "model", "gpu", "max_tokens", "threads")}\n'
            '        cfg["vision"] = {"exe": "/opt/strata/engine/strata-vision",\n'
            '                         "mmproj": "/weights/mmproj.gguf",\n'
            '                         "model": "/weights/shard-00001-of-00003.gguf",\n'
            '                         "gpu": value != "cpu", "max_tokens": 1536, **kept}\n'
            '    else:\n'
            '        cfg.pop("vision", None)\n'
            # setup.py always writes the argv the engine is started from, and the window it was asked
            # for is in it as --max-context. An existing list is carried over, which is what the pass
            # does for a workspace it is only being asked to add a section to.
            '    ctx = args[args.index("--context") + 1] if "--context" in args else "32768"\n'
            '    argv = old.get("args")\n'
            '    if not isinstance(argv, list) or not argv:\n'
            '        argv = ["--serve", "--model", "/weights/shard-00001-of-00003.gguf",\n'
            '                "--api-type", "enterprise", "--port", "8080", "--max-context", ctx]\n'
            '    cfg["args"] = argv\n'
            '    with open("strata-%s.json" % tag, "w") as f:\n'
            '        json.dump(cfg, f, indent=1)\n')
        stub.chmod(0o755)

    def given(self, config):
        if config is not None:
            self.config.parent.mkdir(parents=True, exist_ok=True)
            self.config.write_text(json.dumps(config))

    def start(self, device='4', family='unsloth', model='UD-IQ4_XS', extra=None, expect_fail=False):
        env = {**os.environ, 'STRATA_DATA': str(self.workspace), 'FAMILY': family, 'MODEL': model,
               'VISION': 'yes', 'SETUP_CALL_LOG': str(self.calls)}
        # Statements the host shell may happen to carry and this entry point must not read silently
        # from the environment: only what a test states below is in force.
        for key in ('VISION_DEVICE', 'SAMPLING', 'CONTEXT', 'PLE_GGUF', 'CONVERSATION_CACHE_MIB',
                    'CONVERSATION_CACHE_MIN_FREE_MIB', 'CONVERSATION_CACHE_SLOTS',
                    'CONVERSATION_CACHE_MIN_TOKENS', 'ROPE_SCALING', 'ROPE_SCALE', 'YARN_ORIG_CTX',
                    'VRAM_RESERVE_MIB', 'CONTROL_VECTOR', 'CONTROL_VECTOR_SCALE',
                    'CONTROL_VECTOR_LAYERS', 'STRATA_LOG'):
            env.pop(key, None)
        if extra:
            env.update(extra)
        if device is not None:
            env['VISION_DEVICE'] = device
        run = subprocess.run(['sh', './docker-entrypoint.sh', '--serve'], cwd=self.opt, env=env,
            capture_output=True, text=True)
        if expect_fail:
            self.assertNotEqual(run.returncode, 0)
        else:
            self.assertEqual(run.returncode, 0, run.stderr)
            # Handing over is the entry point's whole second job: the engine has to start with the
            # arguments the container was given, or the placement was applied to a boot that never ran.
            self.assertIn('upstream --serve', run.stdout)
        return run

    def setup_calls(self):
        if not self.calls.exists():
            return []
        return [line.split() for line in self.calls.read_text().splitlines() if line]

    def test_the_declared_card_is_written_into_the_config_the_engine_reads(self):
        self.given({'model': 'UD-IQ4_XS', 'vision': {'cuda_device': 0, 'max_tokens': 1536}})
        run = self.start()
        self.assertEqual(json.loads(self.config.read_text())['vision'],
            {'cuda_device': 4, 'max_tokens': 1536})
        self.assertIn('image encoder on CUDA device 4', run.stdout)
        self.assertEqual(self.setup_calls(), [])

    def test_a_workspace_that_has_been_placed_once_is_left_as_it_is(self):
        self.given({'vision': {'cuda_device': 2}})
        self.start(); self.start()
        self.assertEqual(json.loads(self.config.read_text())['vision'], {'cuda_device': 4})
        self.assertIn('already on CUDA device 4', self.start().stdout)

    def test_a_workspace_without_a_config_gets_the_setup_pass_with_vision(self):
        # A fresh workspace has no config, so the upstream entry point would run the setup pass
        # itself after this one -- and the encoder would start on the pass's default card. This
        # entry point runs the pass first, with vision on, so the card the profile states is in the
        # file the engine reads on this very boot, not the next.
        self.start()
        cfg = json.loads(self.config.read_text())
        self.assertEqual(cfg['vision']['cuda_device'], 4)
        self.assertEqual(cfg['vision']['gpu'], True)
        calls = self.setup_calls()
        self.assertEqual(len(calls), 1)
        args = calls[0]
        self.assertIn('--setup', args)
        self.assertIn('--yes', args)
        self.assertIn('--no-start', args)
        self.assertEqual(args[args.index('--vision') + 1], 'yes')
        self.assertIn('--data-dir', args)

    def test_a_config_without_a_vision_section_gets_the_setup_pass(self):
        # The 2026-10-16 state: a workspace set up before vision was asked for. The upstream entry
        # point skips its setup pass for a config that exists and would serve text-only with the
        # encoder never started; this entry point runs the pass instead, so the boot that asks for
        # a card gets the card, and the pass carries the existing config's keys over.
        self.given({'model': 'UD-IQ4_XS', 'allowed_hosts': ['qwen38-daytime']})
        run = self.start()
        cfg = json.loads(self.config.read_text())
        self.assertEqual(cfg['vision']['cuda_device'], 4)
        self.assertEqual(cfg['allowed_hosts'], ['qwen38-daytime'])
        self.assertEqual(len(self.setup_calls()), 1)
        self.assertIn('no vision section', run.stdout)

    def test_the_setup_pass_runs_once_and_then_the_patch_keeps_the_card(self):
        self.start()
        self.start()
        self.assertEqual(len(self.setup_calls()), 1)
        self.assertEqual(json.loads(self.config.read_text())['vision']['cuda_device'], 4)

    def test_the_setup_pass_takes_the_profile_arguments(self):
        # The pass has to set up the model this container serves, not a default one: the profile's
        # arguments ride the environment the renderer states, and the pass takes them verbatim.
        self.start(extra={'KV': 'k8v4', 'GPUS': '0,1,2,3', 'LAYER_SPLIT': 'auto', 'GGUF_DIR': '/weights',
                          'KV_STREAMING': 'auto', 'LOW_RAM': 'off'})
        args = self.setup_calls()[0]
        for flag, value in (('--kv', 'k8v4'), ('--gpus', '0,1,2,3'), ('--layer-split', 'auto'),
                            ('--gguf-dir', '/weights'), ('--kv-streaming', 'auto'), ('--low-ram', 'off')):
            self.assertEqual(args[args.index(flag) + 1], value)

    def test_a_card_with_vision_off_fails_the_boot(self):
        # A misconfiguration, not a boot state this house renders (the renderer states VISION=yes
        # with VISION_DEVICE): a card for an encoder vision is told not to run must fail the boot,
        # not serve text-only with a log gap.
        self.given({'model': 'UD-IQ4_XS'})
        run = self.start(extra={'VISION': 'no'}, expect_fail=True)
        self.assertNotIn('upstream --serve', run.stdout)
        self.assertIn('still has no vision section', run.stderr)

    def test_the_text_only_container_rewrites_nothing(self):
        # No VISION_DEVICE is the text-only image: it has no encoder, and an entry point that rewrote
        # a workspace it was never asked to configure would be drift with a log line.
        self.given({'vision': {'cuda_device': 2}})
        before = self.config.read_text()
        self.start(device=None)
        self.assertEqual(self.config.read_text(), before)
        self.assertEqual(self.setup_calls(), [])

    def test_the_sampling_preset_is_written_into_the_config_the_engine_reads(self):
        # With no sampling block the engine decodes greedy, and every routed client that sends none of
        # its own is served at that one setting the card does not recommend. The preset is the card's
        # thinking-mode set, and every other key of the file the engine reads goes back as it was.
        self.given({'model': 'UD-IQ4_XS', 'context': 163840, 'allowed_hosts': ['qwen38-daytime']})
        run = self.start(device=None, extra={'VISION': 'no', 'SAMPLING': 'thinking'})
        cfg = json.loads(self.config.read_text())
        self.assertEqual(cfg['sampling'],
            {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0})
        self.assertEqual(cfg['allowed_hosts'], ['qwen38-daytime'])
        self.assertEqual(cfg['context'], 163840)
        self.assertIn('sampling preset thinking', run.stdout)
        self.assertEqual(self.setup_calls(), [])

    def test_the_preset_is_written_once_and_then_left_alone(self):
        self.given({'model': 'UD-IQ4_XS'})
        self.start(device=None, extra={'VISION': 'no', 'SAMPLING': 'thinking'})
        placed = self.config.read_text()
        self.assertIn('already the thinking preset',
            self.start(device=None, extra={'VISION': 'no', 'SAMPLING': 'thinking'}).stdout)
        self.assertEqual(self.config.read_text(), placed)

    def test_a_block_the_profile_does_not_state_is_replaced(self):
        # A hand edit inside the workspace is not a setting this house renders: the profile is the
        # authority, so a block it does not state is replaced and the boot says what it replaced.
        self.given({'model': 'UD-IQ4_XS', 'sampling': {'temperature': 0.6}})
        run = self.start(device=None, extra={'VISION': 'no', 'SAMPLING': 'instruct'})
        self.assertEqual(json.loads(self.config.read_text())['sampling'],
            {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 1.5})
        self.assertIn('replacing a sampling block', run.stdout)

    def test_off_takes_the_block_back_out(self):
        # Going back to greedy has to be a profile change and not a hand edit, because the workspace
        # outlives the image: a block left in it is still read by a text-only image that has no entry
        # point to take it out.
        self.given({'model': 'UD-IQ4_XS', 'sampling': {'temperature': 1.0, 'top_p': 0.95}})
        run = self.start(device=None, extra={'VISION': 'no', 'SAMPLING': 'off'})
        self.assertNotIn('sampling', json.loads(self.config.read_text()))
        self.assertIn('removed the sampling block', run.stdout)
        self.assertIn('no sampling block to remove',
            self.start(device=None, extra={'VISION': 'no', 'SAMPLING': 'off'}).stdout)

    def test_a_preset_this_entry_point_does_not_know_fails_the_boot(self):
        # The engine refuses to start on a sampling value it does not understand, and this resident is
        # exclusive, so a name outside the table fails here rather than after the drain -- and it
        # fails without touching the file, so the boot that follows is the one that was already there.
        self.given({'model': 'UD-IQ4_XS', 'sampling': {'temperature': 0.6}})
        before = self.config.read_text()
        run = self.start(device=None, extra={'VISION': 'no', 'SAMPLING': 'temperature=1.0'},
            expect_fail=True)
        self.assertNotIn('upstream --serve', run.stdout)
        self.assertIn('not a preset', run.stderr)
        self.assertEqual(self.config.read_text(), before)

    def test_a_fresh_workspace_gets_the_setup_pass_and_then_the_block(self):
        # No config yet: the upstream entry point would run its setup pass and write a config out of
        # nothing, so a block written first would be lost. The pass runs here instead, once, with the
        # arguments this container was given, and the block goes into the file it produced.
        run = self.start(device=None, extra={'VISION': 'no', 'SAMPLING': 'thinking'})
        cfg = json.loads(self.config.read_text())
        self.assertEqual(cfg['sampling'],
            {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0})
        self.assertNotIn('vision', cfg)
        calls = self.setup_calls()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][calls[0].index('--vision') + 1], 'no')
        self.assertIn('does not exist', run.stdout)

    def test_the_card_and_the_preset_arrive_on_the_same_boot(self):
        # The two statements this entry point writes are independent of each other: a container told
        # both gets both, and a container told only the card still gets its sampling.
        self.given({'model': 'UD-IQ4_XS', 'vision': {'cuda_device': 0}})
        self.start(extra={'SAMPLING': 'thinking'})
        cfg = json.loads(self.config.read_text())
        self.assertEqual(cfg['vision']['cuda_device'], 4)
        self.assertEqual(cfg['sampling']['temperature'], 1.0)
        # And a boot that restates only the card does not quietly lose the preset: the block it was
        # never told to remove stays, because the profile states it either way.
        self.start()
        self.assertEqual(json.loads(self.config.read_text())['sampling']['temperature'], 1.0)

    def test_tool_call_recovery_on_writes_the_boolean_the_engine_reads(self):
        # serve/server.py reads the run config's top-level key and raises SystemExit unless it is a
        # JSON boolean, so what this writes is `true` and not "true": a string the engine will not read
        # is nothing serving, on the one resident that is exclusive.
        self.given({'model': 'UD-IQ4_XS', 'context': 163840, 'allowed_hosts': ['qwen38-daytime']})
        run = self.start(device=None, extra={'VISION': 'no', 'TOOL_CALL_RECOVERY': 'on'})
        cfg = json.loads(self.config.read_text())
        self.assertIs(cfg['tool_call_recovery'], True)
        self.assertEqual(cfg['allowed_hosts'], ['qwen38-daytime'])
        self.assertEqual(cfg['context'], 163840)
        self.assertIn("tool_call_recovery on (true)", run.stdout)
        self.assertEqual(self.setup_calls(), [])

    def test_off_takes_the_key_back_out_rather_than_writing_false(self):
        # Same reason `off` removes the sampling block: the run config lives in the workspace, which
        # outlives every configuration, so switching back has to be the profile's act and not a hand
        # edit inside a directory. Removing it also leaves the file exactly as setup.py wrote it, which
        # is what the engine's own default looks like.
        self.given({'model': 'UD-IQ4_XS', 'tool_call_recovery': True})
        run = self.start(device=None, extra={'VISION': 'no', 'TOOL_CALL_RECOVERY': 'off'})
        self.assertNotIn('tool_call_recovery', json.loads(self.config.read_text()))
        self.assertIn('removed tool_call_recovery', run.stdout)
        self.assertIn('no tool_call_recovery key to remove',
            self.start(device=None, extra={'VISION': 'no', 'TOOL_CALL_RECOVERY': 'off'}).stdout)

    def test_a_word_the_engine_cannot_read_fails_the_boot_before_the_file(self):
        # Not "true", not 1, not "yes": the two words this entry point knows are the only ones that
        # reach the file, and an unknown one fails here rather than as a SystemExit after the drain.
        self.given({'model': 'UD-IQ4_XS', 'tool_call_recovery': False})
        before = self.config.read_text()
        for stated in ('maybe', 'true', '1'):
            with self.subTest(value=stated):
                run = self.start(device=None, extra={'VISION': 'no', 'TOOL_CALL_RECOVERY': stated},
                    expect_fail=True)
                self.assertNotIn('upstream --serve', run.stdout)
                self.assertIn('not a value this entry point knows', run.stderr)
                self.assertEqual(self.config.read_text(), before)

    def test_a_boot_that_states_nothing_about_recovery_leaves_the_key_alone(self):
        # A profile that has never heard of the key is not a statement about it: an image booted with
        # no TOOL_CALL_RECOVERY neither writes nor removes, so an existing workspace keeps whatever
        # the configuration that wrote it stated until a configuration says otherwise.
        self.given({'model': 'UD-IQ4_XS', 'tool_call_recovery': True})
        before = self.config.read_text()
        self.start(device=None, extra={'VISION': 'no'})
        self.assertEqual(self.config.read_text(), before)

    def test_a_number_in_the_workspace_becomes_the_boolean_the_engine_starts_on(self):
        # A hand edit that wrote 1 instead of true is a boot-faulter waiting to happen -- the engine
        # raises SystemExit unless this key is a JSON boolean -- and `1 == True` in Python is exactly
        # the trap that would leave it there, so the comparison this entry point uses is identity.
        self.given({'model': 'UD-IQ4_XS', 'tool_call_recovery': 1})
        run = self.start(device=None, extra={'VISION': 'no', 'TOOL_CALL_RECOVERY': 'on'})
        cfg = json.loads(self.config.read_text())
        self.assertIs(cfg['tool_call_recovery'], True)
        self.assertIn('replacing a tool_call_recovery the profile does not state: 1', run.stdout)

    def test_the_preset_and_the_recovery_switch_arrive_on_the_same_boot(self):
        # flash-next-strata-solo-thinking-recovery states both, and neither is allowed to disturb the
        # other: two writes to one file, from two profile statements, in one boot.
        self.given({'model': 'UD-IQ4_XS'})
        self.start(device=None, extra={'VISION': 'no', 'SAMPLING': 'thinking',
            'TOOL_CALL_RECOVERY': 'on'})
        cfg = json.loads(self.config.read_text())
        self.assertEqual(cfg['sampling']['temperature'], 1.0)
        self.assertIs(cfg['tool_call_recovery'], True)
        # Restating only the sampling leaves the switch: the profile states it either way, and a
        # configuration that wanted it gone says off.
        self.start(device=None, extra={'VISION': 'no', 'SAMPLING': 'thinking'})
        self.assertIs(json.loads(self.config.read_text())['tool_call_recovery'], True)

    def test_the_stated_window_becomes_the_context_the_engine_starts_on(self):
        # The window a profile states is rendered into the catalog, shown in the portal, and verified
        # against the engine's own /slots before the switch is allowed to publish it. So the number in
        # this file has to be the number the engine boots on, and a file that keeps the window the
        # workspace was born with is a resident refused for reporting another context -- after the
        # drain, which is the expensive place to discover it.
        self.given({'model': 'UD-IQ4_XS', 'context': 163840,
            'args': ['--serve', '--model', '/weights/shard-00001-of-00003.gguf', '--max-context',
                     '163840', '--kv', 'k8v4', '--kv-resident', '32768']})
        run = self.start(device=None, extra={'VISION': 'no', 'CONTEXT': '196608'})
        args = json.loads(self.config.read_text())['args']
        self.assertEqual(args.count('--max-context'), 1)
        self.assertEqual(args[args.index('--max-context') + 1], '196608')
        # Every other argument the engine is started from goes back exactly as it was read: the window
        # is the one thing this boot was told to change, and a resident that lost --kv-resident on the
        # way would pay for it in VRAM it no longer has.
        self.assertEqual([a for a in args if a.startswith('--') and a != '--max-context'],
            ['--serve', '--model', '--kv', '--kv-resident'])
        self.assertEqual(args[args.index('--kv') + 1], 'k8v4')
        self.assertEqual(args[args.index('--kv-resident') + 1], '32768')
        self.assertIn('wrote --max-context 196608', run.stdout)
        self.assertEqual(self.setup_calls(), [])

    def test_the_window_is_written_once_and_then_left_alone(self):
        self.given({'model': 'UD-IQ4_XS', 'args': ['--serve', '--max-context', '163840']})
        self.start(device=None, extra={'VISION': 'no', 'CONTEXT': '196608'})
        placed = self.config.read_text()
        self.assertIn('already in the args list (--max-context 196608)',
            self.start(device=None, extra={'VISION': 'no', 'CONTEXT': '196608'}).stdout)
        self.assertEqual(self.config.read_text(), placed)

    def test_a_workspace_whose_args_list_names_no_window_gets_the_stated_one(self):
        # --max-context absent is the engine's own default of 4096, which is not a session this house
        # would notice, and a profile that states a window does not mean "whatever the file prefers".
        self.given({'model': 'UD-IQ4_XS', 'args': ['--serve', '--kv', 'k8v4']})
        self.start(device=None, extra={'VISION': 'no', 'CONTEXT': '262144'})
        args = json.loads(self.config.read_text())['args']
        self.assertEqual(args[args.index('--max-context') + 1], '262144')

    def test_a_wider_window_reaches_the_engine_on_the_boot_that_asks_for_it(self):
        # The bug this fact exists for: a workspace set up at 160K, asked by a profile edit for 256K.
        # The upstream entry point skips its setup pass (the config exists), so nothing else in the
        # boot ever looks at CONTEXT again, the engine comes up on 163840, and the switch refuses it
        # with reason 'slot context'. One boot has to carry both statements at once -- the encoder's
        # card, which needs the setup pass, and the window, which needs the file that pass wrote.
        self.given({'model': 'UD-IQ4_XS', 'args': ['--serve', '--max-context', '163840']})
        run = self.start(extra={'CONTEXT': '262144'})
        cfg = json.loads(self.config.read_text())
        self.assertEqual(len(self.setup_calls()), 1)
        self.assertEqual(cfg['vision']['cuda_device'], 4)
        self.assertEqual(cfg['args'][cfg['args'].index('--max-context') + 1], '262144')
        self.assertIn('upstream --serve', run.stdout)

    def test_a_fresh_workspace_boots_on_the_window_its_profile_states(self):
        # No config at all: the pass this entry point runs is handed the stated window, and the args
        # list it produces carries it -- written again if the pass decided otherwise, because the
        # profile is the authority and the switch checks the engine and not the intent.
        run = self.start(extra={'CONTEXT': '196608'})
        args = json.loads(self.config.read_text())['args']
        self.assertEqual(args[args.index('--max-context') + 1], '196608')
        calls = self.setup_calls()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][calls[0].index('--context') + 1], '196608')
        self.assertIn('upstream --serve', run.stdout)

    def test_a_window_that_is_not_a_token_count_fails_the_boot_before_the_file(self):
        # A window the engine cannot parse is nothing serving, and a window quietly dropped is a
        # resident refused after the drain for a context nobody asked for. Either way the boot does
        # not happen, and it does not happen with the workspace left as it was.
        self.given({'model': 'UD-IQ4_XS', 'args': ['--serve', '--max-context', '163840']})
        before = self.config.read_text()
        for stated in ('192K', '0', '-1', '163 840'):
            with self.subTest(window=stated):
                run = self.start(device=None, extra={'VISION': 'no', 'CONTEXT': stated},
                    expect_fail=True)
                self.assertNotIn('upstream --serve', run.stdout)
                self.assertIn('CONTEXT=' + stated + ' is not a number of tokens', run.stderr)
                self.assertEqual(self.config.read_text(), before)

    def test_a_container_started_with_no_window_leaves_the_workspace_alone(self):
        # Nothing stated is not a statement. The default this script carries (32768) is for the setup
        # pass of a workspace that has no config; over one that has one it would silently resize a
        # session to a quarter of what the configuration that built it asked for.
        self.given({'model': 'UD-IQ4_XS', 'args': ['--serve', '--max-context', '163840']})
        before = self.config.read_text()
        self.start(device=None, extra={'VISION': 'no'})
        self.assertEqual(self.config.read_text(), before)
        self.assertEqual(self.setup_calls(), [])

    def test_the_stated_ring_is_written_into_the_args_list_the_engine_reads(self):
        # Fact 8. A window past the trained one is YaRN interpolation and the engine only reads that
        # out of its args list, fixed at startup because cached keys are post-RoPE. The three pairs
        # arrive together, which is the only shape that leaves no half a ring behind.
        self.given({'model': 'UD-IQ4_XS', 'args': ['--serve', '--model', '/weights/shard.gguf',
            '--max-context', '262144', '--kv', 'k8v4']})
        run = self.start(device=None, extra={'VISION': 'no', 'CONTEXT': '524288',
            'ROPE_SCALING': 'yarn', 'ROPE_SCALE': '2.0', 'YARN_ORIG_CTX': '262144'})
        args = json.loads(self.config.read_text())['args']
        for flag, value in (('--rope-scaling', 'yarn'), ('--rope-scale', '2.0'),
                            ('--yarn-orig-ctx', '262144')):
            self.assertEqual(args.count(flag), 1)
            self.assertEqual(args[args.index(flag) + 1], value)
        # The window the same boot states is written too, and every unrelated argument goes back
        # byte for byte -- including a rope-sounding neighbour nothing here wrote.
        self.assertEqual(args[args.index('--max-context') + 1], '524288')
        self.assertEqual([a for a in args if a.startswith('--') and a not in
            ('--rope-scaling', '--rope-scale', '--yarn-orig-ctx', '--max-context')],
            ['--serve', '--model', '--kv'])
        self.assertIn('wrote --rope-scaling yarn, --rope-scale 2.0, --yarn-orig-ctx 262144', run.stdout)
        self.assertEqual(self.setup_calls(), [])

    def test_the_ring_is_written_once_and_then_left_alone(self):
        self.given({'model': 'UD-IQ4_XS', 'args': ['--serve', '--max-context', '524288']})
        rope = {'ROPE_SCALING': 'yarn', 'ROPE_SCALE': '2.0', 'YARN_ORIG_CTX': '262144'}
        self.start(device=None, extra={'VISION': 'no', 'CONTEXT': '524288', **rope})
        placed = self.config.read_text()
        self.assertIn('already in the args list',
            self.start(device=None, extra={'VISION': 'no', 'CONTEXT': '524288', **rope}).stdout)
        self.assertEqual(self.config.read_text(), placed)

    def test_a_stranger_s_ring_does_not_survive_the_switch_back(self):
        # The landmine this fact was designed for: the 512K configuration wrote a factor 2.0 into a
        # workspace's args list, the operator selects the 262K sibling, and a 262K window read
        # through somebody else's ring boots, serves, answers wrong, and matches /slots exactly --
        # the one verification the switch has cannot see it, so silence here must strip.
        self.given({'model': 'UD-IQ4_XS', 'args': ['--serve', '--max-context', '262144',
            '--rope-scaling', 'yarn', '--rope-scale', '2.0', '--yarn-orig-ctx', '262144',
            '--kv', 'k8v4']})
        run = self.start(device=None, extra={'VISION': 'no', 'CONTEXT': '262144'})
        args = json.loads(self.config.read_text())['args']
        for flag in ('--rope-scaling', '--rope-scale', '--yarn-orig-ctx'):
            self.assertNotIn(flag, args)
        self.assertIn('removed --rope-scaling, --rope-scale, --yarn-orig-ctx the profile does not state',
            run.stdout)
        # The window and the unrelated neighbours come back exactly as they were read.
        self.assertEqual(args, ['--serve', '--max-context', '262144', '--kv', 'k8v4'])

    def test_off_takes_every_rope_line_back_out_rather_than_writing_words_for_none(self):
        self.given({'model': 'UD-IQ4_XS', 'args': ['--serve', '--max-context', '262144',
            '--rope-scaling', 'linear', '--rope-scale', '1.5']})
        run = self.start(device=None, extra={'VISION': 'no', 'CONTEXT': '262144',
            'ROPE_SCALING': 'off'})
        args = json.loads(self.config.read_text())['args']
        self.assertNotIn('--rope-scaling', args)
        self.assertNotIn('--rope-scale', args)
        self.assertIn('the profile does not state', run.stdout)

    def test_a_ring_the_engine_cannot_read_fails_the_boot_before_the_file(self):
        # Same refusal as the window has, for the same reason: every one of these is an argument the
        # engine parses at startup, and a refused boot costs less than a serving wrong one --
        # especially this one, which the switch cannot catch afterwards at all.
        self.given({'model': 'UD-IQ4_XS', 'args': ['--serve', '--max-context', '524288']})
        before = self.config.read_text()
        base = {'ROPE_SCALING': 'yarn', 'ROPE_SCALE': '2.0', 'YARN_ORIG_CTX': '262144'}
        # Each case is a complete statement of the boot's rope environment; None takes a variable
        # out, which is what "stated no scale" means here.
        for breakage, message in (
                ({'ROPE_SCALING': 'ntk', 'ROPE_SCALE': '2.0', 'YARN_ORIG_CTX': '262144'},
                 'not one of yarn, linear, none'),
                ({'ROPE_SCALING': 'yarn', 'ROPE_SCALE': None, 'YARN_ORIG_CTX': '262144'},
                 'no positive ROPE_SCALE'),
                ({'ROPE_SCALING': 'linear', 'ROPE_SCALE': None, 'YARN_ORIG_CTX': None},
                 'no positive ROPE_SCALE'),
                ({'ROPE_SCALING': 'yarn', 'ROPE_SCALE': 'two', 'YARN_ORIG_CTX': '262144'},
                 'no positive ROPE_SCALE'),
                ({'ROPE_SCALING': 'yarn', 'ROPE_SCALE': '0', 'YARN_ORIG_CTX': '262144'},
                 'no positive ROPE_SCALE'),
                ({'ROPE_SCALING': 'yarn', 'ROPE_SCALE': '2.0.1', 'YARN_ORIG_CTX': '262144'},
                 'no positive ROPE_SCALE'),
                ({'ROPE_SCALING': 'yarn', 'ROPE_SCALE': '2.0', 'YARN_ORIG_CTX': None},
                 'no positive YARN_ORIG_CTX'),
                ({'ROPE_SCALING': 'yarn', 'ROPE_SCALE': '2.0', 'YARN_ORIG_CTX': '524288'},
                 'does not exceed the trained one')):
            with self.subTest(case=str(breakage)):
                env = {key: value for key, value in breakage.items() if value is not None}
                run = self.start(device=None,
                    extra={'VISION': 'no', 'CONTEXT': '524288', **env}, expect_fail=True)
                self.assertNotIn('upstream --serve', run.stdout)
                self.assertIn(message, run.stderr)
                self.assertEqual(self.config.read_text(), before)

    def test_the_ring_and_the_window_arrive_with_the_card_on_the_same_boot(self):
        # A fresh workspace asked for a 512K ring: the setup pass this entry point runs, the window,
        # and the ring all have to land on one boot, because the switch verifies the booted engine
        # against the profile and cannot spend a second drain on the way.
        run = self.start(extra={'CONTEXT': '524288', 'ROPE_SCALING': 'yarn',
            'ROPE_SCALE': '2.0', 'YARN_ORIG_CTX': '262144'})
        cfg = json.loads(self.config.read_text())
        self.assertEqual(cfg['vision']['cuda_device'], 4)
        self.assertEqual(cfg['args'][cfg['args'].index('--max-context') + 1], '524288')
        self.assertEqual(cfg['args'][cfg['args'].index('--rope-scaling') + 1], 'yarn')
        self.assertEqual(len(self.setup_calls()), 1)
        self.assertIn('upstream --serve', run.stdout)

    def test_the_three_hand_edits_are_written_when_a_profile_states_them(self):
        # The reserve, the direction and the log path reached this container by somebody editing a
        # file inside the workspace (docs/deployment.md records the first two as undeclared), which is
        # fine until the day the workspace is reinstalled and the setting is gone with no profile
        # mentioning it ever existed. Stated in the profile, they land in the same run config the
        # engine reads, through the same ceremony as every other fact here.
        vector = self.workspace / 'refusal-direction-r.gguf'
        vector.parent.mkdir(parents=True, exist_ok=True)
        vector.write_text('gguf')
        self.given({'model': 'UD-IQ4_XS', 'vision': {'cuda_device': 4},
            'args': ['--serve', '--model', '/weights/shard-1.gguf', '--max-context', '262144']})
        run = self.start(extra={'VRAM_RESERVE_MIB': '512', 'CONTROL_VECTOR': str(vector),
            'CONTROL_VECTOR_LAYERS': '4:44', 'STRATA_LOG': str(self.workspace / 'engine.log')})
        args = json.loads(self.config.read_text())['args']
        self.assertEqual(args[args.index('--vram-reserve-mib') + 1], '512')
        self.assertEqual(args[args.index('--control-vector-scaled') + 1], '%s:1.0' % vector)
        self.assertEqual(args[args.index('--control-vector-layer-range') + 1:args.index(
            '--control-vector-layer-range') + 3], ['4', '44'])
        self.assertEqual(args[args.index('--cvec-mode') + 1], 'project')
        self.assertEqual(args[args.index('--cvec-dir') + 1], 'per-layer')
        # The log is a key of its own, not an argument: that is how the workspace has carried it.
        self.assertEqual(json.loads(self.config.read_text())['log'], str(self.workspace / 'engine.log'))
        self.assertNotIn('--log', args)
        self.assertIn('upstream --serve', run.stdout)
        # And on the next boot nothing moves, because every one of these writes only on change.
        again = self.start(extra={'VRAM_RESERVE_MIB': '512', 'CONTROL_VECTOR': str(vector),
            'CONTROL_VECTOR_LAYERS': '4:44', 'STRATA_LOG': str(self.workspace / 'engine.log')})
        self.assertIn('already in the args list', again.stdout)
        self.assertIn('engine log already', again.stdout)

    def test_a_stated_nothing_leaves_the_workspace_exactly_as_it_found_it(self):
        # This is the test that makes rebuilding the image safe for the configuration in service: it
        # states none of these three, and its workspace carries all three as hand edits. The new entry
        # point has to leave every one of them alone, or shipping facts 9-11 is itself a change to
        # what is serving -- and it would be a change with no drain and no receipt.
        self.given({'model': 'UD-IQ4_XS', 'vision': {'cuda_device': 4}, 'log': '/data/strata-engine.log',
            'args': ['--serve', '--max-context', '262144', '--vram-reserve-mib', '2048',
                     '--control-vector-scaled', '/data/refusal-direction-r.gguf:1.0',
                     '--control-vector-layer-range', '4', '44', '--cvec-mode', 'project',
                     '--cvec-dir', 'per-layer']})
        before = json.loads(self.config.read_text())
        self.start()
        self.assertEqual(json.loads(self.config.read_text()), before)

    def test_a_reserve_of_zero_is_a_value_and_off_is_the_absence(self):
        self.given({'vision': {'cuda_device': 4}, 'args': ['--serve', '--vram-reserve-mib', '2048']})
        self.start(extra={'VRAM_RESERVE_MIB': '0'})
        args = json.loads(self.config.read_text())['args']
        self.assertEqual(args[args.index('--vram-reserve-mib') + 1], '0')
        self.start(extra={'VRAM_RESERVE_MIB': 'off'})
        self.assertNotIn('--vram-reserve-mib', json.loads(self.config.read_text())['args'])

    def test_a_reserve_the_engine_cannot_parse_does_not_boot(self):
        self.given({'vision': {'cuda_device': 4}, 'args': ['--serve']})
        self.start(extra={'VRAM_RESERVE_MIB': 'two-gig'}, expect_fail=True)

    def test_moving_a_vector_between_layers_leaves_no_second_number_behind(self):
        # The reason the control vector is not written through edit_args. Its layer range is the only
        # two-value flag in this overlay, and a strip that walks one value would leave the high layer
        # sitting in the argv as an orphan, which the engine reads as the next flag's value: a control
        # vector applied to layers 45 and 4 with mode set to "44" is not a crash, it is a quiet lie.
        vector = self.workspace / 'v.gguf'
        vector.parent.mkdir(parents=True, exist_ok=True)
        vector.write_text('gguf')
        self.given({'vision': {'cuda_device': 4}, 'args': ['--serve', '--control-vector-scaled',
            '%s:1.0' % vector, '--control-vector-layer-range', '4', '44', '--cvec-mode', 'project',
            '--cvec-dir', 'per-layer', '--max-context', '262144']})
        self.start(extra={'CONTROL_VECTOR': str(vector), 'CONTROL_VECTOR_LAYERS': '5:45'})
        args = json.loads(self.config.read_text())['args']
        self.assertEqual(args.count('44'), 0)
        self.assertEqual(args[args.index('--control-vector-layer-range') + 1:args.index(
            '--control-vector-layer-range') + 3], ['5', '45'])
        # Drop it, and the whole block goes: no flag, and no orphan.
        self.start(extra={'CONTROL_VECTOR': 'off'})
        args = json.loads(self.config.read_text())['args']
        for flag in ('--control-vector-scaled', '--control-vector-layer-range', '--cvec-mode',
                     '--cvec-dir'):
            self.assertNotIn(flag, args)
        self.assertEqual(args, ['--serve', '--max-context', '262144'])

    def test_a_vector_that_is_not_a_file_and_layers_that_are_not_a_range_refuse_the_boot(self):
        self.given({'vision': {'cuda_device': 4}, 'args': ['--serve']})
        missing = self.workspace / 'nowhere.gguf'
        self.start(extra={'CONTROL_VECTOR': str(missing), 'CONTROL_VECTOR_LAYERS': '4:44'},
            expect_fail=True)
        self.start(extra={'CONTROL_VECTOR': str(missing), 'CONTROL_VECTOR_LAYERS': ''}, expect_fail=True)
        missing.parent.mkdir(parents=True, exist_ok=True)
        missing.write_text('gguf')
        # Which layers is stated, not assumed: the range is where the fitted direction lives.
        self.start(extra={'CONTROL_VECTOR': str(missing)}, expect_fail=True)
        for layers in ('44:4', 'a:b', '4:4:4', ':4', '4:'):
            with self.subTest(layers=layers):
                self.start(extra={'CONTROL_VECTOR': str(missing), 'CONTROL_VECTOR_LAYERS': layers},
                    expect_fail=True)
        self.start(extra={'CONTROL_VECTOR': str(missing), 'CONTROL_VECTOR_LAYERS': '4:44'})
        self.assertIn('--cvec-mode', json.loads(self.config.read_text())['args'])

    def test_the_log_path_is_an_absolute_path_in_a_directory_that_exists(self):
        self.given({'vision': {'cuda_device': 4}, 'args': ['--serve']})
        self.start(extra={'STRATA_LOG': 'engine.log'}, expect_fail=True)
        self.start(extra={'STRATA_LOG': str(self.workspace / 'missing' / 'engine.log')},
            expect_fail=True)
        # The path this house serves with: an existing directory, a file that is not there yet.
        self.start(extra={'STRATA_LOG': str(self.workspace / 'strata-engine.log')})
        self.assertEqual(json.loads(self.config.read_text())['log'],
            str(self.workspace / 'strata-engine.log'))
        self.start(extra={'STRATA_LOG': 'off'})
        self.assertNotIn('log', json.loads(self.config.read_text()))

    def test_the_image_installs_this_entry_point_over_the_upstream_one(self):
        dockerfile = (fixture.ROOT / 'runtime/overlay/Dockerfile').read_text()
        self.assertIn('mv /opt/strata/docker-entrypoint.sh /opt/strata/docker-entrypoint.upstream.sh',
            dockerfile)
        self.assertIn('COPY strata-vision-entrypoint.sh /opt/strata/docker-entrypoint.sh', dockerfile)
        # The overlay is only ever an addition to a pinned upstream image; it never names a base.
        self.assertIn('FROM ${UPSTREAM}', dockerfile)
        self.assertNotIn('FROM local/', dockerfile)


if __name__ == '__main__':
    main()

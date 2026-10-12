"""A self-provisioning engine has to bring a measured window to a switch, not a guessed one.

The Strata trial's first attempt spent its whole readiness window building the state it had never
built before and was abandoned with the router drained. These tests hold the line that grew out of
that: the build and the warm boot are measured on this host, the throwaway run asks for the same
machine the switch would ask for, and the switch refuses to enter a window nobody has timed.
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
from runtime.provision import measurement, throwaway, workspace_target
from runtime.system import atomic_json

PROFILE = 'flash-next-strata-160k'


class ProvisioningSystem(fixture.FakeSystem):
    """The engine as observed from outside: a container that is started, and eventually answers.

    The workspace and the health answer are the only two things the tool can watch, so the fake
    decides when each of them happens, which lets a measurement be tested without an engine.
    """

    def __init__(self, bundle, healthy_after=3):
        super().__init__(bundle)
        self.healthy_after = healthy_after
        self.health_polls = 0
        self.started = []
        self.throwaway_port = None
        self.throwaway_name = None
        self.devices = None

    def docker(self, *args, **kwargs):
        # The measurement starts and stops its own compose project; the fake plays the part of
        # compose reading the file it was handed, including the card reservations in it.
        if args[0] == 'compose' and 'up' in args:
            self.events.append(('docker', args))
            spec = read(args[args.index('-f') + 1])
            role = args[-1]
            service = spec['services'][role]
            self.throwaway_port = service['ports'][0].split(':')[1]
            self.throwaway_name = service['container_name']
            self.devices = service['deploy']['resources']['reservations']['devices'][0]['device_ids']
            self.started.append((service['container_name'], self.throwaway_port, list(self.devices)))
            self.containers[service['container_name']] = {'Id': 'throwaway',
                'Name': service['container_name'], 'State': {'Running': True}, 'Config': {'Cmd': []}}
            return ''
        if args[0] == 'compose' and 'down' in args:
            self.events.append(('docker', args))
            spec = read(args[args.index('-f') + 1])
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
            if self.health_polls < self.healthy_after:
                raise RuntimeError('connection refused')
            return {'status': 'ok'}
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
        self.polls = patch('runtime.provision.time.sleep', side_effect=self.build_workspace)
        self.polls.start(); self.addCleanup(self.polls.stop)

    def build_workspace(self, _seconds):
        """The engine packs the workspace a couple of polls in, then is still loading afterwards."""
        self.poll_number = getattr(self, 'poll_number', 0) + 1
        if self.poll_number >= 2:
            ready = self.workspace / 'config'
            ready.mkdir(parents=True, exist_ok=True)
            (ready / 'strata-ud-iq4_xs.json').write_text('{}')

    def receipt(self, **values):
        engine = self.c.provisioning(self.bundle)[1]
        recorded = {'profile': PROFILE, 'engine_tag': engine['tag'], 'image_id': engine['image_id'],
            'workspace': str(self.workspace), 'measured_at': '2026-10-12T05:00:00+00:00', 'runs': []}
        recorded.update(values)
        atomic_json(self.state / 'provisioning' / (PROFILE + '.json'), recorded)
        return recorded

    def test_a_trial_nobody_has_measured_is_not_switched_to(self):
        """The refusal has to come before anything is drained or recreated, and name the way out."""
        with self.assertRaisesRegex(RuntimeError, 'never been pre-provisioned'):
            self.c.transition(PROFILE)
        self.assertEqual(self.system.events, [])
        self.assertFalse(self.system.draining)

    def test_a_measurement_only_counts_for_the_machine_it_was_taken_on(self):
        engine = self.c.provisioning(self.bundle)[1]
        self.receipt(healthy_warm_boot=True, warm_boot_seconds=100)
        self.assertIsNotNone(self.c.provisioned(self.bundle))
        self.receipt(healthy_warm_boot=True, warm_boot_seconds=100, image_id='sha256:' + '0' * 64)
        self.assertIsNone(self.c.provisioned(self.bundle))
        self.receipt(healthy_warm_boot=True, warm_boot_seconds=100, workspace=str(self.state / 'elsewhere'))
        self.assertIsNone(self.c.provisioned(self.bundle))
        with self.assertRaisesRegex(RuntimeError, 'never been pre-provisioned'):
            self.c.switchable(self.bundle)

    def test_a_workspace_that_never_came_up_healthy_ends_the_trial(self):
        """An engine that cannot boot from its own provisioned state is not switchable, full stop."""
        self.receipt(healthy_warm_boot=False, warm_boot_seconds=2900,
            error='The container never answered its health check from the provisioned workspace')
        with self.assertRaisesRegex(RuntimeError, 'not switchable'):
            self.c.switchable(self.bundle)

    def test_the_window_has_to_be_three_times_the_boot_it_is_guaranteeing(self):
        self.receipt(healthy_warm_boot=True, warm_boot_seconds=1500)
        with self.assertRaisesRegex(RuntimeError, 'under three times'):
            self.c.switchable(self.bundle)
        self.receipt(healthy_warm_boot=True, warm_boot_seconds=1000)
        self.assertEqual(self.c.switchable(self.bundle), None)

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
        for key in ('image', 'entrypoint', 'command', 'environment', 'user', 'volumes', 'ulimits',
                'tmpfs', 'healthcheck', 'read_only', 'init', 'pull_policy', 'logging'):
            self.assertEqual(service[key], cfg[key], key)
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

    def test_the_refusal_is_readable_without_attempting_a_switch(self):
        """A refusal that can only be understood by provoking one is a refusal nobody can plan around."""
        reasons = self.c.verdict(self.bundle)
        self.assertEqual(len(reasons), 1)
        self.assertIn('never been pre-provisioned', reasons[0])
        self.assertIn('--profile ' + PROFILE, reasons[0])
        self.receipt(healthy_warm_boot=False, warm_boot_seconds=2900, error='never answered')
        self.assertIn('not switchable', self.c.verdict(self.bundle)[0])
        self.receipt(healthy_warm_boot=True, warm_boot_seconds=1500)
        refusal = self.c.verdict(self.bundle)[0]
        self.assertIn('at least 4501s', refusal)
        with self.assertRaisesRegex(RuntimeError, 'under three times the measured warm boot'):
            self.c.switchable(self.bundle)
        self.receipt(healthy_warm_boot=True, warm_boot_seconds=1000)
        self.assertEqual(self.c.verdict(self.bundle), [])
        self.c.switchable(self.bundle)

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


if __name__ == '__main__':
    main()

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
from runtime.config import render
from runtime.controller import Controller
from runtime.provision import measurement, run_arguments, workspace_target
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

    def docker(self, *args, **kwargs):
        if args[0] == 'run':
            self.events.append(('docker', args))
            self.started.append(args)
            published = args[args.index('-p') + 1]
            self.throwaway_port = published.split(':')[1]
            self.containers[args[args.index('--name') + 1]] = {'Id': 'throwaway', 'Name': args[args.index('--name') + 1],
                'State': {'Running': True}, 'Config': {'Cmd': []}}
            return 'throwaway'
        if args[0] == 'rm' and args[-1] not in self.containers:
            # A throwaway that never made it is not an error worth raising about; the real thing
            # would have nothing to remove and `check=False` says so.
            self.events.append(('docker', args))
            return ''
        return super().docker(*args, **kwargs)

    def inspect(self, name):
        state = super().inspect(name)
        if state is not None:
            state['State'].setdefault('Health', {'Status': 'healthy'})
        return state

    def http(self, url, body=None, headers=None, timeout=None):
        # Only the throwaway's own answer is simulated; a measurement must not be able to make the
        # serving backend look healthy or unhealthy.
        if self.throwaway_port and f':{self.throwaway_port}/health' in url:
            self.health_polls += 1
            if self.health_polls < self.healthy_after:
                raise RuntimeError('connection refused')
            return {'status': 'ok'}
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

    def test_the_throwaway_run_asks_for_the_same_machine_the_switch_would(self):
        """Anything less and the measurement is about a different container than the one to come."""
        cfg = self.bundle['compose']['services']['coding']
        args = run_arguments(cfg, cfg['container_name'] + '-provision', 18081)
        flags = ' '.join(args)
        devices = cfg['deploy']['resources']['reservations']['devices'][0]['device_ids']
        self.assertIn('--gpus device=' + ','.join(devices), flags)
        self.assertIn(cfg['image'], args)
        self.assertEqual(args[args.index('--entrypoint') + 1], cfg['entrypoint'][0])
        self.assertEqual(args[args.index(cfg['image']):], [cfg['image'], *cfg['command']])
        self.assertEqual(args[args.index(cfg['image']) + 1:], list(cfg['command']))
        for key, value in cfg['environment'].items():
            self.assertIn(f'-e {key}={value}', flags)
        workspace = str(self.workspace)
        self.assertIn(f'-v {workspace}:/data ', flags + ' ')
        self.assertNotIn('-v ' + workspace + ':/data:ro', flags)
        self.assertIn('--network bridge', flags)
        self.assertNotIn('router', args)
        self.assertNotEqual(cfg['container_name'], args[args.index('--name') + 1])

    def test_the_build_is_measured_while_the_serving_backend_is_watched(self):
        """Packing competes with a loaded model; this run is also the answer to whether that matters."""
        result = measurement(self.c, PROFILE, limit=60, poll=0, log=lambda line: None)
        self.assertTrue(result['workspace_ready'])
        self.assertGreaterEqual(result['provision_seconds'], 0)
        self.assertTrue(result['neighbour_healthy'])
        self.assertTrue(self.system.started)
        self.assertTrue(all('qwen38-daytime-provision' == args[args.index('--name') + 1]
            for args in self.system.started))
        # Nothing about the serving pair was touched, and it is still serving.
        self.assertTrue(self.c.serving())
        self.assertFalse(self.system.draining)
        self.assertNotIn('qwen38-daytime-provision', self.system.containers)

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
        self.c.switchable(self.bundle)
        recorded = json.loads((self.state / 'provisioning' / (PROFILE + '.json')).read_text())
        self.assertEqual([run['phase'] for run in recorded['runs']], ['warm'])
        self.assertEqual(recorded['warm_boot_seconds'], 240)
        self.assertEqual(recorded['profile'], PROFILE)


if __name__ == '__main__':
    main()

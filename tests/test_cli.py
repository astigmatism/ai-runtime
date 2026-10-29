"""The command line's dispatch table: every accepted action reaches the operation it names.

`argparse` accepts the action and the code that builds the transition already knows how to derive
its flags from the action, so an action missing from the branch that makes the call fails with
neither a parse error nor a refused transition -- it falls through every branch, prints the result it
never assigned, and dies on the way. `install` lived like that for a whole machine's bring-up: the
runbook command was accepted, ran for thirty seconds, and reported `cannot access local variable
'result'`. These tests drive `main()` for each mutating action so the dispatch table cannot hold a
name it does not route again.
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import machine
from runtime import __main__ as cli
from runtime.config import read
from runtime.system import atomic_json


class DispatchTests(unittest.TestCase):
    def flags(self, *argv):
        seen = {}

        class FakeController:
            def __init__(self, system=None):
                seen['readonly'] = not getattr(system, 'readonly', True)

            def transition(self, profile, deploy=False, adopt=False, install=False):
                seen['flags'] = {'deploy': deploy, 'adopt': adopt, 'install': install}
                return {'ok': True}

        with mock.patch.object(cli, 'Controller', FakeController), \
                mock.patch.object(cli, 'System', lambda **kwargs: type('S', (), {'readonly': False})()), \
                mock.patch.object(sys, 'argv', ['ai-runtime', *argv]):
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main()
        return seen['flags']

    def test_install_is_dispatched_as_an_install(self):
        self.assertEqual(self.flags('install'), {'deploy': False, 'adopt': False, 'install': True})

    def test_the_siblings_of_install_keep_their_own_flags(self):
        # The four share one branch and differ only by the flag each names; a branch that gained one
        # and forgot another would otherwise pass the install case and send the wrong verb.
        self.assertEqual(self.flags('apply'), {'deploy': False, 'adopt': False, 'install': False})
        self.assertEqual(self.flags('deploy'), {'deploy': True, 'adopt': False, 'install': False})
        self.assertEqual(self.flags('adopt'), {'deploy': False, 'adopt': True, 'install': False})


if __name__ == '__main__':
    unittest.main()


class PowerCliTests(unittest.TestCase):
    """`power` from a terminal: the same rungs the page offers, and the same refusal for the same cause.

    The dispatch is the risk here, not the write: `power --gpu 3090 --rung sensible` has to arrive at
    the runtime as one validated request, and a flag that is quietly ignored is a card left at whatever
    it was. So these assert what the runtime was asked for, and no fake driver is needed to say it.
    """
    CARDS = [('GPU-358353dd-5933-1dc6-ac0d-e1189b063e4c', 'RTX 3090'),
        ('GPU-91239bb4-0f5b-b148-6d03-c098a5806e13', 'RTX 4080 SUPER'),
        ('GPU-e4252fcb-6c59-6892-c5bb-bf8f75fbe523', 'RTX 3080 Ti')]

    def run_cli(self, *argv):
        asked = []

        class FakePower:
            def cards(self):
                return PowerCliTests.CARDS

            def write(self, body):
                asked.append(body)
                return {'ok': True, 'to': body.get('rung') or body['action']}

            def card(self, wanted):
                """The runtime's own rule: a full UUID, or enough of a name to be exactly one card."""
                cards = PowerCliTests.CARDS
                if wanted and wanted.lower().startswith('gpu-'):
                    return next((uuid for uuid, _ in cards if uuid.lower() == wanted.lower()), wanted)
                matches = [(uuid, name) for uuid, name in cards if wanted.lower() in name.lower()]
                if len(matches) == 1:
                    return matches[0][0]
                raise RuntimeError('no card this machine reports is named %r; give a full UUID or '
                    'enough of a name to be exactly one card' % wanted)

            def write_sets(self, body):
                asked.append(body)
                return {'ok': True, 'set': body['name'], 'action': body['action'], 'applied': 3,
                    'cards': 3, 'skipped': 1, 'left_out': [{'gpu': PowerCliTests.CARDS[2][0],
                        'name': 'RTX 3080 Ti'}], 'note': None}

            def sets_block(self, registry=None):
                return {'available': True, 'error': None, 'limit': 24, 'count': 2, 'sets': [
                    {'name': 'stock', 'builtin': True, 'saved_at': None, 'ready': 3, 'skipped': 0,
                        'cards': [{'gpu': uuid, 'name': name, 'kind': 'stock', 'rung': None,
                            'state': 'ready', 'reason': None, 'moved': False}
                            for uuid, name in PowerCliTests.CARDS], 'left_out': []},
                    {'name': 'quiet-night', 'builtin': False, 'saved_at': 'yesterday', 'ready': 2,
                        'skipped': 1, 'cards': [
                            {'gpu': PowerCliTests.CARDS[0][0], 'name': 'RTX 3090', 'kind': 'rung',
                                'rung': 'sensible', 'state': 'ready', 'reason': None, 'moved': False},
                            {'gpu': PowerCliTests.CARDS[2][0], 'name': 'RTX 3080 Ti', 'kind': 'rung',
                                'rung': 'assertive', 'state': 'absent',
                                'reason': 'left the machine', 'moved': False}],
                        'left_out': [{'gpu': PowerCliTests.CARDS[1][0], 'name': 'RTX 4080 SUPER'}]}]}

            def status(self):
                return {'cards': {uuid: {'name': name, 'state': 'stock', 'rungs': []}
                    for uuid, name in PowerCliTests.CARDS}}

            def registry(self):
                return {'envelope': {'offset_step': 15, 'offset_max': 150, 'clock_min': 210},
                    'plans': {'rtx3090': {'name': 'RTX 3090'}}}

            def registry_error(self):
                return None

            def host_plans(self):
                return {PowerCliTests.CARDS[0][0]: 'rtx3090'}

            def plans_for_cards(self):
                """What each card resolves to: the settings table, plus the model match where not."""
                first = PowerCliTests.CARDS[0][0]
                return {uuid: {'name': name, 'reported': 'NVIDIA GeForce ' + name,
                        'plan': 'rtx3090' if uuid == first else 'rtx3080-ti' if 'Ti' in name else None,
                        'source': 'host' if uuid == first else 'model' if 'Ti' in name else None}
                    for uuid, name in PowerCliTests.CARDS}

        controller = type('C', (), {'power': FakePower()})()
        out = io.StringIO()
        with mock.patch.object(cli, 'Controller', lambda **kwargs: controller), \
                mock.patch.object(cli, 'System', lambda **kwargs: type('S', (), {'readonly': False})()), \
                mock.patch.object(sys, 'argv', ['ai-runtime', *argv]):
            with contextlib.redirect_stdout(out):
                cli.main()
        return asked, out.getvalue()

    def test_every_way_of_asking_arrives_as_one_request(self):
        card = self.CARDS[0][0]
        asked, _ = self.run_cli('power', '--gpu', card, '--rung', 'sensible')
        self.assertEqual(asked[0], {'gpu': card, 'action': 'apply', 'rung': 'sensible',
            'override': False})
        asked, _ = self.run_cli('power', '--gpu', card, '--revert')
        self.assertEqual(asked[0]['action'], 'revert')
        asked, _ = self.run_cli('power', '--gpu', card, '--set', '300,1800,+75')
        self.assertEqual(asked[0]['action'], 'custom')
        self.assertEqual((asked[0]['power'], asked[0]['clock_max'], asked[0]['offset']),
            (300, 1800, 75))
        asked, _ = self.run_cli('power', '--gpu', card, '--rung', 'assertive', '--override')
        self.assertTrue(asked[0]['override'])
        # A card named without a rung is a request to read it, which is what an operator means by
        # typing the name on its own.
        asked, _ = self.run_cli('power', '--gpu', card)
        self.assertEqual(asked[0]['action'], 'read')
        _, printed = self.run_cli('power')
        self.assertIn('RTX 3090', printed)
        self.assertIn('stock', printed)
        _, printed = self.run_cli('power', '--plans')
        self.assertIn('rtx3090', printed)

    def test_a_card_is_named_by_enough_of_a_name_to_be_exactly_one_card(self):
        asked, _ = self.run_cli('power', '--gpu', '3080 ti', '--revert')
        self.assertEqual(asked[0]['gpu'], self.CARDS[2][0])
        asked, _ = self.run_cli('power', '--gpu', 'gpu-358353dd-5933-1dc6-ac0d-e1189b063e4c', '--revert')
        self.assertEqual(asked[0]['gpu'], self.CARDS[0][0])
        for spelling in ('RTX 30', 'nothing like a card here'):
            with self.subTest(spelling=spelling), self.assertRaises(RuntimeError) as refused:
                self.run_cli('power', '--gpu', spelling, '--revert')
            self.assertIn('full UUID', str(refused.exception))

    def test_a_power_set_is_saved_used_and_forgotten_from_the_same_words_as_the_page(self):
        """The three set verbs, arriving as three requests, and a card named the way a person types one.

        `--omit` is the interesting one: it is how an operator saves a machine that has one card sitting
        on hand-written values, and it is refused on an apply, where leaving a card out would mean the
        set was not applied.
        """
        asked, _ = self.run_cli('power', '--save-set', 'quiet-night')
        self.assertEqual(asked[0], {'action': 'save', 'name': 'quiet-night', 'omit': [],
            'override': False})
        asked, _ = self.run_cli('power', '--save-set', 'quiet-night', '--omit', 'RTX 3080 Ti',
            '--omit', self.CARDS[0][0])
        self.assertEqual(asked[0]['omit'], ['RTX 3080 Ti', self.CARDS[0][0]])
        asked, _ = self.run_cli('power', '--use-set', 'quiet-night', '--override')
        self.assertEqual((asked[0]['action'], asked[0]['override']), ('apply', True))
        asked, _ = self.run_cli('power', '--use-set', 'stock')
        self.assertEqual(asked[0]['name'], 'stock')
        asked, _ = self.run_cli('power', '--delete-set', 'quiet-night')
        self.assertEqual(asked[0]['action'], 'delete')
        _, printed = self.run_cli('power', '--save-set', 'quiet-night')
        self.assertIn('quiet-night', printed)

    def test_omitting_a_card_is_a_thing_a_save_offers_and_an_apply_does_not(self):
        for flag in ('--use-set', '--delete-set'):
            with self.subTest(flag=flag), self.assertRaises(RuntimeError) as refused:
                self.run_cli('power', flag, 'quiet-night', '--omit', 'RTX 3080 Ti')
            self.assertIn('save', str(refused.exception))

    def test_listing_the_sets_prints_the_gaps_the_page_prints(self):
        _, printed = self.run_cli('power', '--sets')
        self.assertIn('stock', printed)
        self.assertIn('quiet-night', printed)
        self.assertIn('absent', printed)
        self.assertIn('RTX 4080 SUPER', printed, 'a set does not say which cards it leaves alone')

    def test_a_set_needs_all_three_numbers_or_it_refuses(self):
        with self.assertRaises(RuntimeError) as refused:
            self.run_cli('power', '--gpu', self.CARDS[0][0], '--set', '300,1800')
        self.assertIn('POWER,CLOCK_MAX,OFFSET', str(refused.exception))


class MachineConfigurationActionsTests(unittest.TestCase):
    """`registry` and `retire-profile`: the two actions answered out of the machine's own directory.

    These are the actions the dispatch table can get wrong in the expensive direction. An action that
    falls through to the controller asks a backend what a record says, and a `retire-profile` that reached
    a transition would drain a serving workload to take a definition out of a directory -- the one thing a
    retirement exists to not do. So the controller and the system here are not fakes that answer, they are
    fakes that object the moment they are built, and the cases that pass are the ones that never built one.
    The deployment is written to a temporary directory for the same reason the retirements themselves are
    tested there: the answer these actions give is a file.
    """
    YARN = 'flash-next-strata-solo-thinking-recovery-cache-yarn'
    YARN_REPLACEMENT = 'flash-next-strata-solo-thinking-recovery-cache'
    RENAMED = 'qwen27b-q6k-with-nighttime'
    RENAMED_ONTO = 'qwen27b-q6k-pair'

    def deployment(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        state = root / '.state'
        state.mkdir()
        shutil.copytree(machine.CONFIG, root / 'config')
        home = machine.home(state / 'profiles')
        atomic_json(state / 'host.json', machine.stating(machine.HOST, home, host_id='rosalina'))
        return root / 'config', state, home

    def run_cli(self, argv, config, state, state_dir=None):
        built = []

        def objects(name):
            def factory(*args, **kwargs):
                built.append(name)
                raise AssertionError(name + ' was built to answer ' + argv[0])
            return factory

        out = io.StringIO()
        with mock.patch.object(cli, 'Controller', objects('Controller')), \
                mock.patch.object(cli, 'System', objects('System')), mock.patch.dict(os.environ, {
                    'RUNTIME_CONFIG_DIR': str(config),
                    'RUNTIME_STATE_DIR': str(state) if state_dir is None else state_dir}), \
                mock.patch.object(sys, 'argv', ['ai-runtime', *argv]):
            with contextlib.redirect_stdout(out):
                cli.main()
        self.assertEqual(built, [], argv[0] + ' answered the machine\'s own directory and then built a '
            'backend to ask it the same question')
        return json.loads(out.getvalue())

    def test_registry_is_the_machine_answering_for_its_own_configurations(self):
        config, state, home = self.deployment()
        reported = self.run_cli(['registry'], config, state)
        self.assertEqual(reported['home'], str(home), 'the audit names the directory this machine states, '
            'not the one the source ships')
        self.assertEqual(reported['record'], str(home / 'registry.json'))
        self.assertEqual([entry['profile'] for entry in reported['offers']],
            [name for name in machine.record()['selectable']
                if 'rosalina' in read(machine.HOME / (name + '.json'))['hosts']])
        self.assertIsNone(reported['running'], 'a machine that has never switched has nothing recorded as '
            'active, and the JSON an operator reads says nothing rather than naming its default')

    def test_retire_profile_takes_a_configuration_out_of_service_without_asking_a_backend(self):
        config, state, home = self.deployment()
        result = self.run_cli(['retire-profile', '--profile', self.YARN, '--replacement',
            self.YARN_REPLACEMENT], config, state)
        self.assertEqual(result['act'], {'retired': self.YARN, 'replacement': self.YARN_REPLACEMENT,
            'carried_retirements': [], 'default_followed_replacement': False})
        self.assertEqual(result['drain'], 'none')
        self.assertFalse((home / (self.YARN + '.json')).exists())
        self.assertEqual(read(home / 'registry.json')['retired'][self.YARN], self.YARN_REPLACEMENT)

    def test_the_replacement_is_resolved_the_way_a_name_typed_at_a_keyboard_is(self):
        # An operator types the name they have always typed. The retirement writes what the machine's own
        # record resolves it to, so the row left behind points at a configuration that exists -- which is
        # the rule that makes a refusal survivable -- and not at the spelling they happened to use.
        config, state, home = self.deployment()
        result = self.run_cli(['retire-profile', '--profile', self.YARN, '--replacement', self.RENAMED],
            config, state)
        self.assertEqual(result['act']['replacement'], self.RENAMED_ONTO)
        self.assertEqual(read(home / 'registry.json')['retired'][self.YARN], self.RENAMED_ONTO)

    def test_a_retirement_that_names_only_one_of_the_two_configurations_refuses(self):
        config, state, home = self.deployment()
        with self.assertRaisesRegex(RuntimeError, 'names the configuration it takes out of service with '
                '--profile.*--replacement'):
            self.run_cli(['retire-profile', '--profile', self.YARN], config, state)
        self.assertTrue((home / (self.YARN + '.json')).exists())

    def test_both_actions_asked_of_no_machine_refuse(self):
        config, state, home = self.deployment()
        for argv in (['registry'], ['retire-profile', '--profile', self.YARN, '--replacement',
            self.YARN_REPLACEMENT]):
            with self.subTest(action=argv[0]), self.assertRaisesRegex(RuntimeError,
                    'no machine is being asked here'):
                self.run_cli(argv, config, state, state_dir='')
        self.assertTrue((home / (self.YARN + '.json')).exists(), 'a machine that was not named was not '
            'written to')


if __name__ == '__main__':
    unittest.main()

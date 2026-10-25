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
import sys
import unittest
from unittest import mock

from runtime import __main__ as cli


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

    def test_a_set_needs_all_three_numbers_or_it_refuses(self):
        with self.assertRaises(RuntimeError) as refused:
            self.run_cli('power', '--gpu', self.CARDS[0][0], '--set', '300,1800')
        self.assertIn('POWER,CLOCK_MAX,OFFSET', str(refused.exception))

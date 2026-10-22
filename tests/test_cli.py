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

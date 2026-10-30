import json
from pathlib import Path
import tempfile
import unittest

import machine
from runtime.config import read, render


class ReasoningRetryContract(unittest.TestCase):
    profile = 'flash-next-strata-solo-thinking-recovery'
    engine = 'strata-vision-870aedd9e1eb'

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config, self.home, self.host = machine.tree(Path(self.temporary.name))
        self.path = self.home / (self.profile + '.json')
        self.definition = read(self.path)
        self.definition['engine'] = self.engine
        self.definition['arguments']['--reasoning-close-retry'] = 'on'
        self.definition['argument_order'].append('--reasoning-close-retry')

    def launch(self):
        self.path.write_text(json.dumps(self.definition))
        return render(self.config, self.host, self.profile)['compose']['services']

    def test_a_verified_release_renders_the_opt_in(self):
        service = next(iter(self.launch().values()))
        self.assertEqual(service['environment']['REASONING_CLOSE_RETRY'], 'on')
        self.assertEqual(service['image'], read(self.config / 'shared.json')['engines'][self.engine]['tag'])

    def test_an_unqualified_image_cannot_claim_recovery(self):
        path = self.config / 'shared.json'
        shared = read(path)
        shared['engines'][self.engine]['reasoning_close_retry']['image_id'] = 'sha256:other'
        path.write_text(json.dumps(shared))
        with self.assertRaisesRegex(Exception, 'matching.*receipt'):
            self.launch()

    def test_invalid_values_are_rejected_before_rendering(self):
        self.definition['arguments']['--reasoning-close-retry'] = 'true'
        with self.assertRaisesRegex(Exception, 'must be on or off'):
            self.launch()

    def test_container_environment_cannot_bypass_the_contract(self):
        self.definition['container'].setdefault('environment', {})['REASONING_CLOSE_RETRY'] = 'on'
        with self.assertRaisesRegex(Exception, 'state --reasoning-close-retry in arguments'):
            self.launch()

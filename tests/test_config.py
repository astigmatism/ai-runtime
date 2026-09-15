import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from runtime.config import digest, read, render

ROOT = Path(__file__).resolve().parents[1]
BASELINE = read(ROOT / 'tests/fixtures/legacy-fingerprints.json')


class ConfigurationTests(unittest.TestCase):
    def test_import_matches_independently_recorded_daytime_profile(self):
        for name, expected in [('daytime', BASELINE['profiles']['daytime'])]:
            with self.subTest(profile=name):
                result = render(ROOT / 'config', BASELINE['host'], name)
                self.assertEqual(digest(result['compose']), expected['compose_sha256'])
                catalog = result['catalog']
                for row in [catalog, *catalog['models']]:
                    row.pop('updated_at'); row.pop('source')
                self.assertEqual(digest(catalog), expected['catalog_sha256'])

    def test_context_edit_updates_both_launch_flags_manifest_and_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            path = config / 'profiles/daytime.json'
            definition = read(path); definition['context_tokens'] = 98304
            path.write_text(json.dumps(definition))
            result = render(config, BASELINE['host'], 'daytime')
            argv = result['compose']['services']['coding']['command']
            for flag in ['--ctx-size', '--kv-unified-per-slot']:
                self.assertEqual(argv[argv.index(flag) + 1], '98304')
            self.assertEqual(result['catalog']['context_length'], 98304)
            self.assertEqual(result['catalog']['models'][0]['total_context_length'], 98304)
            self.assertEqual(result['manifest']['services'][0]['context_tokens'], 98304)
            self.assertEqual(result['catalog']['models'][1]['context_length'], 131072)
            self.assertEqual(result['catalog']['display_name'], 'Daytime (96K)')

    def test_gpu_groups_cannot_overlap(self):
        host = copy.deepcopy(BASELINE['host']); host['gpu_ids']['nighttime'][0] = host['gpu_ids']['daytime'][0]
        with self.assertRaisesRegex(RuntimeError, 'four distinct'):
            render(ROOT / 'config', host, 'daytime')

    def test_unknown_profile_and_output_cap_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'retired'):
            render(ROOT / 'config', BASELINE['host'], 'daytime-swift')
        with self.assertRaisesRegex(RuntimeError, 'Unknown'):
            render(ROOT / 'config', BASELINE['host'], '../../secret')
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            path = config / 'shared.json'; value = read(path); value['arguments']['--n-predict'] = '100'
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, 'unsupported policy'):
                render(config, BASELINE['host'], 'daytime')

    def test_no_qualification_is_synthesized(self):
        result = render(ROOT / 'config', BASELINE['host'], 'daytime')
        self.assertNotIn('qualified', json.dumps(result))
        self.assertEqual(len(result['artifacts']), 5)
        self.assertEqual(result['catalog']['aliases'], ['local-active', 'daytime'])

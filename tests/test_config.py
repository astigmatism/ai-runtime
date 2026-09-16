import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from runtime.config import digest, read, render, service_engine

ROOT = Path(__file__).resolve().parents[1]
BASELINE = read(ROOT / 'tests/fixtures/legacy-fingerprints.json')


class ConfigurationTests(unittest.TestCase):
    def test_import_matches_independently_recorded_profiles(self):
        for name, expected in BASELINE['profiles'].items():
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
        def check_keys(value):
            if isinstance(value, dict):
                self.assertTrue({'qualified', 'qualification', 'qualification_receipt'}.isdisjoint(value))
                for child in value.values(): check_keys(child)
            elif isinstance(value, list):
                for child in value: check_keys(child)
        check_keys(result)
        # The imported capability name is historical metadata, not a generated receipt.
        self.assertEqual(result['catalog']['capability_profile'],
            read(ROOT / 'config/profiles/daytime.json')['catalog']['capability_profile'])
        self.assertEqual(len(result['artifacts']), 37)
        self.assertEqual(result['catalog']['aliases'], ['local-active', 'daytime'])

    def test_flash_next_keeps_shards_tensor_placement_and_independent_engines(self):
        result = render(ROOT / 'config', BASELINE['host'], 'daytime')
        argv = result['compose']['services']['coding']['command']
        overrides = [argv[i + 1] for i, x in enumerate(argv) if x == '--override-tensor']
        self.assertEqual(len(overrides), 33)
        self.assertEqual(overrides[0], r'blk\.15\.ffn_down.*=CUDA0')
        self.assertEqual(overrides[-1], r'blk\.48\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU')
        shards = [a for a in result['artifacts'] if '-of-00033.gguf' in a['target']]
        self.assertEqual(len(shards), 33)
        self.assertNotEqual(service_engine(result, 'coding')['image_id'], service_engine(result, 'everyday')['image_id'])
        for role, model in zip(('coding', 'everyday'), result['catalog']['models']):
            self.assertEqual(model['backend_revision'], service_engine(result, role)['revision'])
        self.assertEqual(result['catalog']['mtp']['device'], 'CUDA0')
        self.assertEqual(result['catalog']['mtp']['max_draft_tokens'], 2)

    def test_recovery_profile_preserves_nighttime_and_original_identity(self):
        current = render(ROOT / 'config', BASELINE['host'], 'daytime')
        recovery = render(ROOT / 'config', BASELINE['host'], 'daytime-27b')
        self.assertEqual(current['compose']['services']['everyday'], recovery['compose']['services']['everyday'])
        self.assertEqual(recovery['catalog']['model'], 'qwen3.8-27b-q8_0')
        self.assertEqual(recovery['catalog']['context_length'], 163840)
        self.assertEqual(recovery['catalog']['mtp']['max_draft_tokens'], 3)
        self.assertEqual(recovery['catalog']['mtp']['device'], 'CUDA1')
        self.assertNotIn('qwen3.8-27b-q8_0', current['catalog']['aliases'])

    def test_invalid_repeated_arguments_or_missing_model_mount_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            path = config / 'profiles/daytime.json'; original = read(path)
            for key, value, message in [('--temp', ['1', '2'], 'Only --override-tensor'),
                    ('--override-tensor', [], 'nonempty'), ('--model', '/weights/missing.gguf', 'declared artifact')]:
                with self.subTest(key=key):
                    definition = copy.deepcopy(original); definition['arguments'][key] = value
                    path.write_text(json.dumps(definition))
                    with self.assertRaisesRegex(RuntimeError, message): render(config, BASELINE['host'], 'daytime')
            definition = copy.deepcopy(original)
            definition['artifacts'].pop(20)
            path.write_text(json.dumps(definition))
            with self.assertRaisesRegex(RuntimeError, 'missing a declared shard'):
                render(config, BASELINE['host'], 'daytime')

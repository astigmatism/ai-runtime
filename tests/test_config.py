import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from runtime.config import (DAYTIME_PROFILES, NIGHTTIME_PROFILE, available_profiles, digest, read,
    render, service_engine)

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
        for flag, value in [('--cache-type-k', 'q8_0'), ('--cache-type-v', 'q8_0'),
                ('--ubatch-size', '2048')]:
            self.assertEqual(argv[argv.index(flag) + 1], value)
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

    def test_flash_f16_candidate_changes_only_main_cache_and_microbatch_launch_settings(self):
        baseline = read(ROOT / 'config/profiles/daytime.json')
        candidate = read(ROOT / 'config/profiles/daytime-flash-f16.json')
        shared = read(ROOT / 'config/shared.json')
        self.assertEqual(candidate['id'], 'daytime-flash-f16')
        self.assertEqual(candidate['engine'], 'flash-next-mtp')
        for key in baseline.keys() - {'id', 'display_name', 'arguments', 'catalog'}:
            with self.subTest(field=key):
                self.assertEqual(candidate[key], baseline[key])
        baseline_args = {**shared['arguments'], **baseline['arguments']}
        candidate_args = {**shared['arguments'], **candidate['arguments']}
        self.assertEqual({key for key in candidate_args if candidate_args[key] != baseline_args[key]},
            {'--cache-type-k', '--cache-type-v', '--ubatch-size'})
        for flag, value in [('--cache-type-k', 'f16'), ('--cache-type-v', 'f16'),
                ('--ubatch-size', '1024')]:
            self.assertEqual(candidate_args[flag], value)
        self.assertEqual(candidate_args['--spec-draft-type-k'], 'q8_0')
        self.assertEqual(candidate_args['--spec-draft-type-v'], 'q8_0')

        original = render(ROOT / 'config', BASELINE['host'], 'daytime')
        variant = render(ROOT / 'config', BASELINE['host'], 'daytime-flash-f16')
        original_cfg = original['compose']['services']['coding']
        variant_cfg = copy.deepcopy(variant['compose']['services']['coding'])
        for flag in ('--cache-type-k', '--cache-type-v', '--ubatch-size'):
            old_command = original_cfg['command']
            new_command = variant_cfg['command']
            new_command[new_command.index(flag) + 1] = old_command[old_command.index(flag) + 1]
        self.assertEqual(variant_cfg, original_cfg)
        self.assertEqual(variant['compose']['services']['everyday'],
            original['compose']['services']['everyday'])
        self.assertEqual(variant['artifacts'], original['artifacts'])
        self.assertEqual(service_engine(variant, 'coding'), service_engine(original, 'coding'))
        self.assertEqual(variant['catalog']['model'], original['catalog']['model'])
        self.assertEqual(variant['catalog']['aliases'], original['catalog']['aliases'])
        self.assertEqual(variant['catalog']['context_length'], 131072)
        self.assertEqual(variant['catalog']['mtp'], original['catalog']['mtp'])
        self.assertEqual(variant['catalog']['kv_cache'],
            {'unified': False, 'key_type': 'f16', 'value_type': 'f16'})
        self.assertNotEqual(variant['catalog']['capability_profile']['name'],
            original['catalog']['capability_profile']['name'])
        warning = ' '.join(variant['catalog']['deployment_warnings']).lower()
        self.assertIn('not been qualified', warning)
        self.assertIn('vram', warning)
        self.assertIn('throughput', warning)

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


class RegistryTests(unittest.TestCase):
    def test_every_selectable_configuration_agrees_with_its_rendered_catalog(self):
        registry = available_profiles(ROOT / 'config', BASELINE['host'])
        self.assertEqual(DAYTIME_PROFILES,
            ('daytime', 'daytime-27b', 'daytime-flash-f16'))
        self.assertEqual([x['profile'] for x in registry['selectable']], list(DAYTIME_PROFILES))
        self.assertEqual([x['profile'] for x in registry['always_included']], [NIGHTTIME_PROFILE])
        night = registry['always_included'][0]
        baseline_night = render(ROOT / 'config', BASELINE['host'], 'daytime')['compose']['services']['everyday']
        for name in DAYTIME_PROFILES:
            with self.subTest(profile=name):
                entry = next(x for x in registry['selectable'] if x['profile'] == name)
                rendered = render(ROOT / 'config', BASELINE['host'], name)
                model = next(m for m in rendered['catalog']['models'] if m['model'] == entry['model'])
                engine = service_engine(rendered, 'coding')
                self.assertEqual(entry['display_name'], model['display_name'])
                self.assertEqual(entry['context_tokens'], model['context_length'])
                self.assertEqual(entry['engine_tag'], engine['tag'])
                self.assertEqual(entry['backend_revision'], engine['revision'])
                # The paired backend is identical whichever Daytime configuration is selected.
                everyday = service_engine(rendered, 'everyday')
                night_entry = next(m for m in rendered['catalog']['models'] if m['model'] == night['model'])
                self.assertEqual(night['engine_tag'], everyday['tag'])
                self.assertEqual(night['backend_revision'], everyday['revision'])
                self.assertEqual(night['context_tokens'], night_entry['context_length'])
                self.assertEqual(night['display_name'], night_entry['display_name'])
                self.assertEqual(rendered['compose']['services']['everyday'], baseline_night)

    def test_registry_offers_only_registered_profiles_and_names_configured_gpus(self):
        definitions = {path.stem: read(path) for path in (ROOT / 'config/profiles').glob('*.json')}
        self.assertEqual(sorted(k for k, v in definitions.items() if v['role'] == 'coding'),
            sorted(DAYTIME_PROFILES))  # A new profile file must be registered in DAYTIME_PROFILES.
        selectable = [x['profile'] for x in available_profiles(ROOT / 'config')['selectable']]
        for retired in ('daytime-swift', 'daytime-256', 'nighttime-256', 'nighttime', 'primary'):
            self.assertNotIn(retired, selectable)
        self.assertEqual(available_profiles(ROOT / 'config')['selectable'][0]['gpu_names'], [])
        listed = available_profiles(ROOT / 'config', BASELINE['host'])
        self.assertEqual(listed['selectable'][0]['gpu_names'], BASELINE['host']['gpu_names']['daytime'])
        self.assertEqual(listed['always_included'][0]['gpu_names'], BASELINE['host']['gpu_names']['nighttime'])

    def test_registry_publishes_no_host_paths_artifacts_or_checksums(self):
        registry = available_profiles(ROOT / 'config', BASELINE['host'])
        def check(value):
            if isinstance(value, dict):
                self.assertTrue({'model_path', 'mmproj_path', 'source', 'sources', 'artifacts',
                    'mounts', 'sha256', 'bytes', 'arguments'}.isdisjoint(value))
                for child in value.values(): check(child)
            elif isinstance(value, list):
                for child in value: check(child)
            elif isinstance(value, str):
                self.assertFalse(value.startswith('/'), value)
        check(registry)
        self.assertNotIn(BASELINE['host']['model_root'], json.dumps(registry))

    def test_cli_list_is_generated_from_the_same_registry(self):
        from runtime.__main__ import listed
        lines = listed(ROOT / 'config').splitlines()
        self.assertEqual([line.split(':')[0] for line in lines],
            ['primary', *DAYTIME_PROFILES, NIGHTTIME_PROFILE])
        registry = available_profiles(ROOT / 'config')
        self.assertIn(registry['always_included'][0]['display_name'], lines[0])
        for entry in registry['selectable'] + registry['always_included']:
            row = next(line for line in lines if line.startswith(entry['profile'] + ':'))
            self.assertIn(entry['display_name'], row)
            self.assertIn(entry['model'], row)

    def registry_with(self, name, mutate):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            path = config / 'profiles' / (name + '.json')
            definition = read(path)
            if mutate is None:
                path.write_text('{')
            else:
                mutate(definition); path.write_text(json.dumps(definition))
            return available_profiles(config)

    def test_registry_reports_an_unreadable_or_invalid_profile_instead_of_guessing(self):
        with self.subTest(case='unreadable file'):
            with self.assertRaises(json.JSONDecodeError):
                self.registry_with('daytime-27b', None)
        for name, mutate, expected in [
                ('daytime', lambda definition: definition.update(engine='no-such-engine'), 'unknown engine'),
                ('nighttime', lambda definition: definition.update(id='renamed'), 'differs from its filename'),
                ('daytime', lambda definition: definition.update(context_tokens=262144), 'router contract')]:
            with self.subTest(profile=name):
                with self.assertRaisesRegex(RuntimeError, expected):
                    self.registry_with(name, mutate)

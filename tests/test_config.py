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

    def test_ram_prompt_cache_is_a_uniform_shared_default_with_no_profile_overrides(self):
        shared = read(ROOT / 'config/shared.json')
        self.assertEqual(shared['arguments']['--cache-ram'], '49152')
        # No profile may override the shared RAM prompt cache cap: every backend inherits the
        # same 48 GiB lazy LRU limit (zero idle cost), so the worst-case combined host-RAM
        # pressure of the daytime + nighttime pair stays bounded at 96 GiB.
        for name in [*DAYTIME_PROFILES, NIGHTTIME_PROFILE]:
            with self.subTest(profile=name):
                definition = read(ROOT / 'config/profiles' / (name + '.json'))
                self.assertNotIn('--cache-ram', definition['arguments'])
        for name in DAYTIME_PROFILES:
            with self.subTest(profile=name):
                rendered = render(ROOT / 'config', BASELINE['host'], name)
                self.assertEqual(rendered['catalog']['global_ram_prompt_cache_mib'], 49152)
                for model in rendered['catalog']['models']:
                    self.assertEqual(model['global_ram_prompt_cache_mib'], 49152)

    def test_q6k_experiment_changes_only_the_main_weights(self):
        baseline = read(ROOT / 'config/profiles/daytime-27b.json')
        candidate = read(ROOT / 'config/profiles/daytime-27b-q6k.json')
        self.assertEqual(candidate.keys(), baseline.keys())
        self.assertEqual(candidate['id'], 'daytime-27b-q6k')
        self.assertEqual(candidate['display_name'], 'Daytime-27B Q6_K')
        # Engine, placement, argument order, and container are inherited unchanged.
        for key in baseline.keys() - {'id', 'display_name', 'arguments', 'artifacts', 'catalog'}:
            with self.subTest(field=key):
                self.assertEqual(candidate[key], baseline[key])
        # Only the alias changes, because it names the weight quantization.
        self.assertEqual(candidate['arguments'], {**baseline['arguments'], '--alias': 'qwen3.8-27b-ud-q6_k_xl'})
        main_path = ('llm/Qwen3.8-27B-GGUF/fallbacks/4ca720788d1e01f1bff70c033e0d0028fd02e502/'
            'Qwen3.8-27B-UD-Q6_K_XL.gguf')
        self.assertEqual(candidate['artifacts'][0], {'path': main_path, 'target': '/weights/main.gguf',
            'bytes': 25299061664, 'sha256': '701d8fa9ed214ab21bfc130cd2a7df19ca89bbef7713e2dfb19f3c63696aa917'})
        self.assertEqual(baseline['artifacts'][0]['target'], '/weights/main.gguf')
        self.assertEqual(candidate['artifacts'][1:], baseline['artifacts'][1:])  # Q4_0 MTP draft and projector
        expected_catalog = copy.deepcopy(baseline['catalog'])
        expected_catalog['quantization'] = 'UD-Q6_K_XL'
        expected_catalog['capability_profile']['name'] = 'qwen38-27b-golden-vision-tools-q6k'
        self.assertEqual(candidate['catalog'], expected_catalog)

        original = render(ROOT / 'config', BASELINE['host'], 'daytime-27b')
        variant = render(ROOT / 'config', BASELINE['host'], 'daytime-27b-q6k')
        main_source = str(Path(BASELINE['host']['model_root']) / main_path)
        original_cfg = original['compose']['services']['coding']
        variant_cfg = copy.deepcopy(variant['compose']['services']['coding'])
        mount = next(m for m in variant_cfg['volumes'] if m['target'] == '/weights/main.gguf')
        self.assertEqual(mount['source'], main_source)
        mount['source'] = next(m['source'] for m in original_cfg['volumes'] if m['target'] == '/weights/main.gguf')
        argv = variant_cfg['command']
        self.assertEqual(argv[argv.index('--alias') + 1], 'qwen3.8-27b-ud-q6_k_xl')
        argv[argv.index('--alias') + 1] = 'qwen3.8-27b-q8_0'
        self.assertEqual(variant_cfg, original_cfg)  # identical devices, ports, other mounts, and other argv
        self.assertEqual(variant['compose']['services']['everyday'], original['compose']['services']['everyday'])
        self.assertEqual(service_engine(variant, 'coding'), service_engine(original, 'coding'))
        self.assertEqual(service_engine(variant, 'everyday'), service_engine(original, 'everyday'))
        changed = [a for a in variant['artifacts'] if a not in original['artifacts']]
        self.assertEqual([a['source'] for a in changed], [main_source])
        self.assertEqual(len(variant['artifacts']), len(original['artifacts']))
        self.assertEqual(variant['catalog']['model'], 'qwen3.8-27b-ud-q6_k_xl')
        self.assertEqual(variant['catalog']['model_path'], main_source)
        self.assertEqual(variant['catalog']['quantization'], 'UD-Q6_K_XL')
        self.assertEqual(variant['catalog']['aliases'], original['catalog']['aliases'])
        self.assertEqual(variant['catalog']['context_length'], 163840)
        self.assertEqual(variant['catalog']['kv_cache'], original['catalog']['kv_cache'])
        self.assertEqual(variant['catalog']['mtp'], original['catalog']['mtp'])  # same Q4_0 draft on CUDA1
        self.assertEqual(variant['catalog']['display_name'], 'Daytime-27B Q6_K (160K)')
        self.assertEqual(variant['catalog']['capability_profile'], {
            **original['catalog']['capability_profile'], 'name': 'qwen38-27b-golden-vision-tools-q6k'})
        self.assertEqual(variant['catalog']['models'][1], original['catalog']['models'][1])

    def test_q4k_experiment_changes_only_the_main_weights(self):
        baseline = read(ROOT / 'config/profiles/daytime-27b.json')
        candidate = read(ROOT / 'config/profiles/daytime-27b-q4k.json')
        self.assertEqual(candidate.keys(), baseline.keys())
        self.assertEqual(candidate['id'], 'daytime-27b-q4k')
        self.assertEqual(candidate['display_name'], 'Daytime-27B Q4_K')
        # Engine, placement, argument order, and container are inherited unchanged.
        for key in baseline.keys() - {'id', 'display_name', 'arguments', 'artifacts', 'catalog'}:
            with self.subTest(field=key):
                self.assertEqual(candidate[key], baseline[key])
        # Only the alias changes, because it names the weight quantization.
        self.assertEqual(candidate['arguments'], {**baseline['arguments'], '--alias': 'qwen3.8-27b-ud-q4_k_m'})
        main_path = ('llm/Qwen3.8-27B-GGUF/fallbacks/4ca720788d1e01f1bff70c033e0d0028fd02e502/'
            'Qwen3.8-27B-UD-Q4_K_M.gguf')
        self.assertEqual(candidate['artifacts'][0], {'path': main_path, 'target': '/weights/main.gguf',
            'bytes': 16464440224, 'sha256': '322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482'})
        self.assertEqual(baseline['artifacts'][0]['target'], '/weights/main.gguf')
        self.assertEqual(candidate['artifacts'][1:], baseline['artifacts'][1:])  # Q4_0 MTP draft and projector
        expected_catalog = copy.deepcopy(baseline['catalog'])
        expected_catalog['quantization'] = 'UD-Q4_K_M'
        expected_catalog['capability_profile']['name'] = 'qwen38-27b-golden-vision-tools-q4k'
        self.assertEqual(candidate['catalog'], expected_catalog)

        original = render(ROOT / 'config', BASELINE['host'], 'daytime-27b')
        variant = render(ROOT / 'config', BASELINE['host'], 'daytime-27b-q4k')
        main_source = str(Path(BASELINE['host']['model_root']) / main_path)
        original_cfg = original['compose']['services']['coding']
        variant_cfg = copy.deepcopy(variant['compose']['services']['coding'])
        mount = next(m for m in variant_cfg['volumes'] if m['target'] == '/weights/main.gguf')
        self.assertEqual(mount['source'], main_source)
        mount['source'] = next(m['source'] for m in original_cfg['volumes'] if m['target'] == '/weights/main.gguf')
        argv = variant_cfg['command']
        self.assertEqual(argv[argv.index('--alias') + 1], 'qwen3.8-27b-ud-q4_k_m')
        argv[argv.index('--alias') + 1] = 'qwen3.8-27b-q8_0'
        self.assertEqual(variant_cfg, original_cfg)  # identical devices, ports, other mounts, and other argv
        self.assertEqual(variant['compose']['services']['everyday'], original['compose']['services']['everyday'])
        self.assertEqual(service_engine(variant, 'coding'), service_engine(original, 'coding'))
        self.assertEqual(service_engine(variant, 'everyday'), service_engine(original, 'everyday'))
        changed = [a for a in variant['artifacts'] if a not in original['artifacts']]
        self.assertEqual([a['source'] for a in changed], [main_source])
        self.assertEqual(len(variant['artifacts']), len(original['artifacts']))
        self.assertEqual(variant['catalog']['model'], 'qwen3.8-27b-ud-q4_k_m')
        self.assertEqual(variant['catalog']['model_path'], main_source)
        self.assertEqual(variant['catalog']['quantization'], 'UD-Q4_K_M')
        self.assertEqual(variant['catalog']['aliases'], original['catalog']['aliases'])
        self.assertEqual(variant['catalog']['context_length'], 163840)
        self.assertEqual(variant['catalog']['kv_cache'], original['catalog']['kv_cache'])
        self.assertEqual(variant['catalog']['mtp'], original['catalog']['mtp'])  # same Q4_0 draft on CUDA1
        self.assertEqual(variant['catalog']['display_name'], 'Daytime-27B Q4_K (160K)')
        self.assertEqual(variant['catalog']['capability_profile'], {
            **original['catalog']['capability_profile'], 'name': 'qwen38-27b-golden-vision-tools-q4k'})
        self.assertEqual(variant['catalog']['models'][1], original['catalog']['models'][1])

    def test_q4k_3090_experiment_changes_only_context_and_placement(self):
        baseline = read(ROOT / 'config/profiles/daytime-27b-q4k.json')
        candidate = read(ROOT / 'config/profiles/daytime-27b-q4k-3090.json')
        self.assertEqual(candidate.keys(), baseline.keys())
        self.assertEqual(candidate['id'], 'daytime-27b-q4k-3090')
        self.assertEqual(candidate['display_name'], 'Daytime-27B Q4_K 3090')
        self.assertEqual(candidate['context_tokens'], 131072)
        # Engine, artifacts (same UD-Q4_K_M weights, Q4_0 MTP draft, projector), argument order,
        # GPU group, and container are inherited unchanged.
        for key in baseline.keys() - {'id', 'display_name', 'context_tokens', 'arguments', 'catalog'}:
            with self.subTest(field=key):
                self.assertEqual(candidate[key], baseline[key])
        # Every layer goes to the first listed device (CUDA1, the RTX 3090); the alias keeps
        # single-GPU results apart from the two-GPU profile that shares these weights.
        self.assertEqual(candidate['arguments'], {**baseline['arguments'],
            '--alias': 'qwen3.8-27b-ud-q4_k_m-3090', '--tensor-split': '100,0'})
        for key, expected in {'--device': 'CUDA1,CUDA0', '--spec-draft-device': 'CUDA1',
                '--n-gpu-layers': '66', '--spec-draft-ngl': 'all'}.items():
            with self.subTest(argument=key):
                self.assertEqual(candidate['arguments'][key], expected)
        warnings = ['Single-RTX 3090 placement: VRAM headroom at full 128K context and throughput have '
            'not been qualified on the production GPUs.',
            '128K is the total prompt, history, reasoning, and output capacity.']
        expected_catalog = copy.deepcopy(baseline['catalog'])
        expected_catalog['capability_profile']['name'] = 'qwen38-27b-golden-vision-tools-q4k-3090'
        expected_catalog['deployment_warnings'] = warnings
        self.assertEqual(candidate['catalog'], expected_catalog)

        original = render(ROOT / 'config', BASELINE['host'], 'daytime-27b-q4k')
        variant = render(ROOT / 'config', BASELINE['host'], 'daytime-27b-q4k-3090')
        original_cfg = original['compose']['services']['coding']
        variant_cfg = copy.deepcopy(variant['compose']['services']['coding'])
        argv = variant_cfg['command']
        original_argv = original_cfg['command']
        for flag, value in {'--alias': 'qwen3.8-27b-ud-q4_k_m-3090', '--ctx-size': '131072',
                '--kv-unified-per-slot': '131072', '--tensor-split': '100,0'}.items():
            with self.subTest(flag=flag):
                self.assertEqual(argv[argv.index(flag) + 1], value)
                argv[argv.index(flag) + 1] = original_argv[original_argv.index(flag) + 1]
        self.assertEqual(variant_cfg, original_cfg)  # identical devices, mounts, ports, and other argv
        self.assertEqual(variant['compose']['services']['everyday'], original['compose']['services']['everyday'])
        self.assertEqual(service_engine(variant, 'coding'), service_engine(original, 'coding'))
        self.assertEqual(service_engine(variant, 'everyday'), service_engine(original, 'everyday'))
        self.assertEqual(variant['artifacts'], original['artifacts'])
        catalog = variant['catalog']
        self.assertEqual(catalog['model'], 'qwen3.8-27b-ud-q4_k_m-3090')
        self.assertEqual(catalog['model_path'], original['catalog']['model_path'])
        self.assertEqual(catalog['quantization'], 'UD-Q4_K_M')
        self.assertEqual(catalog['aliases'], original['catalog']['aliases'])
        self.assertEqual(catalog['context_length'], 131072)
        self.assertEqual(catalog['total_context_length'], 131072)
        self.assertEqual(catalog['gpu_uuids'], original['catalog']['gpu_uuids'])
        self.assertEqual(catalog['kv_cache'], original['catalog']['kv_cache'])
        self.assertEqual(catalog['mtp'], original['catalog']['mtp'])  # same Q4_0 draft on CUDA1
        self.assertEqual(catalog['display_name'], 'Daytime-27B Q4_K 3090 (128K)')
        self.assertEqual(catalog['capability_profile'], {
            **original['catalog']['capability_profile'], 'name': 'qwen38-27b-golden-vision-tools-q4k-3090'})
        self.assertEqual(catalog['deployment_warnings'], warnings)
        self.assertEqual(catalog['models'][1], original['catalog']['models'][1])

    def test_q4k_256k_experiment_changes_only_the_context(self):
        baseline = read(ROOT / 'config/profiles/daytime-27b-q4k.json')
        candidate = read(ROOT / 'config/profiles/daytime-27b-q4k-256k.json')
        self.assertEqual(candidate.keys(), baseline.keys())
        self.assertEqual(candidate['id'], 'daytime-27b-q4k-256k')
        self.assertEqual(candidate['context_tokens'], 262144)  # the model's native window
        # Engine, artifacts (same UD-Q4_K_M weights, Q4_0 MTP draft, projector), argument order,
        # GPU group, container, and display name are inherited unchanged.
        for key in baseline.keys() - {'id', 'context_tokens', 'arguments', 'catalog'}:
            with self.subTest(field=key):
                self.assertEqual(candidate[key], baseline[key])
        # Placement, cache types, and batch sizes are untouched; only the alias names the experiment.
        self.assertEqual(candidate['arguments'], {**baseline['arguments'],
            '--alias': 'qwen3.8-27b-ud-q4_k_m-256k'})
        warnings = ['256K context: VRAM headroom at full context and long-context throughput have not '
            'been qualified on the production GPUs.',
            '256K is the total prompt, history, reasoning, and output capacity.']
        expected_catalog = copy.deepcopy(baseline['catalog'])
        expected_catalog['capability_profile']['name'] = 'qwen38-27b-golden-vision-tools-q4k-256k'
        expected_catalog['deployment_warnings'] = warnings
        self.assertEqual(candidate['catalog'], expected_catalog)

        original = render(ROOT / 'config', BASELINE['host'], 'daytime-27b-q4k')
        variant = render(ROOT / 'config', BASELINE['host'], 'daytime-27b-q4k-256k')
        original_cfg = original['compose']['services']['coding']
        variant_cfg = copy.deepcopy(variant['compose']['services']['coding'])
        argv = variant_cfg['command']
        original_argv = original_cfg['command']
        for flag, value in {'--alias': 'qwen3.8-27b-ud-q4_k_m-256k', '--ctx-size': '262144',
                '--kv-unified-per-slot': '262144'}.items():
            with self.subTest(flag=flag):
                self.assertEqual(argv[argv.index(flag) + 1], value)
                argv[argv.index(flag) + 1] = original_argv[original_argv.index(flag) + 1]
        self.assertEqual(variant_cfg, original_cfg)  # identical devices, split, KV types, mounts, and other argv
        self.assertEqual(variant['compose']['services']['everyday'], original['compose']['services']['everyday'])
        self.assertEqual(service_engine(variant, 'coding'), service_engine(original, 'coding'))
        self.assertEqual(service_engine(variant, 'everyday'), service_engine(original, 'everyday'))
        self.assertEqual(variant['artifacts'], original['artifacts'])
        catalog = variant['catalog']
        self.assertEqual(catalog['model'], 'qwen3.8-27b-ud-q4_k_m-256k')
        self.assertEqual(catalog['model_path'], original['catalog']['model_path'])
        self.assertEqual(catalog['quantization'], 'UD-Q4_K_M')
        self.assertEqual(catalog['aliases'], original['catalog']['aliases'])
        self.assertEqual(catalog['context_length'], 262144)
        self.assertEqual(catalog['total_context_length'], 262144)
        self.assertEqual(variant['manifest']['services'][0]['context_tokens'], 262144)
        self.assertEqual(catalog['gpu_uuids'], original['catalog']['gpu_uuids'])
        self.assertEqual(catalog['kv_cache'], {'unified': False, 'key_type': 'q8_0', 'value_type': 'q8_0'})
        self.assertEqual(catalog['kv_cache'], original['catalog']['kv_cache'])
        self.assertEqual(catalog['global_ram_prompt_cache_mib'], original['catalog']['global_ram_prompt_cache_mib'])
        self.assertEqual(catalog['mtp'], original['catalog']['mtp'])  # same Q4_0 draft on CUDA1
        self.assertEqual(catalog['display_name'], 'Daytime-27B Q4_K (256K)')
        self.assertEqual(catalog['capability_profile'], {
            **original['catalog']['capability_profile'], 'name': 'qwen38-27b-golden-vision-tools-q4k-256k'})
        self.assertEqual(catalog['deployment_warnings'], warnings)
        self.assertEqual(catalog['models'][1], original['catalog']['models'][1])

    def test_context_ceiling_is_the_router_contract_native_window(self):
        from runtime.config import ROUTER_CONTEXT_LIMIT
        self.assertEqual(ROUTER_CONTEXT_LIMIT, 262144)
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            path = config / 'profiles/daytime-27b-q4k.json'
            definition = read(path)
            for tokens in (262145, 0, '262144'):
                with self.subTest(tokens=tokens):
                    definition['context_tokens'] = tokens
                    path.write_text(json.dumps(definition))
                    with self.assertRaisesRegex(RuntimeError, 'router contract: 1–262144 tokens'):
                        render(config, BASELINE['host'], 'daytime-27b-q4k')
                    with self.assertRaisesRegex(RuntimeError, 'router contract'):
                        available_profiles(config)
            definition['context_tokens'] = 262144
            path.write_text(json.dumps(definition))
            self.assertEqual(render(config, BASELINE['host'], 'daytime-27b-q4k')['catalog']['context_length'], 262144)

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
            ('daytime', 'daytime-27b', 'daytime-flash-f16', 'daytime-27b-q6k', 'daytime-27b-q4k',
             'daytime-27b-q4k-3090', 'daytime-27b-q4k-256k'))
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
                ('daytime', lambda definition: definition.update(context_tokens=262145), 'router contract')]:
            with self.subTest(profile=name):
                with self.assertRaisesRegex(RuntimeError, expected):
                    self.registry_with(name, mutate)

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
VISION = 'GPU-465a9a1e-fcb9-27f2-8f6a-cca1c30de981'
# Exclusive profiles hold all four text GPUs and run without Nighttime; every other Daytime
# profile is paired with the unchanged Nighttime backend.
EXCLUSIVE = tuple(n for n in DAYTIME_PROFILES if read(ROOT / 'config/profiles' / (n + '.json')).get('exclusive'))
PAIRED = tuple(n for n in DAYTIME_PROFILES if n not in EXCLUSIVE)


def vision_host(host=None):
    host = copy.deepcopy(host or BASELINE['host'])
    host.update(vision_gpu_id=VISION, vision_gpu_name='RTX 3080',
        cuda_order={'daytime': host['gpu_ids']['daytime'][::-1], 'nighttime': host['gpu_ids']['nighttime'][:]})
    return host


def host_for(profile):
    # The exclusive profiles place their projector and MTP draft on the vision GPU.
    return vision_host() if profile in EXCLUSIVE else BASELINE['host']


class ConfigurationTests(unittest.TestCase):
    def test_renamed_profiles_launch_exactly_the_backends_of_their_previous_names(self):
        # The 2026-10-07 consolidation renamed the kept configurations. Their backends and model aliases
        # (benchmark history) are unchanged, so switching from an old name to its new one recreates nothing.
        recorded = read(ROOT / 'tests/fixtures/renamed-profiles.json')
        self.assertEqual(set(recorded), set(DAYTIME_PROFILES) - {'daytime-flash-solo-tuned-mtp3-160k'})
        for name, expected in recorded.items():
            with self.subTest(profile=name, previous=expected['previous_id']):
                self.assertEqual(digest(render(ROOT / 'config', vision_host(), name)['compose']), expected['compose_sha256'])

    def test_context_edit_updates_both_launch_flags_manifest_and_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            path = config / 'profiles/qwen27b-q8-with-nighttime.json'
            definition = read(path); definition['context_tokens'] = 98304
            path.write_text(json.dumps(definition))
            result = render(config, BASELINE['host'], 'qwen27b-q8-with-nighttime')
            argv = result['compose']['services']['coding']['command']
            for flag in ['--ctx-size', '--kv-unified-per-slot']:
                self.assertEqual(argv[argv.index(flag) + 1], '98304')
            self.assertEqual(result['catalog']['context_length'], 98304)
            self.assertEqual(result['catalog']['models'][0]['total_context_length'], 98304)
            self.assertEqual(result['manifest']['services'][0]['context_tokens'], 98304)
            self.assertEqual(result['catalog']['models'][1]['context_length'], 131072)
            self.assertEqual(result['catalog']['display_name'], 'Qwen3.8 27B Q8 (96K)')

    def test_gpu_groups_cannot_overlap(self):
        host = copy.deepcopy(BASELINE['host']); host['gpu_ids']['nighttime'][0] = host['gpu_ids']['daytime'][0]
        with self.assertRaisesRegex(RuntimeError, 'four distinct'):
            render(ROOT / 'config', host, 'qwen27b-q8-with-nighttime')

    def test_unknown_retired_profile_and_output_cap_are_rejected(self):
        for retired in ('daytime', 'daytime-27b', 'daytime-flash-solo', 'daytime-swift'):
            with self.subTest(profile=retired), self.assertRaisesRegex(RuntimeError, 'retired'):
                render(ROOT / 'config', BASELINE['host'], retired)
        with self.assertRaisesRegex(RuntimeError, 'Unknown'):
            render(ROOT / 'config', BASELINE['host'], '../../secret')
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            path = config / 'shared.json'; value = read(path); value['arguments']['--n-predict'] = '100'
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, 'unsupported policy'):
                render(config, BASELINE['host'], 'qwen27b-q8-with-nighttime')

    def test_no_qualification_is_synthesized(self):
        result = render(ROOT / 'config', BASELINE['host'], 'qwen27b-q8-with-nighttime')
        def check_keys(value):
            if isinstance(value, dict):
                self.assertTrue({'qualified', 'qualification', 'qualification_receipt'}.isdisjoint(value))
                for child in value.values(): check_keys(child)
            elif isinstance(value, list):
                for child in value: check_keys(child)
        check_keys(result)
        # The imported capability name is historical metadata, not a generated receipt.
        self.assertEqual(result['catalog']['capability_profile'],
            read(ROOT / 'config/profiles/qwen27b-q8-with-nighttime.json')['catalog']['capability_profile'])
        self.assertEqual(len(result['artifacts']), 5)  # 27B weights, MTP draft, projector; Nighttime weights, projector
        self.assertEqual(result['catalog']['aliases'], ['local-active', 'daytime'])

    def test_27b_profiles_run_tensor_parallel_on_the_daytime_pair_beside_nighttime(self):
        engine = read(ROOT / 'config/shared.json')['engines']['qwen38-dual-836d571']
        q8 = render(ROOT / 'config', BASELINE['host'], 'qwen27b-q8-with-nighttime')
        q6k = render(ROOT / 'config', BASELINE['host'], 'qwen27b-q6k-with-nighttime')
        for bundle, alias, quant, display in ((q8, 'qwen3.8-27b-q8_0-tensor-next', 'Q8_0', 'Qwen3.8 27B Q8 (160K)'),
                (q6k, 'qwen3.8-27b-ud-q6_k_xl-tensor-next', 'UD-Q6_K_XL', 'Qwen3.8 27B Q6_K (160K)')):
            with self.subTest(alias=alias):
                self.assertEqual(list(bundle['compose']['services']), ['coding', 'everyday'])
                argv = bundle['compose']['services']['coding']['command']
                # Tensor mode divides every weight between the RTX 3090 and RTX 4080 SUPER, balanced by bandwidth.
                for flag, value in {'--alias': alias, '--split-mode': 'tensor', '--tensor-split': '55,45', '--fit': 'off',
                        '--ctx-size': '163840', '--spec-draft-n-max': '3', '--spec-draft-device': 'CUDA1'}.items():
                    self.assertEqual(argv[argv.index(flag) + 1], value)
                self.assertEqual(service_engine(bundle, 'coding'), engine)
                catalog = bundle['catalog']
                self.assertEqual((catalog['model'], catalog['quantization'], catalog['display_name']), (alias, quant, display))
                self.assertEqual((catalog['context_length'], catalog['mtp']['max_draft_tokens'], catalog['mtp']['device']), (163840, 3, 'CUDA1'))
                self.assertEqual(catalog['models'][1]['display_name'], 'Qwen3.8 27B Abliterated Q6_K (128K)')
        # Q6_K differs from Q8 only in its main weights, alias, and catalog identity.
        q8_cfg, q6k_cfg = q8['compose']['services']['coding'], copy.deepcopy(q6k['compose']['services']['coding'])
        argv = q6k_cfg['command']; argv[argv.index('--alias') + 1] = 'qwen3.8-27b-q8_0-tensor-next'
        for mount in q6k_cfg['volumes']:
            if mount['target'] == '/weights/main.gguf':
                self.assertTrue(mount['source'].endswith('Qwen3.8-27B-UD-Q6_K_XL.gguf'))
                mount['source'] = next(m['source'] for m in q8_cfg['volumes'] if m['target'] == '/weights/main.gguf')
        self.assertEqual(q6k_cfg, q8_cfg)
        self.assertEqual(q6k['compose']['services']['everyday'], q8['compose']['services']['everyday'])

    def test_flash_next_solo_profiles_hold_every_gpu_with_the_measured_settings(self):
        engine = read(ROOT / 'config/shared.json')['engines']['qwen38-dual-43fe9c6']
        self.assertEqual((engine['revision'], engine['tag'], engine['image_id']), ('43fe9c64281ef735046adc025e9e7559a1f659a5',
            'local/llama.cpp:qwen38-dual-43fe9c64281e', 'sha256:5a25a54554f2ba945c0761ad67b329ae9cfa0c18cacf1aa890d549d03a9b4546'))
        host = vision_host()
        text = [*host['gpu_ids']['daytime'], *host['gpu_ids']['nighttime']]  # CUDA0-3: 3090, 4080S, 4080, 3080 Ti
        common = {'--split-mode': 'layer', '--device': 'CUDA1,CUDA2,CUDA3,CUDA0', '--fit': 'off', '--n-gpu-layers': 'all',
            # A fixed whole-layer split (48 blocks + output); the input-embedding override only restates its default
            # CPU placement and keeps pipeline parallelism, whose larger buffers do not fit, disabled.
            '--tensor-split': '11,12,8,18', '--override-tensor': 'token_embd\\.weight=CPU', '--lazy-mode': 'off',
            '--spec-draft-n-max': '3', '--spec-draft-device': 'CUDA4', '--mmproj-device': 'CUDA4',
            # Engine 43fe9c6 runs a small CPU input split each step; more OpenMP threads only busy-wait.
            '--threads': '1', '--threads-batch': '1'}
        cases = {'flash-next-solo-128k': ('Qwen3.8 Flash-Next (128K)', 'qwen3.8-flash-next-ad4.27-solo-tuned-mtp3',
                    {'--ctx-size': '131072', '--parallel': '1', '--cache-type-k': 'f16', '--cache-type-v': 'f16', '--ubatch-size': '1024'}),
            # F16 K/V at 160K left the RTX 4080 without room for the indexer top-k scratch and aborted a prefill.
            'flash-next-solo-160k': ('Qwen3.8 Flash-Next (160K)', 'qwen3.8-flash-next-ad4.27-solo-tuned-mtp3-160k',
                    {'--ctx-size': '163840', '--parallel': '1', '--cache-type-k': 'q8_0', '--cache-type-v': 'q8_0', '--ubatch-size': '1024'}),
            # Two 128K slots: q8_0 K/V keeps the pool at one F16 slot's size; the 512 microbatch fits the top-k scratch.
            'flash-next-solo-two-requests': ('Qwen3.8 Flash-Next, two requests at once (128K)', 'qwen3.8-flash-next-ad4.27-solo-tuned-mtp3-2slot',
                    {'--ctx-size': '262144', '--kv-unified-per-slot': '131072', '--parallel': '2', '--cache-type-k': 'q8_0',
                     '--cache-type-v': 'q8_0', '--ubatch-size': '512'})}
        for name, (display, alias, specific) in cases.items():
            with self.subTest(profile=name):
                bundle = render(ROOT / 'config', host, name)
                self.assertEqual(list(bundle['compose']['services']), ['coding'])
                cfg = bundle['compose']['services']['coding']
                self.assertEqual(cfg['image'], engine['tag'])
                self.assertEqual(cfg['deploy']['resources']['reservations']['devices'][0]['device_ids'], [*text, VISION])
                argv = cfg['command']
                for flag, value in {**common, **specific, '--alias': alias}.items():
                    self.assertEqual(argv.count(flag), 1, flag)
                    self.assertEqual(argv[argv.index(flag) + 1], value, flag)
                self.assertEqual(sum(int(x) for x in common['--tensor-split'].split(',')), 49)
                model = bundle['catalog']['models'][0]
                slots = int(specific['--parallel'])
                self.assertEqual((model['display_name'], model['model'], model['exclusive'], model['max_active_requests']), (display, alias, True, slots))
                self.assertEqual((model['mtp']['max_draft_tokens'], model['mtp']['device'], model['fit_target']), (3, 'CUDA4', 'off'))
                self.assertEqual(model['total_context_length'], model['context_length'] * slots)
                self.assertEqual(model['backend_revision'], engine['revision'])
                self.assertEqual(bundle['manifest']['services'][0]['parallel_slots'], slots)
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            for name, change, message in [('flash-next-solo-two-requests', {'parallel_slots': 3}, 'must be 1 or 2'),
                    ('qwen27b-q8-with-nighttime', {'parallel_slots': 2}, 'only an exclusive profile')]:
                target = config / 'profiles' / (name + '.json'); original = target.read_text()
                definition = read(target); definition.update(change); target.write_text(json.dumps(definition))
                with self.subTest(profile=name), self.assertRaisesRegex(RuntimeError, message):
                    render(config, vision_host(), name)
                target.write_text(original)

    def test_exclusive_profiles_require_vision_gpu_and_keep_the_model_on_text_gpus(self):
        with self.assertRaisesRegex(RuntimeError, 'configured vision GPU'):
            render(ROOT / 'config', BASELINE['host'], 'flash-next-solo-128k')
        host = vision_host(); host['gpu_ids']['nighttime'][0] = host['gpu_ids']['daytime'][0]
        with self.assertRaisesRegex(RuntimeError, 'four distinct'):
            render(ROOT / 'config', host, 'flash-next-solo-128k')
        host = vision_host(); host['gpu_ids']['nighttime'] = host['gpu_ids']['nighttime'][:1]
        with self.assertRaisesRegex(RuntimeError, 'two real GPU UUIDs'):
            render(ROOT / 'config', host, 'flash-next-solo-128k')
        host = vision_host(); host['vision_gpu_id'] = 'GPU-night-2'
        with self.assertRaisesRegex(RuntimeError, 'full GPU UUID'):
            render(ROOT / 'config', host, 'flash-next-solo-128k')
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            path = config / 'profiles/flash-next-solo-128k.json'; original = read(path)
            for key, value, message in [
                    ('--device', 'CUDA0,CUDA1,CUDA2', 'exactly once'),
                    ('--device', 'CUDA0,CUDA1,CUDA2,CUDA4', 'exactly once'),
                    ('--device', 'CUDA0,CUDA1,CUDA2,CUDA2', 'exactly once'),
                    ('--tensor-split', '30,30,40', 'one value per text GPU'),
                    ('--override-tensor', ['blk.*=CUDA4'], 'never the vision GPU'),
                    ('--spec-draft-device', 'CUDA5', 'reserved GPU'),
                    ('--main-gpu', '4', 'text GPU')]:
                definition = copy.deepcopy(original); definition['arguments'][key] = value
                path.write_text(json.dumps(definition))
                with self.subTest(key=key, value=value), self.assertRaisesRegex(RuntimeError, message):
                    render(config, vision_host(), 'flash-next-solo-128k')
            for name, change, message in [
                    ('flash-next-solo-128k', {'exclusive': 'yes'}, 'true or false'),
                    ('flash-next-solo-160k', {'gpu_group': 'daytime'}, '"all" GPU group'),
                    ('qwen27b-q8-with-nighttime', {'gpu_group': 'all'}, 'unknown GPU group'),
                    ('nighttime', {'exclusive': True}, '"all" GPU group')]:
                shutil.rmtree(config); shutil.copytree(ROOT / 'config', config)
                target = config / 'profiles' / (name + '.json')
                definition = read(target); definition.update(change); target.write_text(json.dumps(definition))
                with self.subTest(profile=name, change=change), self.assertRaisesRegex(RuntimeError, message):
                    render(config, vision_host(), name if name.startswith('flash-next') else 'qwen27b-q8-with-nighttime')

    def test_ram_prompt_cache_is_shared_for_daytime_and_lowered_only_for_nighttime(self):
        shared = read(ROOT / 'config/shared.json')
        self.assertEqual(shared['arguments']['--cache-ram'], '49152')
        # Every Daytime profile inherits the shared 48 GiB lazy LRU cap. Nighttime alone overrides
        # it to 24 GiB: on the 123 GiB host its cache had grown to ~50 GB with ~8 GB swapped and
        # swap exhausted, so the worst-case combined pair is bounded at 72 GiB instead of 96 GiB.
        for name in DAYTIME_PROFILES:
            with self.subTest(profile=name):
                definition = read(ROOT / 'config/profiles' / (name + '.json'))
                self.assertNotIn('--cache-ram', definition['arguments'])
        night = read(ROOT / 'config/profiles' / (NIGHTTIME_PROFILE + '.json'))
        self.assertEqual(night['arguments']['--cache-ram'], '24576')
        self.assertIn('--cache-ram', night['argument_order'])
        for name in DAYTIME_PROFILES:
            with self.subTest(profile=name):
                rendered = render(ROOT / 'config', host_for(name), name)
                self.assertEqual(rendered['catalog']['global_ram_prompt_cache_mib'], 49152)
                coding, *paired = rendered['catalog']['models']
                self.assertEqual(coding['global_ram_prompt_cache_mib'], 49152)
                self.assertEqual([m['global_ram_prompt_cache_mib'] for m in paired], [] if name in EXCLUSIVE else [24576])
                for role, expected in (('coding', '49152'), ('everyday', '24576'))[:1 if name in EXCLUSIVE else 2]:
                    argv = rendered['compose']['services'][role]['command']
                    self.assertEqual(argv.count('--cache-ram'), 1)
                    self.assertEqual(argv[argv.index('--cache-ram') + 1], expected)

    def test_every_backend_uses_one_of_two_pinned_engines(self):
        engines = read(ROOT / 'config/shared.json')['engines']
        # The Flash-Next branch engine retired with the paired Flash-Next profiles; the 27B pair and
        # Nighttime use 836d571, and the Flash-Next solo profiles use 43fe9c6.
        self.assertEqual(set(engines), {'qwen38-dual-836d571', 'qwen38-dual-43fe9c6'})
        engine = engines['qwen38-dual-836d571']
        self.assertEqual(engine['revision'], '836d57176dc699a726c55418e4f96b8ca628e1bf')
        self.assertEqual(engine['tag'], 'local/llama.cpp:qwen38-dual-836d57176dc6')
        self.assertEqual(engine['image_id'], 'sha256:9f7179f568e4aa1004c8af9b613b65417e6f0e451f8cfbc774d6b0d0f50a0ecd')
        for key in ('repository', 'cuda_architectures', 'build_base_digest', 'runtime_base_digest'):
            self.assertEqual(engines['qwen38-dual-43fe9c6'][key], engine[key])  # same Dockerfile inputs, newer source
        expected = {'qwen27b-q8-with-nighttime': 'qwen38-dual-836d571', 'qwen27b-q6k-with-nighttime': 'qwen38-dual-836d571',
            'flash-next-solo-128k': 'qwen38-dual-43fe9c6', 'flash-next-solo-160k': 'qwen38-dual-43fe9c6',
            'flash-next-solo-two-requests': 'qwen38-dual-43fe9c6', 'daytime-flash-solo-tuned-mtp3-160k': 'qwen38-dual-43fe9c6',
            NIGHTTIME_PROFILE: 'qwen38-dual-836d571'}
        self.assertEqual(set(expected), {*DAYTIME_PROFILES, NIGHTTIME_PROFILE})
        for name, engine_name in expected.items():
            with self.subTest(profile=name):
                self.assertEqual(read(ROOT / 'config/profiles' / (name + '.json'))['engine'], engine_name)
        for name in PAIRED:
            bundle = render(ROOT / 'config', BASELINE['host'], name)
            for role in ('coding', 'everyday'):
                with self.subTest(profile=name, role=role):
                    self.assertEqual(bundle['compose']['services'][role]['image'], engine['tag'])
                    self.assertEqual(service_engine(bundle, role), engine)

    def test_nighttime_uses_tensor_parallel_placement_across_its_pair(self):
        night = read(ROOT / 'config/profiles' / (NIGHTTIME_PROFILE + '.json'))
        self.assertEqual(night['arguments']['--split-mode'], 'tensor')
        self.assertIn('--split-mode', night['argument_order'])
        for flag, value in {'--tensor-split': '60,40', '--device': 'CUDA0,CUDA1', '--n-gpu-layers': 'all',
                '--cache-ram': '24576'}.items():
            with self.subTest(argument=flag):
                self.assertEqual(night['arguments'][flag], value)
        self.assertNotIn('--spec-type', night['arguments'])  # Nighttime has no MTP draft
        for name in PAIRED:
            with self.subTest(profile=name):
                argv = render(ROOT / 'config', BASELINE['host'], name)['compose']['services']['everyday']['command']
                self.assertEqual(argv.count('--split-mode'), 1)
                self.assertEqual(argv[argv.index('--split-mode') + 1], 'tensor')
                self.assertEqual(argv[argv.index('--flash-attn') + 1], 'on')  # required by tensor mode
                self.assertEqual(argv[argv.index('--fit') + 1], 'off')  # fitting is not implemented for tensor mode

    def test_context_ceiling_is_the_router_contract_native_window(self):
        from runtime.config import ROUTER_CONTEXT_LIMIT
        self.assertEqual(ROUTER_CONTEXT_LIMIT, 262144)
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            path = config / 'profiles/qwen27b-q6k-with-nighttime.json'
            definition = read(path)
            for tokens in (262145, 0, '262144'):
                with self.subTest(tokens=tokens):
                    definition['context_tokens'] = tokens
                    path.write_text(json.dumps(definition))
                    with self.assertRaisesRegex(RuntimeError, 'router contract: 1–262144 tokens'):
                        render(config, BASELINE['host'], 'qwen27b-q6k-with-nighttime')
                    with self.assertRaisesRegex(RuntimeError, 'router contract'):
                        available_profiles(config)
            definition['context_tokens'] = 262144
            path.write_text(json.dumps(definition))
            self.assertEqual(render(config, BASELINE['host'], 'qwen27b-q6k-with-nighttime')['catalog']['context_length'], 262144)

    def test_invalid_repeated_arguments_or_missing_model_mount_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            path = config / 'profiles/flash-next-solo-128k.json'; original = read(path)
            for key, value, message in [('--temp', ['1', '2'], 'Only --override-tensor'),
                    ('--override-tensor', [], 'nonempty'), ('--model', '/weights/missing.gguf', 'declared artifact')]:
                with self.subTest(key=key):
                    definition = copy.deepcopy(original); definition['arguments'][key] = value
                    path.write_text(json.dumps(definition))
                    with self.assertRaisesRegex(RuntimeError, message): render(config, vision_host(), 'flash-next-solo-128k')
            definition = copy.deepcopy(original)
            definition['artifacts'].pop(20)
            path.write_text(json.dumps(definition))
            with self.assertRaisesRegex(RuntimeError, 'missing a declared shard'):
                render(config, vision_host(), 'flash-next-solo-128k')


class RegistryTests(unittest.TestCase):
    def test_every_selectable_configuration_agrees_with_its_rendered_catalog(self):
        registry = available_profiles(ROOT / 'config', BASELINE['host'])
        self.assertEqual(DAYTIME_PROFILES, ('qwen27b-q8-with-nighttime', 'qwen27b-q6k-with-nighttime', 'flash-next-solo-128k',
            'flash-next-solo-160k', 'flash-next-solo-two-requests', 'daytime-flash-solo-tuned-mtp3-160k'))
        self.assertEqual(PAIRED, DAYTIME_PROFILES[:2])
        self.assertEqual([x['profile'] for x in registry['selectable']], list(DAYTIME_PROFILES))
        self.assertEqual([x['profile'] for x in registry['always_included']], [NIGHTTIME_PROFILE])
        night = registry['always_included'][0]
        baseline_night = render(ROOT / 'config', BASELINE['host'], 'qwen27b-q8-with-nighttime')['compose']['services']['everyday']
        for name in DAYTIME_PROFILES:
            with self.subTest(profile=name):
                entry = next(x for x in registry['selectable'] if x['profile'] == name)
                rendered = render(ROOT / 'config', host_for(name), name)
                model = next(m for m in rendered['catalog']['models'] if m['model'] == entry['model'])
                engine = service_engine(rendered, 'coding')
                self.assertEqual(entry['display_name'], model['display_name'])
                self.assertEqual(entry['context_tokens'], model['context_length'])
                self.assertEqual(entry['engine_tag'], engine['tag'])
                self.assertEqual(entry['backend_revision'], engine['revision'])
                self.assertEqual(entry['exclusive'], name in EXCLUSIVE)
                if name in EXCLUSIVE:
                    # Nighttime is neither launched nor published while an exclusive profile is selected.
                    self.assertEqual(list(rendered['compose']['services']), ['coding'])
                    self.assertEqual([m['model'] for m in rendered['catalog']['models']], [entry['model']])
                    continue
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
            self.assertEqual(row.endswith('(all four text GPUs; stops Nighttime)'), entry['profile'] in EXCLUSIVE)

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
                self.registry_with('qwen27b-q6k-with-nighttime', None)
        for name, mutate, expected in [
                ('qwen27b-q8-with-nighttime', lambda definition: definition.update(engine='no-such-engine'), 'unknown engine'),
                ('nighttime', lambda definition: definition.update(id='renamed'), 'differs from its filename'),
                ('qwen27b-q8-with-nighttime', lambda definition: definition.update(context_tokens=262145), 'router contract')]:
            with self.subTest(profile=name):
                with self.assertRaisesRegex(RuntimeError, expected):
                    self.registry_with(name, mutate)

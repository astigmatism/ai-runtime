import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from runtime.config import (CONTAINER_CONTRACT, DAYTIME_PROFILES, NIGHTTIME_PROFILE, NIGHTTIME_PROFILES,
    available_profiles, digest, engine_interface, merge_settings, read, render, service_engine)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = read(ROOT / 'tests/fixtures/legacy-fingerprints.json')
VISION = 'GPU-465a9a1e-fcb9-27f2-8f6a-cca1c30de981'
# Exclusive profiles hold all four text GPUs and run without Nighttime; every other Daytime
# profile is paired with the unchanged Nighttime backend.
EXCLUSIVE = tuple(n for n in DAYTIME_PROFILES if read(ROOT / 'config/profiles' / (n + '.json')).get('exclusive'))
PAIRED = tuple(n for n in DAYTIME_PROFILES if n not in EXCLUSIVE)
ENGINES = read(ROOT / 'config/shared.json')['engines']
# The one profile that is not llama.cpp. Its engine is configured by the environment, so the
# catalog fields that describe a flag the engine was handed (KV types, --fit, the host-RAM prompt
# cache budget) are not published for it. It is reached through its own service alias, TRIAL_SERVICE,
# which is nobody's default and is asserted on that entry by ServiceIdentityTests below.
TRIAL = 'flash-next-strata-160k'
TRIAL_SERVICE = 'trial'
# The alias every configuration's Daytime entry carries for the legacy Ollama surface. It is the
# compatibility ID, not the one a contract client sends, so it is never the only alias an entry has.
COMPATIBILITY_ALIAS = 'local-active'
# The service IDs a resident answers to here. The trial claims none of them: it replaces the Daytime
# resident and stops Nighttime, so an entry answering to any of these would be reachable by clients
# that were never told to try it.
RESIDENT_SERVICES = ('daytime', COMPATIBILITY_ALIAS, 'nighttime')


def interface_of(profile):
    return engine_interface(ENGINES[read(ROOT / 'config/profiles' / (profile + '.json'))['engine']])


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
        # compose_sha256 is the whole backend set at the rename; Nighttime has since gained MTP3 (2026-10-07),
        # so the Daytime service is what stays pinned for the paired configurations.
        recorded = read(ROOT / 'tests/fixtures/renamed-profiles.json')
        self.assertEqual(set(recorded), {'qwen27b-q8-with-nighttime', 'qwen27b-q6k-with-nighttime', 'flash-next-solo-128k', 'flash-next-solo-160k'})
        for name, expected in recorded.items():
            with self.subTest(profile=name, previous=expected['previous_id']):
                compose = render(ROOT / 'config', vision_host(), name)['compose']
                self.assertEqual(digest(compose['services']['coding']), expected['coding_service_sha256'])
                if name in EXCLUSIVE:
                    self.assertEqual(digest(compose), expected['compose_sha256'])

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
            self.assertEqual(result['catalog']['models'][1]['context_length'], 98304)
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
                self.assertEqual(catalog['models'][1]['display_name'], 'Qwen3.8 27B Abliterated Q6_K (96K)')
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
                    {'--ctx-size': '163840', '--parallel': '1', '--cache-type-k': 'q8_0', '--cache-type-v': 'q8_0', '--ubatch-size': '1024'})}
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
            # No configuration runs two slots now; the capability stays (LLM Router admits two requests for
            # such a resident). Two 128K slots size the KV pool for both and keep the per-request window.
            target = config / 'profiles/flash-next-solo-128k.json'; original = target.read_text()
            definition = read(target); definition['parallel_slots'] = 2; target.write_text(json.dumps(definition))
            bundle = render(config, host, 'flash-next-solo-128k')
            argv = bundle['compose']['services']['coding']['command']
            self.assertEqual([argv[argv.index(f) + 1] for f in ('--parallel', '--ctx-size', '--kv-unified-per-slot')], ['2', '262144', '131072'])
            model = bundle['catalog']['models'][0]
            self.assertEqual((model['context_length'], model['total_context_length'], model['max_active_requests']), (131072, 262144, 2))
            self.assertEqual(bundle['manifest']['services'][0]['parallel_slots'], 2)
            target.write_text(original)
            for name, change, message in [('flash-next-solo-128k', {'parallel_slots': 3}, 'must be 1 or 2'),
                    ('qwen27b-q8-with-nighttime', {'parallel_slots': 2}, 'only an exclusive profile')]:
                target = config / 'profiles' / (name + '.json'); original = target.read_text()
                definition = read(target); definition.update(change); target.write_text(json.dumps(definition))
                with self.subTest(profile=name), self.assertRaisesRegex(RuntimeError, message):
                    render(config, vision_host(), name)
                target.write_text(original)

    def test_nighttime_runs_the_base_mtp3_head_on_the_shared_vision_gpu(self):
        night = read(ROOT / 'config/profiles/nighttime.json')
        spec = {'--spec-type': 'draft-mtp', '--spec-draft-model': '/weights/mtp.gguf', '--spec-draft-n-max': '3',
            '--spec-draft-ngl': 'all', '--spec-draft-device': 'CUDA2', '--spec-draft-type-k': 'q8_0', '--spec-draft-type-v': 'q8_0'}
        self.assertEqual({k: night['arguments'][k] for k in spec}, spec)
        self.assertEqual((night['display_name'], night['arguments']['--alias'], night['draft_on_vision_gpu']),
            ('Qwen3.8 27B Abliterated Q6_K', 'qwen3.8-27b-abliterated-q6_k', True))
        # At 128K, speculative decoding needed another 564 MiB on the RTX 4080 than the pair had free.
        self.assertEqual(night['context_tokens'], 98304)
        head = next(a for a in read(ROOT / 'config/profiles/qwen27b-q6k-with-nighttime.json')['artifacts'] if a['target'] == '/weights/mtp.gguf')
        self.assertIn(head, night['artifacts'])  # the same self-contained 27B MTP head Daytime uses
        host = vision_host()
        for name in PAIRED:
            with self.subTest(profile=name):
                bundle = render(ROOT / 'config', host, name)
                self.assertNotIn('nighttime_profile', bundle)
                everyday = bundle['compose']['services']['everyday']
                argv = everyday['command']
                for flag, value in {**spec, '--split-mode': 'tensor', '--device': 'CUDA0,CUDA1', '--mmproj-device': 'CUDA2'}.items():
                    self.assertEqual(argv[argv.index(flag) + 1], value)
                self.assertEqual(everyday['deploy']['resources']['reservations']['devices'][0]['device_ids'], [*host['gpu_ids']['nighttime'], VISION])
                model = bundle['catalog']['models'][1]
                self.assertEqual((model['model'], model['mtp']['device'], model['mtp']['max_draft_tokens'], model['display_name']),
                    ('qwen3.8-27b-abliterated-q6_k', 'CUDA2', 3, 'Qwen3.8 27B Abliterated Q6_K (96K)'))
        # Without a vision GPU there is nowhere to put the draft: Nighttime runs without it rather than failing.
        cpu = render(ROOT / 'config', BASELINE['host'], 'qwen27b-q8-with-nighttime')
        argv = cpu['compose']['services']['everyday']['command']
        self.assertFalse([x for x in argv if x.startswith('--spec-')])
        self.assertNotIn('/weights/mtp.gguf', [m['target'] for m in cpu['compose']['services']['everyday']['volumes']])
        self.assertEqual(cpu['catalog']['models'][1]['mtp'], {'enabled': False})
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config_dir)
            for name, change, message, target in [
                    ('nighttime', {'draft_on_vision_gpu': False}, 'Draft model must remain on a text GPU', 'qwen27b-q8-with-nighttime'),
                    ('qwen27b-q8-with-nighttime', {'nighttime': 'nighttime-retired'}, 'unknown Nighttime profile', 'qwen27b-q8-with-nighttime'),
                    ('flash-next-solo-128k', {'nighttime': 'nighttime'}, 'runs no Nighttime', 'flash-next-solo-128k')]:
                path = config_dir / 'profiles' / (name + '.json'); original = path.read_text()
                definition = read(path); definition.update(change); path.write_text(json.dumps(definition))
                with self.subTest(profile=name), self.assertRaisesRegex(RuntimeError, message):
                    render(config_dir, host, target)
                path.write_text(original)

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
        # Only the flag-configured engines are given a cache budget at all, so only they publish one.
        for name in [n for n in DAYTIME_PROFILES if interface_of(n) == 'argv']:
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

    def test_every_backend_uses_one_of_the_pinned_engines(self):
        engines = read(ROOT / 'config/shared.json')['engines']
        # The Flash-Next branch engine retired with the paired Flash-Next profiles; the 27B pair and
        # Nighttime use 836d571, the Flash-Next solo profiles use 43fe9c6, and the Strata trial is the
        # only non-llama.cpp engine in the house.
        #
        # Two Strata engines are registered on purpose, and only for as long as the 0.1.44.1 candidate
        # is under evaluation (since 2026-10-15): 7695859aa13a is the one that was serving when the
        # candidate was built, and the agreed rollback for a candidate that fails its gates is to
        # repoint the profile, not to re-add an entry and pay a second drain to reconstruct it. When
        # the candidate is decided either way, this set goes back to three keys and the losing entry
        # is removed reference-aware (ops/reports/20261015-strata-engine-0.1.44.1-build/README.md).
        self.assertEqual(set(engines), {'qwen38-dual-836d571', 'qwen38-dual-43fe9c6',
                                        'strata-7695859aa13a', 'strata-296b59d2f73c'})
        engine = engines['qwen38-dual-836d571']
        self.assertEqual(engine['revision'], '836d57176dc699a726c55418e4f96b8ca628e1bf')
        self.assertEqual(engine['tag'], 'local/llama.cpp:qwen38-dual-836d57176dc6')
        self.assertEqual(engine['image_id'], 'sha256:9f7179f568e4aa1004c8af9b613b65417e6f0e451f8cfbc774d6b0d0f50a0ecd')
        for key in ('repository', 'cuda_architectures', 'build_base_digest', 'runtime_base_digest'):
            self.assertEqual(engines['qwen38-dual-43fe9c6'][key], engine[key])  # same Dockerfile inputs, newer source
        expected = {'qwen27b-q8-with-nighttime': 'qwen38-dual-836d571', 'qwen27b-q6k-with-nighttime': 'qwen38-dual-836d571',
            'flash-next-solo-128k': 'qwen38-dual-43fe9c6', 'flash-next-solo-160k': 'qwen38-dual-43fe9c6',
            'flash-next-strata-160k': 'strata-296b59d2f73c',
            NIGHTTIME_PROFILE: 'qwen38-dual-836d571'}
        self.assertEqual(set(expected), {*DAYTIME_PROFILES, *NIGHTTIME_PROFILES})
        for name, engine_name in expected.items():
            with self.subTest(profile=name):
                self.assertEqual(read(ROOT / 'config/profiles' / (name + '.json'))['engine'], engine_name)
        for name in PAIRED:
            bundle = render(ROOT / 'config', host_for(name), name)
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
        self.assertEqual(night['arguments']['--spec-type'], 'draft-mtp')  # MTP3 since 2026-10-07, on the vision GPU
        for name in PAIRED:
            with self.subTest(profile=name):
                argv = render(ROOT / 'config', host_for(name), name)['compose']['services']['everyday']['command']
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
            'flash-next-solo-160k', 'flash-next-strata-160k'))
        self.assertEqual(PAIRED, ('qwen27b-q8-with-nighttime', 'qwen27b-q6k-with-nighttime'))
        self.assertEqual([x['profile'] for x in registry['selectable']], list(DAYTIME_PROFILES))
        # Every Nighttime variant a configuration names is listed, the default first.
        self.assertEqual([x['profile'] for x in registry['always_included']], list(NIGHTTIME_PROFILES))
        variants = {x['profile']: x for x in registry['always_included']}
        night = variants[NIGHTTIME_PROFILE]
        baseline = {name: render(ROOT / 'config', vision_host(), cfg)['compose']['services']['everyday']
            for name, cfg in (('nighttime', 'qwen27b-q8-with-nighttime'),)}
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
                self.assertEqual(rendered['catalog']['configuration'], {'id': name, 'exclusive': name in EXCLUSIVE})
                if name in EXCLUSIVE:
                    # Nighttime is neither launched nor published while an exclusive profile is selected;
                    # the router reports it as offline for this configuration instead of unknown.
                    self.assertEqual(list(rendered['compose']['services']), ['coding'])
                    self.assertEqual([m['model'] for m in rendered['catalog']['models']], [entry['model']])
                    self.assertEqual(rendered['catalog']['offline_services'], [{'model': night['model'],
                        'aliases': ['nighttime'], 'display_name': night['display_name'], 'role': 'everyday',
                        'reason': 'exclusive_configuration'}])
                    continue
                self.assertEqual(rendered['catalog']['offline_services'], [])
                # The paired backend is the Nighttime variant the configuration names, identical whichever
                # Daytime configuration names it.
                paired = variants[entry['nighttime']]
                everyday = service_engine(rendered, 'everyday')
                night_entry = next(m for m in rendered['catalog']['models'] if m['model'] == paired['model'])
                self.assertEqual(paired['engine_tag'], everyday['tag'])
                self.assertEqual(paired['backend_revision'], everyday['revision'])
                self.assertEqual(paired['context_tokens'], night_entry['context_length'])
                self.assertEqual(paired['display_name'], night_entry['display_name'])
                self.assertEqual(render(ROOT / 'config', vision_host(), name)['compose']['services']['everyday'], baseline[entry['nighttime']])

    def test_catalog_names_each_backends_text_gpus_and_vision_gpu(self):
        host = vision_host()
        names = host['gpu_names']
        paired = render(ROOT / 'config', host, 'qwen27b-q8-with-nighttime')['catalog']
        coding, everyday = paired['models']
        self.assertEqual(coding['gpu_names'], names['daytime'])
        self.assertEqual(everyday['gpu_names'], names['nighttime'])
        self.assertEqual(len(coding['gpu_names']), len(coding['text_gpu_uuids']))
        self.assertEqual((coding['vision_gpu_name'], coding['vision_gpu_shared']), ('RTX 3080', True))
        self.assertEqual(paired['configuration'], {'id': 'qwen27b-q8-with-nighttime', 'exclusive': False})
        # The root coding projection carries the same names as its entry.
        self.assertEqual(paired['gpu_names'], coding['gpu_names'])
        [solo] = render(ROOT / 'config', host, 'flash-next-solo-128k')['catalog']['models']
        self.assertEqual(solo['gpu_names'], [*names['daytime'], *names['nighttime']])
        self.assertEqual((solo['vision_gpu_name'], solo['vision_gpu_shared']), ('RTX 3080', False))
        # Without GPU vision the projector stays on the CPU and no vision card is named.
        cpu = render(ROOT / 'config', BASELINE['host'], 'qwen27b-q8-with-nighttime')['catalog']['models'][0]
        self.assertNotIn('vision_gpu_name', cpu)
        self.assertEqual(cpu['gpu_names'], BASELINE['host']['gpu_names']['daytime'])

    def test_every_model_declares_whether_it_is_nsfw(self):
        # Router clients pick "the most capable NSFW model" from this flag; only Nighttime is abliterated.
        for name in DAYTIME_PROFILES:
            with self.subTest(profile=name):
                catalog = render(ROOT / 'config', host_for(name), name)['catalog']
                flags = {m['model']: m['capability_profile']['nsfw'] for m in catalog['models']}
                self.assertEqual(sorted(flags.values()), [False] if name in EXCLUSIVE else [False, True])
                self.assertIs(catalog['models'][0]['capability_profile']['nsfw'], False)
                self.assertIs(catalog['capability_profile']['nsfw'], False)
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            path = config / 'profiles' / 'nighttime.json'
            definition = read(path)
            for value in (None, 'yes'):
                if value is None:
                    definition['catalog']['capability_profile'].pop('nsfw', None)
                else:
                    definition['catalog']['capability_profile']['nsfw'] = value
                path.write_text(json.dumps(definition))
                with self.subTest(nsfw=value), self.assertRaisesRegex(RuntimeError, 'capability_profile.nsfw must be declared'):
                    render(config, BASELINE['host'], 'qwen27b-q8-with-nighttime')

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
            ['primary', *DAYTIME_PROFILES, *NIGHTTIME_PROFILES])
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


class EnvironmentEngineTests(unittest.TestCase):
    """The trial engine is configured by the environment instead of an argument vector."""

    def bundle(self):
        return render(ROOT / 'config', host_for(TRIAL), TRIAL)

    def test_every_argument_is_either_an_environment_variable_or_house_metadata(self):
        definition = read(ROOT / 'config/profiles' / (TRIAL + '.json'))
        engine = ENGINES[definition['engine']]
        self.assertEqual(engine_interface(engine), 'environment')
        for key in definition['argument_order']:
            with self.subTest(argument=key):
                if key in ('--alias', '--model'):
                    self.assertNotIn(key, engine['environment'])
                else:
                    self.assertIn(key, engine['environment'])

    def test_the_service_is_rendered_from_its_own_arguments_and_mounts(self):
        definition = read(ROOT / 'config/profiles' / (TRIAL + '.json'))
        service = self.bundle()['compose']['services']['coding']
        self.assertEqual(service['command'], [])
        self.assertEqual(service['entrypoint'], ['./docker-entrypoint.sh'])
        self.assertIs(service['read_only'], False)
        environment = service['environment']
        for key, variable in ENGINES[definition['engine']]['environment'].items():
            with self.subTest(argument=key):
                self.assertEqual(environment[variable], str(definition['arguments'][key]))
        # The folder the engine reads is the folder the shards are mounted in, and the window is the
        # profile's context: neither is a value a profile could quietly disagree with.
        self.assertEqual(environment['GGUF_DIR'], str(Path(definition['arguments']['--model']).parent))
        self.assertEqual(environment['CONTEXT'], str(definition['context_tokens']))
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '0,1,2,3')
        self.assertEqual(service['ulimits']['memlock'], -1)
        # Only the engine's own workspace is writable; every model file stays a read-only bind.
        writable = [x['target'] for x in service['volumes'] if not x['read_only']]
        self.assertEqual(writable, ['/data'])
        self.assertEqual([x['target'] for x in service['volumes'] if x['read_only']],
            [x['target'] for x in definition['artifacts']])

    def test_the_trial_is_published_under_its_own_service_id_and_without_llama_cpp_only_claims(self):
        bundle = self.bundle()
        entry = bundle['catalog']['models'][0]
        definition = read(ROOT / 'config/profiles' / (TRIAL + '.json'))
        self.assertEqual(entry['model'], definition['arguments']['--alias'])
        self.assertEqual(entry['aliases'], [TRIAL_SERVICE])
        self.assertEqual(entry['context_length'], definition['context_tokens'])
        self.assertEqual(entry['total_context_length'], definition['context_tokens'])
        self.assertEqual(entry['input_modalities'], ['text'])
        self.assertIs(entry['capability_profile']['vision'], False)
        self.assertIs(entry['capability_profile']['nsfw'], False)
        # The house contract still holds: no server-side output ceiling, and the full window.
        self.assertEqual(entry['output_policy'], 'unrestricted')
        # Nothing is claimed about caches and projectors this engine was never given.
        for field in ('kv_cache', 'fit_target', 'global_ram_prompt_cache_mib', 'mmproj_path', 'vision_gpu_uuid'):
            with self.subTest(field=field):
                self.assertNotIn(field, entry)
        # The RTX 3080 stays out of the container entirely.
        self.assertNotIn(VISION, entry['gpu_uuids'])
        self.assertEqual(len(entry['gpu_uuids']), 4)
        self.assertTrue(any('not been attested' in warning for warning in entry['deployment_warnings']))


    def test_the_trial_service_keeps_the_house_container_settings(self):
        """Attempt one of this switch died on KeyError: 'restart'.

        The engine entry declares its own container_defaults and they replaced the shared ones, so the
        trial service arrived with no restart policy, no init, no logging and no pull policy. The merge
        is the fix; this is the test that would have caught it, stating both halves: what the house
        provides and what the engine is allowed to narrow.
        """
        shared = read(ROOT / 'config/shared.json')['container_defaults']
        cfg = self.bundle()['compose']['services']['coding']
        for key in CONTAINER_CONTRACT:
            with self.subTest(setting=key):
                self.assertIsNotNone(cfg.get(key))
        self.assertEqual(cfg['pull_policy'], shared['pull_policy'])
        self.assertEqual(cfg['restart'], shared['restart'])
        self.assertIs(cfg['init'], shared['init'])
        self.assertEqual(cfg['logging'], shared['logging'])
        # What the engine narrowed still survives the merge.
        self.assertEqual(cfg['entrypoint'], ['./docker-entrypoint.sh'])
        self.assertIs(cfg['read_only'], False)
        self.assertEqual(cfg['healthcheck']['start_period'], '3600s')
        self.assertEqual(cfg['healthcheck']['retries'], 8)
        self.assertEqual(cfg['ulimits'], {'core': 0, 'memlock': -1})
        self.assertEqual(cfg['stop_grace_period'], '180s')
    def test_the_engine_identity_is_pinned(self):
        engine = ENGINES[read(ROOT / 'config/profiles' / (TRIAL + '.json'))['engine']]
        self.assertEqual(engine['revision'], '296b59d2f73c8f04a805df29c2fa97655adace78')
        self.assertEqual(engine['tag'], 'local/strata:296b59d2f73c')
        self.assertEqual(engine['release'], 'v0.1.44.1')
        # Built on this host from the pinned revision, not downloaded: the upstream releases ship
        # Windows zips only. Receipt and image digest in
        # ops/reports/20261015-strata-engine-0.1.44.1-build/README.md.
        self.assertEqual(engine['image_id'],
                         'sha256:2c6b8332effd8630cabe6ab033ee953ed9fdaecf7d979f5dba36380db94ad057')
        self.assertEqual(engine['repository'], 'https://github.com/Niko1221/Strata')
        self.assertEqual(engine['cuda_architectures'], ['86', '89'])
        self.assertEqual(engine['interface'], 'environment')
        self.assertGreater(engine['readiness_wait_seconds'], 450)


class ServiceIdentityTests(unittest.TestCase):
    """What a client has to send to reach a backend, and what must never reach the trial."""

    def test_every_profile_publishes_a_service_id_a_contract_client_can_send(self):
        # LLM Router derives the client-facing ID from these aliases and falls back to the canonical
        # model id when it finds none (llm-router `src/capabilities.js`: `service: aliases.find(...)
        # ?? entry.id`), and its client contract forbids a client from sending, configuring or
        # comparing canonical ids. An entry with no service alias is therefore running, selected, and
        # impossible to ask for — a host whose only lever moves nothing, which is the failure AGENTS.md
        # forbids in a drain, reached here through model identity instead. It is not theoretical: the
        # trial published `aliases: []` from 2026-10-11 until it was given its own alias on 2026-10-13,
        # and the primary client on this machine had no model at all to use while it was selected.
        for name in DAYTIME_PROFILES:
            catalog = render(ROOT / 'config', host_for(name), name)['catalog']
            with self.subTest(profile=name):
                # Publication merges the default's entry over the document root, so a client that reads
                # the root has to be told the same service ID as the entry it describes.
                self.assertEqual(catalog['aliases'], catalog['models'][0]['aliases'])
            for entry in catalog['models']:
                with self.subTest(profile=name, model=entry['model']):
                    service = [alias for alias in entry['aliases']
                        if isinstance(alias, str) and alias.strip() and alias != COMPATIBILITY_ALIAS]
                    # The compatibility alias is the legacy Ollama surface. An entry carrying only it
                    # still has nothing a contract client may send.
                    self.assertTrue(service, 'Profile ' + name +
                        ' publishes a catalog no client following the router contract can address')
                    # No two residents answer to the same service ID, whichever configuration is
                    # selected: `trial` is only ever the trial, `daytime` only ever a Daytime model.
                    for other in catalog['models']:
                        if other is entry:
                            continue
                        self.assertTrue(set(service).isdisjoint(
                            [alias for alias in other['aliases'] if alias != COMPATIBILITY_ALIAS]),
                            'Service ID shared between ' + entry['model'] + ' and ' + other['model'])

    def test_the_trial_never_shadows_a_resident_service(self):
        """`aliases: []` was never about being unreachable. It was about being nobody's default.

        The trial replaces the Daytime resident on the same cards and stops Nighttime, so a `daytime`,
        `local-active` or `nighttime` alias on this entry would move an existing client onto a 4-bit,
        text-only, unqualified engine with nobody having chosen it — contradicting that selecting the
        trial, and switching back, are the owner's acts. Its own alias is what makes it addressable;
        these three are what keep it a deliberate choice, so the protection is asserted against the
        alias set rather than depending on that set being empty.
        """
        catalog = render(ROOT / 'config', host_for(TRIAL), TRIAL)['catalog']
        entry = catalog['models'][0]
        self.assertEqual(entry['aliases'], [TRIAL_SERVICE])
        self.assertTrue(set(RESIDENT_SERVICES).isdisjoint(entry['aliases']))
        # `trial` is not a service anything else claims, and the trial is not declared offline.
        self.assertNotIn(TRIAL_SERVICE, [alias for service in catalog['offline_services']
            for alias in service['aliases']])
        # And the other way round: no other configuration answers to `trial`, so a client that kept
        # sending it after a switch away is refused rather than quietly left on the trial engine.
        for name in DAYTIME_PROFILES:
            if name == TRIAL:
                continue
            with self.subTest(profile=name):
                aliases = [alias for model in render(ROOT / 'config', host_for(name), name)['catalog']['models']
                    for alias in model['aliases']]
                self.assertNotIn(TRIAL_SERVICE, aliases)


class ContainerContractTests(unittest.TestCase):
    """How a backend runs here is set by the house, so a merge must not be able to lose it."""

    def test_partial_blocks_merge_instead_of_replacing_them(self):
        base = {'restart': 'unless-stopped', 'ulimits': {'core': 0},
            'healthcheck': {'interval': '15s', 'retries': 4}, 'tmpfs': ['/tmp'],
            'logging': {'driver': 'json-file', 'options': {'max-size': '25m'}}}
        merged = merge_settings(base, {'ulimits': {'memlock': -1},
            'healthcheck': {'start_period': '3600s'}, 'tmpfs': ['/run']})
        self.assertEqual(merged['ulimits'], {'core': 0, 'memlock': -1})
        self.assertEqual(merged['healthcheck'], {'interval': '15s', 'retries': 4, 'start_period': '3600s'})
        # A list is the override's own statement, never appended to what it inherits.
        self.assertEqual(merged['tmpfs'], ['/run'])
        self.assertEqual(merged['restart'], 'unless-stopped')
        self.assertEqual(base['ulimits'], {'core': 0}, 'merge must not mutate what it inherited from')

    def test_every_service_carries_the_house_container_settings(self):
        for name in DAYTIME_PROFILES:
            for role, cfg in render(ROOT / 'config', host_for(name), name)['compose']['services'].items():
                for key in CONTAINER_CONTRACT:
                    with self.subTest(profile=name, role=role, setting=key):
                        self.assertIsNotNone(cfg.get(key))

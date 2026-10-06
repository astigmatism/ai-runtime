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

    def test_flash_next_engine_candidate_changes_only_engine_alias_and_mtp_head(self):
        baseline = read(ROOT / 'config/profiles/daytime.json')
        candidate = read(ROOT / 'config/profiles/daytime-flash-next.json')
        engines = read(ROOT / 'config/shared.json')['engines']
        self.assertEqual(candidate.keys(), baseline.keys())
        self.assertEqual(candidate['id'], 'daytime-flash-next')
        self.assertEqual(candidate['display_name'], 'FlashNext Next Engine')
        self.assertEqual(candidate['engine'], 'qwen38-dual-836d571')
        for key in baseline.keys() - {'id', 'display_name', 'engine', 'arguments', 'artifacts', 'catalog'}:
            with self.subTest(field=key):
                self.assertEqual(candidate[key], baseline[key])
        # Same layer-split placement, overrides, batch sizes, and draft settings as daytime. Tensor
        # mode is deliberately not used: 31 of 48 expert blocks live on the CPU, and the meta device
        # used by --split-mode tensor cannot offload CPU-resident expert weights for prompt batches.
        self.assertEqual(candidate['arguments'], {**baseline['arguments'], '--alias': 'qwen3.8-flash-next-ad4.27-next'})
        self.assertNotIn('--split-mode', candidate['arguments'])
        # Upstream qwen4exp MTP cannot borrow the target's tensors, so the shared-Q4_K_M head is
        # replaced by the mainline-converted, self-contained ggml-org Q4_0 head.
        head = {'path': 'llm/ggml-org/Qwen3.8-Flash-Next-GGUF/revisions/052beeaca7bec4a303e59cc7bc630c4f3a1b845d/'
                'mtp-Qwen3.8-Flash-Next-Q4_0.gguf', 'target': '/weights/mtp.gguf',
                'bytes': 2199652640, 'sha256': 'd484e6e541a36b714976f72f7d1a86ae2cc450cff110b81e3656512784abd1c6'}
        self.assertEqual(candidate['artifacts'][:-1], baseline['artifacts'][:-1])  # 33 shards and projector
        self.assertEqual(baseline['artifacts'][-1]['target'], '/weights/mtp.gguf')
        self.assertEqual(candidate['artifacts'][-1], head)
        expected_catalog = copy.deepcopy(baseline['catalog'])
        expected_catalog['mtp'].update(revision='052beeaca7bec4a303e59cc7bc630c4f3a1b845d', quantization='Q4_0')
        expected_catalog['capability_profile']['name'] = 'qwen38-flash-next-ad427-next-engine-128k'
        expected_catalog['deployment_warnings'] = [
            'Newer pinned llama.cpp engine (qwen38-dual-836d571, 836d571 of 2026-10-03) with the mainline '
            'ggml-org Q4_0 MTP head: load, VRAM headroom at full 128K context, and throughput have not been '
            'qualified on the production GPUs.',
            '128K is the total prompt, history, reasoning, and output capacity.']
        self.assertEqual(candidate['catalog'], expected_catalog)

        original = render(ROOT / 'config', BASELINE['host'], 'daytime')
        variant = render(ROOT / 'config', BASELINE['host'], 'daytime-flash-next')
        original_cfg = original['compose']['services']['coding']
        variant_cfg = copy.deepcopy(variant['compose']['services']['coding'])
        self.assertEqual(variant_cfg['image'], engines['qwen38-dual-836d571']['tag'])
        variant_cfg['image'] = original_cfg['image']
        argv, original_argv = variant_cfg['command'], original_cfg['command']
        self.assertEqual(argv[argv.index('--split-mode') + 1], 'layer')
        self.assertEqual(argv[argv.index('--alias') + 1], 'qwen3.8-flash-next-ad4.27-next')
        argv[argv.index('--alias') + 1] = original_argv[original_argv.index('--alias') + 1]
        head_source = str(Path(BASELINE['host']['model_root']) / head['path'])
        mount = next(m for m in variant_cfg['volumes'] if m['target'] == '/weights/mtp.gguf')
        self.assertEqual(mount['source'], head_source)
        mount['source'] = next(m['source'] for m in original_cfg['volumes'] if m['target'] == '/weights/mtp.gguf')
        self.assertEqual(variant_cfg, original_cfg)  # same devices, overrides, draft on CUDA0, KV types, other argv
        self.assertEqual(variant['compose']['services']['everyday'], original['compose']['services']['everyday'])
        self.assertEqual(service_engine(variant, 'coding'), engines['qwen38-dual-836d571'])
        self.assertEqual(service_engine(variant, 'everyday'), service_engine(original, 'everyday'))
        catalog = variant['catalog']
        self.assertEqual(catalog['model'], 'qwen3.8-flash-next-ad4.27-next')
        self.assertEqual(catalog['backend_revision'], '836d57176dc699a726c55418e4f96b8ca628e1bf')
        self.assertEqual(catalog['context_length'], 131072)
        self.assertEqual(catalog['kv_cache'], original['catalog']['kv_cache'])
        self.assertEqual(catalog['aliases'], original['catalog']['aliases'])
        self.assertEqual(catalog['display_name'], 'FlashNext Next Engine (128K)')
        self.assertEqual(catalog['mtp'], {**original['catalog']['mtp'], 'model_path': head_source,
            'revision': '052beeaca7bec4a303e59cc7bc630c4f3a1b845d', 'quantization': 'Q4_0'})
        self.assertEqual((catalog['mtp']['device'], catalog['mtp']['max_draft_tokens']), ('CUDA0', 2))
        self.assertEqual(catalog['models'][1], original['catalog']['models'][1])
        self.assertEqual(len(variant['artifacts']), 37)

    def test_flash_solo_is_flash_next_on_all_four_text_gpus_without_nighttime(self):
        baseline = read(ROOT / 'config/profiles/daytime-flash-next.json')
        candidate = read(ROOT / 'config/profiles/daytime-flash-solo.json')
        self.assertEqual(candidate.keys(), baseline.keys() | {'exclusive'})
        self.assertIs(candidate['exclusive'], True)
        self.assertEqual((candidate['id'], candidate['display_name'], candidate['gpu_group']),
            ('daytime-flash-solo', 'FlashNext Solo 4-GPU', 'all'))
        # Same engine, container, port, context, weights, projector, and MTP head as the paired
        # next-engine profile; only placement changes.
        for key in baseline.keys() - {'id', 'display_name', 'gpu_group', 'argument_order', 'arguments', 'catalog'}:
            with self.subTest(field=key):
                self.assertEqual(candidate[key], baseline[key])
        # Every expert block fits in VRAM, so the CPU overrides and hand split go away and
        # automatic fitting distributes the 48 layers across the four text GPUs.
        removed = ('--n-gpu-layers', '--tensor-split', '--override-tensor')
        order = [x for x in baseline['argument_order'] if x not in removed]
        order.insert(order.index('--fit') + 1, '--fit-target')
        self.assertEqual(candidate['argument_order'], order)
        self.assertEqual(candidate['arguments'], {**{k: v for k, v in baseline['arguments'].items() if k not in removed},
            '--alias': 'qwen3.8-flash-next-ad4.27-solo', '--device': 'CUDA1,CUDA2,CUDA3,CUDA0',
            '--ubatch-size': '1024', '--fit': 'on', '--fit-target': '1024', '--spec-draft-device': 'CUDA4'})
        expected_catalog = copy.deepcopy(baseline['catalog'])
        expected_catalog['mtp']['device'] = 'CUDA4'
        expected_catalog['capability_profile']['name'] = 'qwen38-flash-next-ad427-solo-128k'
        expected_catalog['deployment_warnings'] = candidate['catalog']['deployment_warnings']
        self.assertEqual(candidate['catalog'], expected_catalog)
        self.assertIn('Nighttime is stopped', candidate['catalog']['deployment_warnings'][0])

        host = vision_host()
        bundle = render(ROOT / 'config', host, 'daytime-flash-solo')
        self.assertEqual(list(bundle['compose']['services']), ['coding'])
        cfg = bundle['compose']['services']['coding']
        text = [*host['gpu_ids']['daytime'], *host['gpu_ids']['nighttime']]  # CUDA0-3: 3090, 4080S, 4080, 3080 Ti
        self.assertEqual(cfg['deploy']['resources']['reservations']['devices'][0]['device_ids'], [*text, VISION])
        self.assertEqual(cfg['environment'], {'CUDA_VISIBLE_DEVICES': ','.join([*text, VISION])})
        argv = cfg['command']
        for flag, value in {'--split-mode': 'layer', '--device': 'CUDA1,CUDA2,CUDA3,CUDA0', '--fit': 'on',
                '--fit-target': '1024', '--ubatch-size': '1024', '--batch-size': '2048', '--ctx-size': '131072',
                '--mmproj-device': 'CUDA4', '--spec-draft-device': 'CUDA4', '--spec-draft-n-max': '2',
                '--cache-type-k': 'q8_0', '--cache-type-v': 'q8_0', '--lazy-mode': 'on'}.items():
            with self.subTest(flag=flag):
                self.assertEqual(argv.count(flag), 1)
                self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertIn('--mmproj-offload', argv)
        for flag in (*removed, '--no-mmproj-offload'):
            self.assertNotIn(flag, argv)
        paired = render(ROOT / 'config', host, 'daytime-flash-next')['compose']['services']['coding']
        for key in cfg.keys() - {'command', 'deploy', 'environment'}:
            with self.subTest(compose=key):
                self.assertEqual(cfg[key], paired[key])  # image, container, ports, mounts, network, health
        catalog = bundle['catalog']
        self.assertEqual(len(catalog['models']), 1)
        model = catalog['models'][0]
        self.assertEqual(catalog['default_model'], model['model'])
        for key, value in {'model': 'qwen3.8-flash-next-ad4.27-solo', 'display_name': 'FlashNext Solo 4-GPU (128K)',
                'exclusive': True, 'vision_gpu_shared': False, 'vision_device': 'CUDA4', 'vision_gpu_uuid': VISION,
                'mmproj_offload': 'gpu', 'text_gpu_uuids': text, 'gpu_uuids': [*text, VISION],
                'cuda_visible_devices': [*text, VISION], 'fit_target': 'on', 'context_length': 131072,
                'aliases': ['local-active', 'daytime'],
                'backend_revision': '836d57176dc699a726c55418e4f96b8ca628e1bf'}.items():
            with self.subTest(catalog=key):
                self.assertEqual(model[key], value)
        self.assertEqual((model['mtp']['device'], model['mtp']['max_draft_tokens']), ('CUDA4', 2))
        self.assertEqual([a['path'] for a in bundle['artifacts']], [a['path'] for a in baseline['artifacts']])

    def test_tuned_solo_profiles_combine_the_measured_winners(self):
        # Measured one change at a time against daytime-flash-solo on a fixed session replay; these settings
        # each helped and are combined. See docs/deployment.md for the numbers and the rejected experiments.
        engines = read(ROOT / 'config/shared.json')['engines']
        engine = engines['qwen38-dual-43fe9c6']
        self.assertEqual((engine['revision'], engine['tag'], engine['image_id']), ('43fe9c64281ef735046adc025e9e7559a1f659a5',
            'local/llama.cpp:qwen38-dual-43fe9c64281e', 'sha256:5a25a54554f2ba945c0761ad67b329ae9cfa0c18cacf1aa890d549d03a9b4546'))
        for key in ('repository', 'cuda_architectures', 'build_base_digest', 'runtime_base_digest'):
            self.assertEqual(engine[key], engines['qwen38-dual-836d571'][key])  # same Dockerfile inputs, newer source
        solo = read(ROOT / 'config/profiles/daytime-flash-solo.json')
        order = [x for x in solo['argument_order'] if x != '--fit-target']
        order[order.index('--fit') + 1:order.index('--fit') + 1] = ['--n-gpu-layers', '--tensor-split', '--override-tensor']
        for name, depth in (('daytime-flash-solo-tuned', 2), ('daytime-flash-solo-tuned-mtp3', 3)):
            with self.subTest(profile=name):
                tuned = read(ROOT / 'config/profiles' / (name + '.json'))
                for key in ('role', 'container_name', 'host_port', 'gpu_group', 'exclusive', 'context_tokens', 'artifacts', 'container'):
                    self.assertEqual(tuned[key], solo[key])
                self.assertEqual(tuned['engine'], 'qwen38-dual-43fe9c6')
                self.assertEqual(tuned['argument_order'], order)
                expected = {k: v for k, v in solo['arguments'].items() if k != '--fit-target'}
                expected.update({'--alias': 'qwen3.8-flash-next-ad4.27-solo-' + name.removeprefix('daytime-flash-solo-'),
                    # A fixed whole-layer split (48 blocks + output) replaces automatic fitting, which split two layers
                    # across devices. The input-embedding override only restates its default CPU placement; any override
                    # keeps pipeline parallelism, whose larger buffers do not fit at the 1024 microbatch, disabled.
                    '--fit': 'off', '--n-gpu-layers': 'all', '--tensor-split': '11,12,8,18',
                    '--override-tensor': ['token_embd\\.weight=CPU'],
                    '--lazy-mode': 'off', '--cache-type-k': 'f16', '--cache-type-v': 'f16', '--spec-draft-n-max': str(depth),
                    # Engine 43fe9c6 adds DUP/SET_ROWS input nodes to the CPU split; with 18 OpenMP threads the team
                    # busy-waited between steps (about 9 cores at the same speed), so one host thread runs that split.
                    '--threads': '1', '--threads-batch': '1'})
                self.assertEqual(tuned['arguments'], expected)
                self.assertEqual(sum(int(x) for x in expected['--tensor-split'].split(',')), 49)
                bundle = render(ROOT / 'config', vision_host(), name)
                cfg = bundle['compose']['services']['coding']
                self.assertEqual(cfg['image'], engine['tag'])
                self.assertEqual(cfg['command'][cfg['command'].index('--device') + 1], 'CUDA1,CUDA2,CUDA3,CUDA0')
                model = bundle['catalog']['models'][0]
                self.assertEqual((model['kv_cache']['key_type'], model['kv_cache']['value_type']), ('f16', 'f16'))
                self.assertEqual((model['mtp']['max_draft_tokens'], model['mtp']['device'], model['fit_target']), (depth, 'CUDA4', 'off'))
                self.assertEqual(model['backend_revision'], engine['revision'])
                self.assertEqual(list(bundle['compose']['services']), ['coding'])

    def test_two_slot_and_160k_experiments(self):
        tuned = read(ROOT / 'config/profiles/daytime-flash-solo-tuned-mtp3.json')
        two = read(ROOT / 'config/profiles/daytime-flash-solo-tuned-mtp3-2slot.json')
        self.assertEqual(two['parallel_slots'], 2)
        self.assertEqual(two['arguments'], {**tuned['arguments'], '--alias': 'qwen3.8-flash-next-ad4.27-solo-tuned-mtp3-2slot',
            '--cache-type-k': 'q8_0', '--cache-type-v': 'q8_0'})
        bundle = render(ROOT / 'config', vision_host(), 'daytime-flash-solo-tuned-mtp3-2slot')
        argv = bundle['compose']['services']['coding']['command']
        # Two 128K slots: the KV pool is sized for both, each request still sees 128K.
        self.assertEqual([argv[argv.index(f) + 1] for f in ('--parallel', '--ctx-size', '--kv-unified-per-slot')], ['2', '262144', '131072'])
        model = bundle['catalog']['models'][0]
        self.assertEqual((model['context_length'], model['total_context_length'], model['max_active_requests'],
            model['backend_parallel_slots']), (131072, 131072, 1, 2))
        self.assertEqual(bundle['manifest']['services'][0]['parallel_slots'], 2)
        wide = read(ROOT / 'config/profiles/daytime-flash-solo-tuned-mtp3-160k.json')
        self.assertEqual((wide['context_tokens'], wide['arguments']), (163840, {**tuned['arguments'], '--alias': 'qwen3.8-flash-next-ad4.27-solo-tuned-mtp3-160k'}))
        argv = render(ROOT / 'config', vision_host(), 'daytime-flash-solo-tuned-mtp3-160k')['compose']['services']['coding']['command']
        self.assertEqual([argv[argv.index(f) + 1] for f in ('--parallel', '--ctx-size', '--kv-unified-per-slot')], ['1', '163840', '163840'])
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            for name, change, message in [('daytime-flash-solo-tuned-mtp3-2slot', {'parallel_slots': 3}, 'must be 1 or 2'),
                    ('daytime', {'parallel_slots': 2}, 'only an exclusive profile')]:
                target = config / 'profiles' / (name + '.json'); original = target.read_text()
                definition = read(target); definition.update(change); target.write_text(json.dumps(definition))
                with self.subTest(profile=name), self.assertRaisesRegex(RuntimeError, message):
                    render(config, vision_host(), name)
                target.write_text(original)

    def test_exclusive_profiles_require_vision_gpu_and_keep_the_model_on_text_gpus(self):
        with self.assertRaisesRegex(RuntimeError, 'configured vision GPU'):
            render(ROOT / 'config', BASELINE['host'], 'daytime-flash-solo')
        host = vision_host(); host['gpu_ids']['nighttime'][0] = host['gpu_ids']['daytime'][0]
        with self.assertRaisesRegex(RuntimeError, 'four distinct'):
            render(ROOT / 'config', host, 'daytime-flash-solo')
        host = vision_host(); host['gpu_ids']['nighttime'] = host['gpu_ids']['nighttime'][:1]
        with self.assertRaisesRegex(RuntimeError, 'two real GPU UUIDs'):
            render(ROOT / 'config', host, 'daytime-flash-solo')
        host = vision_host(); host['vision_gpu_id'] = 'GPU-night-2'
        with self.assertRaisesRegex(RuntimeError, 'full GPU UUID'):
            render(ROOT / 'config', host, 'daytime-flash-solo')
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            path = config / 'profiles/daytime-flash-solo.json'; original = read(path)
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
                    render(config, vision_host(), 'daytime-flash-solo')
            for name, change, message in [
                    ('daytime-flash-solo', {'exclusive': 'yes'}, 'true or false'),
                    ('daytime-flash-solo', {'gpu_group': 'daytime'}, '"all" GPU group'),
                    ('daytime', {'gpu_group': 'all'}, 'unknown GPU group'),
                    ('nighttime', {'exclusive': True}, '"all" GPU group')]:
                shutil.rmtree(config); shutil.copytree(ROOT / 'config', config)
                target = config / 'profiles' / (name + '.json')
                definition = read(target); definition.update(change); target.write_text(json.dumps(definition))
                with self.subTest(profile=name, change=change), self.assertRaisesRegex(RuntimeError, message):
                    render(config, vision_host(), 'daytime-flash-solo' if name == 'daytime-flash-solo' else 'daytime')

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

    def test_every_qwen_backend_uses_the_single_pinned_engine(self):
        engines = read(ROOT / 'config/shared.json')['engines']
        # The original qwen38-dual (8ea2902) engine is retired; Flash-Next keeps its own engine, and the
        # daytime-flash-next experiment runs the same Flash-Next weights on the 836d571 engine.
        self.assertEqual(set(engines), {'qwen38-dual-836d571', 'qwen38-dual-43fe9c6', 'flash-next-mtp'})
        engine = engines['qwen38-dual-836d571']
        self.assertEqual(engine['revision'], '836d57176dc699a726c55418e4f96b8ca628e1bf')
        self.assertEqual(engine['tag'], 'local/llama.cpp:qwen38-dual-836d57176dc6')
        self.assertEqual(engine['image_id'], 'sha256:9f7179f568e4aa1004c8af9b613b65417e6f0e451f8cfbc774d6b0d0f50a0ecd')
        expected = {'daytime': 'flash-next-mtp', 'daytime-flash-f16': 'flash-next-mtp',
            'daytime-27b': 'qwen38-dual-836d571', 'daytime-27b-tensor-next': 'qwen38-dual-836d571',
            'daytime-27b-q6k-tensor-next': 'qwen38-dual-836d571', 'daytime-flash-next': 'qwen38-dual-836d571',
            'daytime-flash-solo': 'qwen38-dual-836d571', 'daytime-flash-solo-tuned': 'qwen38-dual-43fe9c6',
            'daytime-flash-solo-tuned-mtp3': 'qwen38-dual-43fe9c6', 'daytime-flash-solo-tuned-mtp3-2slot': 'qwen38-dual-43fe9c6',
            'daytime-flash-solo-tuned-mtp3-160k': 'qwen38-dual-43fe9c6',
            NIGHTTIME_PROFILE: 'qwen38-dual-836d571'}
        self.assertEqual(set(expected), {*DAYTIME_PROFILES, NIGHTTIME_PROFILE})
        for name, engine_name in expected.items():
            with self.subTest(profile=name):
                self.assertEqual(read(ROOT / 'config/profiles' / (name + '.json'))['engine'], engine_name)
        for name in ('daytime-27b', 'daytime-27b-tensor-next'):
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

    def assert_tensor_profile(self, candidate_id, display, alias, capability, quant, main=None):
        baseline = read(ROOT / 'config/profiles/daytime-27b.json')
        candidate = read(ROOT / 'config/profiles' / (candidate_id + '.json'))
        self.assertEqual(candidate.keys(), baseline.keys())
        self.assertEqual(candidate['id'], candidate_id)
        self.assertEqual(candidate['display_name'], display)
        # Same engine, context, GPU group, container, and argument order as the saved Q8 recovery profile.
        for key in baseline.keys() - {'id', 'display_name', 'arguments', 'artifacts', 'catalog'}:
            with self.subTest(field=key):
                self.assertEqual(candidate[key], baseline[key])
        # Tensor-parallel placement: split mode tensor with bandwidth-balanced 55,45 proportions.
        self.assertEqual(candidate['arguments'], {**baseline['arguments'], '--alias': alias,
            '--split-mode': 'tensor', '--tensor-split': '55,45'})
        self.assertIn('--split-mode', candidate['argument_order'])
        if main is None:
            self.assertEqual(candidate['artifacts'], baseline['artifacts'])
        else:
            self.assertEqual(candidate['artifacts'][0], main)
            self.assertEqual(candidate['artifacts'][1:], baseline['artifacts'][1:])  # Q4_0 MTP draft and projector
        warnings = ['Newer pinned llama.cpp engine (qwen38-dual-836d571, 836d571 of 2026-10-03) with ' + quant
            + ' tensor-parallel placement: load, VRAM headroom at full 160K context, and throughput have not '
            'been qualified on the production GPUs.',
            '160K is the total prompt, history, reasoning, and output capacity.']
        expected_catalog = copy.deepcopy(baseline['catalog'])
        expected_catalog['capability_profile']['name'] = capability
        expected_catalog['deployment_warnings'] = warnings
        if main is not None:
            expected_catalog['quantization'] = quant
        self.assertEqual(candidate['catalog'], expected_catalog)

        original = render(ROOT / 'config', BASELINE['host'], 'daytime-27b')
        variant = render(ROOT / 'config', BASELINE['host'], candidate_id)
        original_cfg = original['compose']['services']['coding']
        variant_cfg = copy.deepcopy(variant['compose']['services']['coding'])
        argv, original_argv = variant_cfg['command'], original_cfg['command']
        self.assertEqual(argv.count('--split-mode'), 1)
        self.assertEqual(original_argv[original_argv.index('--split-mode') + 1], 'layer')
        for flag, value in {'--alias': alias, '--split-mode': 'tensor', '--tensor-split': '55,45'}.items():
            with self.subTest(flag=flag):
                self.assertEqual(argv[argv.index(flag) + 1], value)
                argv[argv.index(flag) + 1] = original_argv[original_argv.index(flag) + 1]
        if main is not None:
            mount = next(m for m in variant_cfg['volumes'] if m['target'] == '/weights/main.gguf')
            self.assertEqual(mount['source'], str(Path(BASELINE['host']['model_root']) / main['path']))
            mount['source'] = next(m['source'] for m in original_cfg['volumes'] if m['target'] == '/weights/main.gguf')
        self.assertEqual(variant_cfg, original_cfg)  # identical image, devices, draft on CUDA1, KV types, other argv
        self.assertEqual(variant['compose']['services']['everyday'], original['compose']['services']['everyday'])
        self.assertEqual(service_engine(variant, 'coding'), service_engine(original, 'coding'))
        catalog = variant['catalog']
        self.assertEqual(catalog['model'], alias)
        self.assertEqual(catalog['quantization'], quant)
        self.assertEqual(catalog['backend_revision'], '836d57176dc699a726c55418e4f96b8ca628e1bf')
        self.assertEqual(catalog['context_length'], 163840)
        self.assertEqual(catalog['kv_cache'], original['catalog']['kv_cache'])
        self.assertEqual(catalog['mtp'], original['catalog']['mtp'])  # same Q4_0 draft on CUDA1
        self.assertEqual(catalog['aliases'], original['catalog']['aliases'])
        self.assertEqual(catalog['display_name'], display + ' (160K)')
        self.assertEqual(catalog['deployment_warnings'], warnings)
        self.assertEqual(catalog['models'][1], original['catalog']['models'][1])

    def test_q8_tensor_next_is_daytime_27b_with_tensor_parallel_placement(self):
        self.assert_tensor_profile('daytime-27b-tensor-next', 'Daytime-27B Q8 Tensor Next',
            'qwen3.8-27b-q8_0-tensor-next', 'qwen38-27b-golden-vision-tools-tensor-next', 'Q8_0')

    def test_q6k_tensor_next_is_daytime_27b_with_q6_weights_and_tensor_parallel_placement(self):
        main = {'path': 'llm/Qwen3.8-27B-GGUF/fallbacks/4ca720788d1e01f1bff70c033e0d0028fd02e502/'
                'Qwen3.8-27B-UD-Q6_K_XL.gguf', 'target': '/weights/main.gguf',
                'bytes': 25299061664, 'sha256': '701d8fa9ed214ab21bfc130cd2a7df19ca89bbef7713e2dfb19f3c63696aa917'}
        self.assert_tensor_profile('daytime-27b-q6k-tensor-next', 'Daytime-27B Q6_K Tensor Next',
            'qwen3.8-27b-ud-q6_k_xl-tensor-next', 'qwen38-27b-golden-vision-tools-q6k-tensor-next', 'UD-Q6_K_XL', main)

    def test_context_ceiling_is_the_router_contract_native_window(self):
        from runtime.config import ROUTER_CONTEXT_LIMIT
        self.assertEqual(ROUTER_CONTEXT_LIMIT, 262144)
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            path = config / 'profiles/daytime-27b.json'
            definition = read(path)
            for tokens in (262145, 0, '262144'):
                with self.subTest(tokens=tokens):
                    definition['context_tokens'] = tokens
                    path.write_text(json.dumps(definition))
                    with self.assertRaisesRegex(RuntimeError, 'router contract: 1–262144 tokens'):
                        render(config, BASELINE['host'], 'daytime-27b')
                    with self.assertRaisesRegex(RuntimeError, 'router contract'):
                        available_profiles(config)
            definition['context_tokens'] = 262144
            path.write_text(json.dumps(definition))
            self.assertEqual(render(config, BASELINE['host'], 'daytime-27b')['catalog']['context_length'], 262144)

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
            ('daytime', 'daytime-27b', 'daytime-flash-f16', 'daytime-27b-tensor-next',
             'daytime-27b-q6k-tensor-next', 'daytime-flash-next', 'daytime-flash-solo', 'daytime-flash-solo-tuned',
             'daytime-flash-solo-tuned-mtp3', 'daytime-flash-solo-tuned-mtp3-2slot', 'daytime-flash-solo-tuned-mtp3-160k'))
        self.assertEqual(EXCLUSIVE, DAYTIME_PROFILES[6:])
        self.assertEqual([x['profile'] for x in registry['selectable']], list(DAYTIME_PROFILES))
        self.assertEqual([x['profile'] for x in registry['always_included']], [NIGHTTIME_PROFILE])
        night = registry['always_included'][0]
        baseline_night = render(ROOT / 'config', BASELINE['host'], 'daytime')['compose']['services']['everyday']
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
                self.registry_with('daytime-27b', None)
        for name, mutate, expected in [
                ('daytime', lambda definition: definition.update(engine='no-such-engine'), 'unknown engine'),
                ('nighttime', lambda definition: definition.update(id='renamed'), 'differs from its filename'),
                ('daytime', lambda definition: definition.update(context_tokens=262145), 'router contract')]:
            with self.subTest(profile=name):
                with self.assertRaisesRegex(RuntimeError, expected):
                    self.registry_with(name, mutate)

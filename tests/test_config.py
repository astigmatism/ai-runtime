import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from runtime.config import (CONFIGURATIONS, CONTAINER_CONTRACT, DAYTIME_PROFILES, DEFAULT_PROFILE, EVERYDAY_PROFILE,
    EVERYDAY_PROFILES, PROFILE_ALIASES, RETIRED_PROFILES, available_profiles, digest, engine_interface,
    merge_settings, read, render, resolve_profile, service_engine, text_gpus, unknown_profile)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = read(ROOT / 'tests/fixtures/legacy-fingerprints.json')
VISION = 'GPU-465a9a1e-fcb9-27f2-8f6a-cca1c30de981'
# A solo configuration reserves all four text GPUs and loads no second resident; a paired one takes
# one GPU pair and loads the Everyday resident beside it.
EXCLUSIVE = tuple(n for n in CONFIGURATIONS if read(ROOT / 'config/profiles' / (n + '.json')).get('exclusive'))
PAIRED = tuple(n for n in CONFIGURATIONS if n not in EXCLUSIVE)
ENGINES = read(ROOT / 'config/shared.json')['engines']
# The one configuration that is not llama.cpp, and the only solo one. Its engine is configured by the
# environment, so the catalog fields that describe a flag the engine was handed (KV types, --fit, the
# host-RAM prompt cache budget) are not published for it.
SOLO = 'flash-next-strata-solo'
# The alias every configuration's coding entry carries for the legacy Ollama surface. It is the
# compatibility ID, not the one a contract client sends, so it is never the only alias an entry has.
COMPATIBILITY_ALIAS = 'local-active'
# The service IDs clients address residents by. Every configuration publishes all three: a solo
# configuration answers `nighttime` from its own model, which is what keeps a client that only ever
# sends one of them working whichever configuration is selected.
RESIDENT_SERVICES = ('daytime', COMPATIBILITY_ALIAS, 'nighttime')


def interface_of(profile):
    return engine_interface(ENGINES[read(ROOT / 'config/profiles' / (profile + '.json'))['engine']])


def vision_host(host=None):
    host = copy.deepcopy(host or BASELINE['host'])
    host.update(vision_gpu_id=VISION, vision_gpu_name='RTX 3080', vision_cuda_device=4,
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
        self.assertEqual(set(recorded), set(CONFIGURATIONS))
        for name, expected in recorded.items():
            with self.subTest(profile=name, previous=expected['previous_id']):
                compose = render(ROOT / 'config', vision_host(), name)['compose']
                self.assertEqual(digest(compose['services']['coding']), expected['coding_service_sha256'])
                if name in EXCLUSIVE:
                    self.assertEqual(digest(compose), expected['compose_sha256'])

    def test_context_edit_updates_both_launch_flags_manifest_and_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            path = config / 'profiles/qwen27b-q8-pair.json'
            definition = read(path); definition['context_tokens'] = 98304
            path.write_text(json.dumps(definition))
            result = render(config, BASELINE['host'], 'qwen27b-q8-pair')
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
            render(ROOT / 'config', host, 'qwen27b-q8-pair')

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
                render(config, BASELINE['host'], 'qwen27b-q8-pair')

    def test_no_qualification_is_synthesized(self):
        result = render(ROOT / 'config', BASELINE['host'], 'qwen27b-q8-pair')
        def check_keys(value):
            if isinstance(value, dict):
                self.assertTrue({'qualified', 'qualification', 'qualification_receipt'}.isdisjoint(value))
                for child in value.values(): check_keys(child)
            elif isinstance(value, list):
                for child in value: check_keys(child)
        check_keys(result)
        # The imported capability name is historical metadata, not a generated receipt.
        self.assertEqual(result['catalog']['capability_profile'],
            read(ROOT / 'config/profiles/qwen27b-q8-pair.json')['catalog']['capability_profile'])
        self.assertEqual(len(result['artifacts']), 5)  # 27B weights, MTP draft, projector; Nighttime weights, projector
        self.assertEqual(result['catalog']['aliases'], ['local-active', 'daytime'])

    def test_27b_profiles_run_tensor_parallel_on_the_daytime_pair_beside_nighttime(self):
        engine = read(ROOT / 'config/shared.json')['engines']['qwen38-dual-836d571']
        q8 = render(ROOT / 'config', BASELINE['host'], 'qwen27b-q8-pair')
        q6k = render(ROOT / 'config', BASELINE['host'], 'qwen27b-q6k-pair')
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

    def test_the_everyday_resident_runs_the_base_mtp3_head_on_the_shared_vision_gpu(self):
        night = read(ROOT / 'config/profiles' / (EVERYDAY_PROFILE + '.json'))
        spec = {'--spec-type': 'draft-mtp', '--spec-draft-model': '/weights/mtp.gguf', '--spec-draft-n-max': '3',
            '--spec-draft-ngl': 'all', '--spec-draft-device': 'CUDA2', '--spec-draft-type-k': 'q8_0', '--spec-draft-type-v': 'q8_0'}
        self.assertEqual({k: night['arguments'][k] for k in spec}, spec)
        self.assertEqual((night['display_name'], night['arguments']['--alias'], night['draft_on_vision_gpu']),
            ('Qwen3.8 27B Abliterated Q6_K', 'qwen3.8-27b-abliterated-q6_k', True))
        # At 128K, speculative decoding needed another 564 MiB on the RTX 4080 than the pair had free.
        self.assertEqual(night['context_tokens'], 98304)
        head = next(a for a in read(ROOT / 'config/profiles/qwen27b-q6k-pair.json')['artifacts'] if a['target'] == '/weights/mtp.gguf')
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
        # Without a vision GPU there is nowhere to put the draft: the Everyday resident runs without
        # it rather than failing.
        cpu = render(ROOT / 'config', BASELINE['host'], 'qwen27b-q8-pair')
        argv = cpu['compose']['services']['everyday']['command']
        self.assertFalse([x for x in argv if x.startswith('--spec-')])
        self.assertNotIn('/weights/mtp.gguf', [m['target'] for m in cpu['compose']['services']['everyday']['volumes']])
        self.assertEqual(cpu['catalog']['models'][1]['mtp'], {'enabled': False})
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config_dir)
            for name, change, message, target in [
                    (EVERYDAY_PROFILE, {'draft_on_vision_gpu': False}, 'Draft model must remain on a text GPU', 'qwen27b-q8-pair'),
                    ('qwen27b-q8-pair', {'everyday': 'everyday-retired'}, 'unknown everyday resident profile', 'qwen27b-q8-pair'),
                    (SOLO, {'everyday': EVERYDAY_PROFILE}, 'runs no second resident', SOLO)]:
                path = config_dir / 'profiles' / (name + '.json'); original = path.read_text()
                definition = read(path); definition.update(change); path.write_text(json.dumps(definition))
                with self.subTest(profile=name), self.assertRaisesRegex(RuntimeError, message):
                    render(config_dir, host, target)
                path.write_text(original)

    def test_exclusive_configurations_keep_the_model_on_text_gpus(self):
        # The solo configuration in the house is Strata: it is handed an environment, not an argument
        # vector, and it puts nothing on the vision card. The llama.cpp placement rules below still
        # guard the next four-card model, so the exemplar that exercises them is built here from the
        # 27B pair rather than the coverage retiring along with a profile.
        text = [*BASELINE['host']['gpu_ids']['daytime'], *BASELINE['host']['gpu_ids']['nighttime']]
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'

            def solo_llama(host=None, **change):
                shutil.rmtree(config, ignore_errors=True); shutil.copytree(ROOT / 'config', config)
                path = config / 'profiles/qwen27b-q8-pair.json'
                definition = read(path)
                definition.update(exclusive=True, gpu_group='all')
                definition['arguments'].update({'--device': 'CUDA0,CUDA1,CUDA2,CUDA3',
                    '--tensor-split': '1,1,1,1', '--spec-draft-device': 'CUDA4'})
                definition['arguments'].update(change)
                path.write_text(json.dumps(definition))
                return render(config, vision_host() if host is None else host, 'qwen27b-q8-pair')

            reserved = solo_llama()['compose']['services']['coding']['deploy']['resources']['reservations']['devices'][0]
            self.assertEqual(reserved['device_ids'], [*text, VISION])
            # With no second resident the vision card is not shared, so it is reserved last and the
            # language model never lands on it.
            argv = solo_llama()['compose']['services']['coding']['command']
            self.assertEqual(argv[argv.index('--device') + 1], 'CUDA0,CUDA1,CUDA2,CUDA3')
            for change, message in [({'--device': 'CUDA0,CUDA1,CUDA2'}, 'exactly once'),
                    ({'--device': 'CUDA0,CUDA1,CUDA2,CUDA4'}, 'exactly once'),
                    ({'--device': 'CUDA0,CUDA1,CUDA2,CUDA2'}, 'exactly once'),
                    ({'--tensor-split': '30,30,40'}, 'one value per text GPU'),
                    ({'--spec-draft-device': 'CUDA5'}, 'reserved GPU')]:
                with self.subTest(arguments=change), self.assertRaisesRegex(RuntimeError, message):
                    solo_llama(**change)
            # The projector and the draft live on the one card no text model uses, so a host that was
            # never given one cannot run a solo llama.cpp configuration at all.
            with self.assertRaisesRegex(RuntimeError, 'configured vision GPU'):
                solo_llama(BASELINE['host'])
            thin = vision_host(); thin['vision_gpu_id'] = 'GPU-night-2'
            with self.assertRaisesRegex(RuntimeError, 'full GPU UUID'):
                solo_llama(thin)
            for name, change, message in [
                    (SOLO, {'exclusive': 'yes'}, 'true or false'),
                    (SOLO, {'gpu_group': 'daytime'}, '"all" GPU group'),
                    ('qwen27b-q8-pair', {'gpu_group': 'all'}, 'unknown GPU group'),
                    (EVERYDAY_PROFILE, {'exclusive': True}, '"all" GPU group')]:
                shutil.rmtree(config); shutil.copytree(ROOT / 'config', config)
                target = config / 'profiles' / (name + '.json')
                definition = read(target); definition.update(change); target.write_text(json.dumps(definition))
                with self.subTest(profile=name, change=change), self.assertRaisesRegex(RuntimeError, message):
                    render(config, vision_host(), name if name == SOLO else 'qwen27b-q8-pair')

    def test_a_solo_configuration_needs_every_text_card_named(self):
        # Both text pairs are reserved, so a host that cannot name four distinct cards is refused
        # before a drain is taken rather than after a backend has been half-built on three of them.
        host = vision_host(); host['gpu_ids']['nighttime'][0] = host['gpu_ids']['daytime'][0]
        with self.assertRaisesRegex(RuntimeError, 'four distinct'):
            render(ROOT / 'config', host, SOLO)
        host = vision_host(); host['gpu_ids']['nighttime'] = host['gpu_ids']['nighttime'][:1]
        with self.assertRaisesRegex(RuntimeError, 'two real GPU UUIDs'):
            render(ROOT / 'config', host, SOLO)

    def test_ram_prompt_cache_is_shared_for_daytime_and_lowered_only_for_nighttime(self):
        shared = read(ROOT / 'config/shared.json')
        self.assertEqual(shared['arguments']['--cache-ram'], '49152')
        # Every Daytime profile inherits the shared 48 GiB lazy LRU cap. Nighttime alone overrides
        # it to 24 GiB: on the 123 GiB host its cache had grown to ~50 GB with ~8 GB swapped and
        # swap exhausted, so the worst-case combined pair is bounded at 72 GiB instead of 96 GiB.
        for name in CONFIGURATIONS:
            with self.subTest(profile=name):
                definition = read(ROOT / 'config/profiles' / (name + '.json'))
                self.assertNotIn('--cache-ram', definition['arguments'])
        night = read(ROOT / 'config/profiles' / (EVERYDAY_PROFILE + '.json'))
        self.assertEqual(night['arguments']['--cache-ram'], '24576')
        self.assertIn('--cache-ram', night['argument_order'])
        # Only the flag-configured engines are given a cache budget at all, so only they publish one.
        for name in [n for n in CONFIGURATIONS if interface_of(n) == 'argv']:
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
        # The Flash-Next branch engine retired with the llama.cpp Flash-Next configurations; the 27B
        # pair and the Everyday resident use 836d571, and the Strata engine is the only non-llama.cpp
        # one in the house.
        #
        # Two Strata engines are registered on purpose, and only for as long as the 0.1.44.1 candidate
        # is under evaluation (since 2026-10-15): 7695859aa13a is the one that was serving when the
        # candidate was built, and the agreed rollback for a candidate that fails its gates is to
        # repoint the profile, not to re-add an entry and pay a second drain to reconstruct it. When
        # the candidate is decided either way, this set goes back to three keys and the losing entry
        # is removed reference-aware (ops/reports/20261015-strata-engine-0.1.44.1-build/README.md).
        #
        # A fourth, dated 2026-10-16: the same revision with the image encoder compiled in, which is a
        # build-time decision (BUILD_VISION) and not a runtime switch, so giving the resident its
        # picture encoder means a second image. The text-only entry stays registered for exactly the
        # same reason the 0.1.43 one does -- undoing the vision step is a repoint, not a rebuild.
        self.assertEqual(set(engines), {'qwen38-dual-836d571',
                                        'strata-7695859aa13a', 'strata-296b59d2f73c',
                                        'strata-vision-296b59d2f73c'})
        engine = engines['qwen38-dual-836d571']
        self.assertEqual(engine['revision'], '836d57176dc699a726c55418e4f96b8ca628e1bf')
        self.assertEqual(engine['tag'], 'local/llama.cpp:qwen38-dual-836d57176dc6')
        self.assertEqual(engine['image_id'], 'sha256:9f7179f568e4aa1004c8af9b613b65417e6f0e451f8cfbc774d6b0d0f50a0ecd')
        expected = {'qwen27b-q8-pair': 'qwen38-dual-836d571', 'qwen27b-q6k-pair': 'qwen38-dual-836d571',
            'flash-next-strata-solo': 'strata-vision-296b59d2f73c', EVERYDAY_PROFILE: 'qwen38-dual-836d571'}
        self.assertEqual(set(expected), {*CONFIGURATIONS, *EVERYDAY_PROFILES})
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
        night = read(ROOT / 'config/profiles' / (EVERYDAY_PROFILE + '.json'))
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
            path = config / 'profiles/qwen27b-q6k-pair.json'
            definition = read(path)
            for tokens in (262145, 0, '262144'):
                with self.subTest(tokens=tokens):
                    definition['context_tokens'] = tokens
                    path.write_text(json.dumps(definition))
                    with self.assertRaisesRegex(RuntimeError, 'router contract: 1–262144 tokens'):
                        render(config, BASELINE['host'], 'qwen27b-q6k-pair')
                    with self.assertRaisesRegex(RuntimeError, 'router contract'):
                        available_profiles(config)
            definition['context_tokens'] = 262144
            path.write_text(json.dumps(definition))
            self.assertEqual(render(config, BASELINE['host'], 'qwen27b-q6k-pair')['catalog']['context_length'], 262144)

    def test_invalid_repeated_arguments_or_missing_model_mount_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            path = config / 'profiles/qwen27b-q8-pair.json'; original = read(path)
            for key, value, message in [('--temp', ['1', '2'], 'Only --override-tensor'),
                    ('--cache-type-k', [], 'nonempty'), ('--model', '/weights/missing.gguf', 'declared artifact')]:
                with self.subTest(key=key):
                    definition = copy.deepcopy(original); definition['arguments'][key] = value
                    path.write_text(json.dumps(definition))
                    with self.assertRaisesRegex(RuntimeError, message): render(config, vision_host(), 'qwen27b-q8-pair')
            # A sharded model is the one case where an undeclared file is not simply an argument that
            # points nowhere: the engine reads the folder, so a shard nobody declared still loads.
            path = config / 'profiles' / (SOLO + '.json'); original = read(path)
            definition = copy.deepcopy(original); definition['artifacts'].pop(1)
            path.write_text(json.dumps(definition))
            with self.assertRaisesRegex(RuntimeError, 'missing a declared shard'):
                render(config, vision_host(), SOLO)


class RegistryTests(unittest.TestCase):
    def test_every_selectable_configuration_agrees_with_its_rendered_catalog(self):
        registry = available_profiles(ROOT / 'config', BASELINE['host'])
        self.assertEqual(CONFIGURATIONS, ('qwen27b-q8-pair', 'qwen27b-q6k-pair', 'flash-next-strata-solo'))
        self.assertEqual(PAIRED, ('qwen27b-q8-pair', 'qwen27b-q6k-pair'))
        self.assertEqual([x['profile'] for x in registry['selectable']], list(CONFIGURATIONS))
        # Every second resident a configuration names is listed, the default first.
        self.assertEqual([x['profile'] for x in registry['always_included']], list(EVERYDAY_PROFILES))
        variants = {x['profile']: x for x in registry['always_included']}
        night = variants[EVERYDAY_PROFILE]
        baseline = {name: render(ROOT / 'config', vision_host(), cfg)['compose']['services']['everyday']
            for name, cfg in ((EVERYDAY_PROFILE, 'qwen27b-q8-pair'),)}
        for name in CONFIGURATIONS:
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
                # Nothing a client sends is left unpublished, in any configuration: a solo one answers
                # `nighttime` from the one model it loads, so the router has no offline service to
                # report for it either. What differs is only how many residents are launched.
                self.assertEqual(rendered['catalog']['offline_services'], [])
                self.assertEqual(list(rendered['compose']['services']),
                    ['coding'] if name in EXCLUSIVE else ['coding', 'everyday'])
                self.assertEqual([m['model'] for m in rendered['catalog']['models']],
                    [entry['model']] if name in EXCLUSIVE else [entry['model'], night['model']])
                if name in EXCLUSIVE:
                    continue
                # The second backend is the Everyday variant the configuration names, identical
                # whichever coding configuration names it.
                paired = variants[entry['everyday']]
                everyday = service_engine(rendered, 'everyday')
                night_entry = next(m for m in rendered['catalog']['models'] if m['model'] == paired['model'])
                self.assertEqual(paired['engine_tag'], everyday['tag'])
                self.assertEqual(paired['backend_revision'], everyday['revision'])
                self.assertEqual(paired['context_tokens'], night_entry['context_length'])
                self.assertEqual(paired['display_name'], night_entry['display_name'])
                self.assertEqual(render(ROOT / 'config', vision_host(), name)['compose']['services']['everyday'], baseline[entry['everyday']])

    def test_catalog_names_each_backends_text_gpus_and_vision_gpu(self):
        host = vision_host()
        names = host['gpu_names']
        paired = render(ROOT / 'config', host, 'qwen27b-q8-pair')['catalog']
        coding, everyday = paired['models']
        self.assertEqual(coding['gpu_names'], names['daytime'])
        self.assertEqual(everyday['gpu_names'], names['nighttime'])
        self.assertEqual(len(coding['gpu_names']), len(coding['text_gpu_uuids']))
        self.assertEqual((coding['vision_gpu_name'], coding['vision_gpu_shared']), ('RTX 3080', True))
        self.assertEqual(paired['configuration'], {'id': 'qwen27b-q8-pair', 'exclusive': False})
        # The root coding projection carries the same names as its entry.
        self.assertEqual(paired['gpu_names'], coding['gpu_names'])
        [solo] = render(ROOT / 'config', host, SOLO)['catalog']['models']
        self.assertEqual(solo['gpu_names'], [*names['daytime'], *names['nighttime']])
        # The solo configuration holds the shared card to itself from 2026-10-16: it loads no second
        # resident, so the encoder that runs there shares it with nothing, and `vision_gpu_shared`
        # says so rather than repeating the paired configurations' answer.
        self.assertEqual((solo['vision_gpu_name'], solo['vision_gpu_shared']), ('RTX 3080', False))
        self.assertEqual(solo['vision_device'], 'CUDA4')
        self.assertTrue(solo['capability_profile']['vision'])
        self.assertEqual(solo['input_modalities'], ['text', 'image'])
        # Without GPU vision the projector stays on the CPU and no vision card is named.
        cpu = render(ROOT / 'config', BASELINE['host'], 'qwen27b-q8-pair')['catalog']['models'][0]
        self.assertNotIn('vision_gpu_name', cpu)
        self.assertEqual(cpu['gpu_names'], BASELINE['host']['gpu_names']['daytime'])
        # The same profile on a host with no vision GPU publishes text, which is what it was until
        # the shared card was lent to it: the promise and the fact are never allowed to disagree.
        [text_only] = render(ROOT / 'config', BASELINE['host'], SOLO)['catalog']['models']
        self.assertNotIn('vision_gpu_uuid', text_only)
        self.assertFalse(text_only['capability_profile']['vision'])
        self.assertEqual(text_only['input_modalities'], ['text'])

    def test_every_model_declares_whether_it_is_nsfw(self):
        # Router clients pick "the most capable NSFW model" from this flag; only the Everyday
        # resident is abliterated, and a solo configuration answers `nighttime` with a model that is
        # not, which is a fact the flag states rather than one the service ID implies.
        for name in CONFIGURATIONS:
            with self.subTest(profile=name):
                catalog = render(ROOT / 'config', host_for(name), name)['catalog']
                flags = {m['model']: m['capability_profile']['nsfw'] for m in catalog['models']}
                self.assertEqual(sorted(flags.values()), [False] if name in EXCLUSIVE else [False, True])
                self.assertIs(catalog['models'][0]['capability_profile']['nsfw'], False)
                self.assertIs(catalog['capability_profile']['nsfw'], False)
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            path = config / 'profiles' / (EVERYDAY_PROFILE + '.json')
            definition = read(path)
            for value in (None, 'yes'):
                if value is None:
                    definition['catalog']['capability_profile'].pop('nsfw', None)
                else:
                    definition['catalog']['capability_profile']['nsfw'] = value
                path.write_text(json.dumps(definition))
                with self.subTest(nsfw=value), self.assertRaisesRegex(RuntimeError, 'capability_profile.nsfw must be declared'):
                    render(config, BASELINE['host'], 'qwen27b-q8-pair')

    def test_registry_offers_only_registered_profiles_and_names_configured_gpus(self):
        definitions = {path.stem: read(path) for path in (ROOT / 'config/profiles').glob('*.json')}
        self.assertEqual(sorted(k for k, v in definitions.items() if v['role'] == 'coding'),
            sorted(CONFIGURATIONS))  # A new profile file must be registered in CONFIGURATIONS.
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
            ['primary', *CONFIGURATIONS, *EVERYDAY_PROFILES])
        registry = available_profiles(ROOT / 'config')
        self.assertIn(registry['always_included'][0]['display_name'], lines[0])
        for entry in registry['selectable'] + registry['always_included']:
            row = next(line for line in lines if line.startswith(entry['profile'] + ':'))
            self.assertIn(entry['display_name'], row)
            self.assertIn(entry['model'], row)
            self.assertEqual(row.endswith('(all four text GPUs; runs without a second resident)'),
            entry['profile'] in EXCLUSIVE)

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
                self.registry_with('qwen27b-q6k-pair', None)
        for name, mutate, expected in [
                ('qwen27b-q8-pair', lambda definition: definition.update(engine='no-such-engine'), 'unknown engine'),
                (EVERYDAY_PROFILE, lambda definition: definition.update(id='renamed'), 'differs from its filename'),
                # A solo configuration has no second resident to name, and saying it does is a
                # configuration that lies about what it will launch.
                (SOLO, lambda definition: definition.update(everyday=EVERYDAY_PROFILE), 'runs no second resident'),
                ('qwen27b-q8-pair', lambda definition: definition.update(context_tokens=262145), 'router contract')]:
            with self.subTest(profile=name):
                with self.assertRaisesRegex(RuntimeError, expected):
                    self.registry_with(name, mutate)


class EnvironmentEngineTests(unittest.TestCase):
    """The solo engine is configured by the environment instead of an argument vector."""

    def bundle(self):
        return render(ROOT / 'config', host_for(SOLO), SOLO)

    def test_every_argument_is_either_an_environment_variable_or_house_metadata(self):
        definition = read(ROOT / 'config/profiles' / (SOLO + '.json'))
        engine = ENGINES[definition['engine']]
        self.assertEqual(engine_interface(engine), 'environment')
        for key in definition['argument_order']:
            with self.subTest(argument=key):
                if key in ('--alias', '--model'):
                    self.assertNotIn(key, engine['environment'])
                else:
                    self.assertIn(key, engine['environment'])

    def test_the_service_is_rendered_from_its_own_arguments_and_mounts(self):
        definition = read(ROOT / 'config/profiles' / (SOLO + '.json'))
        service = self.bundle()['compose']['services']['coding']
        self.assertEqual(service['command'], [])
        self.assertEqual(service['entrypoint'], ['./docker-entrypoint.sh'])
        self.assertIs(service['read_only'], False)
        environment = service['environment']
        # Two of the mapped variables are the renderer's answer to the host, not the profile's: the
        # shared vision GPU turns the profile's text-only default on, and names the one card its
        # encoder runs on. The profile keeps the cheaper default; host.json decides (docs/shared-vision).
        for key, variable in ENGINES[definition['engine']]['environment'].items():
            if key in ('--vision', '--vision-device'):
                continue
            with self.subTest(argument=key):
                self.assertEqual(environment[variable], str(definition['arguments'][key]))
        self.assertEqual(definition['arguments']['--vision'], 'no')
        self.assertEqual((environment['VISION'], environment['VISION_DEVICE']), ('yes', '4'))
        # The folder the engine reads is the folder the shards are mounted in, and the window is the
        # profile's context: neither is a value a profile could quietly disagree with.
        self.assertEqual(environment['GGUF_DIR'], str(Path(definition['arguments']['--model']).parent))
        self.assertEqual(environment['CONTEXT'], str(definition['context_tokens']))
        # Five cards are in the container and four are the language model's: --gpus is the engine's
        # own card list and stops where the text cards stop, so the encoder cannot spill onto one.
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '0,1,2,3,4')
        self.assertEqual(environment['GPUS'], '0,1,2,3')
        self.assertEqual(service['ulimits']['memlock'], -1)
        # Only the engine's own workspace is writable; every model file stays a read-only bind.
        writable = [x['target'] for x in service['volumes'] if not x['read_only']]
        self.assertEqual(writable, ['/data'])
        self.assertEqual([x['target'] for x in service['volumes'] if x['read_only']],
            [x['target'] for x in definition['artifacts']])

    def test_the_shared_vision_gpu_is_a_fifth_card_the_language_model_never_sees(self):
        """The card the encoder runs on joins the container; the text placement does not move.

        This is the whole point of putting the encoder on the shared GPU: the four cards the language
        model is laid out across keep every byte they have today, and the encoder gets a card of its
        own instead of a reservation taken out of each of those four.
        """
        bundle = self.bundle()
        service = bundle['compose']['services']['coding']
        host = host_for(SOLO)
        text = text_gpus(host, read(ROOT / 'config/profiles' / (SOLO + '.json')))
        self.assertEqual(service['deploy']['resources']['reservations']['devices'][0]['device_ids'],
            [*text, VISION])
        self.assertNotIn(VISION, service['environment']['GPUS'].split(','))
        entry = bundle['catalog']['models'][0]
        self.assertEqual((entry['text_gpu_uuids'], entry['vision_gpu_uuid'], entry['vision_device']),
            (text, VISION, 'CUDA4'))
        # CUDA_VISIBLE_DEVICES is this container's own numbering and the encoder is placed by a number
        # in it, so the catalog states what the container was told, not a UUID list it was never given.
        self.assertEqual(entry['cuda_visible_devices'], '0,1,2,3,4')
        self.assertEqual(int(service['environment']['VISION_DEVICE']), len(text))

    def test_a_host_with_no_shared_vision_gpu_renders_the_text_only_resident_it_was(self):
        """host.json is the switch, and the profile is the same file whichever way it is set."""
        definition = read(ROOT / 'config/profiles' / (SOLO + '.json'))
        service = render(ROOT / 'config', BASELINE['host'], SOLO)['compose']['services']['coding']
        self.assertEqual(service['environment']['CUDA_VISIBLE_DEVICES'], '0,1,2,3')
        self.assertEqual(service['environment']['VISION'], 'no')
        self.assertNotIn('VISION_DEVICE', service['environment'])
        self.assertEqual(service['deploy']['resources']['reservations']['devices'][0]['device_ids'],
            text_gpus(BASELINE['host'], definition))
        entry = render(ROOT / 'config', BASELINE['host'], SOLO)['catalog']['models'][0]
        self.assertEqual(entry['mmproj_offload'], 'none')
        self.assertNotIn('vision_gpu_uuid', entry)

    def test_the_encoder_is_placed_on_the_card_the_text_model_leaves_free(self):
        # The encoder takes its card from a number, and that number is the shared card only while the
        # shared card is the last of the ones this container reserves. Any other value puts the encoder
        # on a text card and leaves a fifth card idle, which is expensive, quiet, and a render error:
        # a host that states a number it cannot mean is refused rather than obeyed.
        for device in (0, 3, 5, '4', None):
            host = vision_host()
            host['vision_cuda_device'] = device
            with self.subTest(vision_cuda_device=device), \
                self.assertRaisesRegex(RuntimeError, 'vision_cuda_device'):
                render(ROOT / 'config', host, SOLO)

    def test_an_engine_that_cannot_name_the_encoder_card_is_lent_no_card(self):
        # Rolling this profile back to the text-only image of the same revision is a profile change, and
        # it has to render the way it did before the encoder: four cards, no VISION_DEVICE, text in the
        # catalog. An image with no way to place an encoder is never handed a fifth GPU to wander onto.
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            path = config / ('profiles/' + SOLO + '.json')
            definition = read(path)
            definition['engine'] = 'strata-296b59d2f73c'
            path.write_text(json.dumps(definition))
            bundle = render(config, vision_host(), SOLO)
            service = bundle['compose']['services']['coding']
            self.assertEqual(service['environment']['CUDA_VISIBLE_DEVICES'], '0,1,2,3')
            self.assertEqual(service['environment']['VISION'], 'no')
            self.assertNotIn('VISION_DEVICE', service['environment'])
            self.assertEqual(
                len(service['deploy']['resources']['reservations']['devices'][0]['device_ids']), 4)
            self.assertEqual(bundle['catalog']['models'][0]['input_modalities'], ['text'])

    def test_the_solo_resident_is_published_without_llama_cpp_only_claims(self):
        bundle = self.bundle()
        entry = bundle['catalog']['models'][0]
        definition = read(ROOT / 'config/profiles' / (SOLO + '.json'))
        self.assertEqual(entry['model'], definition['arguments']['--alias'])
        # A resident, so it answers the service IDs a resident answers -- including `nighttime`,
        # which it answers itself rather than leaving a client with nothing to send.
        self.assertEqual(set(entry['aliases']), set(RESIDENT_SERVICES))
        # The router takes the first alias that is not the compatibility ID as the service a client
        # sends, so the order is load-bearing, not cosmetic.
        self.assertEqual(next(a for a in entry['aliases'] if a != COMPATIBILITY_ALIAS), 'daytime')
        self.assertEqual(entry['context_length'], definition['context_tokens'])
        self.assertEqual(entry['total_context_length'], definition['context_tokens'])
        self.assertEqual(entry['input_modalities'], ['text', 'image'])
        self.assertIs(entry['capability_profile']['vision'], True)
        self.assertIs(entry['capability_profile']['nsfw'], False)
        # The house contract still holds: no server-side output ceiling, and the full window.
        self.assertEqual(entry['output_policy'], 'unrestricted')
        # Nothing is claimed about caches and projectors this engine was never given. `mmproj_path` is
        # the llama.cpp projector file the server was handed on its command line; this engine is told
        # to build an encoder and finds its own weights, so the catalog says which card runs it
        # instead of which path was passed.
        for field in ('kv_cache', 'fit_target', 'global_ram_prompt_cache_mib', 'mmproj_path'):
            with self.subTest(field=field):
                self.assertNotIn(field, entry)
        # Four text cards plus the shared one, and the entry says which is which.
        self.assertEqual(len(entry['text_gpu_uuids']), 4)
        self.assertEqual(entry['gpu_uuids'], [*entry['text_gpu_uuids'], VISION])
        self.assertEqual(len(entry['gpu_uuids']), 5)
        self.assertTrue(any('not been attested' in warning for warning in entry['deployment_warnings']))


    def test_the_solo_service_keeps_the_house_container_settings(self):
        """Attempt one of this switch died on KeyError: 'restart'.

        The engine entry declares its own container_defaults and they replaced the shared ones, so the
        solo service arrived with no restart policy, no init, no logging and no pull policy. The merge
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
        engine = ENGINES[read(ROOT / 'config/profiles' / (SOLO + '.json'))['engine']]
        self.assertEqual(engine['revision'], '296b59d2f73c8f04a805df29c2fa97655adace78')
        self.assertEqual(engine['tag'], 'local/strata:296b59d2f73c-vision')
        self.assertEqual(engine['release'], 'v0.1.44.1')
        # Built on this host from the pinned revision with the image encoder compiled in
        # (BUILD_VISION=1), not downloaded: the upstream releases ship Windows zips only. Receipt and
        # image digest in ops/reports/20261016-strata-vision/README.md. The text-only image of the
        # same revision is a different image with a different digest, because the encoder is compiled
        # in at build time and is not a runtime switch.
        self.assertEqual(engine['image_id'],
                         'sha256:294c987a8282b63a58a2e631c5f3c5653baa4eeeda3e6a91a7f4e62902504608')
        self.assertEqual(engine['repository'], 'https://github.com/Niko1221/Strata')
        self.assertEqual(engine['cuda_architectures'], ['86', '89'])
        self.assertEqual(engine['interface'], 'environment')
        self.assertGreater(engine['readiness_wait_seconds'], 450)
        # Being told which card its encoder runs on is the whole difference between an engine the
        # renderer may lend the shared vision GPU to and one it has to leave the card out of.
        self.assertEqual(engine['environment']['--vision-device'], 'VISION_DEVICE')

    # flash-next-strata-solo answers to these names through a hand edit inside its own workspace's
    # engine config (strata-unsloth-ud-iq4_xs.json), not through source. Neither of that file's
    # two backups records when the names went in, which is part of why they are easy to lose. It is
    # exempt only because declaring them in the profile recreates the container that is serving
    # right now. The next time that profile restarts for any reason, put the variable in its
    # container.environment and drop it from this map.
    #
    # The value is the hand edit as it stands in the workspace, read off its live config. An
    # exemption that records no content is a hole wearing a check: this is
    # the only environment-interface profile left, so the loop below would compare nothing at all,
    # and an alias rename would leave the serving container refusing the very name the controller
    # probes for it -- the failure this test exists to catch, arriving by a different route.
    WORKSPACE_CARRIED_HOSTS = {'flash-next-strata-solo': ('qwen38-daytime', 'strata-trial', 'qwen38-strata')}

    def test_an_environment_engine_names_the_hosts_its_server_will_answer(self):
        """A Strata server refuses a request whose Host name it was never told to answer.

        DNS rebinding protection (serve/server.py: allowed_hosts_of, applied where the service is
        built) checks the Host header against the config's allowed_hosts plus $STRATA_ALLOWED_HOSTS,
        and the controller probes a backend by its container name on the router network. A workspace
        built from scratch has no config to put those names in, so the profile has to carry them.

        Selecting flash-next-strata-xl-160k on 2026-10-15 loaded 71.73 GiB of experts into RAM, came
        ready on :8080, and then failed the controller's readiness verification with

            refused a request for Host 'qwen38-daytime:8080' from 172.18.0.4: not a name this server
            answers to (add it to "allowed_hosts" in the config or STRATA_ALLOWED_HOSTS, ...)

        The transition recovered by itself. The model was never the problem: the profile could not
        have passed, and nothing in source said which names it needed. They are the container name and
        every alias the profile claims on the router, because those are exactly the names the
        controller and clients can reach it by. Equality, not containment, so an alias rename that
        leaves a stale name behind fails too.

        A profile whose names are carried by its workspace instead of its source is held to the same
        equality, against the edit recorded in WORKSPACE_CARRIED_HOSTS: the deviation is allowed to
        exist, not to go quietly stale.
        """
        environment = sorted(n for n in CONFIGURATIONS if interface_of(n) == 'environment')
        self.assertEqual(set(self.WORKSPACE_CARRIED_HOSTS).difference(environment), set(),
            'WORKSPACE_CARRIED_HOSTS exempts a profile that is not an environment-interface Daytime '
            'profile; retire the exemption with the profile')
        for name in environment:
            definition = read(ROOT / 'config/profiles' / (name + '.json'))
            container = definition['container']
            reached = {definition['container_name'], *container['networks']['router']['aliases']}
            with self.subTest(profile=name):
                if name in self.WORKSPACE_CARRIED_HOSTS:
                    self.assertEqual(set(self.WORKSPACE_CARRIED_HOSTS[name]), reached,
                        '%s answers to %s in its workspace and is reached by %s: the hand edit no '
                        'longer covers this profile, so declare STRATA_ALLOWED_HOSTS in the profile '
                        '(that recreates the container) rather than editing the workspace again'
                        % (name, sorted(self.WORKSPACE_CARRIED_HOSTS[name]), sorted(reached)))
                    continue
                declared = {h.strip().lower() for h in container['environment']['STRATA_ALLOWED_HOSTS'].split(',')}
                self.assertEqual(declared, reached)


class ProvisioningWorkspaceTests(unittest.TestCase):
    """Two profiles may share an engine. They may not share a provisioning workspace.

    provisioning() reads the contract off the engine, so the ready marker must not name a quant: on
    2026-10-15 one Strata engine served two profiles with two quants, and listing one marker per
    quant is no fix either, because workspace_ready() requires every declared pattern to match, which
    would leave the first profile watching for a file the second one will never write. What makes a
    workspace-relative marker say what the literal one said is that no two provisioning profiles look
    at the same directory. If they ever do, whichever finishes first declares the other's half-built
    workspace ready, and the switch spends its readiness window on a machine that has not packed its
    model yet. The second quant that exposed this was retired the same day; the invariant and the
    broadened marker stay, because the next quant trial is exactly what they were written for.
    """

    def test_every_provisioning_profile_has_a_workspace_of_its_own(self):
        engines = read(ROOT / 'config/shared.json')['engines']
        seen = {}
        for name in (*CONFIGURATIONS, *EVERYDAY_PROFILES):
            profile = read(ROOT / 'config/profiles' / (name + '.json'))
            if not engines[profile['engine']].get('provisioning'):
                continue
            workspace = profile['workspace']['path']
            self.assertNotIn(workspace, seen,
                '%s and %s share workspace %s: a ready marker read off the engine would let either '
                'one answer for the other' % (name, seen.get(workspace), workspace))
            seen[workspace] = name
        self.assertEqual(sorted(seen.values()), ['flash-next-strata-solo'])


class ServiceIdentityTests(unittest.TestCase):
    """What a client has to send to reach a backend, and what may never answer it."""

    def test_every_profile_publishes_a_service_id_a_contract_client_can_send(self):
        # LLM Router derives the client-facing ID from these aliases and falls back to the canonical
        # model id when it finds none (llm-router `src/capabilities.js`: `service: aliases.find(...)
        # ?? entry.id`), and its client contract forbids a client from sending, configuring or
        # comparing canonical ids. An entry with no service alias is therefore running, selected, and
        # impossible to ask for — a host whose only lever moves nothing, which is the failure AGENTS.md
        # forbids in a drain, reached here through model identity instead. It is not theoretical: the
        # trial published `aliases: []` from 2026-10-11 until it was given its own alias on 2026-10-13,
        # and the primary client on this machine had no model at all to use while it was selected.
        for name in CONFIGURATIONS:
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
                    # selected: `daytime` is only ever the coding resident, `nighttime` only ever one
                    # thing, in exactly one direction.
                    for other in catalog['models']:
                        if other is entry:
                            continue
                        self.assertTrue(set(service).isdisjoint(
                            [alias for alias in other['aliases'] if alias != COMPATIBILITY_ALIAS]),
                            'Service ID shared between ' + entry['model'] + ' and ' + other['model'])

    def test_every_service_id_is_answered_or_declared_offline(self):
        """A service ID is answered here or says so; it is never quietly missing.

        A client that has `nighttime` configured cannot discover that this configuration stopped
        loading that resident — its request just fails, forever, with nothing telling it why. So for
        every configuration each ID a client may send is asserted to be either published by a
        resident or listed in `offline_services` with a reason a status page can show. Nothing is
        offline in any configuration today: the solo configuration answers `nighttime` from the one
        model it does load, which is the whole reason it is a configuration and not an experiment.
        """
        for name in CONFIGURATIONS:
            catalog = render(ROOT / 'config', host_for(name), name)['catalog']
            published = {alias for model in catalog['models'] for alias in model['aliases']}
            offline = {alias for service in catalog['offline_services'] for alias in service['aliases']}
            with self.subTest(profile=name):
                for service_id in RESIDENT_SERVICES:
                    self.assertIn(service_id, published | offline,
                        'Configuration ' + name + ' neither answers ' + service_id +
                        ' nor declares it offline: a client holding that name fails with no reason')
                self.assertEqual(published & offline, set(),
                    'Configuration ' + name + ' answers ' + str(published & offline) + ' and also '
                    'declares it offline')
                self.assertEqual(offline, set(), 'Configuration ' + name +
                    ' declares a service offline that every resident here answers')

    def test_a_retired_service_id_is_answered_by_nothing(self):
        """`trial` addressed the Strata engine while it was an experiment and nobody's default.

        Promoting that engine to a resident gave it the resident IDs and took `trial` away, so a
        client still sending it is refused. Refusing is right — the alternative is some other model
        answering a name that meant something specific — but it is only honest because no *other*
        model picked the name up either, which is what this pins for every configuration.
        """
        for name in CONFIGURATIONS:
            with self.subTest(profile=name):
                catalog = render(ROOT / 'config', host_for(name), name)['catalog']
                self.assertNotIn('trial', {alias for model in catalog['models'] for alias in model['aliases']})


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
        for name in CONFIGURATIONS:
            for role, cfg in render(ROOT / 'config', host_for(name), name)['compose']['services'].items():
                for key in CONTAINER_CONTRACT:
                    with self.subTest(profile=name, role=role, setting=key):
                        self.assertIsNotNone(cfg.get(key))


class ConfigurationNamingTests(unittest.TestCase):
    """What a configuration is called, and what happens to a name that stops meaning one."""

    def test_a_renamed_configuration_renders_its_new_name_exactly(self):
        # A rename is a naming change, not a redeploy: Compose services are keyed by role and the
        # canonical model id is the model alias, so the old name has to produce byte-identical
        # backends. If this ever differs, a host that renames while running would be drained for a
        # backend that is not the one it already has.
        self.assertEqual(set(PROFILE_ALIASES.values()) & set(PROFILE_ALIASES), set(),
            'a rename must land on a current id, never on another alias')
        for old, new in PROFILE_ALIASES.items():
            with self.subTest(profile=old):
                self.assertIn(new, CONFIGURATIONS)
                self.assertEqual(resolve_profile(old), new)
                self.assertEqual(render(ROOT / 'config', host_for(new), old),
                    render(ROOT / 'config', host_for(new), new))

    def test_a_retired_configuration_refuses_and_names_the_current_ones(self):
        # A retired id is not an alias: its backends are described nowhere, so launching one from
        # memory is the drift this house refuses. It says what to use instead, because the operator
        # reading the refusal is the only lever left.
        for old, replacement in RETIRED_PROFILES.items():
            with self.subTest(profile=old):
                self.assertNotIn(old, CONFIGURATIONS)
                self.assertNotIn(old, PROFILE_ALIASES)
                self.assertIn(replacement, CONFIGURATIONS)
                message = unknown_profile(old)
                self.assertIn(old, message)
                self.assertIn(replacement, message)
                for name in CONFIGURATIONS:
                    self.assertIn(name, message)
        # The default is what a fresh host starts on, so it has to be a configuration and not a name
        # that only resolves by the grace of a transitional table.
        self.assertIn(DEFAULT_PROFILE, CONFIGURATIONS)
        self.assertNotIn(DEFAULT_PROFILE, PROFILE_ALIASES)
        self.assertIn('never-existed', unknown_profile('never-existed'))

    def test_every_name_the_transitional_constant_accepts_resolves(self):
        # DAYTIME_PROFILES is read by an updater that is already running, before this revision's code
        # exists, and it refuses the release when the configuration it is running is not in it. So the
        # tuple may be a superset during a rename release -- it may not contain a name that then
        # refuses to mean anything, which would let the release in and strand the running host.
        self.assertTrue(set(CONFIGURATIONS) <= set(DAYTIME_PROFILES),
            'the transitional constant must still offer every current configuration')
        for name in DAYTIME_PROFILES:
            with self.subTest(profile=name):
                target = resolve_profile(name)
                self.assertIn(target, CONFIGURATIONS,
                    name + ' is accepted by the transitional constant and resolves to nothing')
        self.assertEqual(DEFAULT_PROFILE, CONFIGURATIONS[0] if DEFAULT_PROFILE in PROFILE_ALIASES
            else DEFAULT_PROFILE)

    def test_the_updater_reads_whichever_constant_a_revision_uses(self):
        # The updater doing this work is the revision already deployed, parsing the source of the one
        # it is about to become. A rename of the constant itself is only deployable if the old code
        # can still find the new one, and a rename of a configuration is only deployable if the old
        # code accepts the name it is running.
        from runtime.update import Updater

        class Source(Updater):
            def __init__(self, source):
                self.source = source

            def git(self, *args, **kwargs):
                return self.source

        cases = {
            'both constants': ("CONFIGURATIONS = ('a', 'b')\nDAYTIME_PROFILES = ('a', 'b')\n"
                "PROFILE_ALIASES = {}\n", ('a', 'b'), ('a', 'b')),
            # The revision that renames the constant is the one that has to be deployable.
            'transitional binding': ("CONFIGURATIONS = ('a2', 'b2')\nDAYTIME_PROFILES = CONFIGURATIONS\n"
                "PROFILE_ALIASES = {'a': 'a2', 'b': 'b2'}\n", ('a2', 'b2'), ('a2', 'b2', 'a', 'b')),
            'a revision before either': ("DAYTIME_PROFILES = ('a', 'b')\n", ('a', 'b'), ('a', 'b')),
            # A retired id has no backends left to launch, so it still refuses.
            'retired without an alias': ("CONFIGURATIONS = ('a',)\nDAYTIME_PROFILES = ('a',)\n"
                "PROFILE_ALIASES = {}\n", ('a',), ('a',)),
        }
        for label, (source, selectable, accepted) in cases.items():
            with self.subTest(revision=label):
                self.assertEqual(Source(source).target_profiles('target'), (selectable, accepted))
        for source in ("DAYTIME_PROFILES = 5\n", "OTHER = ('a',)\n", "CONFIGURATIONS = ()\n"):
            with self.subTest(source=source), self.assertRaises(RuntimeError):
                Source(source).target_profiles('target')

    def test_every_engine_is_used_and_nothing_is_configured_by_an_absent_one(self):
        # A profile naming an engine that is not registered fails at render, and an engine nothing
        # references is an image kept alive by a config file. The one exception is the engine held for
        # a live rollback, named here so that retiring it is a decision and not an oversight.
        engines = read(ROOT / 'config/shared.json')['engines']
        # The 0.1.43 image, until the 0.1.44.1 candidate is decided; and the text-only image of
        # 0.1.44.1, until the encoder that joined the resident on 2026-10-16 has earned its place.
        rollback = {'strata-7695859aa13a', 'strata-296b59d2f73c'}
        used = set()
        for name in (*CONFIGURATIONS, *EVERYDAY_PROFILES):
            definition = read(ROOT / 'config/profiles' / (name + '.json'))
            self.assertIn(definition['engine'], engines, name + ' names an engine that is not registered')
            used.add(definition['engine'])
        self.assertEqual(set(engines) - used, rollback,
            'unreferenced engine: ' + str(set(engines) - used - rollback))

import copy
import json
import re
from pathlib import Path
import shutil
import tempfile
import unittest

from models import CODING, CODING_Q6K, EVERYDAY, STRATA_DAISY, STRATA_PAIR, STRATA_SOLO
from runtime.system import atomic_json
from runtime.config import (ALIAS_MAX, CONFIGURATIONS, CONTAINER_CONTRACT,
    DEFAULT_PROFILE, PROFILE_ALIASES, RETIRED_PROFILES,
    ALIASES_VERSION, available_profiles, belongs_to_host, catalog_names, clean_alias,
    configurations_for, is_exclusive, model_id, models_of, nameable_models,
    default_name, default_profile, digest, display_name, engine_interface, merge_settings, read,
    read_aliases, render, resolve_profile, service_engine, service_key, text_gpus, unknown_profile)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = read(ROOT / 'tests/fixtures/legacy-fingerprints.json')
VISION = 'GPU-465a9a1e-fcb9-27f2-8f6a-cca1c30de981'


def profile_file(name):
    """What one profile file says, since half of these tests ask what a file states."""
    return read(ROOT / 'config/profiles' / (name + '.json'))


def config_with(name, **changes):
    """The shipped configuration directory with one profile file edited.

    A refusal worth reaching through a real file: the same edit applied to a dict in memory skips the
    part of the rule that lives in how a profile is found and read on disk.
    """
    directory = Path(tempfile.mkdtemp()) / 'config'
    shutil.copytree(ROOT / 'config', directory)
    path = directory / 'profiles' / (name + '.json')
    definition = read(path)
    definition.update(changes)
    path.write_text(json.dumps(definition))
    return directory


def one_pair_host(machine):
    """A machine with one text pair, for a configuration that loads one model on it.

    The alternative is four cards and one model, which render refuses -- correctly: a declared card
    with nothing placed on it is a card the operator believes is serving.
    """
    host = copy.deepcopy(BASELINE['host'])
    return {'model_root': host['model_root'], 'router_state_dir': host['router_state_dir'],
        'router_admin_url': host['router_admin_url'], 'router_admin_header': host['router_admin_header'],
        'uid': host['uid'], 'gid': host['gid'], 'host_id': machine,
        'gpu_ids': {'daytime': host['gpu_ids']['daytime']},
        'gpu_names': {'daytime': host['gpu_names']['daytime']}}


def exclusive(name):
    """Whether the model a configuration is named after owns every text GPU of this host."""
    return is_exclusive(profile_file(name))


# How many models a configuration loads is a fact in its own `models` list, and the two shapes the
# shipped set is in follow from reading it: one model across every card, or several with the cards
# divided. Nothing here may assume the answer the way a `role` used to.
NAMED = {name: [model_id(model) for model in models_of(ROOT / 'config', profile_file(name))]
    for name in CONFIGURATIONS}
ALONE = tuple(n for n in CONFIGURATIONS if len(NAMED[n]) == 1)
# Which configurations load a model, read off the lists rather than from a role every file used to
# carry: a test that says "every configuration that loads this model" says which model it means.
LOADS = {model: tuple(n for n in CONFIGURATIONS if model in NAMED[n])
    for model in {model for models in NAMED.values() for model in models}}
# The file the Everyday model is defined in. It is a profile file like any other, and what makes it a
# model rather than a configuration is that it names no `models` list of its own -- which is also why
# no configuration may load it as one.
EVERYDAY_FILE = 'everyday'
# A configuration is filed under the model it is named after, and a model is filed under the id a
# client routes on, so `service_key(model) is model_id(model)` is the whole identity rule; the tests
# read it once here instead of trusting a key that used to be spelled `coding`.
OWN = {name: service_key(profile_file(name)) for name in CONFIGURATIONS}
# Every profile file the revision ships: the configurations, and the models their lists name.
DEFINITIONS = tuple(sorted(path.stem for path in (ROOT / 'config/profiles').glob('*.json')))
ENGINES = read(ROOT / 'config/shared.json')['engines']
# The one configuration that is not llama.cpp, and the only one that owns every card. Its engine is
# configured by the environment, so the catalog fields that describe a flag the engine was handed (KV
# types, --fit, the host-RAM prompt cache budget) are not published for it.
SOLO = 'flash-next-strata-solo-thinking-recovery'
RECOVERY = SOLO
# The alias the household's default model carries for the legacy Ollama surface. It is the
# compatibility ID, not the one a contract client sends, so it is never the only alias an entry has.
COMPATIBILITY_ALIAS = 'local-active'
# The service IDs a client of this machine sends. Every configuration of Rosalina publishes all three:
# the configuration that loads one model answers `nighttime` from that model, which is what keeps a
# client that only ever sends one of them working whichever configuration is selected.
CLIENT_SERVICES = ('daytime', COMPATIBILITY_ALIAS, 'nighttime')
# The machine the shipped private settings name, and the machine every fixture host that does not name
# itself stands for: absent `host_id` says nothing, which is the whole registry with no guard.
ROSALINA = 'rosalina'
# The second machine, now that it has a configuration of its own: its id is the one its private
# settings state and the one its definitions name.
DAISY = 'daisy'


def interface_of(profile):
    return engine_interface(ENGINES[profile_file(profile)['engine']])


def vision_host(host=None):
    host = copy.deepcopy(host or BASELINE['host'])
    host.update(vision_gpu_id=VISION, vision_gpu_name='RTX 3080', vision_cuda_device=4,
        cuda_order={'daytime': host['gpu_ids']['daytime'][::-1], 'nighttime': host['gpu_ids']['nighttime'][:]})
    return host


def deployed(host, machine):
    """A host that states which machine it is, which is what a deployed `host.json` does.

    A host that states nothing is shown every machine's configurations, and then a service ID another
    machine publishes really is one nothing loaded here answers -- which is what `offline_services`
    reports, and what a pin recorded on the machine its own definition names does not contain.
    """
    return {**host, 'host_id': machine}


def fixture_host(profile):
    """The host a pinned digest was recorded on, which for the second machine's configuration is not
    the four-card host the rest of these pins were."""
    hosts = profile_file(profile).get('hosts')
    if hosts and ROSALINA not in hosts:
        return second_machine(gpu_architectures=['75'])
    return deployed(vision_host(), ROSALINA)


def host_for(profile):
    """The machine a definition was written for, because it cannot be rendered on any other.

    Before a second machine existed this chose between four cards and four cards-minus-a-pair; with
    `hosts` on every definition it has to keep choosing that, and additionally give a configuration of
    another machine the host it was measured on, since `--gpus 0,1` on four cards is a refusal and not
    a rendering.
    """
    hosts = profile_file(profile).get('hosts')
    if hosts and ROSALINA not in hosts:
        return second_machine(gpu_architectures=['75'])
    # A model that owns every text GPU places its projector and MTP draft on the vision GPU.
    return deployed(vision_host() if exclusive(profile) else BASELINE['host'], ROSALINA)


def second_machine(**changes):
    """A second machine reading the same source: two cards, one pair, no card to lend for vision.

    Written out rather than derived from Rosalina's settings, because the point of the host is that it
    is not a Rosalina with a key removed: different cards, one text pair, nothing to lend for vision,
    and a name of its own -- which is what turns host scoping on. The architectures default to the
    ones the shipped engines carry, so a test that means to exercise the architecture guard says so.
    """
    host = {'model_root': '/models', 'router_state_dir': '/router',
        'router_admin_url': 'http://ai-router:11435', 'router_admin_header': 'X-Admin-Token',
        'uid': 1000, 'gid': 1000, 'host_id': 'daisy', 'gpu_architectures': ['86', '89'],
        'gpu_ids': {'daytime': ['GPU-daisy-1', 'GPU-daisy-2']},
        'gpu_names': {'daytime': ['RTX 2080 Ti', 'RTX 2080 Ti']}}
    host.update(changes)
    return host


# Both fixture files under tests/fixtures/ are published artifacts of the revision that keyed a Compose
# service by its role, and a fixture is never re-cut to let a test pass: `renamed-profiles.json` holds
# the digests a rename must keep reproducing and `active-bundles.json` the whole-bundle digests that
# notice an unintended launch change. A service is now filed under the model it loads, and the container
# it starts no longer carries a `local-ai.role` label: nothing read that label -- not the controller,
# not the portal, not the router -- so it described a concept this runtime no longer has, and a label
# nobody reads is not worth a container's metadata. The pins therefore want two things back and only
# those two: the old spelling of the service key, and the label that spelled it. Putting exactly that
# back is what keeps these pins the check they were cut to be: an argv, a mount, an environment
# variable, a catalog field or a GPU moved by accident still breaks the digest.
def role(service):
    """The key the pinned revision filed this model's launch under."""
    return 'everyday' if service == EVERYDAY else 'coding'


def stamped(service, cfg):
    """The container as the pinned revision rendered it, label and all."""
    return {**cfg, 'labels': {**cfg.get('labels', {}), 'local-ai.role': role(service)}}


def filed_by_role(bundle):
    bundle = copy.deepcopy(bundle)
    bundle.pop('config_sha256', None)
    services = bundle['compose']['services']
    bundle['compose']['services'] = {role(service): stamped(service, cfg)
        for service, cfg in services.items()}
    for entry in bundle['manifest']['services']:
        entry['role'] = role(entry.pop('service'))
    return bundle


class ConfigurationTests(unittest.TestCase):
    def test_renamed_profiles_launch_exactly_the_backends_of_their_previous_names(self):
        # The 2026-10-07 consolidation renamed the kept configurations. Their backends and model aliases
        # (benchmark history) are unchanged, so switching from an old name to its new one recreates nothing.
        # compose_sha256 is the whole backend set at the rename; Nighttime has since gained MTP3 (2026-10-07),
        # so the configuration's own service is what stays pinned for the configurations that load another
        # model beside it.
        #
        # Every selectable configuration is pinned here, renamed or not -- an entry whose previous_id is
        # null says so in its note -- because the digest is the cheapest thing that notices a profile edit
        # moved a launch flag, a mount or a variable along with it. The per-service digest is the one that
        # says a launch did not move, and it survives the day a service is filed under a different name:
        # what is digested is the container, not the key it sits under.
        recorded = read(ROOT / 'tests/fixtures/renamed-profiles.json')
        self.assertEqual(set(recorded), set(CONFIGURATIONS))
        for name, expected in recorded.items():
            with self.subTest(profile=name, previous=expected['previous_id']):
                bundle = render(ROOT / 'config', fixture_host(name), name)
                self.assertEqual(digest(stamped(OWN[name], bundle['compose']['services'][OWN[name]])),
                    expected['coding_service_sha256'])
                if name in ALONE:
                    self.assertEqual(digest(filed_by_role(bundle)['compose']), expected['compose_sha256'])

    def test_nothing_in_service_is_relaunched_by_a_revision_that_did_not_mean_to(self):
        # A revision that changes the launch of what is serving is a deliberate act, not a side effect
        # of adding a neighbour. config_sha256 is what the controller compares against the live
        # container, so a change here does not refuse -- it recreates a running model. Two of these
        # bundles also carry the Everyday model's launch inside them, so the night-side model's
        # projector and MTP draft are pinned here too: they ride on the index the shared vision GPU is
        # assigned, which is exactly the thing a change to the vision placement moves without any new
        # profile mentioning it.
        pinned = read(ROOT / 'tests/fixtures/active-bundles.json')
        self.assertEqual(set(pinned['bundles']), set(CONFIGURATIONS))
        for name, expected in sorted(pinned['bundles'].items()):
            with self.subTest(profile=name):
                self.assertEqual(digest(filed_by_role(
                    render(ROOT / 'config', fixture_host(name), name))), expected)

    def test_context_edit_updates_both_launch_flags_manifest_and_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            path = config / 'profiles/qwen27b-q8-pair.json'
            definition = read(path); definition['context_tokens'] = 98304
            path.write_text(json.dumps(definition))
            result = render(config, BASELINE['host'], 'qwen27b-q8-pair')
            argv = result['compose']['services'][CODING]['command']
            for flag in ['--ctx-size', '--kv-unified-per-slot']:
                self.assertEqual(argv[argv.index(flag) + 1], '98304')
            self.assertEqual(result['catalog']['context_length'], 98304)
            self.assertEqual(result['catalog']['models'][0]['total_context_length'], 98304)
            self.assertEqual(result['manifest']['services'][0]['context_tokens'], 98304)
            # An edit to one model's window is an edit to that model: the other model this
            # configuration loads keeps the window its own file states, which 98304 alone cannot show
            # (the Everyday model is served at that width) and the source of it can.
            self.assertEqual([m['context_length'] for m in result['catalog']['models']],
                [98304, read(ROOT / 'config/profiles/everyday.json')['context_tokens']])
            self.assertEqual(result['catalog']['display_name'], 'Qwen3.8 27B Q8 (96K)')

    def test_gpu_groups_cannot_overlap(self):
        host = copy.deepcopy(BASELINE['host']); host['gpu_ids']['nighttime'][0] = host['gpu_ids']['daytime'][0]
        with self.assertRaisesRegex(RuntimeError, 'reserves a card twice'):
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
        for bundle, alias, quant, display in ((q8, CODING, 'Q8_0', 'Qwen3.8 27B Q8 (160K)'),
                (q6k, CODING_Q6K, 'UD-Q6_K_XL', 'Qwen3.8 27B Q6_K (160K)')):
            with self.subTest(alias=alias):
                # Two models named, two services, each filed under the model it loads.
                self.assertEqual(list(bundle['compose']['services']), [alias, EVERYDAY])
                argv = bundle['compose']['services'][alias]['command']
                # Tensor mode divides every weight between the RTX 3090 and RTX 4080 SUPER, balanced by bandwidth.
                for flag, value in {'--alias': alias, '--split-mode': 'tensor', '--tensor-split': '55,45', '--fit': 'off',
                        '--ctx-size': '163840', '--spec-draft-n-max': '3', '--spec-draft-device': 'CUDA1'}.items():
                    self.assertEqual(argv[argv.index(flag) + 1], value)
                self.assertEqual(service_engine(bundle, alias), engine)
                catalog = bundle['catalog']
                self.assertEqual((catalog['model'], catalog['quantization'], catalog['display_name']), (alias, quant, display))
                self.assertEqual((catalog['context_length'], catalog['mtp']['max_draft_tokens'], catalog['mtp']['device']), (163840, 3, 'CUDA1'))
                self.assertEqual(catalog['models'][1]['display_name'], 'Qwen3.8 27B Abliterated Q6_K (96K)')
        # Q6_K differs from Q8 only in its main weights, alias, and catalog identity.
        q8_cfg, q6k_cfg = q8['compose']['services'][CODING], copy.deepcopy(q6k['compose']['services'][CODING_Q6K])
        argv = q6k_cfg['command']; argv[argv.index('--alias') + 1] = CODING
        for mount in q6k_cfg['volumes']:
            if mount['target'] == '/weights/main.gguf':
                self.assertTrue(mount['source'].endswith('Qwen3.8-27B-UD-Q6_K_XL.gguf'))
                mount['source'] = next(m['source'] for m in q8_cfg['volumes'] if m['target'] == '/weights/main.gguf')
        self.assertEqual(q6k_cfg, q8_cfg)
        # The model both configurations load beside themselves is one launch, not two renderings of it.
        self.assertEqual(q6k['compose']['services'][EVERYDAY], q8['compose']['services'][EVERYDAY])

    def test_the_everyday_model_runs_the_base_mtp3_head_on_the_shared_vision_gpu(self):
        night = profile_file(EVERYDAY_FILE)
        spec = {'--spec-type': 'draft-mtp', '--spec-draft-model': '/weights/mtp.gguf', '--spec-draft-n-max': '3',
            '--spec-draft-ngl': 'all', '--spec-draft-device': 'CUDA2', '--spec-draft-type-k': 'q8_0', '--spec-draft-type-v': 'q8_0'}
        self.assertEqual({k: night['arguments'][k] for k in spec}, spec)
        self.assertEqual((night['display_name'], night['arguments']['--alias'], night['draft_on_vision_gpu']),
            ('Qwen3.8 27B Abliterated Q6_K', EVERYDAY, True))
        # At 128K, speculative decoding needed another 564 MiB on the RTX 4080 than the pair had free.
        self.assertEqual(night['context_tokens'], 98304)
        head = next(a for a in profile_file('qwen27b-q6k-pair')['artifacts'] if a['target'] == '/weights/mtp.gguf')
        self.assertIn(head, night['artifacts'])  # the same self-contained 27B MTP head Daytime uses
        host = vision_host()
        for name in LOADS[EVERYDAY]:
            with self.subTest(profile=name):
                bundle = render(ROOT / 'config', host, name)
                # The bundle says what it loads by loading it. A top-level key pointing at whatever
                # runs beside the selected configuration -- in either spelling this house has used,
                # `nighttime_profile` or `everyday_profile` -- would be a second source for what the
                # `models` list already states, and the two would eventually disagree.
                self.assertEqual(set(bundle), {'schema_version', 'profile', 'compose', 'catalog',
                    'manifest', 'artifacts', 'config_sha256'})
                everyday = bundle['compose']['services'][EVERYDAY]
                argv = everyday['command']
                for flag, value in {**spec, '--split-mode': 'tensor', '--device': 'CUDA0,CUDA1', '--mmproj-device': 'CUDA2'}.items():
                    self.assertEqual(argv[argv.index(flag) + 1], value)
                self.assertEqual(everyday['deploy']['resources']['reservations']['devices'][0]['device_ids'], [*host['gpu_ids']['nighttime'], VISION])
                model = next(m for m in bundle['catalog']['models'] if m['model'] == EVERYDAY)
                self.assertEqual((model['model'], model['mtp']['device'], model['mtp']['max_draft_tokens'], model['display_name']),
                    (EVERYDAY, 'CUDA2', 3, 'Qwen3.8 27B Abliterated Q6_K (96K)'))
        # Without a vision GPU there is nowhere to put the draft: the Everyday model runs without
        # it rather than failing.
        cpu = render(ROOT / 'config', BASELINE['host'], 'qwen27b-q8-pair')
        argv = cpu['compose']['services'][EVERYDAY]['command']
        self.assertFalse([x for x in argv if x.startswith('--spec-')])
        self.assertNotIn('/weights/mtp.gguf', [m['target'] for m in cpu['compose']['services'][EVERYDAY]['volumes']])
        self.assertEqual(cpu['catalog']['models'][1]['mtp'], {'enabled': False})
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config_dir)
            for name, change, message, target in [
                    (EVERYDAY_FILE, {'draft_on_vision_gpu': False}, 'Draft model must remain on a text GPU', 'qwen27b-q8-pair'),
                    # A model that owns every text GPU cannot be named beside another one: the
                    # configuration would be promising two launches that each need the whole machine.
                    (SOLO, {'models': [SOLO, EVERYDAY_FILE]},
                        'owns every text GPU of this host, so no other model can be loaded beside it', SOLO),
                    # And the things a configuration names have to be models: a configuration in that
                    # list is a decision, not something to load.
                    ('qwen27b-q8-pair', {'models': ['qwen27b-q8-pair', SOLO]},
                        'is a configuration and cannot be loaded as a model', 'qwen27b-q8-pair')]:
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
                # A model that owns every text GPU of this host owns it alone, so this exemplar loads
                # nothing beside it -- which is the rule, not an inconvenience of the fixture.
                definition['models'] = ['qwen27b-q8-pair']
                definition['arguments'].update({'--device': 'CUDA0,CUDA1,CUDA2,CUDA3',
                    '--tensor-split': '1,1,1,1', '--spec-draft-device': 'CUDA4'})
                definition['arguments'].update(change)
                path.write_text(json.dumps(definition))
                return render(config, vision_host() if host is None else host, 'qwen27b-q8-pair')

            reserved = solo_llama()['compose']['services'][CODING]['deploy']['resources']['reservations']['devices'][0]
            self.assertEqual(reserved['device_ids'], [*text, VISION])
            # Nothing else is loaded to share the vision card, so it is reserved last and the
            # language model never lands on it.
            argv = solo_llama()['compose']['services'][CODING]['command']
            self.assertEqual(argv[argv.index('--device') + 1], 'CUDA0,CUDA1,CUDA2,CUDA3')
            for change, message in [({'--device': 'CUDA0,CUDA1,CUDA2'}, 'exactly once'),
                    ({'--device': 'CUDA0,CUDA1,CUDA2,CUDA4'}, 'exactly once'),
                    ({'--device': 'CUDA0,CUDA1,CUDA2,CUDA2'}, 'exactly once'),
                    ({'--tensor-split': '30,30,40'}, 'one value per text GPU'),
                    ({'--spec-draft-device': 'CUDA5'}, 'reserved GPU')]:
                with self.subTest(arguments=change), self.assertRaisesRegex(RuntimeError, message):
                    solo_llama(**change)
            # The projector and the draft live on the one card no text model uses, so a host that was
            # never given one cannot run a llama.cpp model that owns the machine at all.
            with self.assertRaisesRegex(RuntimeError, 'configured vision GPU'):
                solo_llama(BASELINE['host'])
            thin = vision_host(); thin['vision_gpu_id'] = 'GPU-night-2'
            with self.assertRaisesRegex(RuntimeError, 'full GPU UUID'):
                solo_llama(thin)
            for name, change, message in [
                    (SOLO, {'exclusive': 'yes'}, 'true or false'),
                    (SOLO, {'gpu_group': 'daytime'}, '"all" GPU group'),
                    ('qwen27b-q8-pair', {'gpu_group': 'all'}, 'unknown GPU group'),
                    # Exclusivity is a fact about one model and its memory: the Everyday model naming
                    # it without naming every card is refused through the list that loads it.
                    (EVERYDAY_FILE, {'exclusive': True}, '"all" GPU group')]:
                shutil.rmtree(config); shutil.copytree(ROOT / 'config', config)
                target = config / 'profiles' / (name + '.json')
                definition = read(target); definition.update(change); target.write_text(json.dumps(definition))
                with self.subTest(profile=name, change=change), self.assertRaisesRegex(RuntimeError, message):
                    render(config, vision_host(), name if name == SOLO else 'qwen27b-q8-pair')

    def test_a_solo_configuration_needs_every_text_card_named(self):
        # Every text card is reserved, so a host that cannot name four distinct cards is refused
        # before a drain is taken rather than after a backend has been half-built on three of them.
        host = vision_host(); host['gpu_ids']['nighttime'][0] = host['gpu_ids']['daytime'][0]
        with self.assertRaisesRegex(RuntimeError, 'reserves a card twice'):
            render(ROOT / 'config', host, SOLO)
        host = vision_host(); host['gpu_ids']['nighttime'] = host['gpu_ids']['nighttime'][:1]
        with self.assertRaisesRegex(RuntimeError, 'two real GPU UUIDs'):
            render(ROOT / 'config', host, SOLO)

    def test_ram_prompt_cache_is_shared_for_daytime_and_lowered_only_for_nighttime(self):
        shared = read(ROOT / 'config/shared.json')
        self.assertEqual(shared['arguments']['--cache-ram'], '49152')
        # Every configuration's own model inherits the shared 48 GiB lazy LRU cap. The abliterated
        # model alone overrides it to 24 GiB: on the 123 GiB host its cache had grown to ~50 GB with
        # ~8 GB swapped and swap exhausted, so the worst-case combined pair is bounded at 72 GiB
        # instead of 96 GiB.
        for name in CONFIGURATIONS:
            with self.subTest(profile=name):
                definition = profile_file(name)
                self.assertNotIn('--cache-ram', definition['arguments'])
        night = profile_file(EVERYDAY_FILE)
        self.assertEqual(night['arguments']['--cache-ram'], '24576')
        self.assertIn('--cache-ram', night['argument_order'])
        # Only the flag-configured engines are given a cache budget at all, so only they publish one.
        for name in [n for n in CONFIGURATIONS if interface_of(n) == 'argv']:
            with self.subTest(profile=name):
                rendered = render(ROOT / 'config', host_for(name), name)
                self.assertEqual(rendered['catalog']['global_ram_prompt_cache_mib'], 49152)
                # A model is published with the budget its own file states, whichever configuration is
                # asking for it: the lowered one travels with the abliterated model and not with the
                # configuration that happens to load it.
                self.assertEqual({m['model']: m['global_ram_prompt_cache_mib']
                    for m in rendered['catalog']['models']},
                    {model: (24576 if model == EVERYDAY else 49152) for model in NAMED[name]})
                for model, cfg in rendered['compose']['services'].items():
                    argv = cfg['command']
                    self.assertEqual(argv.count('--cache-ram'), 1)
                    self.assertEqual(argv[argv.index('--cache-ram') + 1],
                        '24576' if model == EVERYDAY else '49152')

    def test_every_backend_uses_one_of_the_pinned_engines(self):
        engines = read(ROOT / 'config/shared.json')['engines']
        # The Flash-Next branch engine retired with the llama.cpp Flash-Next configurations; the 27B
        # pair and the abliterated model use 836d571, and the Strata engine is the only non-llama.cpp
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
        # build-time decision (BUILD_VISION) and not a runtime switch, so giving a launch its
        # picture encoder means a second image. The text-only entry stays registered for exactly the
        # same reason the 0.1.43 one does -- undoing the vision step is a repoint, not a rebuild.
        #
        # A fifth, dated 2026-10-20: the same revision compiled for sm_75, built on the machine that has
        # the cards for it. A separate entry rather than a widened `cuda_architectures` on the one in
        # service, because an engine pins one image id -- and because that image carries no SASS a
        # Turing card can execute: it starts, passes its own health check, and dies at its first kernel
        # launch, which is what require_engine_architecture refuses at render now.
        self.assertEqual(set(engines), {'qwen38-dual-836d571',
                                        'strata-7695859aa13a', 'strata-296b59d2f73c',
                                        'strata-vision-296b59d2f73c', 'strata-sm75-296b59d2f73c'})
        engine = engines['qwen38-dual-836d571']
        self.assertEqual(engine['revision'], '836d57176dc699a726c55418e4f96b8ca628e1bf')
        self.assertEqual(engine['tag'], 'local/llama.cpp:qwen38-dual-836d57176dc6')
        self.assertEqual(engine['image_id'], 'sha256:9f7179f568e4aa1004c8af9b613b65417e6f0e451f8cfbc774d6b0d0f50a0ecd')
        expected = {'qwen27b-q8-pair': 'qwen38-dual-836d571', 'qwen27b-q6k-pair': 'qwen38-dual-836d571',
            'flash-next-strata-solo-thinking-recovery': 'strata-vision-296b59d2f73c',
            'flash-next-strata-solo-thinking-recovery-cache': 'strata-vision-296b59d2f73c',
            'flash-next-strata-solo-thinking-recovery-cache-yarn': 'strata-vision-296b59d2f73c',
            'flash-next-strata-two-card-iq4-xs': 'strata-vision-296b59d2f73c',
            'flash-next-strata-two-card-iq3-s': 'strata-vision-296b59d2f73c',
            'flash-next-strata-iq3-xxs-daisy': 'strata-sm75-296b59d2f73c',
            EVERYDAY_FILE: 'qwen38-dual-836d571'}
        # Every profile file in the tree is accounted for: a configuration, or a model one names. A
        # file that is neither is a launch no configuration renders and no test renders either.
        self.assertEqual(set(expected), {path.stem for path in (ROOT / 'config/profiles').glob('*.json')})
        for name, engine_name in expected.items():
            with self.subTest(profile=name):
                self.assertEqual(profile_file(name)['engine'], engine_name)
        for name in CONFIGURATIONS:
            bundle = render(ROOT / 'config', host_for(name), name)
            for model in models_of(ROOT / 'config', profile_file(name)):
                service = service_key(model)
                with self.subTest(profile=name, service=service):
                    # Loading two models is not the same as being one kind of engine: the two-card
                    # Strata probe loads the abliterated model on the other pair exactly as the 27B
                    # pair does, and runs a different image as itself. Each model is held to its own
                    # engine, which is the stronger reading and the one that stays true as the house
                    # grows -- and it is only readable because the service is filed under the model.
                    want = engines[model['engine']]
                    self.assertEqual(bundle['compose']['services'][service]['image'], want['tag'])
                    self.assertEqual(service_engine(bundle, service), want)

    def test_nighttime_uses_tensor_parallel_placement_across_its_pair(self):
        night = profile_file(EVERYDAY_FILE)
        self.assertEqual(night['arguments']['--split-mode'], 'tensor')
        self.assertIn('--split-mode', night['argument_order'])
        for flag, value in {'--tensor-split': '60,40', '--device': 'CUDA0,CUDA1', '--n-gpu-layers': 'all',
                '--cache-ram': '24576'}.items():
            with self.subTest(argument=flag):
                self.assertEqual(night['arguments'][flag], value)
        self.assertEqual(night['arguments']['--spec-type'], 'draft-mtp')  # MTP3 since 2026-10-07, on the vision GPU
        for name in LOADS[EVERYDAY]:
            with self.subTest(profile=name):
                argv = render(ROOT / 'config', host_for(name), name)['compose'][
                    'services'][EVERYDAY]['command']
                self.assertEqual(argv.count('--split-mode'), 1)
                self.assertEqual(argv[argv.index('--split-mode') + 1], 'tensor')
                self.assertEqual(argv[argv.index('--flash-attn') + 1], 'on')  # required by tensor mode
                self.assertEqual(argv[argv.index('--fit') + 1], 'off')  # fitting is not implemented for tensor mode

    def test_context_ceiling_is_the_router_contract(self):
        # The ceiling is the coordinated number it shares with LLM Router's RESIDENT_CONTEXT_LIMIT:
        # widening it is a two-repo release, and the router half shipped first (llm-router 0d9dc83,
        # 2026-10-18) because publishing a catalog above a router's ceiling 503s every model route.
        # The native window of the Qwen3.8 pack is 262,144; everything above it is rope scaling and
        # the profile must state the ring (engine_rope_scaling refuses the rest at render).
        from runtime.config import ROUTER_CONTEXT_LIMIT
        self.assertEqual(ROUTER_CONTEXT_LIMIT, 524288)
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            path = config / 'profiles/qwen27b-q6k-pair.json'
            definition = read(path)
            for tokens in (524289, 0, '524288'):
                with self.subTest(tokens=tokens):
                    definition['context_tokens'] = tokens
                    path.write_text(json.dumps(definition))
                    with self.assertRaisesRegex(RuntimeError, 'router contract: 1–524288 tokens'):
                        render(config, BASELINE['host'], 'qwen27b-q6k-pair')
                    with self.assertRaisesRegex(RuntimeError, 'router contract'):
                        available_profiles(config)
            definition['context_tokens'] = 524288
            path.write_text(json.dumps(definition))
            self.assertEqual(render(config, BASELINE['host'], 'qwen27b-q6k-pair')['catalog']['context_length'], 524288)

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


class AConfigurationNamesItsModels(unittest.TestCase):
    """How many models a configuration loads is written in its own file, and it has to be written well.

    Every one of these refusals is a configuration that would otherwise render something other than
    what its author read: a list that does not include the file it is written in, a model named twice
    onto two cards, a configuration loaded as if it were a model. None of them can serve, and a
    machine that started one would be running a decision nobody selected.
    """

    def test_the_models_a_configuration_names_are_loaded_in_that_order(self):
        # The switch brings a configuration's models up one at a time in the order the file names
        # them, so the list is load-bearing for the drain: a renderer that sorted the models, or that
        # always put the configuration's own model first, would change what comes up when without
        # changing a byte of any profile.
        config = config_with('qwen27b-q8-pair', models=[EVERYDAY_FILE, 'qwen27b-q8-pair'])
        bundle = render(config, BASELINE['host'], 'qwen27b-q8-pair')
        self.assertEqual(list(bundle['compose']['services']), [EVERYDAY, CODING])
        self.assertEqual([s['service'] for s in bundle['manifest']['services']], [EVERYDAY, CODING])
        self.assertEqual([m['model'] for m in bundle['catalog']['models']], [EVERYDAY, CODING])

    def test_a_configuration_that_names_two_models_loads_two_services(self):
        # One file, as many launches as it names: the number of Compose services is the number of
        # entries in the list, which is the whole content of the change and what every old assertion
        # about "the coding half and the everyday half" used to stand in for.
        for name in CONFIGURATIONS:
            with self.subTest(profile=name):
                bundle = render(ROOT / 'config', host_for(name), name)
                self.assertEqual(sorted(bundle['compose']['services']), sorted(NAMED[name]),
                    name + ' does not load one service per model it names')
        # And the count comes out of the list rather than out of a rule about configurations: the 27B
        # pair, told to load only its own model, loads one service. It needs a machine with the one
        # pair it keeps, because a declared card with nothing placed on it is refused -- as it should
        # be, and that refusal is the other half of what the list means.
        only = render(config_with('qwen27b-q8-pair', models=['qwen27b-q8-pair']),
            one_pair_host(ROSALINA), 'qwen27b-q8-pair')
        self.assertEqual(list(only['compose']['services']), [CODING])

    def test_a_configuration_must_name_the_models_it_loads(self):
        for models, message in [
                ([], 'must be a non-empty list'),
                ('everyday', 'must be a non-empty list'),
                (['qwen27b-q6k-pair'], 'must name itself'),
                (['qwen27b-q8-pair', 'qwen27b-q8-pair'], 'twice'),
                (['qwen27b-q8-pair', SOLO], 'is a configuration and cannot be loaded as a model')]:
            with self.subTest(models=models):
                with self.assertRaisesRegex(RuntimeError, message):
                    render(config_with('qwen27b-q8-pair', models=models),
                        BASELINE['host'], 'qwen27b-q8-pair')
        # A configuration file with the list taken out is not "one model, implicitly". A file that
        # states no decision is refused, because the answer used to be that it was stated twice.
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            path = config / 'profiles/qwen27b-q8-pair.json'
            definition = read(path); definition.pop('models')
            path.write_text(json.dumps(definition))
            with self.assertRaisesRegex(RuntimeError, 'a configuration must name the models it loads'):
                render(config, BASELINE['host'], 'qwen27b-q8-pair')

    def test_a_model_that_names_a_list_of_its_own_is_a_configuration(self):
        # The list is what makes a file a configuration, so naming one is not a detail about who loads
        # it: `everyday` carrying its own list would be a decision somebody can select, filed under
        # the name another configuration is loading as a model.
        with self.assertRaisesRegex(RuntimeError, 'is a configuration and cannot be loaded as a model'):
            render(config_with(EVERYDAY_FILE, models=[EVERYDAY_FILE]),
                BASELINE['host'], 'qwen27b-q8-pair')


class RegistryTests(unittest.TestCase):
    def test_every_selectable_configuration_agrees_with_its_rendered_catalog(self):
        registry = available_profiles(ROOT / 'config', BASELINE['host'])
        self.assertEqual(CONFIGURATIONS, ('qwen27b-q8-pair', 'qwen27b-q6k-pair',
            'flash-next-strata-solo-thinking-recovery', 'flash-next-strata-solo-thinking-recovery-cache',
            'flash-next-strata-solo-thinking-recovery-cache-yarn', 'flash-next-strata-two-card-iq4-xs',
            'flash-next-strata-two-card-iq3-s', 'flash-next-strata-iq3-xxs-daisy'))
        # Loading more than one model is a statement about cards -- one model on one text pair and
        # another on the other -- and not about which engine the first of them runs, which is why a
        # Strata configuration joined this set on 2026-10-20 without either 27B pair changing.
        self.assertEqual(tuple(n for n in CONFIGURATIONS if len(NAMED[n]) > 1),
            ('qwen27b-q8-pair', 'qwen27b-q6k-pair',
             'flash-next-strata-two-card-iq4-xs', 'flash-next-strata-two-card-iq3-s'))
        self.assertEqual([x['profile'] for x in registry['selectable']], list(CONFIGURATIONS))
        # The models a configuration loads are listed with it, in the order it names them. A model an
        # operator cannot see they are about to load is a model they did not choose.
        self.assertEqual([[m['model'] for m in x['models']] for x in registry['selectable']],
            [NAMED[name] for name in CONFIGURATIONS])
        for name in CONFIGURATIONS:
            with self.subTest(profile=name):
                entry = next(x for x in registry['selectable'] if x['profile'] == name)
                rendered = render(ROOT / 'config', host_for(name), name)
                model = next(m for m in rendered['catalog']['models'] if m['model'] == entry['model'])
                engine = service_engine(rendered, OWN[name])
                self.assertEqual(entry['display_name'], model['display_name'])
                self.assertEqual(entry['context_tokens'], model['context_length'])
                self.assertEqual(entry['engine_tag'], engine['tag'])
                self.assertEqual(entry['backend_revision'], engine['revision'])
                self.assertEqual(entry['exclusive'], exclusive(name))
                self.assertEqual(rendered['catalog']['configuration'],
                    {'id': name, 'exclusive': exclusive(name)})
                # Nothing a client sends is left unpublished, in any configuration: the one that loads
                # a single model answers `nighttime` from that model, so the router has no offline
                # service to report for it either. What differs is only how many models are launched.
                self.assertEqual(rendered['catalog']['offline_services'], [])
                # One service per model the file names, in the order it names them, each filed under
                # the model it loads -- which is the claim this whole change rests on.
                self.assertEqual(list(rendered['compose']['services']), NAMED[name])
                self.assertEqual([m['model'] for m in rendered['catalog']['models']], NAMED[name])
                for listed in entry['models']:
                    published = next(m for m in rendered['catalog']['models']
                        if m['model'] == listed['model'])
                    with self.subTest(model=listed['model']):
                        self.assertEqual(listed['display_name'], published['display_name'])
                        self.assertEqual(listed['context_tokens'], published['context_length'])
                        self.assertEqual(listed['service_ids'], sorted(published['aliases']))
                        self.assertEqual(listed['exclusive'], published.get('exclusive', False))
                # What a client can ask for is the union over the models loaded; what the
                # configuration is named after is a separate fact, because "what would answer" and
                # "what is loaded" are different questions and neither answers the other.
                self.assertEqual(entry['service_ids'],
                    sorted({sid for m in entry['models'] for sid in m['service_ids']}))
                self.assertEqual(entry['model_service_ids'], sorted(model['aliases']))
        # The model several configurations share is one launch, not one rendering per configuration
        # that names it: identical image, identical argv, identical GPU reservation.
        launches = {json.dumps(render(ROOT / 'config', vision_host(), name)['compose']['services'][EVERYDAY],
                   sort_keys=True) for name in LOADS[EVERYDAY]}
        self.assertEqual(len(launches), 1, 'the Everyday model is a different launch depending on '
            'which configuration asks for it')

    def test_catalog_names_each_backends_text_gpus_and_vision_gpu(self):
        host = vision_host()
        names = host['gpu_names']
        paired = render(ROOT / 'config', host, 'qwen27b-q8-pair')['catalog']
        coding, everyday = paired['models']
        self.assertEqual((coding['model'], everyday['model']), (CODING, EVERYDAY))
        self.assertEqual(coding['gpu_names'], names['daytime'])
        self.assertEqual(everyday['gpu_names'], names['nighttime'])
        self.assertEqual(len(coding['gpu_names']), len(coding['text_gpu_uuids']))
        self.assertEqual((coding['vision_gpu_name'], coding['vision_gpu_shared']), ('RTX 3080', True))
        self.assertEqual(paired['configuration'], {'id': 'qwen27b-q8-pair', 'exclusive': False})
        # The root projection is the first model's entry, names and all.
        self.assertEqual(paired['gpu_names'], coding['gpu_names'])
        [solo] = render(ROOT / 'config', host, SOLO)['catalog']['models']
        self.assertEqual(solo['gpu_names'], [*names['daytime'], *names['nighttime']])
        # A model that owns every text GPU holds the shared card to itself: nothing else is loaded to
        # share it with, so `vision_gpu_shared` says so rather than repeating what the configurations
        # that divide the machine say.
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
        # Router clients pick "the most capable NSFW model" from this flag. It is a fact about one
        # model and its pack -- only the abliterated one is -- stated the same way whichever
        # configuration loads it, and a configuration that answers `nighttime` from a model that is
        # not abliterated says so rather than letting the service ID imply it.
        for name in CONFIGURATIONS:
            with self.subTest(profile=name):
                catalog = render(ROOT / 'config', host_for(name), name)['catalog']
                flags = {m['model']: m['capability_profile']['nsfw'] for m in catalog['models']}
                self.assertEqual(flags, {model: model == EVERYDAY for model in NAMED[name]})
                # The document root is the first model published, and it is never the abliterated one.
                self.assertIs(catalog['models'][0]['capability_profile']['nsfw'], False)
                self.assertIs(catalog['capability_profile']['nsfw'], False)
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            path = config / 'profiles' / (EVERYDAY_FILE + '.json')
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
        # A profile that names the models it loads is a configuration, and every one has to be
        # registered; everything else in the directory is a model, and it is reachable only because
        # some configuration names it. A file that is neither is a launch nothing renders.
        configurations = {name for name, value in definitions.items() if 'models' in value}
        self.assertEqual(sorted(configurations), sorted(CONFIGURATIONS))
        named = {model for name in CONFIGURATIONS for model in profile_file(name)['models']}
        self.assertEqual(configurations | named, set(definitions),
            'a profile file no configuration names or loads is dead configuration')
        selectable = [x['profile'] for x in available_profiles(ROOT / 'config')['selectable']]
        for retired in ('daytime-swift', 'daytime-256', 'nighttime-256', 'nighttime', 'primary'):
            self.assertNotIn(retired, selectable)
        self.assertEqual(available_profiles(ROOT / 'config')['selectable'][0]['gpu_names'], [])
        listed = available_profiles(ROOT / 'config', BASELINE['host'])
        # Each model of a configuration is drawn with the cards it runs on, in the order the file
        # names them: the page reads the cards off this and not out of its own head.
        self.assertEqual([m['gpu_names'] for m in listed['selectable'][0]['models']],
            [BASELINE['host']['gpu_names']['daytime'], BASELINE['host']['gpu_names']['nighttime']])

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
        registry = available_profiles(ROOT / 'config')
        lines = listed(ROOT / 'config').splitlines()
        # One row per configuration, and under it one indented row per model it loads: `list` is the
        # command that answers "what could this machine be doing", and the answer is models -- one,
        # two, or as many as its cards hold -- rather than a configuration with a footnote.
        at = 0
        for entry in registry['selectable']:
            row = lines[at]; at += 1
            with self.subTest(profile=entry['profile']):
                self.assertTrue(row.startswith(entry['profile'] + ':'), row)
                self.assertIn(entry['display_name'], row)
                self.assertIn(entry['model'], row)
            for model in entry['models']:
                row = lines[at]; at += 1
                with self.subTest(model=model['model']):
                    self.assertTrue(row.startswith('    '), row)
                    self.assertIn(model['display_name'], row)
                    self.assertIn('every text GPU of this host' if model['exclusive']
                        else 'cards unknown', row)
                    self.assertIn(', '.join(model['service_ids']) or 'no service ID', row)
        self.assertEqual(at, len(lines), 'the listing says something the registry does not know')

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
                (EVERYDAY_FILE, lambda definition: definition.update(id='renamed'), 'differs from its filename'),
                # A model that owns every text GPU of this host cannot be named beside another one, and
                # a configuration that does so anyway is a configuration that lies about what it will
                # launch -- it would drain onto two launches that each need the whole machine.
                (SOLO, lambda definition: definition.update(models=[SOLO, EVERYDAY_FILE]),
                    'no other model can be loaded beside it'),
                ('qwen27b-q8-pair', lambda definition: definition.update(context_tokens=524289), 'router contract')]:
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
        service = self.bundle()['compose']['services'][STRATA_SOLO]
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
                if key not in definition['arguments']:
                    # An argument a profile may leave unstated -- the two refinements of a
                    # conversation cache this configuration has not switched on -- is one the renderer
                    # invents no variable for, because an invented variable is the engine being told
                    # something no configuration stated.
                    self.assertNotIn(variable, environment)
                else:
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
        service = bundle['compose']['services'][STRATA_SOLO]
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

    def test_a_host_with_no_shared_vision_gpu_renders_the_text_only_model_it_was(self):
        """host.json is the switch, and the profile is the same file whichever way it is set."""
        definition = read(ROOT / 'config/profiles' / (SOLO + '.json'))
        service = render(ROOT / 'config', BASELINE['host'], SOLO)['compose']['services'][STRATA_SOLO]
        self.assertEqual(service['environment']['CUDA_VISIBLE_DEVICES'], '0,1,2,3')
        self.assertEqual(service['environment']['VISION'], 'no')
        self.assertNotIn('VISION_DEVICE', service['environment'])
        self.assertEqual(service['deploy']['resources']['reservations']['devices'][0]['device_ids'],
            text_gpus(BASELINE['host'], definition))
        entry = render(ROOT / 'config', BASELINE['host'], SOLO)['catalog']['models'][0]
        self.assertEqual(entry['mmproj_offload'], 'none')
        self.assertNotIn('vision_gpu_uuid', entry)

    def test_the_sampling_preset_the_profile_states_is_the_variable_the_image_reads(self):
        # The engine's own default is greedy and its setup never writes the sampling block, so the
        # only thing that decides what a routed request that sends no sampling gets is this variable
        # and the entry point that writes it into the run config. Every configuration still selectable
        # states the preset the card asks for; a profile that states nothing renders nothing, which
        # leaves whatever the last-selected configuration wrote sitting in the workspace.
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            service = render(config, BASELINE['host'], SOLO)['compose']['services'][STRATA_SOLO]
            self.assertEqual(service['environment']['SAMPLING'], 'thinking')
            self.assertNotIn('VISION_DEVICE', service['environment'])
            path = config / ('profiles/' + SOLO + '.json')
            definition = read(path)
            definition['arguments'].pop('--sampling')
            definition['argument_order'].remove('--sampling')
            path.write_text(json.dumps(definition))
            service = render(config, BASELINE['host'], SOLO)['compose']['services'][STRATA_SOLO]
            self.assertNotIn('SAMPLING', service['environment'])

    def test_a_sampling_preset_the_entry_point_cannot_write_is_refused_before_any_drain(self):
        # A preset is a name from a table that lives in the image, because the engine refuses to
        # start on a sampling value it does not understand and this model is exclusive: a config
        # it refuses is nothing serving. So a value outside the table is a render error, not a boot
        # that dies after the drain, and numbers are refused for the same reason.
        for preset in ('temperature=1.0', '1.0', 'Thinking', 'greedy', '', True):
            with self.subTest(sampling=preset):
                with tempfile.TemporaryDirectory() as tmp:
                    config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
                    path = config / ('profiles/' + SOLO + '.json')
                    definition = read(path)
                    definition['arguments']['--sampling'] = preset
                    path.write_text(json.dumps(definition))
                    with self.assertRaisesRegex(RuntimeError, '--sampling states'):
                        render(config, BASELINE['host'], SOLO)
        # And stating nothing at all is the engine's own default, which is what every profile that
        # has never heard of this key gets.
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            path = config / ('profiles/' + SOLO + '.json')
            definition = read(path)
            definition['arguments'].pop('--sampling')
            definition['argument_order'].remove('--sampling')
            path.write_text(json.dumps(definition))
            service = render(config, BASELINE['host'], SOLO)['compose']['services'][STRATA_SOLO]
            self.assertNotIn('SAMPLING', service['environment'])

    def test_the_encoder_is_placed_at_the_end_of_whatever_cards_the_container_is_given(self):
        # The encoder takes its card from a number in the container's own CUDA_VISIBLE_DEVICES, and the
        # number is derived from how many text cards this configuration reserves -- not from a host
        # integer. One host states one host.json, and it now serves a solo configuration whose text
        # model takes four cards and a paired one whose text model takes two, so the same physical
        # card is CUDA4 in one container and CUDA2 in the other. What keeps the encoder off a text
        # card is that the shared card is refused as a text-pair member and appended last: a host
        # stating any other number is ignored, and only a value that cannot be an index at all is a
        # typo worth refusing over.
        for stated in (0, 3, 4, 9):
            host = vision_host()
            host['vision_cuda_device'] = stated
            with self.subTest(vision_cuda_device=stated):
                bundle = render(ROOT / 'config', host, SOLO)
                service = bundle['compose']['services'][STRATA_SOLO]
                visible = [int(x) for x in service['environment']['CUDA_VISIBLE_DEVICES'].split(',')]
                # The number is derived, so every int the host states is ignored. What places the
                # encoder is that the shared card is the last card this container is given, and the
                # index names it: the reserved list ends on the vision GPU, and the container numbers
                # what it is given from 0.
                self.assertEqual(visible, list(range(len(visible))))
                self.assertEqual(bundle['manifest']['services'][0]['gpu_device_ids'][-1], VISION)
                self.assertEqual(service['environment']['VISION_DEVICE'], str(len(visible) - 1))
        with self.subTest(vision_cuda_device='4'), self.assertRaisesRegex(RuntimeError, 'is an int'):
            host = vision_host(); host['vision_cuda_device'] = '4'
            render(ROOT / 'config', host, SOLO)

    def test_a_two_card_strata_layout_places_its_encoder_beside_two_text_cards(self):
        # The rendering this whole change exists for: the same host.json, the same image, and a Strata
        # configuration whose text model takes half the cards. Before 2026-10-20 the two layouts were
        # mutually exclusive in one host file, because the solo layout's number was the only one the
        # host could state.
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            path = config / ('profiles/' + SOLO + '.json')
            definition = read(path)
            definition.pop('exclusive'); definition['gpu_group'] = 'daytime'
            # Half the cards for this model means the other half is somebody's: a configuration that
            # left a declared card unplugged leaves a card the operator believes is serving with
            # nothing that can be placed on it.
            definition['models'] = [SOLO, EVERYDAY_FILE]
            definition['arguments']['--gpus'] = '0,1'
            definition['container']['environment']['CUDA_VISIBLE_DEVICES'] = '0,1'
            path.write_text(json.dumps(definition))
            service = render(config, vision_host(), SOLO)['compose']['services'][STRATA_SOLO]
            self.assertEqual(service['environment']['CUDA_VISIBLE_DEVICES'], '0,1,2')
            self.assertEqual(service['environment']['VISION_DEVICE'], '2')
            self.assertEqual(service['environment']['GPUS'], '0,1')
            self.assertEqual(service['environment']['VISION'], 'yes')
            # And the solo layout still renders on that same host, at its own number.
            solo = render(ROOT / 'config', vision_host(), SOLO)['compose']['services'][STRATA_SOLO]
            self.assertEqual(solo['environment']['CUDA_VISIBLE_DEVICES'], '0,1,2,3,4')
            self.assertEqual(solo['environment']['VISION_DEVICE'], '4')

    def test_an_engine_that_cannot_name_the_encoder_card_is_lent_no_card(self):
        # Rolling this profile back to the text-only image of the same revision is a profile change, and
        # it has to render the way it did before the encoder: four cards, no VISION_DEVICE, text in the
        # catalog. An image with no way to place an encoder is never handed a fifth GPU to wander onto.
        # The statements this house's entry point writes -- the encoder's card, the sampling preset,
        # the tool-call switch, the conversation cache and the PLE table -- are the image's, so the
        # rollback drops those lines with it rather than rendering variables nothing in that image reads.
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            path = config / ('profiles/' + SOLO + '.json')
            definition = read(path)
            definition['engine'] = 'strata-296b59d2f73c'
            path.write_text(json.dumps(definition))
            # Each of the statements this entry point writes refuses on its own, and only after the one
            # before it has been taken out: the refusal is raised at the first argument the image cannot
            # be told, so a profile that states several names only the first.
            for flag in ('--sampling', '--tool-call-recovery', '--conversation-cache-mib',
                         '--conversation-cache-min-free-mib', '--ple-gguf'):
                with self.subTest(argument=flag):
                    with self.assertRaisesRegex(RuntimeError, flag + ' has no environment variable'):
                        render(config, vision_host(), SOLO)
                    definition['arguments'].pop(flag, None)
                    if flag in definition['argument_order']:
                        definition['argument_order'].remove(flag)
                    path.write_text(json.dumps(definition))
            for key in ('--sampling', '--tool-call-recovery', '--vision-device'):
                definition['arguments'].pop(key, None)
                if key in definition['argument_order']:
                    definition['argument_order'].remove(key)
            path.write_text(json.dumps(definition))
            bundle = render(config, vision_host(), SOLO)
            service = bundle['compose']['services'][STRATA_SOLO]
            self.assertEqual(service['environment']['CUDA_VISIBLE_DEVICES'], '0,1,2,3')
            self.assertEqual(service['environment']['VISION'], 'no')
            self.assertNotIn('VISION_DEVICE', service['environment'])
            self.assertNotIn('SAMPLING', service['environment'])
            self.assertEqual(
                len(service['deploy']['resources']['reservations']['devices'][0]['device_ids']), 4)
            self.assertEqual(bundle['catalog']['models'][0]['input_modalities'], ['text'])

    def test_the_rope_statements_refuse_on_the_image_without_the_entry_point(self):
        # The rollback story for the ring is the story for every entry-point fact: the text-only tag
        # runs upstream's entry point, there is nothing in that image to write a rope into the run
        # config, and the refusal names what a profile can do instead (drop the line, which rolls a
        # 512K promise back to a trained-window one -- and the ceiling refuses the window by itself).
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            name = 'flash-next-strata-solo-thinking-recovery-cache-yarn'
            path = config / ('profiles/' + name + '.json')
            definition = read(path)
            definition['engine'] = 'strata-296b59d2f73c'
            path.write_text(json.dumps(definition))
            for flag in ('--sampling', '--tool-call-recovery', '--conversation-cache-mib',
                         '--conversation-cache-min-free-mib', '--conversation-cache-slots',
                         '--conversation-cache-min-tokens', '--ple-gguf',
                         '--rope-scaling', '--rope-scale', '--yarn-orig-ctx'):
                with self.subTest(argument=flag):
                    with self.assertRaisesRegex(RuntimeError, flag + ' has no environment variable'):
                        render(config, vision_host(), name)
                    definition['arguments'].pop(flag, None)
                    definition['argument_order'].remove(flag)
                    path.write_text(json.dumps(definition))
            # With every entry-point statement gone the image renders -- the window is what is left
            # of the 512K claim, and it is the contract that answers: 262,144 or nothing.
            bundle = render(config, vision_host(), name)
            environment = bundle['compose']['services'][STRATA_SOLO]['environment']
            for variable in ('ROPE_SCALING', 'ROPE_SCALE', 'YARN_ORIG_CTX', 'SAMPLING'):
                self.assertNotIn(variable, environment)
            definition['context_tokens'] = 262144
            path.write_text(json.dumps(definition))
            self.assertEqual(render(config, vision_host(), name)['catalog']['context_length'], 262144)

    def test_an_exclusive_model_is_published_without_llama_cpp_only_claims(self):
        bundle = self.bundle()
        entry = bundle['catalog']['models'][0]
        definition = read(ROOT / 'config/profiles' / (SOLO + '.json'))
        self.assertEqual(entry['model'], definition['arguments']['--alias'])
        # A model of this machine, so it answers the service IDs a client of this machine sends --
        # including `nighttime`, which it answers itself rather than leaving a client with nothing.
        self.assertEqual(set(entry['aliases']), set(CLIENT_SERVICES))
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
        cfg = self.bundle()['compose']['services'][STRATA_SOLO]
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
        # image digest in ops/reports/20261017-strata-sampling/README.md. The text-only image of the
        # same revision is a different image with a different digest, because the encoder is compiled
        # in at build time and is not a runtime switch.
        #
        # Rebuilt on 2026-10-17 for one layer and nothing else: the same upstream base
        # (strata-upstream:296b59d2f73c-vision, both base digests unchanged) with this house's entry
        # point, whose sha256 (8b8a5ef3862ce0ac42fd0e4338b933bd303a68ae9005178ff36b605dbe5745a3) is
        # verified equal to the file in runtime/overlay/ inside the built image, and whose upstream
        # half (e85f40f57d036d1247804321a1164be707873712851e093998df97a2398b76b2) is the file it execs.
        # The first rebuild of that day lost the identity labels and this line caught it: the id was
        # pinned to an image whose inspect output carried no org.opencontainers.image.revision, the
        # candidate built, and the deploy was refused as `pinned inference engine identity changed`
        # with the box still serving the revision before it. The recipe in runtime/overlay/Dockerfile
        # takes the labels as required build arguments and refuses a build that omits them, and
        # test_the_overlay_layer_declares_the_identity_the_runtime_reads_back keeps it that way.
        # The pin below is the id that recipe builds today, entry point included: the id covers the
        # text of the Dockerfile as well as its layers, so editing the recipe above -- even a comment
        # -- means rebuilding and re-pinning this line, never remembering an old build.
        # The image this line pointed at before the rebuild is kept on the box as
        # local/strata:296b59d2f73c-vision-pre-cache, so this line is the rollback: one re-pin, one
        # deploy, and the box is running the entry point that knows nothing about a conversation cache.
        #
        # Rebuilt again on 2026-10-18, again for one layer: the entry point, now
        # 7adc6bae12393a6298a560fda0e6ff5af7606407fa69ad02bc0ba9abd5c0dcb5, which writes the session
        # window the profile states into the args list the engine reads it from. The upstream half is
        # byte for byte the file above (e85f40f57d036d1247804321a1164be707873712851e093998df97a2398b76b2),
        # the base and both base digests are the same, and 22 layers come back identical at every
        # index but the one that carries the entry point. Proven by running the entry point in the
        # built image against a copy of this box's real run config, at the production environment with
        # CONTEXT raised: the args list came back with two entries different, `163840` out and
        # `196608` in, every other key of the file byte for byte what it was, and the line "already"
        # for the encoder's card, the sampling preset, the recovery switch and the cache budget. Built
        # from the canonical build dir on the box (~/ops/strata-build/overlay, the recipe in
        # runtime/overlay/), log build-0.1.44.1-vision-context-20261018.log beside it. Receipt
        # ops/reports/20261018-strata-vision-context/README.md. The image before it is kept as
        # local/strata:296b59d2f73c-vision-pre-context.
        #
        # And again on 2026-10-18 for the fact-8 layer: the entry point, now
        # ee4cf6b909700eb58ce27ddf98f00c650cbb3c5698e76d98baee6939c81a0495, writes the position ring
        # a rope-scaled window is read through (ROPE_SCALING/ROPE_SCALE/YARN_ORIG_CTX -> --rope-scaling
        #/--rope-scale/--yarn-orig-ctx) and strips any ring the profile does not state, because a
        # 262K sibling must never boot through the 512K configuration's factor in silence. Same base,
        # same upstream half, built from /tmp/overlay-rope with the canonical recipe; the image before
        # it is kept as local/strata:296b59d2f73c-vision-pre-rope, which is also what a rollback of
        # this pin boots.
        # And again on 2026-10-20 for facts 9-11: the entry point, now
        # 8dfd5dd2680b53c7526987c9d5536c2cca637310862a988404d7b58cc1846124, writes the per-card VRAM
        # reserve, the control vector with its layer range, and the engine's durable log path -- the
        # three settings this box had been carrying as hand edits inside the workspace. Built from
        # ~/tmp/overlay-build-20261020 on the same base and the same upstream half (receipt
        # ops/reports/20261020-strata-vision/); the image before it is kept as
        # local/strata:296b59d2f73c-vision-pre-cvec, which is also what a rollback of this pin boots.
        # Note what a pin change is: image_id is inside the bundle, so this moved the config_sha256 of
        # the three configurations that run this image and of none that does not -- tests/fixtures/
        # active-bundles.json is where that shows up, and the container in service keeps running the
        # image it was made from until a switch moves it. BuildKit attaches a per-build attestation,
        # so this id is this box's build of that entry point and not a reproducible digest: the entry
        # point's own sha256, verified inside the image, is what identifies the content.
        self.assertEqual(engine['image_id'],
                         'sha256:79fc1949f635359f4428b174b4fbc4ed5f5ea0077591a1e7067a8dc1cdcefc62')
        self.assertEqual(engine['repository'], 'https://github.com/Niko1221/Strata')
        self.assertEqual(engine['cuda_architectures'], ['86', '89'])
        self.assertEqual(engine['interface'], 'environment')
        self.assertGreater(engine['readiness_wait_seconds'], 450)
        # Being told which card its encoder runs on is the whole difference between an engine the
        # renderer may lend the shared vision GPU to and one it has to leave the card out of.
        self.assertEqual(engine['environment']['--vision-device'], 'VISION_DEVICE')

    def test_a_new_lever_ships_with_its_measurement_and_its_instrument(self):
        # A configuration that states a hypothesis is a question the portal shows as an answer, and
        # the two levers this one added were exactly the kind that get selected on the prose. So each
        # carries the run that decided it, in the warning the portal renders, and the scripts that
        # produced it are in the repo: the route-skip verdict is unreadable without the comparator
        # that prints confidence intervals, and the conversation cache is invisible to the benchmark
        # altogether, because a benchmark never lets its own conversations go cold.
        for profile in sorted(n for n in CONFIGURATIONS if interface_of(n) == 'environment'):
            # Derived from the engine interface rather than a list of names: a configuration added
            # tomorrow cannot opt out of carrying the measurement it was built on by not being in a
            # tuple somebody remembered to edit. The date and the percentage are looked for anywhere in
            # a warning, because this house writes a verdict as prose ("STRATA_HC_Q8=1 was measured on
            # 2026-10-14 and reverted: ...") and not as a header format.
            warnings = read(ROOT / 'config/profiles' / (profile + '.json'))['catalog']['deployment_warnings']
            measured = [w for w in warnings
                        if re.search(r'(?:Measured|measured|tried) on 2026-\d\d-\d\d', w)
                        and re.search(r'[-+]?\d[\d.]*\s*%', w)]
            self.assertTrue(measured, profile + ' ships a lever with no dated measurement in its '
                'warnings: the portal renders this prose as the reason to select it')
        # The instruments themselves live in scripts/, which the runtime image does not copy -- the
        # candidate image this test also runs inside ships runtime, config and tests -- so their
        # presence is asserted where they are read, from a checkout, while the verdict above is
        # asserted everywhere because the catalog the portal renders is in the image.
        if (ROOT / 'scripts').is_dir():
            for script, evidence in (('strata_bench_compare.py', 'invisible'), ('strata_park_probe.py', 'restored')):
                path = ROOT / 'scripts' / script
                compile(path.read_text(), str(path), 'exec')
                self.assertIn(evidence, path.read_text())
    def test_the_overlay_layer_declares_the_identity_the_runtime_reads_back(self):
        # Strata's own Dockerfile labels only the NVIDIA base, so the image this house ships is the
        # only place the identity labels exist, and a rebuild of the house layer that forgets them
        # produces an image that runs perfectly and deploys not at all: the switch guard, the update
        # cutover, the migration check and the recovery path each read org.opencontainers.image.
        # revision out of `docker image inspect` and refuse it against the manifest. So the recipe is
        # pinned here, against the engine it has to produce, and not only in prose.
        overlay = (ROOT / 'runtime' / 'overlay' / 'Dockerfile').read_text()
        stage = overlay.split('FROM ${UPSTREAM}', 1)[1]
        for label in ('org.opencontainers.image.revision=${REVISION}',
                      'org.opencontainers.image.source=${REPOSITORY}',
                      'io.service-portal.trial=${TRIAL}'):
            self.assertIn('LABEL ' + label, overlay)
        # An ARG above FROM reaches only the FROM line, and BuildKit expands one the stage never
        # declared to the empty string -- labels that build, ship and read back blank.
        for name in ('REVISION', 'REPOSITORY', 'TRIAL'):
            self.assertIn('ARG ' + name, stage)
        self.assertIn('test -n "$REVISION"', stage)
        engine = ENGINES[read(ROOT / 'config/profiles' / (SOLO + '.json'))['engine']]
        self.assertIn('--build-arg REVISION=' + engine['revision'], overlay)
        self.assertIn('--build-arg REPOSITORY=' + engine['repository'], overlay)

    # Closed 2026-10-17. flash-next-strata-solo used to answer to its host names through a hand edit
    # inside its own workspace's engine config (strata-unsloth-ud-iq4_xs.json) rather than through
    # source, and was exempt here only because declaring them would recreate the container that was
    # serving. The sampling change recreates that container anyway, so the names are declared now --
    # in both the configuration and its sampling variant, which is the same backend -- and the
    # exemption is gone. Nothing in the workspace has to survive for a backend to answer the names
    # the controller probes it by.

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

        A profile whose names were carried by its workspace instead of by source is held to the same
        equality; that deviation was closed on 2026-10-17, and the check is what keeps it closed.
        """
        environment = sorted(n for n in CONFIGURATIONS if interface_of(n) == 'environment')
        self.assertEqual(len(environment), 6, 'the environment-interface configurations are the '
            'call-recovery, conversation-cache and rope-rung ones, the two two-card arms, and the second '
            'machine; the greedy preset and the route-tail skip left with their configurations on '
            '2026-10-18, and their measurements are in docs/deployment.md')
        for name in environment:
            definition = read(ROOT / 'config/profiles' / (name + '.json'))
            container = definition['container']
            reached = {definition['container_name'], *container['networks']['router']['aliases']}
            with self.subTest(profile=name):
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
    broadened marker stay, because the next quant trial is exactly what they were written for -- and it
    arrived on 2026-10-20 as flash-next-strata-two-card-iq4-xs, which is why the declared set below has
    a workspace of its own in it rather than a fourth name on the resident's line: a trial that shares
    a workspace with the configuration it is being compared against cannot be re-provisioned without
    touching what is serving, which is the one thing a trial may not do.
    """

    # One workspace may be shared by a declared variant pair. The invariant is about a pack: two
    # profiles whose packs differ must not watch the same directory, because whichever finishes first
    # declares the other's half-built workspace ready. The two Strata configurations (shared since
    # 2026-10-17) mount the same shards on the same engine and differ only in cache variables, so
    # they build the same pack -- and giving the variant its own directory would buy a second copy of a
    # tens-of-gigabytes pack, and an hour of nothing serving on the first switch, for a workspace that
    # comes out identical. The sharing is not taken on trust: the test below refuses the chain the
    # moment two neighbouring configurations differ in anything but the one variable between them --
    # the sampling preset, then the recovery switch -- and a different quant or window
    # in either one fails that check and restores this invariant by hand.
    SHARED_WORKSPACES = {'strata-trial/data': ('flash-next-strata-solo-thinking-recovery',
        'flash-next-strata-solo-thinking-recovery-cache',
        'flash-next-strata-solo-thinking-recovery-cache-yarn')}

    def test_every_provisioning_profile_has_a_workspace_of_its_own(self):
        engines = read(ROOT / 'config/shared.json')['engines']
        seen = {}
        for name in DEFINITIONS:
            profile = read(ROOT / 'config/profiles' / (name + '.json'))
            if not engines[profile['engine']].get('provisioning'):
                continue
            workspace = profile['workspace']['path']
            # The message is built with the same default as the check: a workspace of its own is the
            # normal case, and a guard that raises KeyError on it instead of passing would have hidden
            # this class of failure until the first profile that owned a directory (2026-10-20).
            self.assertIn(name, self.SHARED_WORKSPACES.get(workspace, (name,)),
                '%s provisions into %s, which is only ever shared by %s: a ready marker read off the '
                'engine would let either one answer for the other'
                % (name, workspace, ', '.join(self.SHARED_WORKSPACES.get(workspace, (name,)))))
            seen.setdefault(workspace, []).append(name)
        for workspace, names in seen.items():
            declared = self.SHARED_WORKSPACES.get(workspace, (names[0],))
            self.assertEqual(sorted(names), sorted(declared),
                '%s is provisioned by %s and declared for %s; a variant that leaves stops sharing and '
                'needs a workspace of its own' % (workspace, sorted(names), sorted(declared)))
        self.assertEqual(sorted(seen), ['strata-daisy-iq3xxs/data', 'strata-pair-iq3s/data',
            'strata-pair-iq4xs/data', 'strata-trial/data'])


class AVariantIsOneLineApartFromItsOriginal(unittest.TestCase):
    """A configuration that differs from another in one switch must differ in nothing else.

    `flash-next-strata-solo-thinking-recovery-cache` is offered so the operator can take the parked
    conversations -- and put them back -- at the portal, without an edit and a deploy, starting from
    `flash-next-strata-solo-thinking-recovery`. That is only honest while both launch the same backend:
    same image, same shards, same mounts, same GPUs, same window, same service IDs, same workspace --
    one variable apart. The moment they diverge, the shared provisioning workspace above
    stops being safe and this test stops the pair.

    The chain was four steps deep until 2026-10-18, and this check is a second reason retiring three of
    those steps released nothing they mounted: it is what proves a variant differs from its neighbour
    in the named variable and nowhere else, so a name going takes a switch with it and leaves the
    image, the shards and the workspace exactly where the sibling can still see them.

    The comparison is on the rendered bundle rather than on the two files, because the files also carry
    prose that is *supposed* to differ: what each configuration warns about is exactly the difference.
    """

    # Each step names the configuration it is one variable from, and the variable that may differ. An
    # operator moving along the chain changes one thing at a time, and every step is checked against
    # the configuration before it rather than against the first one -- so the day a lever is measured
    # and earns its own configuration, it is checked against what is in service and not against a
    # description of it. A step may be carried by more than one variable where one of them is
    # meaningless without another -- a parked conversation's slot count and minimum size say nothing
    # about a cache that is off -- and then all of them come out together, because the question is
    # unchanged: does anything else move?
    STEPS = (('flash-next-strata-solo-thinking-recovery-cache', 'flash-next-strata-solo-thinking-recovery',
        ('CONVERSATION_CACHE_MIB', 'CONVERSATION_CACHE_SLOTS', 'CONVERSATION_CACHE_MIN_TOKENS')),
        ('flash-next-strata-solo-thinking-recovery-cache-yarn', 'flash-next-strata-solo-thinking-recovery-cache',
        ('ROPE_SCALING', 'ROPE_SCALE', 'YARN_ORIG_CTX', 'CONTEXT')))

    def rendered(self, name, variables):
        # host_for, not the baseline host: a configuration of the second machine cannot be rendered on
        # four cards, and this class asks what a configuration differs *in*, which presupposes it renders.
        bundle = render(ROOT / 'config', host_for(name), name)
        variables = (variables,) if isinstance(variables, str) else variables
        for variable in variables:
            bundle['compose']['services'][service_key(profile_file(name))]['environment'].pop(variable, None)
            bundle['manifest']['services'][0]['recommended_environment'].pop(variable, None)
        if 'CONTEXT' in variables:
            # A window step carries one number, and CONTEXT is where the launch receives it; the
            # same number then goes out through the published contract (catalog and manifest) and
            # into the display name and capability name that name it. A step that changes the
            # window must set that number aside wherever it lands, or the test measures the echo
            # and not the change -- and the echo is exactly what the step is allowed to move.
            for contract in [bundle['catalog'], *bundle['catalog']['models'],
                    *bundle['manifest']['services']]:
                for key in ('context_length', 'total_context_length', 'context_tokens', 'display_name'):
                    contract.pop(key, None)
                capability = contract.get('capability_profile')
                if capability is not None:
                    capability.pop('name', None)
        bundle.pop('config_sha256')
        # The identity, and the prose that names the difference, are the only other things allowed to
        # differ; both are set aside so a launch change cannot hide behind them.
        bundle['profile'] = bundle['catalog']['configuration']['id'] = 'variant'
        bundle['catalog'].pop('deployment_warnings')
        for model in bundle['catalog']['models']:
            model.pop('deployment_warnings')
        return bundle

    def test_each_step_differs_in_one_variable_and_nowhere_else(self):
        for name, previous, variable in self.STEPS:
            with self.subTest(step=name):
                self.assertEqual(self.rendered(name, variable), self.rendered(previous, variable),
                    name + ' must render identically to ' + previous + ' once ' +
                    ', '.join(variable) + ' are taken back out; only those variables, the profile id '
                    'and the warning that names the difference may differ')

    def test_every_configuration_states_both_switches_rather_than_leaving_one_out(self):
        """`off` and not an omitted line: the workspace outlives the profile, and switching back has
        to take out of the run config a block another configuration wrote. Every configuration of
        this engine therefore states both switches, whichever way it wants them."""
        stated = {'flash-next-strata-solo-thinking-recovery': ('thinking', 'on'),
            'flash-next-strata-solo-thinking-recovery-cache': ('thinking', 'on'),
            'flash-next-strata-solo-thinking-recovery-cache-yarn': ('thinking', 'on'),
            'flash-next-strata-two-card-iq4-xs': ('thinking', 'on'),
            'flash-next-strata-two-card-iq3-s': ('thinking', 'on'),
            'flash-next-strata-iq3-xxs-daisy': ('thinking', 'on')}
        # Every configuration this engine serves, derived rather than remembered: the two-card probe is
        # a different topology, and a topology is not an exemption from stating a switch the run config
        # will otherwise inherit from whoever booted into this workspace last -- nor is being a
        # different machine, whose workspace is a different disk.
        self.assertEqual(set(stated), {n for n in CONFIGURATIONS if interface_of(n) == 'environment'})
        for name, expectations in stated.items():
            with self.subTest(profile=name):
                environment = self.rendered(name, '-')\
                    ['compose']['services'][service_key(profile_file(name))]['environment']
                self.assertEqual((environment['SAMPLING'], environment['TOOL_CALL_RECOVERY']),
                    expectations)

    def test_every_configuration_states_the_cache_and_the_table_it_wants(self):
        """The same rule one switch on: a parked conversation and a PLE table are written into the run
        config the engine reads from its own workspace, so a configuration that leaves them unsaid
        inherits whatever the last one that started left behind. The budget and the table are stated
        by every configuration of this engine, and the two refinements appear only where a budget
        makes them mean something."""
        stated = {'flash-next-strata-solo-thinking-recovery': ('off', 'off'),
            'flash-next-strata-solo-thinking-recovery-cache': ('16384', 'off'),
            'flash-next-strata-solo-thinking-recovery-cache-yarn': ('16384', 'off'),
            'flash-next-strata-two-card-iq4-xs': ('16384', 'off'),
            'flash-next-strata-two-card-iq3-s': ('16384', 'off'),
            'flash-next-strata-iq3-xxs-daisy': ('off', 'off')}
        self.assertEqual(set(stated), {n for n in CONFIGURATIONS if interface_of(n) == 'environment'})
        for name, (budget, table) in stated.items():
            with self.subTest(profile=name):
                definition = read(ROOT / 'config/profiles' / (name + '.json'))
                arguments = definition['arguments']
                self.assertEqual((arguments['--conversation-cache-mib'], arguments['--ple-gguf']),
                    (budget, table))
                # A floor is not optional next to a budget: it is the ceiling that stops a park
                # displacing the page-locked expert complement this box measures at 32.93 GiB. Beside
                # `off` it is nothing to bound, and it is still stated: what the entry point writes
                # sits in the workspace until a configuration says otherwise, so a configuration that
                # switched the cache back on has to find a floor here that means something on *its*
                # machine -- on the second one, 12,288 MiB of the 14.7 GiB it leaves free is what
                # stops a parked conversation bidding against the page-locked arena.
                self.assertIn('--conversation-cache-min-free-mib', definition['argument_order'])
                positive = str(arguments['--conversation-cache-min-free-mib'])
                self.assertTrue(positive.isdigit() and int(positive) > 0)
                for refinement in ('--conversation-cache-slots', '--conversation-cache-min-tokens'):
                    if budget == 'off':
                        self.assertNotIn(refinement, arguments,
                            name + ' refines a conversation cache it has switched off')
                    else:
                        self.assertIn(refinement, arguments)

    def test_the_two_card_probe_carries_the_resident_tuning_and_moves_only_its_cards(self):
        """What "all the fine tuning is carried over" means as a check and not as a sentence.

        The two-card probe answers one question -- what do two of the four cards cost -- and the answer
        is readable only if every other decision the configuration in service makes is the same
        decision in this one: the greediness the sampling preset sets, the parked conversations and
        their floor, the recovery switch, the speculation, the KV placement, the window, the pack, the
        fitted direction. So the two launches are compared with the card difference taken out, on the
        host that has the shared encoder, and anything else that moves is refused.

        It is the mirror image of the STEPS check above. That one allows one variable between two
        configurations of one topology; this one allows a topology and nothing else, which is also why
        the variables it takes out are named as a set rather than discovered from the diff: a tuning
        flag that drifted into the card column would otherwise stop being a difference by being
        reclassified.
        """
        # The comparison is against the configuration in service, not against its sibling one step
        # up the chain: a trial is measured against what is running, and what is running is the
        # conversation cache.
        probe, resident = ('flash-next-strata-two-card-iq4-xs',
            'flash-next-strata-solo-thinking-recovery-cache')
        host = vision_host()
        p, r = render(ROOT / 'config', host, probe), render(ROOT / 'config', host, resident)
        pc, rc = p['compose']['services'][STRATA_SOLO], r['compose']['services'][STRATA_SOLO]
        # The card column, and the five settings the probe declares in source that the configuration
        # in service still carries as hand edits inside its workspace (overlay facts 9-11, and the
        # budget the pair shares with the abliterated model).
        cards = ('CUDA_VISIBLE_DEVICES', 'GPUS', 'VISION_DEVICE')
        declared = ('RESIDENT_BUDGET_GIB', 'VRAM_RESERVE_MIB', 'CONTROL_VECTOR',
                    'CONTROL_VECTOR_LAYERS', 'STRATA_LOG')
        # And the two memory numbers are the tuned ones rather than a fresh guess, because they were
        # re-invented once already: 512 MiB reserved and a 32 GiB resident budget, both derived from the
        # cards' VRAM, and the pair then reported '172 MiB of VRAM free with everything loaded' and
        # 'CUDA1: 69 MiB of VRAM free', asked for --vram-reserve-mib 852 and --vram-reserve-later-mib
        # 1211, streamed 26.6 GB of experts from storage across 41,262 major faults, and died in
        # 'ggml-cuda (strata mmq): out of memory: cudaMalloc' on its first real request. A reserve is
        # what prefill staging and window capture live in, and the resident budget is a ceiling on *RAM*
        # that a smaller card set needs more of, not less: neither is a fraction of a card's VRAM.
        self.assertEqual(pc['environment']['VRAM_RESERVE_MIB'], '2048')
        self.assertEqual(pc['environment']['RESIDENT_BUDGET_GIB'], '55')
        self.assertEqual({k: v for k, v in pc['environment'].items() if k not in cards + declared},
            {k: v for k, v in rc['environment'].items() if k not in cards},
            probe + ' differs from the configuration in service in its cards and in the five settings '
            'it brings out of the workspace; a sixth difference is a tuning decision nobody agreed to '
            'change')
        # What the probe adds to the container is one read-only weight: the control vector, staged
        # outside the workspace so the controller verifies it before a switch drains anything.
        vector = [v for v in pc['volumes'] if str(v['target']).endswith('refusal-direction-r.gguf')]
        self.assertEqual(len(vector), 1)
        self.assertTrue(vector[0]['read_only'])
        # Everything else it mounts is the resident's own pack, and its /data is the one directory it
        # owns alone: a trial that shares a provisioning workspace with the configuration it is being
        # measured against cannot be re-provisioned without touching what is serving.
        weights = lambda volumes: [v for v in volumes if v['target'] != '/data']
        self.assertEqual(weights([v for v in pc['volumes'] if v is not vector[0]]), weights(rc['volumes']))
        home = lambda service: [v for v in service['volumes'] if v['target'] == '/data'][0]
        self.assertFalse(home(pc)['read_only'])
        self.assertTrue(home(pc)['source'].endswith('strata-pair-iq4xs/data'))
        self.assertNotEqual(home(pc)['source'], home(rc)['source'])
        # The published contract moves only where the cards are named. quantization, revision, the
        # capability profile and the window are the resident's own words, reused verbatim: two
        # configurations of one model that disagree about what the model is would make the portal
        # describe a different model every time the cards were moved.
        stated = ('aliases', 'configuration', 'cuda_visible_devices', 'deployment_warnings',
            'display_name', 'exclusive', 'gpu_names', 'gpu_uuids', 'models', 'text_gpu_uuids',
            'vision_device', 'vision_gpu_shared')
        self.assertEqual({k: v for k, v in p['catalog'].items() if k not in stated},
            {k: v for k, v in r['catalog'].items() if k not in stated})
        self.assertEqual(p['catalog']['models'][0]['quantization'],
            r['catalog']['models'][0]['quantization'])
        # The two session lengths the trial runs at, read off the published contract and not from a
        # launch flag: 262,144 for the model `daytime` answers with, and the abliterated model's own
        # 98,304 for `nighttime`. A configuration that names both publishes both entries, which is
        # what makes the second number visible here at all -- a configuration that names one model
        # cannot show it, because there the same backend answers `nighttime` at its own width.
        self.assertEqual(p['catalog']['models'][0]['context_length'], 262144)
        self.assertEqual(r['catalog']['models'][0]['context_length'], 262144)
        self.assertEqual([m['context_length'] for m in p['catalog']['models']
                          if 'nighttime' in m['aliases']], [98304])
        # And the four cards are still exactly two + two + the encoder, shared with nothing: the
        # night-side model this configuration names beside its own is the launch the 27B pair loads,
        # byte for byte, which is the claim the shared-vision placement was built to allow.
        night = render(ROOT / 'config', host, 'qwen27b-q8-pair')['compose']['services'][EVERYDAY]
        self.assertEqual(p['compose']['services'][EVERYDAY], night)
        self.assertEqual(p['catalog']['text_gpu_uuids'], host['gpu_ids']['daytime'])
        self.assertEqual(p['manifest']['services'][0]['parallel_slots'], 1)
        self.assertNotIn('exclusive', read(ROOT / 'config/profiles' / (probe + '.json')))

    def test_the_two_card_quant_arm_is_the_probe_with_its_pack_swapped(self):
        """The 3-bit arm of the trial is one artifact set deep from the two-card probe.

        `flash-next-strata-two-card-iq3-s` exists to answer a question about a quantization, so the only
        thing allowed to differ between it and `flash-next-strata-two-card-iq4-xs` is the pack: two
        GSQ-RCO shards for the three UD-IQ4_XS ones, the quant and family the engine is told, and the
        model id and workspace that follow from a different pack. Everything the first arm was built to
        hold still -- the cards, the reserve, the resident budget, the window, the sampling preset, the
        parked conversations, the recovery switch, the speculation, the control vector, the night-side
        resident -- has to be held by this one too, or the two arms measure different things and neither
        answers the question. This is what makes "we only changed the quant" a fact and not an intention.
        """
        host = vision_host()
        iq4 = render(ROOT / 'config', host, 'flash-next-strata-two-card-iq4-xs')
        iq3 = render(ROOT / 'config', host, 'flash-next-strata-two-card-iq3-s')
        a, b = iq4['compose']['services'][STRATA_SOLO], iq3['compose']['services'][STRATA_PAIR]
        # What a pack is, as far as the launch is concerned: which quantization and family the engine is
        # told -- and note that the profile's own --model path never reaches the engine at all, because
        # the environment interface carries the quant name and GGUF_DIR and the engine assembles the
        # filenames from its family table.
        self.assertEqual((b['environment']['MODEL'], b['environment']['FAMILY']), ('IQ3_S', 'qwen'))
        self.assertEqual((a['environment']['MODEL'], a['environment']['FAMILY']), ('UD-IQ4_XS', 'unsloth'))
        pack = ('MODEL', 'FAMILY')
        self.assertEqual({k: v for k, v in b['environment'].items() if k not in pack},
            {k: v for k, v in a['environment'].items() if k not in pack},
            'the 3-bit arm differs from the 4-bit arm in the pack it names; anything else in the '
            'environment is a tuning decision the two arms no longer share')
        weights = lambda service: sorted(v['target'] for v in service['volumes'] if v['target'] != '/data')
        self.assertEqual([v for v in weights(b) if 'IQ3_S' not in v],
            [v for v in weights(a) if 'UD-IQ4_XS' not in v],
            'the encoder projector and the control vector are mounted by both arms: the shards are the swap')
        self.assertEqual([v['read_only'] for v in b['volumes'] if v['target'] != '/data'],
            [True] * (len(b['volumes']) - 1))
        home = lambda service: [v for v in service['volumes'] if v['target'] == '/data'][0]
        self.assertTrue(home(b)['source'].endswith('strata-pair-iq3s/data'))
        self.assertNotEqual(home(b)['source'], home(a)['source'])
        # The published contract: what the model *is* moves with the pack, and what it *can do* does not.
        named = ('configuration', 'default_model', 'deployment_warnings', 'display_name', 'model',
            'model_path', 'models', 'quantization', 'revision', 'capability_profile')
        # The last of those is worth naming twice over: model_path is the primary shard's host path, so
        # a pack change moves it, and it is also the one place the profile's own --model value surfaces
        # in the contract at all.
        self.assertEqual({k: v for k, v in iq3['catalog'].items() if k not in named},
            {k: v for k, v in iq4['catalog'].items() if k not in named})
        m4, m3 = iq4['catalog']['models'][0], iq3['catalog']['models'][0]
        self.assertNotEqual(m4['quantization'], m3['quantization'])
        self.assertEqual(m3['model'], 'qwen3.8-flash-next-iq3_s')
        for key in ('context_length', 'total_context_length', 'input_modalities', 'mtp'):
            self.assertEqual(m3[key], m4[key], key)
        self.assertEqual(m3['capability_profile']['name'], 'qwen3.8-flash-next-iq3-s-strata-256k')
        self.assertEqual(m3['capability_profile']['speculative'], m4['capability_profile']['speculative'])
        # And the night side is the same launch in both arms, so the two configurations differ to an
        # operator by the pack they run and to the router by nothing else.
        self.assertEqual(iq3['compose']['services'][EVERYDAY], iq4['compose']['services'][EVERYDAY])
        self.assertEqual([m['context_length'] for m in iq3['catalog']['models']
                          if 'nighttime' in m['aliases']], [98304])
        self.assertEqual(iq3['catalog']['text_gpu_uuids'], host['gpu_ids']['daytime'])

    def test_a_value_outside_the_two_words_refuses_before_anything_renders(self):
        # The entry point writes a JSON boolean and the engine starts on nothing else, so the words
        # are closed in source: a profile that invents a third one is refused here, with no drain.
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            path = config / 'profiles/flash-next-strata-solo-thinking-recovery.json'
            definition = read(path)
            for stated in ('maybe', 'true', '1'):
                with self.subTest(value=stated):
                    definition['arguments']['--tool-call-recovery'] = stated
                    path.write_text(json.dumps(definition))
                    with self.assertRaisesRegex(RuntimeError,
                            '--tool-call-recovery states ' + stated):
                        render(config, BASELINE['host'], 'flash-next-strata-solo-thinking-recovery')

    def test_a_cache_or_table_that_cannot_be_honoured_refuses_before_anything_renders(self):
        """Values on this switch are refused for the shape of the thing they buy, not for spelling.

        A budget of `8GiB` is a value the entry point could not write and the engine could not read; a
        budget with no floor is a cache that will outbid the page-locked expert complement for RAM on
        the same machine that serves; a slot count beside an off cache is a statement about nothing; a
        table nothing mounts is a boot that either does not happen or happens with one layer's n-gram
        table quietly absent. All four are refused here, before a drain is spent, because this engine's
        exclusive model is the only thing serving and a refused boot is nothing serving.
        """
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            name = 'flash-next-strata-solo-thinking-recovery-cache'
            path = config / ('profiles/' + name + '.json')
            original = read(path)
            for argument, stated, refuses in (
                    ('--conversation-cache-mib', '8GiB', '--conversation-cache-mib states 8GiB'),
                    ('--conversation-cache-mib', '0', '--conversation-cache-mib states 0'),
                    ('--conversation-cache-mib', '-1', '--conversation-cache-mib states -1'),
                    ('--conversation-cache-slots', '0', '--conversation-cache-slots states 0'),
                    ('--conversation-cache-min-tokens', 'half', 'min-tokens states half'),
                    ('--ple-gguf', '/weights/never-arrived.gguf',
                     '--ple-gguf states /weights/never-arrived.gguf')):
                with self.subTest(argument=argument, value=stated):
                    definition = json.loads(json.dumps(original))
                    definition['arguments'][argument] = stated
                    path.write_text(json.dumps(definition))
                    with self.assertRaisesRegex(RuntimeError, refuses.replace('-', r'[-]')):
                        render(config, BASELINE['host'], name)
            # The floor is required by a number and forbidden by nothing: take it out of the
            # configuration that turns the cache on and the budget is an unbounded one.
            definition = json.loads(json.dumps(original))
            del definition['arguments']['--conversation-cache-min-free-mib']
            definition['argument_order'].remove('--conversation-cache-min-free-mib')
            path.write_text(json.dumps(definition))
            with self.assertRaisesRegex(RuntimeError,
                    'states a budget and no [-][-]conversation-cache-min-free-mib'):
                render(config, BASELINE['host'], name)
            # And a refinement is refused on a configuration that has switched the whole thing off,
            # which is where an inherited number would otherwise go unnoticed.
            off = config / ('profiles/' + RECOVERY + '.json')
            definition = read(off)
            definition['arguments']['--conversation-cache-slots'] = '4'
            definition['argument_order'].insert(
                definition['argument_order'].index('--conversation-cache-mib') + 1,
                '--conversation-cache-slots')
            off.write_text(json.dumps(definition))
            with self.assertRaisesRegex(RuntimeError,
                    r'refines a conversation cache this configuration has switched off'):
                render(config, BASELINE['host'], 'flash-next-strata-solo-thinking-recovery')

    def test_a_ring_the_engine_cannot_read_refuses_before_anything_renders(self):
        """The rope statements are refused the way every entry-point fact is: at render, because an
        exclusive model started on a ring the engine refuses is nothing serving on a drain already
        spent. A ring word the engine has never heard; a ring with no scale; a scale that is not a
        positive number; yarn with no trained window named; a yarn window that does not exceed the
        trained one it stretches (the engine refuses the pair); and numbers stated beside `off`,
        where every rope line is being taken back out."""
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            name = 'flash-next-strata-solo-thinking-recovery-cache-yarn'
            path = config / ('profiles/' + name + '.json')
            original = read(path)
            for argument, stated, refuses in (
                    ('--rope-scaling', 'ntk', 'rope-scaling states ntk'),
                    ('--rope-scaling', '2.0', 'rope-scaling states 2.0'),
                    ('--rope-scale', 'two', 'no positive [-][-]rope-scale'),
                    ('--rope-scale', '0', 'no positive [-][-]rope-scale'),
                    ('--rope-scale', '2.0.1', 'no positive [-][-]rope-scale'),
                    ('--yarn-orig-ctx', 'half', 'no positive [-][-]yarn-orig-ctx'),
                    ('--yarn-orig-ctx', '524288', 'does not stretch')):
                with self.subTest(argument=argument, value=stated):
                    definition = json.loads(json.dumps(original))
                    definition['arguments'][argument] = stated
                    path.write_text(json.dumps(definition))
                    with self.assertRaisesRegex(RuntimeError, refuses):
                        render(config, BASELINE['host'], name)
            for gone in ('--rope-scale', '--yarn-orig-ctx'):
                definition = json.loads(json.dumps(original))
                del definition['arguments'][gone]
                definition['argument_order'].remove(gone)
                path.write_text(json.dumps(definition))
                with self.subTest(missing=gone):
                    with self.assertRaisesRegex(RuntimeError, '[-][-]' + gone[2:]):
                        render(config, BASELINE['host'], name)
            # A number beside `off` is a statement about a ring that is being taken back out.
            definition = json.loads(json.dumps(original))
            definition['arguments']['--rope-scaling'] = 'off'
            path.write_text(json.dumps(definition))
            with self.assertRaisesRegex(RuntimeError,
                    'refines a rope scaling this configuration has switched to off'):
                render(config, BASELINE['host'], name)


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
                    # No two models answer to the same service ID, whichever configuration is
                    # selected: `daytime` is only ever one model's, `nighttime` only ever one thing,
                    # in exactly one direction.
                    for other in catalog['models']:
                        if other is entry:
                            continue
                        self.assertTrue(set(service).isdisjoint(
                            [alias for alias in other['aliases'] if alias != COMPATIBILITY_ALIAS]),
                            'Service ID shared between ' + entry['model'] + ' and ' + other['model'])

    def test_every_service_id_is_answered_or_declared_offline(self):
        """A service ID is answered here or says so; it is never quietly missing.

        A client that has `nighttime` configured cannot discover that this configuration stopped
        loading the model that used to answer it — its request just fails, forever, with nothing
        telling it why. So for every configuration each ID a client may send is asserted to be either
        published by a model that is loaded or listed in `offline_services` with a reason a status
        page can show. Nothing is offline in any configuration today: the configuration that loads one
        model answers `nighttime` from that model, which is the whole reason it is a configuration and
        not an experiment.
        """
        for name in CONFIGURATIONS:
            definition = read(ROOT / 'config/profiles' / (name + '.json'))
            catalog = render(ROOT / 'config', host_for(name), name)['catalog']
            published = {alias for model in catalog['models'] for alias in model['aliases']}
            offline = {alias for service in catalog['offline_services'] for alias in service['aliases']}
            with self.subTest(profile=name):
                # These IDs are one machine's namespace. A configuration built for a second
                # machine answers its own name -- which is what lets two machines reach one router
                # without either relaunching -- so it is held to publishing something a client can
                # send, which ServiceIdentityTests does, and not to publishing `daytime`, a name this
                # house has given the first machine until the federation decides who owns what.
                if ROSALINA not in definition['hosts']:
                    self.assertTrue(published - {COMPATIBILITY_ALIAS},
                        'Configuration ' + name + ' publishes no service ID of its own')
                    continue
                for service_id in CLIENT_SERVICES:
                    self.assertIn(service_id, published | offline,
                        'Configuration ' + name + ' neither answers ' + service_id +
                        ' nor declares it offline: a client holding that name fails with no reason')
                self.assertEqual(published & offline, set(),
                    'Configuration ' + name + ' answers ' + str(published & offline) + ' and also '
                    'declares it offline')
                self.assertEqual(offline, set(), 'Configuration ' + name +
                    ' declares a service offline that every model here answers')

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
            for service, cfg in render(ROOT / 'config', host_for(name), name)['compose']['services'].items():
                for key in CONTAINER_CONTRACT:
                    with self.subTest(profile=name, service=service, setting=key):
                        self.assertIsNotNone(cfg.get(key))


class ConfigurationNamingTests(unittest.TestCase):
    """What a configuration is called, and what happens to a name that stops meaning one."""

    def test_a_renamed_configuration_renders_its_new_name_exactly(self):
        # A rename is a naming change, not a redeploy: a Compose service is keyed by the model it
        # loads and the canonical model id is that model's alias, so the old name has to produce
        # byte-identical backends. If this ever differs, a host that renames while running would be
        # drained for a backend that is not the one it already has.
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

    def test_every_retired_and_renamed_name_says_what_to_use_instead(self):
        # A retired configuration has no backends left to launch, so it refuses -- and the refusal is
        # the whole service it still performs, because the operator holding it needs the name of the
        # thing to select instead. A renamed one resolves quietly, for the opposite reason: the same
        # backends are still there. Both tables therefore have to point at something selectable, and
        # a name cannot be in both, which is how the 2026-10-18 consolidation of the Strata chain from
        # five configurations to two is kept honest: three retired ids and their old pre-rename id all
        # say `flash-next-strata-solo-thinking-recovery`, and none of them resolves by accident.
        for table in (RETIRED_PROFILES, PROFILE_ALIASES):
            for retired, replacement in table.items():
                with self.subTest(name=retired):
                    self.assertIn(replacement, CONFIGURATIONS,
                        retired + ' points at ' + replacement + ', which is not selectable')
                    self.assertEqual(resolve_profile(retired) if table is PROFILE_ALIASES else retired,
                        retired if table is RETIRED_PROFILES else replacement)
        self.assertFalse(set(RETIRED_PROFILES) & set(PROFILE_ALIASES),
            'a name cannot both refuse and resolve')
        self.assertEqual(DEFAULT_PROFILE, CONFIGURATIONS[0] if DEFAULT_PROFILE in PROFILE_ALIASES
            else DEFAULT_PROFILE)

    def test_every_engine_is_used_and_nothing_is_configured_by_an_absent_one(self):
        # A profile naming an engine that is not registered fails at render, and an engine nothing
        # references is an image kept alive by a config file. The one exception is the engine held for
        # a live rollback, named here so that retiring it is a decision and not an oversight.
        engines = read(ROOT / 'config/shared.json')['engines']
        # The 0.1.43 image, until the 0.1.44.1 candidate is decided; and the text-only image of
        # 0.1.44.1, until the encoder that joined the image on 2026-10-16 has earned its place.
        rollback = {'strata-7695859aa13a', 'strata-296b59d2f73c'}
        used = set()
        for name in DEFINITIONS:
            definition = read(ROOT / 'config/profiles' / (name + '.json'))
            self.assertIn(definition['engine'], engines, name + ' names an engine that is not registered')
            used.add(definition['engine'])
        self.assertEqual(set(engines) - used, rollback,
            'unreferenced engine: ' + str(set(engines) - used - rollback))


    def test_a_model_is_called_by_its_model_and_its_session_length(self):
        # One naming rule for every surface, because a person meets the name in the page, in a
        # router's model list and in a terminal, and asks the same question each time: which window.
        for name in DEFINITIONS:
            with self.subTest(profile=name):
                definition = read(ROOT / 'config' / 'profiles' / (name + '.json'))
                self.assertEqual(default_name(definition),
                    definition['display_name'] + ' (' + str(definition['context_tokens'] // 1024) + 'K)')
                self.assertEqual(display_name(definition, {}), default_name(definition))
                # And a name hangs on the id a client routes on, never on the file that launches it:
                # the same model in two configurations is one thing to name.
                self.assertEqual(model_id(definition), definition['arguments']['--alias'])

    def test_an_alias_replaces_a_name_and_nothing_else_about_the_launch(self):
        # The refusal this test protects: a rename that changed a launch would mean a machine had to
        # be drained to be renamed, and an operator would stop renaming rather than drain for it.
        plain = render(ROOT / 'config', host_for('qwen27b-q8-pair'), 'qwen27b-q8-pair')
        named = render(ROOT / 'config', host_for('qwen27b-q8-pair'), 'qwen27b-q8-pair',
            {'qwen3.8-27b-q8_0-tensor-next': 'Workhorse',
                'qwen3.8-27b-abliterated-q6_k': 'Night owl'})
        self.assertEqual(named['compose'], plain['compose'])
        self.assertEqual([x['recommended_argv'] for x in named['manifest']['services']],
            [x['recommended_argv'] for x in plain['manifest']['services']])
        self.assertEqual([(x['model'], x['aliases']) for x in named['catalog']['models']],
            [(x['model'], x['aliases']) for x in plain['catalog']['models']])
        self.assertEqual([(x['display_name'], x['model']) for x in named['catalog']['models']],
            [('Workhorse', plain['catalog']['models'][0]['model']),
             ('Night owl', plain['catalog']['models'][1]['model'])])
        # The document as a whole does change, and that is right: a machine that says something
        # different about itself has a different catalog, and it is the publish that carries it.
        self.assertNotEqual(named['config_sha256'], plain['config_sha256'])

    def test_an_alias_is_refused_with_the_rule_it_broke(self):
        nameable = nameable_models(ROOT / 'config')
        coding = 'qwen3.8-27b-q8_0-tensor-next'
        self.assertEqual(clean_alias(coding, '  Workhorse  ', nameable), 'Workhorse')
        self.assertIsNone(clean_alias(coding, '   ', nameable))
        self.assertIsNone(clean_alias(coding, None, nameable))
        for value, reason in [(42, 'must be text'), ('x' * (ALIAS_MAX + 1), 'at most ' + str(ALIAS_MAX)),
                ('two\nlines', 'one printable line'), ('tab\there', 'one printable line')]:
            with self.subTest(value=repr(value)), self.assertRaisesRegex(RuntimeError, reason):
                clean_alias(coding, value, nameable)
        # Every model any configuration of this revision loads can be named, and nothing else: the
        # list is the registry's, so a model retired in a release cannot keep a name nobody can serve.
        self.assertGreaterEqual(set(nameable), {coding, 'qwen3.8-27b-abliterated-q6_k'})
        for model in nameable:
            self.assertEqual(clean_alias(model, 'Any name', nameable), 'Any name')
        self.assertIn('qwen27b-q8-pair', {c for where in nameable.values() for c in where})
        with self.assertRaisesRegex(RuntimeError, 'Unknown model'):
            clean_alias('daytime-27b', 'Old habit', nameable)
        # A configuration is not nameable, because it is a placement decision and not a thing a client
        # asks for by name.
        with self.assertRaisesRegex(RuntimeError, 'Unknown model'):
            clean_alias('qwen27b-q8-pair', 'The pair', nameable)

    def test_the_alias_file_is_read_tolerantly_and_says_what_it_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'aliases.json'
            self.assertEqual(read_aliases(path), ({}, None))  # Absent is the ordinary state.
            path.write_text('{not json')
            aliases, note = read_aliases(path)
            self.assertEqual(aliases, {}); self.assertIn('could not be read', note)
            path.write_text('{"aliases": ["not", "a", "map"]}')
            aliases, note = read_aliases(path)
            self.assertEqual(aliases, {}); self.assertIn('malformed', note)
            nameable = nameable_models(ROOT / 'config')
            atomic_json(path, {'version': ALIASES_VERSION, 'aliases': {
                'qwen3.8-27b-q8_0-tensor-next': '  Workhorse  ', 'gone-today': 'Ghost',
                'qwen3.8-27b-abliterated-q6_k': '   '}})
            aliases, note = read_aliases(path, nameable)
            self.assertEqual(aliases, {'qwen3.8-27b-q8_0-tensor-next': 'Workhorse'})
            self.assertIn('gone-today', note)
            # A file from the release that keyed names by configuration is a file of names for models
            # this machine does not serve, which is what it is told.
            atomic_json(path, {'version': 1, 'aliases': {'qwen27b-q8-pair': 'The pair'}})
            ignored, note = read_aliases(path, nameable)
            self.assertEqual(ignored, {})
            self.assertIn('qwen27b-q8-pair', note)

    def test_the_registry_shows_the_chosen_name_and_the_default_behind_it(self):
        # The page needs both: the default is what clearing the name would bring back, and it is what
        # the editor offers as its placeholder, so a person can see what they are covering over.
        registry = available_profiles(ROOT / 'config', host_for('qwen27b-q8-pair'),
            {'qwen3.8-27b-q8_0-tensor-next': 'Workhorse'})
        entry = next(x for x in registry['selectable'] if x['profile'] == 'qwen27b-q8-pair')
        self.assertEqual((entry['display_name'], entry['default_name'], entry['alias']),
            ('Workhorse', 'Qwen3.8 27B Q8 (160K)', 'Workhorse'))
        other = next(x for x in registry['selectable'] if x['profile'] == 'qwen27b-q6k-pair')
        self.assertEqual((other['display_name'], other['default_name']),
            ('Qwen3.8 27B Q6_K (160K)', 'Qwen3.8 27B Q6_K (160K)'))
        self.assertNotIn('alias', other)
        self.assertEqual(catalog_names(ROOT / 'config', host_for('qwen27b-q8-pair'),
            {'qwen3.8-27b-abliterated-q6_k': 'Night owl'})['qwen3.8-27b-abliterated-q6_k'], 'Night owl')
        # And the CLI says it too: `~/local-ai-config.sh list` and the page are two places a person
        # reads the same name, and a rename that reached only one of them is a rename half done.
        from runtime.__main__ import listed
        named = listed(ROOT / 'config', host_for('qwen27b-q8-pair'),
            {'qwen3.8-27b-q8_0-tensor-next': 'Workhorse'})
        self.assertIn('Workhorse', named)
        self.assertNotIn('Qwen3.8 27B Q8 (160K)', named)
        self.assertIn('Qwen3.8 27B Q8 (160K)', listed(ROOT / 'config', host_for('qwen27b-q8-pair')))

class MoreThanOneMachineTests(unittest.TestCase):
    """A second machine reading this source, and what a configuration of another machine is.

    The host says which machine it is and what its cards can execute; every definition says which
    machines it was built for. These tests are the boundary: a row the operator could select is a
    configuration this machine can run, and a card this machine does not have cannot be reserved by a
    launch that was measured where the card did exist.
    """

    DAISY_CONFIGURATION = 'flash-next-strata-iq3-xxs-daisy'
    TWO_CARD_SOLO = 'flash-next-strata-two-card-iq3-s'

    def config_with(self, name, **changes):
        """The shipped configuration with one definition changed, in a tree of its own."""
        tmp = tempfile.mkdtemp(); self.addCleanup(shutil.rmtree, tmp)
        config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
        path = config / 'profiles' / (name + '.json')
        definition = read(path)
        for key, value in changes.items():
            if value is None:
                definition.pop(key, None)
            else:
                definition[key] = value
        path.write_text(json.dumps(definition))
        return config

    def test_every_shipped_definition_names_the_machine_it_runs_on(self):
        # Nothing ships unscoped, and the split is now a real one: this is where the shipped set says
        # which machine each launch was measured on, so a definition that grows a second host has to
        # be changed here as well as there.
        for name in DEFINITIONS:
            definition = read(ROOT / 'config/profiles' / (name + '.json'))
            with self.subTest(profile=name):
                self.assertTrue(definition['hosts'], name + ' names no machine at all')
                self.assertEqual(belongs_to_host(definition, second_machine(host_id=ROSALINA)),
                    ROSALINA in definition['hosts'])
                self.assertEqual(belongs_to_host(definition, second_machine(host_id=DAISY)),
                    DAISY in definition['hosts'],
                    name + ' belongs to the machine it was measured on and to no other')
        self.assertEqual([name for name in CONFIGURATIONS if DAISY in
                read(ROOT / 'config/profiles' / (name + '.json'))['hosts']],
            ['flash-next-strata-iq3-xxs-daisy'],
            'The second machine has the one configuration it has been measured for; a launch nobody '
            'has run there is not something it is offered')

    def test_a_definition_that_names_no_machine_is_refused(self):
        # A default of "every machine" is how a configuration ends up rendered on hardware it was
        # never measured on, which is the failure this whole boundary exists to make impossible.
        for change in ({'hosts': None}, {'hosts': 'rosalina'}, {'hosts': []}, {'hosts': ['Rosalina']}):
            with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, 'hosts must state'):
                render(self.config_with('qwen27b-q8-pair', **change), BASELINE['host'], 'qwen27b-q8-pair')

    def test_a_host_that_names_no_machine_sees_the_whole_registry(self):
        # The private settings of a host that is already serving are edited by an operator, not by a
        # release: until they name the machine, the release does not filter anything away from them.
        self.assertEqual([x['profile'] for x in available_profiles(ROOT / 'config')['selectable']],
            list(CONFIGURATIONS))
        self.assertEqual([x['profile'] for x in
            available_profiles(ROOT / 'config', BASELINE['host'])['selectable']], list(CONFIGURATIONS))
        self.assertEqual(default_profile(ROOT / 'config', BASELINE['host']), DEFAULT_PROFILE)

    def test_a_named_host_sees_only_the_configurations_that_name_it_back(self):
        registry = available_profiles(ROOT / 'config', second_machine())
        self.assertEqual([x['profile'] for x in registry['selectable']],
            ['flash-next-strata-iq3-xxs-daisy'],
            'a machine is offered the launches it was measured on, which is now one and not none')
        self.assertEqual(sorted({m['model'] for c in registry['selectable'] for m in c['models']}),
            [STRATA_DAISY], 'a machine is shown the models of the configurations it names, and no others')
        # Scoping one of Rosalina's launches to this machine *adds* it: the machine's own configuration
        # does not make way for it, and the order is the registry's, which is the order the page shows.
        self.assertEqual(configurations_for(self.config_with(self.TWO_CARD_SOLO, hosts=['daisy']),
            second_machine()), (self.TWO_CARD_SOLO, 'flash-next-strata-iq3-xxs-daisy'))

    def test_a_host_whose_registry_is_empty_says_so(self):
        with self.assertRaisesRegex(RuntimeError, 'No configuration of this revision names this host'):
            default_profile(ROOT / 'config', second_machine(host_id='no-machine-claims-this-one'))

    def test_another_machines_configuration_refuses_to_render_and_names_both(self):
        with self.assertRaisesRegex(RuntimeError,
                r"qwen27b-q8-pair is a configuration of 'rosalina', and this host is 'daisy'"):
            render(ROOT / 'config', second_machine(), 'qwen27b-q8-pair')

    def test_a_one_pair_host_runs_a_solo_configuration_across_its_two_cards(self):
        # The whole point of the boundary: not a degraded Rosalina, a configuration of this machine.
        config = self.config_with(self.TWO_CARD_SOLO, hosts=['daisy'], exclusive=True, gpu_group='all',
            models=[self.TWO_CARD_SOLO])
        bundle = render(config, second_machine(), self.TWO_CARD_SOLO)
        self.assertEqual(list(bundle['compose']['services']), [STRATA_PAIR])
        self.assertEqual([s['service'] for s in bundle['manifest']['services']], [STRATA_PAIR])
        service = bundle['compose']['services'][STRATA_PAIR]
        pair = second_machine()['gpu_ids']['daytime']
        self.assertEqual(service['deploy']['resources']['reservations']['devices'][0]['device_ids'], pair)
        self.assertEqual(service['environment']['CUDA_VISIBLE_DEVICES'], '0,1')
        self.assertEqual(service['environment']['GPUS'], '0,1')
        model = bundle['catalog']['models'][0]
        self.assertEqual(model['gpu_uuids'], pair)
        self.assertEqual(model['gpu_names'], ['RTX 2080 Ti', 'RTX 2080 Ti'])
        self.assertEqual(model['exclusive'], True)
        # The profile promises pictures and this machine has no card for the encoder, so the contract
        # publishes what the host delivers -- the same rule the shared-vision host works by, read the
        # other way round.
        self.assertEqual(model['input_modalities'], ['text'])
        self.assertEqual(model['capability_profile']['vision'], False)
        self.assertNotIn('vision_gpu_uuid', model)
        # `nighttime` is not offline here: it is a service this machine never offered, and 503-ing a
        # service ID nobody publishes is a different claim from silence. What *is* offline is the name
        # the other configuration of this machine answers -- scoping a Rosalina launch to this host
        # puts two configurations on it, and between them the second machine's name goes unanswered.
        self.assertEqual([service['model'] for service in bundle['catalog']['offline_services']],
            [STRATA_DAISY], 'the service the other configuration of this host publishes, with the reason')
        self.assertEqual(bundle['catalog']['offline_services'][0]['reason'],
            'not_loaded_by_this_configuration')

    def test_a_configuration_that_wants_the_other_pair_refuses_a_one_pair_host(self):
        # Same definition, `exclusive` left off: it names two models again, and two models need two
        # text pairs, which this machine does not have. It refuses twice over, and the first refusal
        # is the one that names the machine: the other model it names is a launch of the other host.
        with self.assertRaisesRegex(RuntimeError, 'cannot bring another machine'):
            render(self.config_with(self.TWO_CARD_SOLO, hosts=['daisy']), second_machine(),
                self.TWO_CARD_SOLO)
        # And with the other model scoped to this machine as well, so neither is foreign, the card
        # shortage still refuses: a host states the GPU groups it has, and this one has one.
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            for name in (self.TWO_CARD_SOLO, EVERYDAY_FILE):
                path = config / 'profiles' / (name + '.json')
                definition = read(path); definition['hosts'] = ['daisy']
                path.write_text(json.dumps(definition))
            with self.assertRaisesRegex(RuntimeError, 'declares no "nighttime" GPU group'):
                render(config, second_machine(), self.TWO_CARD_SOLO)

    def test_a_four_card_launch_refuses_a_host_with_two_cards(self):
        # The guard the two-card trial was the argument for: the container would have booted, been
        # healthy on its own health check, and asked the engine for cards it cannot see.
        config = self.config_with(SOLO, hosts=['daisy'])
        with self.assertRaisesRegex(RuntimeError, r'--gpus must name the text cards in the order'):
            render(config, second_machine(), SOLO)

    def test_an_image_without_sass_for_these_cards_is_refused_where_it_is_rendered(self):
        config = self.config_with(self.TWO_CARD_SOLO, hosts=['daisy'], exclusive=True, gpu_group='all',
            models=[self.TWO_CARD_SOLO])
        with self.assertRaisesRegex(RuntimeError, 'carries SASS for CUDA 86/89'):
            render(config, second_machine(gpu_architectures=['75']), self.TWO_CARD_SOLO)
        # A host that does not state its cards gets no check it cannot answer -- and states them in
        # the form an engine states them when it does.
        self.assertEqual(len(render(config, second_machine(gpu_architectures=None),
            self.TWO_CARD_SOLO)['compose']['services']), 1)
        for bad in ('75', [], ['Turing']):
            with self.subTest(architectures=bad), self.assertRaisesRegex(RuntimeError, 'gpu_architectures'):
                render(config, second_machine(gpu_architectures=bad), self.TWO_CARD_SOLO)

    def test_the_second_machine_runs_the_shape_that_was_measured_on_it(self):
        # Each of these is a measurement and not a preference, and the suite above would stay green if
        # somebody widened one after reading its sibling on the other machine. The window is the abort
        # reproduced twice; the absent reserve is the cost that was declined in favour of that window;
        # the cache budget is off because the pack pins 40 GiB of this machine's 60.7; the projector is
        # not mounted because this image was built with no encoder in it; and the service ID is this
        # machine's own name because two machines cannot both answer `daytime` to one router.
        bundle = render(ROOT / 'config', second_machine(gpu_architectures=['75']),
            self.DAISY_CONFIGURATION)
        service = bundle['compose']['services'][STRATA_DAISY]
        environment = service['environment']
        self.assertEqual([x['service'] for x in bundle['manifest']['services']], [STRATA_DAISY])
        self.assertEqual((environment['CONTEXT'], environment['GPUS'],
            environment['CUDA_VISIBLE_DEVICES']), ('65536', '0,1', '0,1'))
        self.assertNotIn('VRAM_RESERVE_MIB', environment,
            'the window is what was chosen to keep these cards clear, not the reserve')
        self.assertEqual(environment['CONVERSATION_CACHE_MIB'], 'off')
        self.assertEqual(service['ulimits']['memlock'], -1,
            'the arena the engine pins is page-locked, and dockerd holds CAP_SYS_RESOURCE')
        self.assertNotIn('VISION_DEVICE', environment)
        self.assertNotIn('/weights/mmproj-Qwen3.8-Flash-Next-BF16.gguf',
            [volume['target'] for volume in service['volumes']])
        self.assertFalse(bundle['catalog']['capability_profile']['vision'])
        # One name, and it is not the household's default name. `local-active` is what a router
        # answers an unnamed request with, and it answers that from one host; a second catalog
        # claiming it is either refused by the router's collision rule or, without the rule, two
        # machines saying they are the model the house is using. The name moves with the router's
        # default host, and moving it is a switch like any other.
        self.assertEqual([entry['aliases'] for entry in bundle['catalog']['models']], [['daisy']])
        # And the image it runs is the one with SASS this machine can execute: the tag in service on the
        # four-GPU host is refused here by name, which is the guard a second machine needs and the one
        # nothing in the launch itself would ever have noticed.
        config = self.config_with(self.DAISY_CONFIGURATION, engine='strata-vision-296b59d2f73c')
        with self.assertRaisesRegex(RuntimeError, 'carries SASS for'):
            render(config, second_machine(gpu_architectures=['75']), self.DAISY_CONFIGURATION)

    def test_a_model_no_configuration_loads_is_not_offered_to_this_host(self):
        # A model is offered to a host because a configuration of that host names it, and for no other
        # reason. Scoping the abliterated model to this machine does not put it in the listing: nothing
        # this machine can select loads it, and a page that drew it anyway would be drawing a launch
        # an operator could never start.
        from runtime.__main__ import listed
        machine = 'a-machine-of-its-own'
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'; shutil.copytree(ROOT / 'config', config)
            for name, changes in ((self.TWO_CARD_SOLO, {'hosts': [machine], 'exclusive': True,
                    'gpu_group': 'all', 'models': [self.TWO_CARD_SOLO]}),
                    (EVERYDAY_FILE, {'hosts': [machine]})):
                path = config / 'profiles' / (name + '.json')
                definition = read(path); definition.update(changes)
                path.write_text(json.dumps(definition))
            registry = available_profiles(config, second_machine(host_id=machine))
            self.assertEqual([x['profile'] for x in registry['selectable']], [self.TWO_CARD_SOLO])
            self.assertEqual([[m['model'] for m in x['models']] for x in registry['selectable']],
                [[STRATA_PAIR]])
            rows = listed(config, second_machine(host_id=machine)).splitlines()
            self.assertEqual(len(rows), 2,
                'a model no configuration of this host names is listed as if it ran: ' + repr(rows))
            row = rows[1]
            self.assertTrue(row.startswith('    '), row)
            self.assertIn(registry['selectable'][0]['models'][0]['display_name'], row)
            self.assertNotIn(read(config / 'profiles/everyday.json')['display_name'], '\n'.join(rows),
                'the abliterated model is scoped to this machine and loads on nothing it can select')

    def test_the_cli_listing_is_the_host_s_own(self):
        from runtime.__main__ import listed

        def blocks(host=None):
            # A listing is a row per configuration with one indented row per model under it.
            rows, current = [], None
            for line in listed(ROOT / 'config', host).splitlines():
                if line.startswith(' '):
                    current.append(line)
                else:
                    current = [line]
                    rows.append(current)
            return rows

        machine = blocks(second_machine())
        self.assertEqual([block[0].split(':')[0] for block in machine],
            ['flash-next-strata-iq3-xxs-daisy'])
        self.assertEqual(len(machine[0]), 2, 'a configuration is listed with the models it loads')
        self.assertIn('every text GPU of this host', machine[0][1])
        self.assertIn('daisy', machine[0][1], 'the service ID a client of this machine sends')
        self.assertEqual(blocks(second_machine(host_id='no-machine-claims-this-one')), [])
        for block in blocks():
            name = block[0].split(':')[0]
            self.assertEqual(len(block) - 1, len(NAMED[name]),
                name + ' is listed with ' + str(len(block) - 1) + ' models and loads '
                + str(len(NAMED[name])))
        self.assertEqual([block[0].split(':')[0] for block in blocks()], list(CONFIGURATIONS))

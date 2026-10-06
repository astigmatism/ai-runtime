"""Render launch configuration and discovery from the same profile definitions."""
import copy
import hashlib
import json
import re
from pathlib import Path

DAYTIME_PROFILES = ('daytime', 'daytime-27b', 'daytime-flash-f16', 'daytime-27b-tensor-next',
    'daytime-27b-q6k-tensor-next', 'daytime-flash-next', 'daytime-flash-solo', 'daytime-flash-solo-mtp3',
    'daytime-flash-solo-pmin3', 'daytime-flash-solo-pmin4', 'daytime-flash-solo-batch4k', 'daytime-flash-solo-ram',
    'daytime-flash-solo-ub2048', 'daytime-flash-solo-f16kv', 'daytime-flash-solo-diag',
    'daytime-flash-solo-pp512', 'daytime-flash-solo-43fe9c6',
    'daytime-flash-solo-pinned')
NIGHTTIME_PROFILE = 'nighttime'
PAIR_GROUPS = ('daytime', 'nighttime')
# An exclusive Daytime profile reserves both text pairs, in this order, and runs without Nighttime.
EXCLUSIVE_GROUP = 'all'
GPU_UUID = r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}'
# LLM Router's resident-catalog ceiling (RESIDENT_CONTEXT_LIMIT, llm-router 7235329 and later): the
# native 256K window of the Qwen3.8 models. Older routers reject catalogs above 163840 tokens.
ROUTER_CONTEXT_LIMIT = 262144


def require_router_context(tokens):
    require(type(tokens) is int and 0 < tokens <= ROUTER_CONTEXT_LIMIT,
        f'Context must be within the current router contract: 1–{ROUTER_CONTEXT_LIMIT} tokens')


def read(path):
    return json.loads(Path(path).read_text())


def context_label(tokens):
    return f'{tokens // 1024}K'


def display_name(definition):
    """The published display identity; the launch catalog and the status listing share it."""
    return f"{definition['display_name']} ({context_label(definition['context_tokens'])})"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def service_engine(bundle, role):
    """Read both current per-service pins and previous single-engine release state."""
    manifest = bundle['manifest']
    service = next(s for s in manifest['services'] if s['role'] == role)
    return service.get('engine') or manifest['engine']


def is_exclusive(definition):
    """An exclusive profile owns all four text GPUs; Nighttime is not run while it is selected."""
    exclusive = definition.get('exclusive', False)
    require(exclusive is True or exclusive is False, definition['id'] + ': exclusive must be true or false')
    if exclusive:
        require(definition['role'] == 'coding' and definition['gpu_group'] == EXCLUSIVE_GROUP,
            definition['id'] + ': an exclusive profile is a coding profile on the "all" GPU group')
    else:
        require(definition['gpu_group'] in PAIR_GROUPS, definition['id'] + ': unknown GPU group')
    return exclusive


def text_gpus(host, definition):
    """Text GPU UUIDs in CUDA order for an exclusive profile, or the configured pair otherwise."""
    groups = PAIR_GROUPS if is_exclusive(definition) else (definition['gpu_group'],)
    devices = []
    for group in groups:
        pair = host['gpu_ids'][group]
        require(len(pair) == 2 and all(x.startswith('GPU-') and 'REPLACE' not in x for x in pair),
            'Configure two real GPU UUIDs per group in host.json')
        devices.extend(pair)
    return devices


def gpu_names(host, definition):
    """Card names in the same order as text_gpus(); empty when the host does not name them."""
    names = (host or {}).get('gpu_names') or {}
    groups = PAIR_GROUPS if is_exclusive(definition) else (definition['gpu_group'],)
    return [name for group in groups for name in names.get(group, [])]


def vision_devices(host, group, options):
    """A shared encoder GPU is never a member of either text pair."""
    vision = host.get('vision_gpu_id')
    if vision is None:
        return None
    require(isinstance(vision, str) and re.fullmatch(GPU_UUID, vision),
        'vision_gpu_id must be a full GPU UUID')
    pairs = host['gpu_ids']
    require(vision not in [gpu for pair in pairs.values() for gpu in pair],
        'Shared vision GPU must be distinct from all text GPUs')
    order = host.get('cuda_order', {}).get(group)
    require(isinstance(order, list) and len(order) == 2 and len(set(order)) == 2
        and set(order) == set(pairs[group]),
        'cuda_order must explicitly order the two text GPU UUIDs for each group')
    require(set(options.get('--device', '').split(',')) == {'CUDA0', 'CUDA1'},
        'Language-model devices must remain CUDA0 and CUDA1')
    require(len(options.get('--tensor-split', '').split(',')) == 2,
        'Language-model tensor split must contain exactly two text GPUs')
    if '--main-gpu' in options:
        require(options['--main-gpu'] in ('0', '1'), 'Main GPU must remain on a text GPU')
    if '--spec-draft-device' in options:
        require(options['--spec-draft-device'] in ('CUDA0', 'CUDA1'),
            'Draft model must remain on a text GPU')
    overrides = options.get('--override-tensor', [])
    if isinstance(overrides, str):
        overrides = [overrides]
    require(all(re.fullmatch(r'.+=(CPU|CUDA[01])', item) for item in overrides),
        'Tensor overrides must target CPU or a text GPU, never the vision GPU')
    require(options.get('--no-mmproj-offload') is True and options.get('--mmproj-device') == 'none',
        'Profiles must retain their CPU vision defaults; configure GPU vision in host.json')
    return [*order, vision]


def exclusive_devices(host, options, text):
    """CUDA order for an exclusive profile: the four text GPUs, then the vision GPU.

    Nighttime is not running, so the vision GPU is not shared: it carries the projector and
    may also carry the MTP draft. The language model itself stays on the four text GPUs.
    """
    vision = host.get('vision_gpu_id')
    require(vision is not None,
        'Exclusive profiles require a configured vision GPU (vision_gpu_id) for the projector and MTP draft')
    require(isinstance(vision, str) and re.fullmatch(GPU_UUID, vision), 'vision_gpu_id must be a full GPU UUID')
    require(vision not in text, 'Shared vision GPU must be distinct from all text GPUs')
    text_cuda = [f'CUDA{i}' for i in range(len(text))]
    vision_cuda = f'CUDA{len(text)}'
    listed = options.get('--device', '').split(',')
    require(len(listed) == len(text_cuda) and set(listed) == set(text_cuda),
        'Exclusive language-model devices must list each of ' + ','.join(text_cuda) + ' exactly once')
    if '--tensor-split' in options:
        require(len(options['--tensor-split'].split(',')) == len(text),
            'Exclusive tensor split must contain exactly one value per text GPU')
    if '--main-gpu' in options:
        require(options['--main-gpu'] in [str(i) for i in range(len(text))], 'Main GPU must remain on a text GPU')
    if '--spec-draft-device' in options:
        require(options['--spec-draft-device'] in [*text_cuda, vision_cuda], 'Draft model must use a reserved GPU')
    overrides = options.get('--override-tensor', [])
    if isinstance(overrides, str):
        overrides = [overrides]
    require(all(re.fullmatch(r'.+=(CPU|CUDA[0-%d])' % (len(text) - 1), item) for item in overrides),
        'Tensor overrides must target CPU or a text GPU, never the vision GPU')
    require(options.get('--no-mmproj-offload') is True and options.get('--mmproj-device') == 'none',
        'Profiles must retain their CPU vision defaults; configure GPU vision in host.json')
    return [*text, vision], vision_cuda


def profile_summary(config_dir, name, shared, host=None):
    """Public facts about one profile, derived without Docker, network, or artifact I/O.

    Host paths, mount targets, and artifact checksums are deliberately absent: this is
    published by the status API for every configuration, including ones that
    are not running.
    """
    definition = read(Path(config_dir) / 'profiles' / (name + '.json'))
    require(definition.get('id') == name, name + ': profile file id differs from its filename')
    require(definition['engine'] in shared['engines'], name + ': unknown engine ' + str(definition['engine']))
    tokens = definition['context_tokens']
    require_router_context(tokens)
    options = {**shared['arguments'], **definition['arguments']}
    engine = shared['engines'][definition['engine']]
    return {'profile': definition['id'], 'role': definition['role'], 'display_name': display_name(definition),
        'model': options['--alias'], 'context_tokens': tokens, 'engine': definition['engine'],
        'engine_tag': engine['tag'], 'backend_revision': engine['revision'],
        'gpu_group': definition['gpu_group'], 'gpu_names': gpu_names(host, definition),
        'exclusive': is_exclusive(definition)}


def available_profiles(config_dir, host=None):
    """Every configuration this revision can apply: the selectable Daytime set plus the paired Nighttime backend."""
    config_dir = Path(config_dir)
    shared = read(config_dir / 'shared.json')
    require(shared['schema_version'] == 2, 'Unsupported configuration version')
    def describe(name):
        return profile_summary(config_dir, name, shared, host)
    return {'selectable': [describe(name) for name in DAYTIME_PROFILES],
        'always_included': [describe(NIGHTTIME_PROFILE)]}


def render(config_dir, host, profile):
    config_dir = Path(config_dir)
    require(profile in DAYTIME_PROFILES, 'Unknown or retired daytime profile')
    shared = read(config_dir / 'shared.json')
    require(shared['schema_version'] == 2, 'Unsupported configuration version')
    root = Path(host['model_root'])
    require(root.is_absolute() and root != Path('/'), 'Model root must be an absolute non-root directory')
    compose = {'name': shared['backend_project'], 'services': {}, 'networks': {
        'router': {'external': True, 'name': shared['network']}}}
    models, artifacts, manifests, gpu_ids = [], {}, [], []
    exclusive_profile = is_exclusive(read(config_dir / 'profiles' / (profile + '.json')))
    for name in (profile,) if exclusive_profile else (profile, NIGHTTIME_PROFILE):
        definition = read(config_dir / 'profiles' / (name + '.json'))
        role = definition['role']
        engine = shared['engines'][definition['engine']]
        ctx = definition['context_tokens']
        require_router_context(ctx)
        options = {**shared['arguments'], **definition['arguments'],
            '--ctx-size': str(ctx), '--kv-unified-per-slot': str(ctx)}
        order = list(definition['argument_order'])
        exclusive = is_exclusive(definition)
        if exclusive:
            cuda_order, vision_device = exclusive_devices(host, options, text_gpus(host, definition))
        else:
            cuda_order, vision_device = vision_devices(host, definition['gpu_group'], options), 'CUDA2'
        if cuda_order:
            options.pop('--no-mmproj-offload')
            options['--mmproj-offload'] = True
            options['--mmproj-device'] = vision_device
            order[order.index('--no-mmproj-offload')] = '--mmproj-offload'
        require(len(order) == len(set(order)) and set(order) == set(options),
            name + ': argument_order must contain every effective argument exactly once')
        for key, expected in {'--n-predict': '-1', '--reasoning-budget': '-1',
                '--reasoning-effort': 'default', '--parallel': '1', '--offline': True}.items():
            require(options.get(key) == expected, name + ': unsupported policy change: ' + key)
        argv = []
        for key in order:
            value = options[key]
            if isinstance(value, list):
                require(key == '--override-tensor' and bool(value) and
                    all(isinstance(v, str) and v and not v.startswith('--') for v in value),
                    'Only --override-tensor accepts a nonempty list of repeated values')
                for item in value:
                    argv.extend([key, item])
                continue
            require(key.startswith('--') and (isinstance(value, str) or value is True), 'Invalid launch argument')
            argv.append(key)
            if value is not True:
                argv.append(value)
        text_devices = text_gpus(host, definition)
        gpu_ids.extend(text_devices)
        devices = [*text_devices, host['vision_gpu_id']] if cuda_order else list(text_devices)
        mounts, targets = [], {}
        for artifact in definition['artifacts']:
            relative = Path(artifact['path'])
            require(not relative.is_absolute() and '..' not in relative.parts, 'Artifact path escapes model root')
            require(len(artifact['sha256']) == 64 and all(c in '0123456789abcdef' for c in artifact['sha256']),
                'Invalid artifact SHA-256')
            source = str(root / relative)
            require(artifact['target'] not in targets, 'Duplicate artifact mount')
            targets[artifact['target']] = source
            artifacts[source] = {**artifact, 'source': source}
            mounts.append({'type': 'bind', 'source': source, 'target': artifact['target'],
                'read_only': True, 'bind': {'create_host_path': False}})
        cfg = copy.deepcopy(shared['container_defaults'])
        cfg.update(copy.deepcopy(definition['container']))
        cfg.update(image=engine['tag'], container_name=definition['container_name'], command=argv,
            user=f"{host['uid']}:{host['gid']}", volumes=mounts,
            ports=[f"127.0.0.1:{definition['host_port']}:8080"],
            deploy={'resources': {'reservations': {'devices': [{
                'driver': 'nvidia', 'device_ids': devices, 'capabilities': ['gpu']}]}}})
        if cuda_order:
            cfg['environment'] = {**cfg.get('environment', {}),
                'CUDA_VISIBLE_DEVICES': ','.join(cuda_order)}
        require(options['--model'] in targets and options['--mmproj'] in targets,
            'Model and projector arguments must reference declared artifact mounts')
        split_model = re.fullmatch(r'(.+)-(\d{5})-of-(\d{5})\.gguf', options['--model'])
        if split_model:
            prefix, first, total = split_model.groups()
            require(int(first) == 1 and 0 < int(total) <= 99999, 'Split model must start at shard one')
            require(all(f'{prefix}-{i:05d}-of-{total}.gguf' in targets for i in range(1, int(total) + 1)),
                'Split model is missing a declared shard mount')
        compose['services'][role] = cfg
        entry = {**copy.deepcopy(shared['catalog_defaults']), **copy.deepcopy(definition['catalog'])}
        entry.update(model=options['--alias'], context_length=ctx, total_context_length=ctx,
            max_active_requests=int(options['--parallel']), gpu_uuids=devices,
            model_path=targets[options['--model']], mmproj_path=targets[options['--mmproj']],
            backend_revision=engine['revision'], fit_target=options['--fit'],
            global_ram_prompt_cache_mib=int(options['--cache-ram']),
            kv_cache={'unified': False, 'key_type': options['--cache-type-k'], 'value_type': options['--cache-type-v']},
            display_name=display_name(definition),
            source='local-ai-runtime', updated_at=None)
        if cuda_order:
            entry.update(mmproj_offload='gpu', text_gpu_uuids=text_devices,
                vision_gpu_uuid=host['vision_gpu_id'], vision_device=vision_device,
                vision_gpu_shared=not exclusive, cuda_visible_devices=cuda_order)
        if exclusive:
            entry['exclusive'] = True
        if entry['mtp'].get('enabled'):
            target = entry['mtp'].pop('artifact_target')
            require(target == options['--spec-draft-model'] and target in targets,
                'MTP argument and catalog must reference the same declared artifact')
            entry['mtp']['model_path'] = targets[target]
            entry['mtp'].update(max_draft_tokens=int(options['--spec-draft-n-max']),
                gpu_layers=options['--spec-draft-ngl'], device=options['--spec-draft-device'],
                key_cache_type=options['--spec-draft-type-k'], value_cache_type=options['--spec-draft-type-v'])
        require(entry['output_policy'] == 'unrestricted' and entry['default_output_tokens'] is None
            and entry['max_output_tokens'] is None and entry['server_default_output_tokens'] == -1,
            'Catalog must preserve unrestricted output policy')
        require(entry['reasoning_policy']['default_level'] == 'default' and all(
            level.get('default_output_tokens') is None and level.get('max_output_tokens') is None
            and (not level['enabled'] or level['reasoning_budget_tokens'] == -1)
            for level in entry['reasoning_policy']['levels'].values()), 'Catalog reasoning policy differs')
        models.append(entry)
        manifests.append({'role': role, 'container_name': cfg['container_name'],
            'engine': copy.deepcopy(engine),
            'model_alias': entry['model'], 'context_tokens': ctx, 'parallel_slots': 1,
            'display_name': entry['display_name'], 'recommended_argv': argv,
            'gpu_device_ids': devices, 'mounts': mounts})
    require(len(gpu_ids) == 4 and len(set(gpu_ids)) == 4, 'Backends require four distinct GPUs')
    catalog = {**copy.deepcopy(models[0]), 'schema_version': 3,
        'default_model': models[0]['model'], 'models': models}
    bundle = {'schema_version': 1, 'profile': profile, 'compose': compose, 'catalog': catalog,
        'manifest': {'schema_version': 2, 'services': manifests},
        'artifacts': list(artifacts.values())}
    bundle['config_sha256'] = digest(bundle)
    return bundle

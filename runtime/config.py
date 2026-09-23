"""Render launch configuration and discovery from the same profile definitions."""
import copy
import hashlib
import json
import re
from pathlib import Path

DAYTIME_PROFILES = ('daytime', 'daytime-27b')
NIGHTTIME_PROFILE = 'nighttime'


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


def vision_devices(host, group, options):
    """A shared encoder GPU is never a member of either exclusive text pair."""
    vision = host.get('vision_gpu_id')
    if vision is None:
        return None
    require(isinstance(vision, str) and re.fullmatch(
        r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', vision),
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


def profile_summary(config_dir, name, shared, gpu_names=None):
    """Public facts about one profile, derived without Docker, network, or artifact I/O.

    Host paths, mount targets, and artifact checksums are deliberately absent: this is
    published by the status API for every configuration, including ones that
    are not running.
    """
    definition = read(Path(config_dir) / 'profiles' / (name + '.json'))
    require(definition.get('id') == name, name + ': profile file id differs from its filename')
    require(definition['engine'] in shared['engines'], name + ': unknown engine ' + str(definition['engine']))
    tokens = definition['context_tokens']
    require(type(tokens) is int and 0 < tokens <= 163840, 'Context must be within the current router contract: 1–163840 tokens')
    options = {**shared['arguments'], **definition['arguments']}
    engine = shared['engines'][definition['engine']]
    return {'profile': definition['id'], 'role': definition['role'], 'display_name': display_name(definition),
        'model': options['--alias'], 'context_tokens': tokens, 'engine': definition['engine'],
        'engine_tag': engine['tag'], 'backend_revision': engine['revision'],
        'gpu_group': definition['gpu_group'], 'gpu_names': list((gpu_names or {}).get(definition['gpu_group'], []))}


def available_profiles(config_dir, host=None):
    """Every configuration this revision can apply: the selectable Daytime set plus the paired Nighttime backend."""
    config_dir = Path(config_dir)
    shared = read(config_dir / 'shared.json')
    require(shared['schema_version'] == 2, 'Unsupported configuration version')
    gpu_names = (host or {}).get('gpu_names') or {}
    def describe(name):
        return profile_summary(config_dir, name, shared, gpu_names)
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
    for name in (profile, NIGHTTIME_PROFILE):
        definition = read(config_dir / 'profiles' / (name + '.json'))
        role = definition['role']
        engine = shared['engines'][definition['engine']]
        ctx = definition['context_tokens']
        require(type(ctx) is int and 0 < ctx <= 163840, 'Context must be within the current router contract: 1–163840 tokens')
        options = {**shared['arguments'], **definition['arguments'],
            '--ctx-size': str(ctx), '--kv-unified-per-slot': str(ctx)}
        order = list(definition['argument_order'])
        cuda_order = vision_devices(host, definition['gpu_group'], options)
        if cuda_order:
            options.pop('--no-mmproj-offload')
            options['--mmproj-offload'] = True
            options['--mmproj-device'] = 'CUDA2'
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
        devices = host['gpu_ids'][definition['gpu_group']]
        require(len(devices) == 2 and all(x.startswith('GPU-') and 'REPLACE' not in x for x in devices),
            'Configure two real GPU UUIDs per group in host.json')
        gpu_ids.extend(devices)
        text_devices = list(devices)
        devices = [*devices, host['vision_gpu_id']] if cuda_order else devices
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
                vision_gpu_uuid=host['vision_gpu_id'], vision_device='CUDA2',
                vision_gpu_shared=True, cuda_visible_devices=cuda_order)
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

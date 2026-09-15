"""Render launch configuration and discovery from the same profile definitions."""
import copy
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def render(config_dir, host, profile):
    config_dir = Path(config_dir)
    require(profile in ('daytime', 'daytime-swift'), 'Unknown daytime profile')
    shared = read(config_dir / 'shared.json')
    require(shared['schema_version'] == 1, 'Unsupported configuration version')
    root = Path(host['model_root'])
    require(root.is_absolute() and root != Path('/'), 'Model root must be an absolute non-root directory')
    compose = {'name': shared['backend_project'], 'services': {}, 'networks': {
        'router': {'external': True, 'name': shared['network']}}}
    models, artifacts, manifests, gpu_ids = [], {}, [], []
    for name in (profile, 'nighttime'):
        definition = read(config_dir / 'profiles' / (name + '.json'))
        role = definition['role']
        ctx = definition['context_tokens']
        require(type(ctx) is int and 0 < ctx <= 163840, 'Context must be within the current router contract: 1–163840 tokens')
        options = {**shared['arguments'], **definition['arguments'],
            '--ctx-size': str(ctx), '--kv-unified-per-slot': str(ctx)}
        order = definition['argument_order']
        require(len(order) == len(set(order)) and set(order) == set(options),
            name + ': argument_order must contain every effective argument exactly once')
        for key, expected in {'--n-predict': '-1', '--reasoning-budget': '-1',
                '--reasoning-effort': 'default', '--parallel': '1', '--offline': True}.items():
            require(options.get(key) == expected, name + ': unsupported policy change: ' + key)
        argv = []
        for key in order:
            value = options[key]
            require(key.startswith('--') and (isinstance(value, str) or value is True), 'Invalid launch argument')
            argv.append(key)
            if value is not True:
                argv.append(value)
        devices = host['gpu_ids'][definition['gpu_group']]
        require(len(devices) == 2 and all(x.startswith('GPU-') and 'REPLACE' not in x for x in devices),
            'Configure two real GPU UUIDs per group in host.json')
        gpu_ids.extend(devices)
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
        cfg.update(container_name=definition['container_name'], command=argv,
            user=f"{host['uid']}:{host['gid']}", volumes=mounts,
            ports=[f"127.0.0.1:{definition['host_port']}:8080"],
            deploy={'resources': {'reservations': {'devices': [{
                'driver': 'nvidia', 'device_ids': devices, 'capabilities': ['gpu']}]}}})
        require(cfg['image'] == shared['engine']['tag'], 'Engine tag differs from pinned engine')
        compose['services'][role] = cfg
        entry = {**copy.deepcopy(shared['catalog_defaults']), **copy.deepcopy(definition['catalog'])}
        entry.update(model=options['--alias'], context_length=ctx, total_context_length=ctx,
            max_active_requests=int(options['--parallel']), gpu_uuids=devices,
            model_path=targets['/weights/main.gguf'], mmproj_path=targets['/weights/projector.gguf'],
            backend_revision=shared['engine']['revision'], fit_target=options['--fit'],
            global_ram_prompt_cache_mib=int(options['--cache-ram']),
            kv_cache={'unified': False, 'key_type': options['--cache-type-k'], 'value_type': options['--cache-type-v']},
            display_name=f"{definition['display_name']} ({ctx // 1024}K)",
            source='local-ai-runtime', updated_at=None)
        if entry['mtp'].get('enabled'):
            target = entry['mtp'].pop('artifact_target')
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
            'model_alias': entry['model'], 'context_tokens': ctx, 'parallel_slots': 1,
            'display_name': entry['display_name'], 'recommended_argv': argv,
            'gpu_device_ids': devices, 'mounts': mounts})
    require(len(gpu_ids) == 4 and len(set(gpu_ids)) == 4, 'Backends require four distinct GPUs')
    catalog = {**copy.deepcopy(models[0]), 'schema_version': 3,
        'default_model': models[0]['model'], 'models': models}
    bundle = {'schema_version': 1, 'profile': profile, 'compose': compose, 'catalog': catalog,
        'manifest': {'schema_version': 1, 'engine': shared['engine'], 'services': manifests},
        'artifacts': list(artifacts.values())}
    bundle['config_sha256'] = digest(bundle)
    return bundle

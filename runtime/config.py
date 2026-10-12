"""Render launch configuration and discovery from the same profile definitions."""
import copy
import hashlib
import json
import re
from pathlib import Path

# Selectable configurations. Paired profiles run on the Daytime pair beside Nighttime; exclusive
# (solo) profiles hold every GPU and stop Nighttime. Display names come from each profile file.
# The Strata entry is a trial: it is selectable by name only and carries no aliases, so selecting
# something else never routes to it and it never shadows the active profile.
DAYTIME_PROFILES = ('qwen27b-q8-with-nighttime', 'qwen27b-q6k-with-nighttime', 'flash-next-solo-128k',
    'flash-next-solo-160k', 'flash-next-strata-160k')
# Started when no release has been recorded yet (a new or rebuilt host without an initial_profile).
DEFAULT_PROFILE = 'qwen27b-q8-with-nighttime'
NIGHTTIME_PROFILE = 'nighttime'
# Nighttime variants a paired configuration may name with "nighttime"; NIGHTTIME_PROFILE is the default.
NIGHTTIME_PROFILES = ('nighttime',)
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


def parallel_slots(definition):
    """Backend slots: one by default; an exclusive profile, which has every GPU, may run two."""
    slots = definition.get('parallel_slots', 1)
    require(type(slots) is int and slots in (1, 2), definition['id'] + ': parallel_slots must be 1 or 2')
    require(slots == 1 or is_exclusive(definition), definition['id'] + ': only an exclusive profile may run two slots')
    return slots


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


def nighttime_for(definition):
    """The Nighttime profile a paired configuration runs beside it."""
    name = definition.get('nighttime', NIGHTTIME_PROFILE)
    require(name in NIGHTTIME_PROFILES, definition['id'] + ': unknown Nighttime profile ' + str(name))
    require(not is_exclusive(definition) or 'nighttime' not in definition, definition['id'] + ': an exclusive profile runs no Nighttime')
    return name


def vision_devices(host, group, options, draft_on_vision=False):
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
        # A profile that declares draft_on_vision_gpu may put its draft beside the shared projectors;
        # the text model itself still never uses the vision GPU.
        require(options['--spec-draft-device'] in (('CUDA0', 'CUDA1', 'CUDA2') if draft_on_vision else ('CUDA0', 'CUDA1')),
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


def engine_interface(engine):
    """How a backend takes its configuration: the llama.cpp argument vector, or a declared
    environment.

    Every engine in this house so far is llama.cpp: it is configured by a long argv, its shared
    argument vocabulary in shared.json, and a fixed set of flags the controller and the router
    policy read out of it. An engine that configures itself from the environment (Strata's
    container entrypoint reads env variables and writes its own config file) declares
    "interface": "environment" plus the argument-to-variable map, and is then rendered from its
    own profile arguments alone: the llama.cpp vocabulary does not apply to it, and neither do the
    llama.cpp-only catalog fields the controller used to insist on. The llama.cpp path is
    untouched by this: an engine without an interface is rendered exactly as before.
    """
    interface = engine.get('interface', 'argv')
    require(interface in ('argv', 'environment'), 'Unknown engine interface: ' + str(interface))
    if interface == 'environment':
        require(isinstance(engine.get('environment'), dict) and engine['environment'],
            'An environment-configured engine must map its arguments to environment variables')
    return interface


def merge_settings(base, override):
    """Layer container settings on top of the ones they inherit, so an override may name part of a block.

    Dict values are merged key by key and everything else is replaced whole: a profile that adds one
    ulimit must not have to restate the ones it inherits, and a list such as tmpfs, a command or a
    mount list is the profile's own, never appended to the defaults.
    """
    merged = copy.deepcopy(base)
    for key, value in copy.deepcopy(override).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_settings(merged[key], value)
        else:
            merged[key] = value
    return merged


# Every rendered service carries these. They are how a backend runs here rather than what it serves:
# never pulled at deploy time, restarted under it, given a real pid one, kept to its own log files and
# core-dump limits, and watched by a healthcheck. The controller's drift check reads most of them
# directly, so a service rendered without one is a configuration bug that has to surface as a
# configuration error - not as a KeyError partway through a profile switch.
CONTAINER_CONTRACT = ('pull_policy', 'entrypoint', 'restart', 'read_only', 'init', 'ulimits',
    'healthcheck', 'logging')


def require_container_contract(role, cfg):
    missing = [key for key in CONTAINER_CONTRACT if cfg.get(key) is None]
    require(not missing, 'Service {} is missing container settings: {}'.format(role, ', '.join(missing)))


def environment_service(name, definition, engine, shared, host, root, ctx, slots, exclusive):
    """One service, its catalog entry, and its mounts for an engine configured from the environment.

    The profile still states its settings in the house's "--flag value" form, so a profile reads the
    same whichever engine it selects and the same validation applies; the engine entry says which
    variable each flag becomes. Only flags the engine maps reach the container, and a mapped flag
    whose value the renderer cannot state as a string is a refusal rather than a silent drop.
    """
    options = dict(definition['arguments'])
    order = list(definition['argument_order'])
    require(len(order) == len(set(order)) and set(order) == set(options),
        name + ': argument_order must contain every effective argument exactly once')
    environment = {}
    model_argument = engine.get('model_argument', '--model')
    for key in order:
        value = options[key]
        require(isinstance(value, (str, int, float)) or value is True, name + ': invalid engine argument ' + key)
        variable = engine['environment'].get(key)
        if variable is None:
            # Two arguments are not settings the engine is told about directly: --alias is this
            # house's discovery name, and the model path becomes a variable only as the folder
            # derived from it, so a mount and an engine setting cannot disagree about which
            # revision is being served.
            require(key in ('--alias', model_argument), name + ': argument ' + key +
                ' has no environment variable in engine ' + definition['engine'])
            continue
        require(variable not in environment, name + ': two arguments map to ' + variable)
        environment[variable] = '' if value is True else str(value)
    model_target = options[model_argument]
    require(model_target in [a['target'] for a in definition['artifacts']],
        name + ': the model argument must reference a declared artifact mount')
    split_model = re.fullmatch(r'(.+)-(\d{5})-of-(\d{5})\.gguf', model_target.rsplit('/', 1)[-1])
    require(split_model is not None, name + ': an environment-configured model must be a GGUF shard set')
    prefix, first, total = split_model.groups()
    require(int(first) == 1 and 0 < int(total) <= 99999, 'Split model must start at shard one')
    require(all(f'{prefix}-{i:05d}-of-{total}.gguf' in
        [a['target'].rsplit('/', 1)[-1] for a in definition['artifacts']] for i in range(1, int(total) + 1)),
        'Split model is missing a declared shard mount')
    # Where the engine looks for GGUF files it already has: the folder holding the shards. It is
    # derived from the model path instead of being written twice, so the mount and the engine can
    # not disagree about which revision is being served.
    if engine.get('model_directory'):
        environment[engine['model_directory']] = model_target.rsplit('/', 1)[0]
    if engine.get('context_environment'):
        environment[engine['context_environment']] = str(ctx * slots)
    text_devices = text_gpus(host, definition)
    mounts, targets = [], {}
    for artifact in definition['artifacts']:
        relative = Path(artifact['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Artifact path escapes model root')
        require(len(artifact['sha256']) == 64 and all(c in '0123456789abcdef' for c in artifact['sha256']),
            'Invalid artifact SHA-256')
        source = str(root / relative)
        require(artifact['target'] not in targets, 'Duplicate artifact mount')
        targets[artifact['target']] = source
        mounts.append({'type': 'bind', 'source': source, 'target': artifact['target'],
            'read_only': True, 'bind': {'create_host_path': False}})
    workspace = definition.get('workspace')
    if workspace:
        relative = Path(workspace['path'])
        require(not relative.is_absolute() and '..' not in relative.parts and relative.parts,
            'Workspace path escapes model root')
        require(workspace['target'].startswith('/') and workspace['target'] != '/',
            'Workspace mount target must be an absolute path')
        mounts.append({'type': 'bind', 'source': str(root / relative), 'target': workspace['target'],
            'read_only': False, 'bind': {'create_host_path': False}})
    # Three layers deep-merged, in this order: the house defaults, what this engine narrows, what the
    # profile states. Replace-semantics here is the trial's first switch failure - an engine entry that
    # declared only its entrypoint, root, ulimits and healthcheck silently arrived with no restart
    # policy, no init, no logging and no pull policy, and the observer crashed on the missing key.
    cfg = merge_settings(shared['container_defaults'], engine.get('container_defaults', {}))
    cfg = merge_settings(cfg, definition['container'])
    # What the profile states about the environment comes first: the engine's own settings are the
    # rendered ones and win, so a profile cannot accidentally overwrite a derived variable.
    environment = {**cfg.get('environment', {}), **environment}
    cfg.update(image=engine['tag'], container_name=definition['container_name'], command=[],
        environment=environment, volumes=mounts,
        ports=[f"127.0.0.1:{definition['host_port']}:8080"],
        deploy={'resources': {'reservations': {'devices': [{
            'driver': 'nvidia', 'device_ids': text_devices, 'capabilities': ['gpu']}]}}})
    # The house default is the deployment user, which can write the workspace and the model files but
    # nothing inside a prebuilt image. An engine that installs into its own virtualenv on first start
    # therefore says so on its engine entry, and gets the user the image was built for instead. It is
    # stated on the engine and not inferred from the interface, because it is a property of what is
    # baked into the image, and it is a wider grant: the container runs with that image's privileges.
    if engine.get('runs_as_image_user'):
        cfg.pop('user', None)
    else:
        cfg['user'] = f"{host['uid']}:{host['gid']}"
    require(cfg.get('read_only') is False,
        name + ': an environment-configured engine writes its own configuration and needs a writable root')
    require(cfg.get('environment', {}).get('CUDA_VISIBLE_DEVICES'),
        name + ': the container-visible GPU list must be stated, since the engine numbers cards itself')
    entry = {**copy.deepcopy(shared['catalog_defaults']), **copy.deepcopy(definition['catalog'])}
    # Same rule as the llama.cpp path: the router never infers this from a model name.
    require(type(entry['capability_profile'].get('nsfw')) is bool,
        name + ': capability_profile.nsfw must be declared true or false')
    entry.update(model=options['--alias'], context_length=ctx,
        total_context_length=ctx * slots, max_active_requests=slots, gpu_uuids=text_devices,
        model_path=targets[model_target], backend_revision=engine['revision'],
        display_name=display_name(definition), gpu_names=gpu_names(host, definition),
        source='local-ai-runtime', updated_at=None)
    if exclusive:
        entry['exclusive'] = True
    require_open_generation_policy(name, entry)
    return cfg, entry, mounts, targets, text_devices


def profile_summary(config_dir, name, shared, host=None):
    """Public facts about one profile, derived without Docker, network, or artifact I/O.

    Host paths, mount targets, and artifact checksums are deliberately absent: this is
    published by the status API for every configuration, including ones that
    are not running.
    """
    definition = read(Path(config_dir) / 'profiles' / (name + '.json'))
    require(definition.get('id') == name, name + ': profile file id differs from its filename')
    require(definition['engine'] in shared['engines'], name + ': unknown engine ' + str(definition['engine']))
    engine = shared['engines'][definition['engine']]
    tokens = definition['context_tokens']
    require_router_context(tokens)
    # The shared llama.cpp vocabulary says nothing about an engine that configures itself from the
    # environment, so it is only merged for the engines that actually receive it.
    options = {**shared['arguments'], **definition['arguments']} if engine_interface(engine) == 'argv' \
        else dict(definition['arguments'])
    return {'profile': definition['id'], 'role': definition['role'], 'display_name': display_name(definition),
        'model': options['--alias'], 'context_tokens': tokens, 'engine': definition['engine'],
        'engine_tag': engine['tag'], 'backend_revision': engine['revision'],
        'gpu_group': definition['gpu_group'], 'gpu_names': gpu_names(host, definition),
        'exclusive': is_exclusive(definition), 'parallel_slots': definition.get('parallel_slots', 1),
        **({'nighttime': nighttime_for(definition)} if definition['role'] == 'coding' and not is_exclusive(definition) else {}),
        **({'variant_label': definition['variant_label']} if definition.get('variant_label') else {})}


def available_profiles(config_dir, host=None):
    """Every configuration this revision can apply: the selectable Daytime set plus the paired Nighttime backend."""
    config_dir = Path(config_dir)
    shared = read(config_dir / 'shared.json')
    require(shared['schema_version'] == 2, 'Unsupported configuration version')
    def describe(name):
        return profile_summary(config_dir, name, shared, host)
    selectable = [describe(name) for name in DAYTIME_PROFILES]
    used = {entry['nighttime'] for entry in selectable if 'nighttime' in entry}
    return {'selectable': selectable,
        'always_included': [describe(name) for name in NIGHTTIME_PROFILES if name == NIGHTTIME_PROFILE or name in used]}


def offline_services(config_dir, shared, host, exclusive):
    """Services this configuration deliberately does not run, so the router can report them as
    temporarily offline rather than unknown. An exclusive profile stops Nighttime."""
    if not exclusive:
        return []
    night = read(Path(config_dir) / 'profiles' / (NIGHTTIME_PROFILE + '.json'))
    summary = profile_summary(config_dir, NIGHTTIME_PROFILE, shared, host)
    return [{'model': summary['model'], 'aliases': list(night['catalog'].get('aliases', [])),
        'display_name': summary['display_name'], 'role': night['role'], 'reason': 'exclusive_configuration'}]


def require_open_generation_policy(name, entry):
    """The router contract every served model is published under, whichever engine serves it.

    No server-side output ceiling and no reasoning budget: the controller verifies the live
    backend really runs unbounded, and the router refuses a catalog that narrows either. It reads
    only published catalog fields, so it holds for an engine that is not llama.cpp too.
    """
    require(entry['output_policy'] == 'unrestricted' and entry['default_output_tokens'] is None
        and entry['max_output_tokens'] is None and entry['server_default_output_tokens'] == -1,
        'Catalog must preserve unrestricted output policy')
    require(entry['reasoning_policy']['default_level'] == 'default' and all(
        level.get('default_output_tokens') is None and level.get('max_output_tokens') is None
        and (not level['enabled'] or level['reasoning_budget_tokens'] == -1)
        for level in entry['reasoning_policy']['levels'].values()), 'Catalog reasoning policy differs')


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
    selected = read(config_dir / 'profiles' / (profile + '.json'))
    exclusive_profile = is_exclusive(selected)
    night_profile = nighttime_for(selected)  # also refuses a Nighttime named by an exclusive profile
    night_profile = None if exclusive_profile else night_profile
    for name in (profile,) if exclusive_profile else (profile, night_profile):
        definition = read(config_dir / 'profiles' / (name + '.json'))
        role = definition['role']
        engine = shared['engines'][definition['engine']]
        ctx = definition['context_tokens']
        require_router_context(ctx)
        slots = parallel_slots(definition)
        if engine_interface(engine) == 'environment':
            # A non-llama.cpp backend is rendered from its own profile arguments and the engine's
            # variable map: nothing in the llama.cpp vocabulary, its artifact-only mount list, or its
            # per-slot KV flags applies to it, and pretending otherwise would silently drop settings
            # the engine needs. Everything the router and the controller still have to agree on —
            # image identity, mounts, ports, the GPU reservation, the discovery entry — is stated here
            # in the same shape as the llama.cpp path.
            cfg, entry, mounts, targets, devices = environment_service(name, definition, engine, shared,
                host, root, ctx, slots, exclusive_profile)
            for artifact in definition['artifacts']:
                source = str(root / Path(artifact['path']))
                artifacts[source] = {**artifact, 'source': source}
            gpu_ids.extend(devices)
            require_container_contract(role, cfg)
            compose['services'][role] = cfg
            models.append(entry)
            manifests.append({'role': role, 'container_name': cfg['container_name'],
                'engine': copy.deepcopy(engine), 'model_alias': entry['model'], 'context_tokens': ctx,
                'parallel_slots': slots, 'display_name': entry['display_name'],
                'recommended_argv': [], 'recommended_environment': dict(cfg['environment']),
                'gpu_device_ids': devices, 'mounts': mounts})
            continue
        options = {**shared['arguments'], **definition['arguments']}
        require(options.get('--parallel') == '1', name + ': unsupported policy change: --parallel')
        # context_tokens is the per-request (per-slot) window; separate KV per slot sums to the total.
        options.update({'--ctx-size': str(ctx * slots), '--kv-unified-per-slot': str(ctx), '--parallel': str(slots)})
        order = list(definition['argument_order'])
        exclusive = is_exclusive(definition)
        if exclusive:
            cuda_order, vision_device = exclusive_devices(host, options, text_gpus(host, definition))
        else:
            cuda_order, vision_device = vision_devices(host, definition['gpu_group'], options,
                definition.get('draft_on_vision_gpu') is True), 'CUDA2'
        draft_dropped = definition.get('draft_on_vision_gpu') is True and not cuda_order
        if draft_dropped:
            # Without a vision GPU there is no room for this profile's draft: it runs without
            # speculative decoding instead (its own GPUs cannot hold the draft beside the model).
            draft_flags = {key for key in options if key.startswith('--spec-')}
            options = {k: v for k, v in options.items() if k not in draft_flags}
            order = [k for k in order if k not in draft_flags]
        if cuda_order:
            options.pop('--no-mmproj-offload')
            options['--mmproj-offload'] = True
            options['--mmproj-device'] = vision_device
            order[order.index('--no-mmproj-offload')] = '--mmproj-offload'
        require(len(order) == len(set(order)) and set(order) == set(options),
            name + ': argument_order must contain every effective argument exactly once')
        for key, expected in {'--n-predict': '-1', '--reasoning-budget': '-1',
                '--reasoning-effort': 'default', '--parallel': str(slots), '--offline': True}.items():
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
        draft_target = definition['catalog'].get('mtp', {}).get('artifact_target') if draft_dropped else None
        for artifact in (a for a in definition['artifacts'] if a['target'] != draft_target):
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
        require_container_contract(role, cfg)
        compose['services'][role] = cfg
        entry = {**copy.deepcopy(shared['catalog_defaults']), **copy.deepcopy(definition['catalog'])}
        # Router clients select models by this flag ("most capable NSFW model"); every profile
        # states it explicitly, because the router never infers it from a model name.
        require(type(entry['capability_profile'].get('nsfw')) is bool,
            name + ': capability_profile.nsfw must be declared true or false')
        if draft_dropped:
            entry['mtp'] = {'enabled': False}
        # LLM Router admits max_active_requests overlapping requests; every slot has the full window.
        entry.update(model=options['--alias'], context_length=ctx, total_context_length=ctx * slots,
            max_active_requests=slots, gpu_uuids=devices,
            model_path=targets[options['--model']], mmproj_path=targets[options['--mmproj']],
            backend_revision=engine['revision'], fit_target=options['--fit'],
            global_ram_prompt_cache_mib=int(options['--cache-ram']),
            kv_cache={'unified': False, 'key_type': options['--cache-type-k'], 'value_type': options['--cache-type-v']},
            display_name=display_name(definition),
            # Card names of the text GPUs, in the same order as text_gpu_uuids/gpu_uuids; the
            # router publishes names and counts but never UUIDs.
            gpu_names=gpu_names(host, definition),
            source='local-ai-runtime', updated_at=None)
        if cuda_order:
            entry.update(mmproj_offload='gpu', text_gpu_uuids=text_devices,
                vision_gpu_uuid=host['vision_gpu_id'], vision_device=vision_device,
                vision_gpu_shared=not exclusive, cuda_visible_devices=cuda_order,
                vision_gpu_name=host.get('vision_gpu_name'))
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
        require_open_generation_policy(name, entry)
        models.append(entry)
        manifests.append({'role': role, 'container_name': cfg['container_name'],
            'engine': copy.deepcopy(engine),
            'model_alias': entry['model'], 'context_tokens': ctx, 'parallel_slots': slots,
            'display_name': entry['display_name'], 'recommended_argv': argv,
            'gpu_device_ids': devices, 'mounts': mounts})
    require(len(gpu_ids) == 4 and len(set(gpu_ids)) == 4, 'Backends require four distinct GPUs')
    catalog = {**copy.deepcopy(models[0]), 'schema_version': 3,
        'default_model': models[0]['model'], 'models': models,
        'configuration': {'id': profile, 'exclusive': exclusive_profile},
        'offline_services': offline_services(config_dir, shared, host, exclusive_profile)}
    bundle = {'schema_version': 1, 'profile': profile, 'compose': compose, 'catalog': catalog,
        'manifest': {'schema_version': 2, 'services': manifests},
        'artifacts': list(artifacts.values())}
    if night_profile not in (None, NIGHTTIME_PROFILE):
        bundle['nighttime_profile'] = night_profile
    bundle['config_sha256'] = digest(bundle)
    return bundle

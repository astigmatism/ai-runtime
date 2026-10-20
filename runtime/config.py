"""Render launch configuration and discovery from the same profile definitions.

Three words carry the whole model, and none of them is a time of day:

* a **configuration** is one selectable set of launch definitions -- what this module used to call a
  "Daytime profile", because it named the resident that happened to be loaded on the first GPU pair;
* a **resident** is one model a configuration keeps loaded, identified by its `role` (`coding`,
  `everyday`), which is also its Compose service key. Two configurations load two residents on two
  GPU pairs; one loads a single resident across all four;
* a **service ID** is the name a client sends. It belongs to the LLM Router contract, not to this
  vocabulary: `daytime` and `nighttime` are service IDs, and a configuration may point any of them at
  whichever residents it actually runs. `nighttime` answering from the same backend as `daytime` is a
  placement fact, not a contradiction.

GPU group keys in `.state/host.json` (`gpu_ids`, `gpu_names`, `cuda_order`), the shared vision GPU
(`vision_gpu_id`, `vision_gpu_name`, `vision_cuda_device`) and the container names are operational
identifiers, deliberately still the historical ones; see docs/deployment.md and docs/shared-vision.md.
"""
import copy
import hashlib
import json
import re
from pathlib import Path

# Selectable configurations. A paired configuration runs its own resident beside the Everyday
# resident; a solo configuration holds every text GPU and runs the Everyday resident's service ID from
# its own backend instead. Display names come from each profile file.
#
# Every one of these publishes a service alias a contract client can address, and every one publishes
# `daytime`: a client that only ever sends `daytime` -- which is what LLM Router's CLIENT_CONTRACT.md
# tells it to do -- always has something to send, whichever configuration is selected. Asserted for
# every configuration by ServiceIdentityTests in tests/test_config.py. Publishing no service alias at
# all is not an option: that is a catalog a contract client cannot address, which is the same hole as
# nothing serving.
CONFIGURATIONS = ('qwen27b-q8-pair', 'qwen27b-q6k-pair',
    'flash-next-strata-solo-thinking-recovery', 'flash-next-strata-solo-thinking-recovery-cache',
    'flash-next-strata-solo-thinking-recovery-cache-yarn')
# Reduced from five Strata configurations to two on 2026-10-18, after both new levers had been
# measured on this box: the greedy step and the card-sampling step are settled policy rather than a
# live question, and the route-tail skip came back a wash on the only axis this machine can read.
# A variant is not free -- each one is a rendered launch, a chain step this suite refuses to let
# drift, and a row an operator reads as a decision to make -- so a lever nobody is testing is
# documentation, and documentation lives in docs/deployment.md with its numbers.
#
# Transitional spelling of the name above, for one release: a literal tuple rather than a binding,
# because the updater running on an already-deployed revision parses the *target* revision's source
# for it with ast.literal_eval. It went the way its own comment promised once a revision reading
# CONFIGURATIONS was deployed (179fce5, 2026-10-17), and target_profiles() takes CONFIGURATIONS
# directly, adding the aliases itself -- so a rename no longer needs scaffolding repeated by hand.
# Started when no release has been recorded yet (a new or rebuilt host without an initial_profile).
DEFAULT_PROFILE = 'qwen27b-q8-pair'
# Configurations that used to be selectable, and what replaced them. A retired name is never
# selectable and never silently resolves -- it refuses and says what to use instead -- because its
# backends are described nowhere any more and launching one from memory is the drift this house
# refuses. A *renamed* one does resolve, for the opposite reason: the same backends are still there.
RETIRED_PROFILES = {
    'flash-next-solo-128k': 'flash-next-strata-solo-thinking-recovery',
    'flash-next-solo-160k': 'flash-next-strata-solo-thinking-recovery',
    # Retired 2026-10-18, the day their levers were measured. The sampling step and the route step
    # are the two named here that were ever selectable; what replaces each is the configuration that
    # no longer carries the switch, which is the only answer a refusal can usefully give.
    # Its own earlier name, which resolved as a rename while the configuration it named existed.
    'flash-next-strata-160k': 'flash-next-strata-solo-thinking-recovery',
    'flash-next-strata-solo': 'flash-next-strata-solo-thinking-recovery',
    'flash-next-strata-solo-thinking': 'flash-next-strata-solo-thinking-recovery',
    'flash-next-strata-solo-thinking-recovery-full-route': 'flash-next-strata-solo-thinking-recovery',
}
# Renamed configurations: old id -> new id. Resolved by render() before a file is read, accepted by
# update.target_profiles() so a rename cannot refuse its own release, and honoured by the home
# wrappers and the status page. Because Compose services are keyed by role and the canonical model id
# is the model alias, a rename launches byte-identical backends; tests/fixtures/renamed-profiles.json
# is what proves it, and it is the reason a rename is a naming change and not a redeploy.
PROFILE_ALIASES = {
    'qwen27b-q8-with-nighttime': 'qwen27b-q8-pair',
    'qwen27b-q6k-with-nighttime': 'qwen27b-q6k-pair',
}
# The second resident: the model a paired configuration loads on the other GPU pair.
EVERYDAY_PROFILE = 'everyday'
# Everyday-resident variants a paired configuration may name with "everyday"; EVERYDAY_PROFILE is the
# default.
EVERYDAY_PROFILES = ('everyday',)
PAIR_GROUPS = ('daytime', 'nighttime')
# A solo configuration reserves both text pairs, in this order, and runs no second resident.
EXCLUSIVE_GROUP = 'all'
GPU_UUID = r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}'
# LLM Router's resident-catalog ceiling (RESIDENT_CONTEXT_LIMIT in the router's model-catalog.js;
# llm-router 7235329 and later carried 262144, the native window of the Qwen3.8 models). The step to
# 524,288 went out as a coordinated release on 2026-10-18 -- router first, then this controller --
# because catalog validation is atomic per read: a 524,288 publication at a router still on 262144
# 503s every model route until it is repointed. A window above 262,144 is rope scaling, and the
# profile must state the ring it ships on (engine_rope_scaling); the measurements are in
# docs/deployment.md. Older routers reject catalogs above 262144, and older still above 163840.
ROUTER_CONTEXT_LIMIT = 524288


def require_router_context(tokens):
    require(type(tokens) is int and 0 < tokens <= ROUTER_CONTEXT_LIMIT,
        f'Context must be within the current router contract: 1–{ROUTER_CONTEXT_LIMIT} tokens')


def read(path):
    return json.loads(Path(path).read_text())


def resolve_profile(profile):
    """Map a renamed configuration id to the id this revision uses.

    Only PROFILE_ALIASES resolves: a retired name is not a rename and must keep refusing, because a
    retired configuration's backends are no longer described anywhere and launching one from memory is
    exactly the drift this house refuses. Callers that report the failure to a person use
    unknown_profile() instead, which says what to select.
    """
    return PROFILE_ALIASES.get(profile, profile)


def unknown_profile(profile):
    """The refusal for a name this revision does not run, naming what it does."""
    message = 'Unknown or retired configuration ' + repr(profile) + '; current configurations: ' \
        + ', '.join(CONFIGURATIONS)
    replacement = RETIRED_PROFILES.get(profile)
    if replacement is not None:
        message += '. ' + str(profile) + ' was retired; select ' + replacement
    return message


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
    """A solo configuration owns all four text GPUs; no second resident runs while it is selected."""
    exclusive = definition.get('exclusive', False)
    require(exclusive is True or exclusive is False, definition['id'] + ': exclusive must be true or false')
    if exclusive:
        require(definition['role'] == 'coding' and definition['gpu_group'] == EXCLUSIVE_GROUP,
            definition['id'] + ': a solo configuration is a coding configuration on the "all" GPU group')
    else:
        require(definition['gpu_group'] in PAIR_GROUPS, definition['id'] + ': unknown GPU group')
    return exclusive


def parallel_slots(definition):
    """Backend slots: one by default; a solo configuration, which has every GPU, may run two."""
    slots = definition.get('parallel_slots', 1)
    require(type(slots) is int and slots in (1, 2), definition['id'] + ': parallel_slots must be 1 or 2')
    require(slots == 1 or is_exclusive(definition), definition['id'] + ': only a solo configuration may run two slots')
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


def everyday_for(definition):
    """The resident a paired configuration loads on the other GPU pair beside its own."""
    name = definition.get('everyday', EVERYDAY_PROFILE)
    require(name in EVERYDAY_PROFILES, definition['id'] + ': unknown everyday resident profile ' + str(name))
    require(not is_exclusive(definition) or 'everyday' not in definition,
        definition['id'] + ': a solo configuration runs no second resident')
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
    """CUDA order for a solo configuration: the four text GPUs, then the vision GPU.

    No second resident is running, so the vision GPU is not shared: it carries the projector and
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


def engine_vision_gpu(name, host, engine, options, environment, text_devices):
    """The shared vision GPU as an engine that configures itself from the environment sees it.

    Two facts have to arrive, and neither of them is an argument this engine takes: the card has to
    be in the container at all, and the image encoder has to be told to run on it alone. The first
    is the reservation plus the visible list; the second is the variable this house's entrypoint
    overlay writes into the engine's own run config, because the engine reads the encoder's card
    from that file and its setup has no flag for it (docs/shared-vision.md).

    The direction is the one the llama.cpp profiles already take: the profile states the cheaper
    default and host.json opts into the shared card. So the profile keeps `--vision no`, and a host
    that names a vision_gpu_id turns it on. Returns the GPUs to reserve and the encoder's device
    index, or the text GPUs and None where the host has no vision GPU or the engine has no switch
    for one, which is a card nothing in that container would ever use.
    """
    vision = host.get('vision_gpu_id')
    if vision is None:
        return list(text_devices), None
    require(re.fullmatch(GPU_UUID, vision) and vision not in text_devices,
        'Shared vision GPU must be distinct from all text GPUs')
    device_variable = engine['environment'].get('--vision-device')
    if device_variable is None:
        # An engine that cannot be told which card its encoder runs on keeps that card out of its
        # container, which is what it was built for: a visible GPU nothing in the container can name
        # is a card taken off the table for no reason. The mapping is the engine's declaration.
        return list(text_devices), None
    require(engine['environment'].get('--vision') is not None, name +
        ': this engine maps --vision-device but not --vision, so it can place an encoder it is '
        'never told to start')
    requested = str(options.get('--vision', 'no')).lower()
    require(requested in ('no', 'none', 'yes', 'gpu'), name + ': --vision states ' + requested +
        '; a profile for a host with a shared vision GPU states the text-only default and lets the '
        'renderer move the encoder onto that card, and a CPU encoder is not what the card is for')
    # The encoder's card is a number in the container's own CUDA_VISIBLE_DEVICES, and it is derived
    # here rather than read from the host. The container is handed the text cards and then the shared
    # card, so the shared card is numbered len(text_devices) in that order -- and one host now serves
    # a solo configuration whose text model takes four cards alongside a paired one whose text model
    # takes two, so the same card is CUDA4 in one container and CUDA2 in the other and a single host
    # integer cannot name both. What keeps the encoder off a text card is not the number: it is the
    # require above refusing the shared card as a text-pair member, and this function appending it
    # last. host.json's vision_cuda_device is a statement about the solo case, which is the number a
    # person reads when they look at the host; nothing here obeys it, so it cannot be wrong in a way
    # that moves the encoder.
    index = len(text_devices)
    stated = host.get('vision_cuda_device')
    require(stated is None or type(stated) is int, name +
        ': vision_cuda_device, if a host states it at all, is an int. It documents the index the '
        'solo configuration numbers the shared vision GPU at; the encoder is placed by the value '
        'derived above, so a host stating a string is a typo, not a placement decision.')
    environment[engine['environment']['--vision']] = 'yes'
    environment[device_variable] = str(index)
    return [*text_devices, vision], index


SAMPLING_PRESETS = ('off', 'thinking', 'instruct')


def engine_sampling_preset(name, engine, options):
    """What a request that sends no sampling of its own gets, checked against what the image can say.

    The engine's own default is greedy - serve/server.py says so in its own words - and its setup
    never writes the run config's optional sampling block, so the only way a routed client that
    omits sampling gets anything else is a preset the profile states and this house's entry point
    writes into that config on every start. A preset is a name and not a set of numbers, because the
    engine refuses to start on a sampling value it does not understand, and for an exclusive resident
    a config it refuses is nothing serving; the numbers behind the names live in the entry point,
    next to the code that writes them. An engine whose image runs upstream's entry point has no map
    entry for this argument at all, and such a profile is refused here rather than rendered into a
    variable nothing reads.
    """
    if engine['environment'].get('--sampling') is None:
        return
    preset = str(options.get('--sampling', 'off'))
    require(preset in SAMPLING_PRESETS, name + ': --sampling states ' + preset + '; the presets are '
        + ', '.join(SAMPLING_PRESETS) + '. The numbers behind a preset belong to the image and not '
        'to a profile, because the engine will not start on a sampling value it does not understand')


TOOL_CALL_RECOVERY_VALUES = ('off', 'on')


def engine_tool_call_recovery(name, engine, options):
    """Whether a declared tool's call written beside the template's form is still a call.

    Same shape as the sampling preset above and for the same two reasons. The engine's own default is
    off - serve/server.py reads the run config's top-level key and defaults it to false, and setup.py
    never writes it - so the only way this house's agent gets the other answer is a switch the profile
    states and this house's entry point writes on every start. And the value is one of two words
    rather than anything a profile can think of, because serve/server.py raises SystemExit unless that
    key is a JSON boolean: for an exclusive resident, a value the engine will not read is nothing
    serving. An engine whose image runs upstream's entry point maps this argument to no variable at
    all, and a profile that states it there is refused when it renders.
    """
    if engine['environment'].get('--tool-call-recovery') is None:
        return
    stated = str(options.get('--tool-call-recovery', 'off'))
    require(stated in TOOL_CALL_RECOVERY_VALUES, name + ': --tool-call-recovery states ' + stated
        + '; the values are ' + ', '.join(TOOL_CALL_RECOVERY_VALUES) + '. The engine reads a JSON true '
        'or false for this key and refuses to start on anything else, which is why a profile states '
        'one of two words and never a value of its own')


CACHE_BUDGET_OFF = 'off'


def engine_conversation_cache(name, engine, options):
    """Whether a conversation that comes back is read again from the start, and what it may spend.

    Two arguments carry the decision and two refine it. The budget is `off` (the engine's own default,
    which re-prefills every returning conversation) or an amount of MiB of host RAM to park
    conversations in; the floor is the RAM the cache must never park beneath, and it is required
    whenever the budget is a number, because this workspace keeps a 32.93 GiB page-locked expert
    complement on the same RAM and a cache that outbids it pays in page faults on the serving path.
    Slots and minimum tokens refine a cache that is on and are refused on one that is off, where they
    are a statement about nothing; unstated, the entry point writes the engine's own defaults, so an
    older profile's number cannot survive a switch in silence. Checked here rather than at boot for
    the standing reason: an exclusive resident whose engine refuses to start is nothing serving, and
    a drain already spent cannot be unspent.
    """
    if engine['environment'].get('--conversation-cache-mib') is None:
        return
    budget = str(options.get('--conversation-cache-mib', CACHE_BUDGET_OFF))
    require(budget == CACHE_BUDGET_OFF or (budget.isdigit() and int(budget) > 0),
        name + ': --conversation-cache-mib states ' + budget + '; it is ' + CACHE_BUDGET_OFF +
        ' or an amount of MiB in whole numbers. Zero is not a small cache: the engine reads 0 as off, '
        'and a profile that means off says ' + CACHE_BUDGET_OFF)
    floor = str(options.get('--conversation-cache-min-free-mib', ''))
    if budget == CACHE_BUDGET_OFF:
        for key in ('--conversation-cache-slots', '--conversation-cache-min-tokens'):
            require(key not in options, name + ': ' + key + ' refines a conversation cache this '
                'configuration has switched off; state a budget before a refinement')
        return
    require(floor.isdigit() and int(floor) > 0, name +
        ': --conversation-cache-mib ' + budget + ' states a budget and no --conversation-cache-min-free-mib '
        'to park nothing beneath. The engine defaults it, and this house does not take that default '
        'beside a 32.93 GiB page-locked expert complement')
    for key in ('--conversation-cache-slots',):
        stated = str(options.get(key, ''))
        require(stated == '' or (stated.isdigit() and int(stated) > 0),
            name + ': ' + key + ' states ' + stated + '; it is a whole number of conversations')
    stated = str(options.get('--conversation-cache-min-tokens', ''))
    require(stated == '' or stated.isdigit(), name +
        ': --conversation-cache-min-tokens states ' + stated + '; it is a whole number of tokens, '
        'and 0 is the engine\'s own answer (park any conversation)')


def engine_ple_table(name, engine, options, artifacts):
    """Which file the n-gram table is read from, or nothing at all.

    The engine reads this table off the SSD on the CPU path at some tens of thousands of rows a second
    and never streams it to a card, so a wider table is the one accuracy purchase on a box at
    `pcie_frac 0.22` that does not go through the expert streaming that cost UD-Q4_K_XL 7-15% of decode
    and this house's STRATA_HC_Q8 trial 15% of prefill. `off` leaves the key unwritten, which at three
    shards means setup.py reads the table out of the shard Unsloth put it in, at that file's own
    quantization. A path is checked against the mounts this profile declares, for the same reason the
    model path is: a table named that nothing mounts is a boot that either does not happen or happens
    with layer one's PLE silently skipped, and a silently skipped layer is the failure this house's
    entry point exists to refuse.
    """
    if engine['environment'].get('--ple-gguf') is None:
        return
    stated = str(options.get('--ple-gguf', 'off'))
    if stated == 'off':
        return
    targets = [artifact['target'] for artifact in artifacts]
    require(stated in targets, name + ': --ple-gguf states ' + stated + ', which is not one of the '
        'mounts this configuration declares (' + ', '.join(targets) + '). The table is a file like the '
        'model is a file: it arrives as a declared artifact, so that a profile naming a table nothing '
        'downloaded fails when it renders and not when the engine starts')


def positive_decimal(value):
    # Digits with at most one point and more than zero -- the shape a rope scale takes. Every other
    # argument validator in this file is isdigit()/positive-int; a rope scale is the first float the
    # house states, so it gets its own check rather than an isdigit that would refuse '2.0'.
    if re.fullmatch(r'[0-9]*\.?[0-9]+', value) is None:
        return False
    try:
        return float(value) > 0
    except ValueError:
        return False


ROPE_SCALINGS = ('yarn', 'linear', 'none', 'off')


def engine_rope_scaling(name, engine, options, context_tokens):
    """Which position ring a session is read through, and why a window past 262,144 has to say so.

    The Qwen3.8 pack is trained to 262,144 and carries no rope scaling of its own; a wider window is
    YaRN interpolation and nothing else. A profile states --rope-scaling (yarn|linear) with a positive
    --rope-scale, and yarn names the trained window it stretches with --yarn-orig-ctx; the entry point
    writes the engine arguments (facts 5/6 write-all-or-remove-all shape, so a wider profile's ring
    cannot survive a switch back to its 256K sibling in silence), and the engine fixes the ring at
    startup because cached keys are post-RoPE. Checked here rather than at boot for the standing
    reason: yarn without a scale, or a scale that is not a positive number, or a yarn window that does
    not exceed the trained one, is an engine that refuses to start -- and for an exclusive resident
    that is nothing serving on a drain already spent. A float validator lives here now too, because
    --rope-scale is the first decimal the house states and every other argument is an integer.
    """
    if engine['environment'].get('--rope-scaling') is None:
        return
    stated = str(options.get('--rope-scaling', 'none'))
    require(stated in ROPE_SCALINGS, name + ': --rope-scaling states ' + stated + '; the values are '
        + ', '.join(ROPE_SCALINGS) + '. A geometry the engine has never heard of is a crash loop, so a '
        'profile states one of these words and never a value of its own')
    if stated in ('none', 'off'):
        for key in ('--rope-scale', '--yarn-orig-ctx'):
            require(key not in options, name + ': ' + key + ' refines a rope scaling this '
                'configuration has switched to ' + stated + '; state a ring before its numbers')
        return
    scale = str(options.get('--rope-scale', ''))
    require(scale != '' and positive_decimal(scale), name + ': --rope-scaling ' + stated +
        ' states no positive --rope-scale to go with it. The scale is the statement; the scaling name '
        'only says which interpolation reads it, and leaving it to the engine default is how a window '
        'gets a ring nobody named')
    if stated == 'yarn':
        orig = str(options.get('--yarn-orig-ctx', ''))
        require(orig != '' and orig.isdigit() and int(orig) > 0, name +
            ': --rope-scaling yarn states no positive --yarn-orig-ctx. YaRN needs to know which '
            'trained window it is stretching, and the engine argument\'s own default is a number to '
            'be stated, not assumed')
        require(int(context_tokens) > int(orig), name + ': --rope-scaling yarn on a ' +
            str(context_tokens) + '-token window with --yarn-orig-ctx ' + orig + ' does not stretch '
            'anywhere: the engine refuses a YaRN window that does not exceed the trained one it scales')


def engine_interface(engine):
    """How a backend takes its configuration: the llama.cpp argument vector, or a declared

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
                ' has no environment variable in engine ' + definition['engine'] +
                ('; only an image that carries this house\'s entry point can be told this, because it '
                 'is written into the engine\'s own run config and not passed to the engine, so a '
                 'profile that states it does not roll back to an image without that entry point '
                 'until this line is dropped'
                 if key in ('--sampling', '--tool-call-recovery', '--conversation-cache-mib',
                            '--conversation-cache-min-free-mib', '--conversation-cache-slots',
                            '--conversation-cache-min-tokens', '--ple-gguf',
                            '--rope-scaling', '--rope-scale', '--yarn-orig-ctx') else ''))
            continue
        require(variable not in environment, name + ': two arguments map to ' + variable)
        environment[variable] = '' if value is True else str(value)
    engine_sampling_preset(name, engine, options)
    engine_tool_call_recovery(name, engine, options)
    engine_conversation_cache(name, engine, options)
    engine_ple_table(name, engine, options, definition['artifacts'])
    engine_rope_scaling(name, engine, options, ctx)
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
    reserved, vision_index = engine_vision_gpu(name, host, engine, options, environment, text_devices)
    if vision_index is not None:
        # The engine numbers the cards itself, so the profile states which of them the language
        # model runs on and the renderer only grows the visible list by the card the host lends.
        # Indices rather than UUIDs, because that is what this engine is addressed in: the encoder
        # is placed by the same numbering, so `--gpus 0,1,2,3` keeps meaning the text cards.
        stated = [item.strip() for item in str(environment.get('CUDA_VISIBLE_DEVICES', '')).split(',')]
        require(stated == [str(index) for index in range(len(text_devices))], name +
            ': the container-visible GPU list must state the text cards as ' +
            ','.join(str(index) for index in range(len(text_devices))) +
            '; the shared vision GPU is appended by the renderer, being a fact of the host')
        environment['CUDA_VISIBLE_DEVICES'] = ','.join(str(index) for index in range(len(reserved)))
    cfg.update(image=engine['tag'], container_name=definition['container_name'], command=[],
        environment=environment, volumes=mounts,
        ports=[f"127.0.0.1:{definition['host_port']}:8080"],
        deploy={'resources': {'reservations': {'devices': [{
            'driver': 'nvidia', 'device_ids': reserved, 'capabilities': ['gpu']}]}}})
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
    declares_vision = list(entry['input_modalities']) == ['text', 'image']
    require(declares_vision == (entry['capability_profile'].get('vision') is True),
        name + ': input_modalities and capability_profile.vision must agree')
    entry.update(model=options['--alias'], context_length=ctx,
        total_context_length=ctx * slots, max_active_requests=slots, gpu_uuids=text_devices,
        model_path=targets[model_target], backend_revision=engine['revision'],
        display_name=display_name(definition), gpu_names=gpu_names(host, definition),
        source='local-ai-runtime', updated_at=None)
    # What the profile declares is what this configuration is for; what the host has is what it
    # delivers, and the router is published the second. A profile that promises pictures on a host
    # with no vision GPU serves text, exactly as a projector stays on the CPU there.
    if vision_index is not None:
        entry.update(gpu_uuids=reserved, text_gpu_uuids=text_devices,
            vision_gpu_uuid=host['vision_gpu_id'], vision_device=f'CUDA{vision_index}',
            vision_gpu_shared=not exclusive,
            # What this container was actually told, in the notation it was told it in. This engine is
            # addressed by index, so its visible list is indices and the encoder's number is a position
            # in it; the argument-vector path exports UUIDs and publishes those. Publishing UUIDs here
            # would state an order the container does not use.
            cuda_visible_devices=environment['CUDA_VISIBLE_DEVICES'],
            vision_gpu_name=host.get('vision_gpu_name'), mmproj_offload='gpu',
            input_modalities=['text', 'image'])
        entry['capability_profile']['vision'] = True
    else:
        entry.update(mmproj_offload='none', input_modalities=['text'])
        entry['capability_profile']['vision'] = False
    if exclusive:
        entry['exclusive'] = True
    require_open_generation_policy(name, entry)
    return cfg, entry, mounts, targets, reserved, text_devices


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
    # Resolved for a solo configuration too, and then dropped: naming a second resident you will
    # never load is a configuration lying about what it launches, and everyday_for refuses it.
    second = everyday_for(definition) if definition['role'] == 'coding' else None
    return {'profile': definition['id'], 'role': definition['role'], 'display_name': display_name(definition),
        'model': options['--alias'], 'context_tokens': tokens, 'engine': definition['engine'],
        'engine_tag': engine['tag'], 'backend_revision': engine['revision'],
        'gpu_group': definition['gpu_group'], 'gpu_names': gpu_names(host, definition),
        # The service IDs a client sends to reach this resident. Listed for every configuration,
        # including the ones not running, because "what would answer" is a different question from
        # "what is loaded" and the status page has to be able to tell them apart.
        'service_ids': sorted(definition['catalog'].get('aliases', [])),
        'exclusive': is_exclusive(definition), 'parallel_slots': definition.get('parallel_slots', 1),
        **({'everyday': second} if second is not None and not is_exclusive(definition) else {}),
        **({'variant_label': definition['variant_label']} if definition.get('variant_label') else {})}


def available_profiles(config_dir, host=None):
    """Every configuration this revision can apply, plus the second resident a paired one loads."""
    config_dir = Path(config_dir)
    shared = read(config_dir / 'shared.json')
    require(shared['schema_version'] == 2, 'Unsupported configuration version')
    def describe(name):
        return profile_summary(config_dir, name, shared, host)
    residents = {name: describe(name) for name in (*EVERYDAY_PROFILES, *CONFIGURATIONS)}
    selectable = [residents[name] for name in CONFIGURATIONS]
    for entry in selectable:
        # What a client can ask for while this configuration is selected. For a paired configuration
        # that is the two residents together: `nighttime` is answered by the second one, and a page
        # that read only the first resident's aliases would report a service as missing while it was
        # answering.
        second = residents.get(entry.get('everyday'))
        if second:
            entry['service_ids'] = sorted(set(entry['service_ids']) | set(second['service_ids']))
    used = {entry['everyday'] for entry in selectable if 'everyday' in entry}
    return {'selectable': selectable,
        'always_included': [residents[name] for name in EVERYDAY_PROFILES
            if name == EVERYDAY_PROFILE or name in used]}


def offline_services(config_dir, shared, host, loaded):
    """Service IDs this configuration deliberately does not run, so the router can report them as
    temporarily offline rather than unknown.

    The rule is about service IDs, not about roles: a service ID is offline here when the resident
    that normally carries it publishes it somewhere in this house and nothing loaded by *this*
    configuration publishes it. A solo configuration that answers `nighttime` from its own backend
    therefore declares nothing offline, and one that stopped offering it would. Roles and service IDs
    are separate facts -- `offline_roles` says what is loaded, this says what answers -- and
    conflating them is how a status page comes to tell you a model is off while it is answering.
    """
    published = {alias for definition in loaded for alias in definition['catalog'].get('aliases', [])}
    offline = []
    for name in EVERYDAY_PROFILES:
        definition = read(Path(config_dir) / 'profiles' / (name + '.json'))
        aliases = [alias for alias in definition['catalog'].get('aliases', []) if alias not in published]
        if not aliases:
            continue
        summary = profile_summary(config_dir, name, shared, host)
        offline.append({'model': summary['model'], 'aliases': aliases,
            'display_name': summary['display_name'], 'role': definition['role'],
            'reason': 'exclusive_configuration'})
    return offline


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
    profile = resolve_profile(profile)
    require(profile in CONFIGURATIONS, unknown_profile(profile))
    shared = read(config_dir / 'shared.json')
    require(shared['schema_version'] == 2, 'Unsupported configuration version')
    root = Path(host['model_root'])
    require(root.is_absolute() and root != Path('/'), 'Model root must be an absolute non-root directory')
    compose = {'name': shared['backend_project'], 'services': {}, 'networks': {
        'router': {'external': True, 'name': shared['network']}}}
    models, artifacts, manifests, gpu_ids = [], {}, [], []
    selected = read(config_dir / 'profiles' / (profile + '.json'))
    exclusive_profile = is_exclusive(selected)
    # Also refuses a second resident named by a solo configuration, which has no other GPU pair.
    # Resolved before the exclusivity check for the same reason profile_summary resolves it: a solo
    # configuration that names a second resident is refused, not quietly ignored.
    second = everyday_for(selected)
    if exclusive_profile:
        second = None
    loaded = [selected] if second is None else [selected,
        read(config_dir / 'profiles' / (second + '.json'))]
    for definition in loaded:
        name = definition['id']
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
            cfg, entry, mounts, targets, devices, text = environment_service(name, definition, engine, shared,
                host, root, ctx, slots, exclusive_profile)
            for artifact in definition['artifacts']:
                source = str(root / Path(artifact['path']))
                artifacts[source] = {**artifact, 'source': source}
            gpu_ids.extend(text)
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
        'offline_services': offline_services(config_dir, shared, host, loaded)}
    bundle = {'schema_version': 1, 'profile': profile, 'compose': compose, 'catalog': catalog,
        'manifest': {'schema_version': 2, 'services': manifests},
        'artifacts': list(artifacts.values())}
    if second not in (None, EVERYDAY_PROFILE):
        bundle['everyday_profile'] = second
    bundle['config_sha256'] = digest(bundle)
    return bundle

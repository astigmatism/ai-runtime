"""Render launch configuration and discovery from the same profile definitions.

Three words carry the whole model, and none of them is a time of day:

* a **configuration** is one selectable decision: the models this machine loads and the cards they
  load on. It names them in its own `models` list, and how many there are is a fact about that file;
* a **model** is one thing loaded on one set of cards, identified by the id a client routes on
  (`qwen3.8-27b-q8_0-tensor-next`), which is also its Compose service key. One model on two cards, one
  model across all four, and two models with the cards divided between them are three configurations,
  not three kinds of anything;
* a **service ID** is the name a client sends. It belongs to the LLM Router contract, not to this
  vocabulary: `daytime` and `nighttime` are service IDs, and a configuration may point any of them at
  whichever models it actually runs. `nighttime` answering from the same backend as `daytime` is a
  placement fact, not a contradiction.

GPU group keys in `.state/host.json` (`gpu_ids`, `gpu_names`, `cuda_order`), the shared vision GPU
(`vision_gpu_id`, `vision_gpu_name`, `vision_cuda_device`) and the container names are operational
identifiers, deliberately still the historical ones; see docs/deployment.md and docs/shared-vision.md.

A second machine is a setting, not a fork. `.state/host.json` names it (`host_id`) and states what its
cards can execute (`gpu_architectures`); every profile states which hosts it was built for (`hosts`),
and a configuration of another machine is refused rather than offered. What the house calls a
configuration is therefore only half the story of what runs: the other half is which machine is
asking, which is why `configurations_for()` -- and never `CONFIGURATIONS` -- is what anything that
offers a choice reads. See docs/deployment.md, "More than one machine".
"""
import copy
import hashlib
import json
import os
import re
from pathlib import Path

# Selectable configurations. Each one names the models it loads in its own `models` list: a
# configuration with two entries runs two models on two GPU pairs, one with `exclusive: true` holds
# every text GPU and loads nothing beside it. Display names come from each profile file.
#
# This is the whole registry of the revision, which is not the same thing as what any one machine can
# select: read configurations_for(host) for that, and read this only when the question is "does this
# revision know this id at all" (a retired name, an alias, the updater reading a target revision).
# Every one of these publishes a service alias a contract client can address, and every one publishes
# `daytime`: a client that only ever sends `daytime` -- which is what LLM Router's CLIENT_CONTRACT.md
# tells it to do -- always has something to send, whichever configuration is selected. Asserted for
# every configuration by ServiceIdentityTests in tests/test_config.py. Publishing no service alias at
# all is not an option: that is a catalog a contract client cannot address, which is the same hole as
# nothing serving.
CONFIGURATIONS = ('qwen27b-q8-pair', 'qwen27b-q6k-pair',
    'flash-next-strata-solo-thinking-recovery', 'flash-next-strata-solo-thinking-recovery-cache',
    'flash-next-strata-solo-thinking-recovery-cache-yarn', 'flash-next-strata-two-card-iq4-xs',
    'flash-next-strata-two-card-iq3-s', 'flash-next-strata-iq3-xxs-daisy',
    # The three trial configurations measured on daisy on 2026-10-25 (~/reports/20261025-free-levers):
    # the tensor-core prompt path, that path widened to the 128K window, and the IQ3_S tier. Each is
    # held to the same service-identity contract as the rest of the list, and each says on its own
    # host line which machine may select it.
    'flash-next-strata-iq3-xxs-daisy-tc', 'flash-next-strata-iq3-xxs-daisy-128k',
    'flash-next-strata-iq3-s-daisy')
# The last entry is the first configuration of a second machine, and it is in this list rather than
# in a list of its own because the updater reads this list out of a *target* revision that has no
# host to ask. What separates it is its own "hosts" line: a machine that does not name itself is
# still shown the whole registry, and a machine that does is shown only its share. See
# docs/deployment.md, "More than one machine".
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
# wrappers and the status page. Because Compose services are keyed by the model id, and the canonical
# model id is the model alias, a rename launches byte-identical backends; tests/fixtures/renamed-profiles.json
# is what proves it, and it is the reason a rename is a naming change and not a redeploy.
PROFILE_ALIASES = {
    'qwen27b-q8-with-nighttime': 'qwen27b-q8-pair',
    'qwen27b-q6k-with-nighttime': 'qwen27b-q6k-pair',
}
PAIR_GROUPS = ('daytime', 'nighttime')
# The GPU group an exclusive model reserves: every text GPU this host declares, in this order, with
# nothing else loaded to share them.
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


# A deployment is one machine, and which machine it is used to be an assumption. Three names carry
# it now: `host_id` in `.state/host.json` names the machine, `gpu_architectures` names what its
# cards can actually execute, and `hosts` on a profile names the machines that profile was built
# for. This is the same rule the house applies to every other fact -- what is not stated is a fact
# that drifts -- aimed at the specific failure of rendering a measured launch onto hardware it was
# never measured on, which spends a drain to arrive at a model that cannot start.
HOST_ID = re.compile(r'[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?')
# One budget for every name a person reads in a hurry -- a machine's label, an operator's rename --
# because they are read in the same picker, and the part of a name that distinguishes two rows is the
# end of it, which is what a long machine label pushes off the line.
NAME_MAX = 48


def host_id(host):
    """The machine these host settings belong to, or None when they do not say.

    Absent is deliberately not an error. The private settings of a host that is already serving are
    edited by an operator and not by a release, so a deployment whose `host.json` names no host
    keeps the registry it has always seen instead of waking up with nothing selectable. Naming
    itself is what turns the filter on, on both sides.
    """
    value = (host or {}).get('host_id')
    if value is None:
        return None
    require(isinstance(value, str) and HOST_ID.fullmatch(value),
        "host_id, if a host states it at all, is a lowercase slug of letters, digits and '-', not a "
        "hostname, a path or a GPU UUID")
    return value


def host_display_name(host):
    """The operator's own word for this machine, or None when they have not chosen one.

    It is the label a shared router shows for the machine and the prefix every model of the machine is
    published with, so it is the one place both read it. It is handed on **trimmed**: a `host` block
    whose display name is blank or padded is a refused publication, and a refusal there costs the
    machine its whole catalog, not just the label an operator fat-fingered into a private setting. A
    label of no usable shape -- blank, not one printable line, longer than any other name here -- is
    not used at all, which leaves the machine called by the slug it also states.
    """
    value = (host or {}).get('display_name')
    if not isinstance(value, str) or not value.strip() or not value.strip().isprintable():
        return None
    value = value.strip()
    return value if len(value) <= NAME_MAX else None


def host_name(host):
    """What this machine is called when something of it needs a name said out loud.

    The operator's label when they gave one and the `host_id` slug otherwise, because both are facts
    the machine states about itself and a slug is still the truth a client can act on. None only when
    the machine names itself neither way: the house rule for an unstated fact is that it is not
    invented, and a model published by a machine that has not said which one it is gets the name it
    always had.

    Read tolerantly, like the rest of what a person edits in `host.json` -- a label that is not a
    printable line falls back to the slug rather than becoming a page that will not load.
    """
    return host_display_name(host) or host_id(host)


ENV_NAME = re.compile(r'[A-Z][A-Z0-9_]*')
URL = re.compile(r'https?://[^\s/]+(?:/\S*)?$')


def router_target(host):
    """Where this machine sends its catalog when the router is not on its own filesystem, or None.

    The router's contract asks a publisher for exactly three facts: the router's address, the machine
    the publisher is speaking for, and the name of the variable holding its credential. Two of the
    three are already in these settings — `host_id` says which machine this is, and a machine that
    published a catalog under somebody else's name would be offering its own backends under another
    machine's addresses — so the definition holds the other two and the host ID is not stated twice.
    Stating the same fact in two files is how one of them ends up wrong.

    The credential is named and never stored: the value arrives in the container environment the way
    every other secret here does, and a repository that held a bearer token would be a repository
    that leaks it. Absent the whole definition means the historical handoff — write the marker into
    `router_state_dir` and tell the router to reload — which is what a machine whose router is the
    machine it runs on still does, and what Rosalina keeps doing beside a push during the migration.

    There are two credentials, because the router has two listeners' worth of trust. `token_env` is the
    publisher: by the router's own contract it is a write credential for one host's catalog, and it
    cannot change drain state — deliberately, so a runtime that is compromised cannot drive the router.
    But a switch behind a remote router *has* to drain it, so the machine needs the control credential
    as well, and `admin_token_env` names it. A deployment with one secret sets the two to the same
    variable and says so in one line; a deployment that cares says which machine can publish and which
    can close the door.
    """
    value = (host or {}).get('router_target')
    if value is None:
        return None
    require(isinstance(value, dict),
        "router_target is an object stating 'url' and 'token_env'; anything else is a typo in a "
        "private setting, not a publication target")
    unknown = sorted(set(value) - {'url', 'token_env', 'admin_token_env'})
    require(not unknown, 'router_target states ' + ', '.join(unknown) +
        '; the router is given a URL, two credential names, and this machine\'s host_id, and nothing else')
    url = value.get('url')
    require(isinstance(url, str) and URL.fullmatch(url),
        "router_target.url is the router's admin base, e.g. http://192.168.1.23:11435")
    token_env = value.get('token_env')
    require(isinstance(token_env, str) and ENV_NAME.fullmatch(token_env),
        "router_target.token_env names the environment variable holding the publisher token, e.g. "
        "CATALOG_PUSH_TOKEN — the name, never the value")
    require(host_id(host),
        'router_target publishes as this machine, so host.json has to state host_id: the router '
        'refuses a push from an id it has not declared, and a machine that guesses its own name '
        'gets a catalog nobody can address')
    admin_token_env = value.get('admin_token_env', 'ADMIN_TOKEN')
    require(isinstance(admin_token_env, str) and ENV_NAME.fullmatch(admin_token_env),
        "router_target.admin_token_env names the variable holding the router's control token, e.g. "
        "ADMIN_TOKEN — the name, and again not the value")
    return {'url': url.rstrip('/'), 'token_env': token_env, 'admin_token_env': admin_token_env,
        'host': host_id(host), 'token': os.environ.get(token_env, ''),
        'admin_token': os.environ.get(admin_token_env, '')}


def backend_bind_address(host):
    """The address a backend's port is published on: loopback until the router lives elsewhere.

    The port has always been published, because the router on this machine's own Docker network was
    never the only way in — but on `127.0.0.1`, which is the honest answer while the router is a
    neighbour container and a lie the moment it is another machine. Naming an address here moves the
    binding and the advertised URL together, which is the pair that has to agree.
    """
    value = (host or {}).get('backend_address')
    if value is None:
        return '127.0.0.1'
    require(isinstance(value, str) and value and not value.startswith('http'),
        "backend_address is a bare address this machine answers on, e.g. 192.168.1.4 — not a URL and "
        "not a hostname someone's Docker network happens to resolve")
    return value


def advertised_backend_url(host, definition):
    """The URL a *reader* of the catalog must use, which is not how this machine reaches itself.

    `http://qwen38-daytime:8080` is correct on the machine that resolves it and meaningless one hop
    away; the router's own documentation names that pair of facts as the classic multi-host mistake.
    A host that states no backend address is a host whose router is a neighbour, and the container
    name stays right.
    """
    if (host or {}).get('backend_address') is None:
        return definition['catalog']['backend_url']
    return 'http://' + backend_bind_address(host) + ':' + str(definition['host_port'])


def gpu_architectures(host):
    """The CUDA architectures this host's cards execute, or None when the host does not say.

    This is the reading that keeps an image built for one generation off another machine: the images
    in service carry sm_86/sm_89 SASS and a Turing card cannot run a line of it. A host that states
    its architectures gets that checked while a configuration renders, before the drain rather than
    at the first kernel launch.
    """
    value = (host or {}).get('gpu_architectures')
    if value is None:
        return None
    require(isinstance(value, list) and value and all(isinstance(v, str) and v.isdigit() for v in value),
        'gpu_architectures, if a host states it at all, is a non-empty list of CUDA architecture '
        'numbers as strings, in the form an engine entry states them')
    return tuple(value)


def require_engine_architecture(name, host, engine):
    """Refuse an image that has no SASS for these cards.

    An engine built for another architecture is not a slow start: the container loads, answers the
    health check its own entry point writes, and dies at its first kernel launch -- for an exclusive
    model is nothing serving, after the drain. Checked against what the engine entry states
    it was built for, which is the same declaration a person reads to decide what to build.
    """
    built = tuple(str(arch) for arch in engine.get('cuda_architectures', []))
    stated = gpu_architectures(host)
    if stated is None or not built:
        return
    require(set(built) & set(stated), name + ': engine ' + engine['tag'] + ' carries SASS for CUDA '
        + '/'.join(built) + ', and this host states cards of CUDA ' + '/'.join(stated) +
        '. The image would start and refuse to compute, so the configuration is refused here rather '
        'than after the drain: build the engine for this architecture, or select a configuration of '
        'this host')


def profile_hosts(definition):
    """The machines a definition says it was built for.

    No default of "every machine": the pair of cards, the memory sizing and the artifact set *are*
    the machine, and a configuration that names no host is a configuration nobody has said where it
    runs. Naming hosts is cheap -- it is one line beside the engine it pins.
    """
    hosts = definition.get('hosts')
    require(isinstance(hosts, list) and hosts and all(
        isinstance(value, str) and HOST_ID.fullmatch(value) for value in hosts),
        str(definition.get('id')) + ': hosts must state at least one host this definition is built '
        'for, as a list of host_id slugs')
    return tuple(hosts)


def belongs_to_host(definition, host):
    """Whether this machine's settings entitle it to select this definition."""
    machine = host_id(host)
    return machine is None or machine in profile_hosts(definition)


def configurations_for(config_dir, host=None):
    """The selectable configuration ids for this host, in registry order.

    The registry stays one list, because `runtime/update.py` reads the `CONFIGURATIONS` tuple out of
    the *target* revision's source with a parser that has no host to ask. This is the one place the
    host's share of that list is derived, and every surface that offers a choice reads it here.
    """
    machine = host_id(host)
    if machine is None:
        return tuple(CONFIGURATIONS)
    config_dir = Path(config_dir)
    return tuple(name for name in CONFIGURATIONS
        if machine in profile_hosts(read(config_dir / 'profiles' / (name + '.json'))))


def default_profile(config_dir, host=None):
    """What a host with no recorded release starts on.

    `DEFAULT_PROFILE` is the house default, and it is only a default where its hardware exists. On a
    host that does not run the paired 27B, the first configuration of its own registry order is the
    default, so a host is never started on a configuration that would refuse its selection.
    """
    selectable = configurations_for(config_dir, host)
    require(selectable, 'No configuration of this revision names this host; every one of '
        + ', '.join(CONFIGURATIONS) + ' belongs to another machine')
    return DEFAULT_PROFILE if DEFAULT_PROFILE in selectable else selectable[0]


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


def model_id(definition):
    """The id a client routes on, which is the only thing a name can sensibly be attached to.

    A configuration is not an identity: two configurations can load the same model at two windows, and
    a machine can load several models at once. What the router addresses, what a request names, and
    what an operator means when they say "that model" is this string, so it is the key for a name too.
    """
    return definition['arguments']['--alias']


def default_name(definition, host=None):
    """What a model is called when nobody has renamed it: the machine, the model, and its session length.

    This is the whole naming rule, and it is one rule on purpose. The pair of parentheses is what a
    person reads first when two configurations are the same model at two windows, which is a question
    they ask from the page, the router's model list, and a terminal at the same rate. The machine in
    front of it is what a household with more than one box needs next: two machines can launch the same
    weights, and a model list that cannot say which row answers from where is a list a person has to
    remember. A machine that has not said which one it is says nothing, which is the name it had
    before there was more than one machine.
    """
    name = f"{definition['display_name']} ({context_label(definition['context_tokens'])})"
    machine = host_name(host)
    return f"{machine}: {name}" if machine else name


def display_name(definition, aliases=None, host=None):
    """The published display identity; the launch catalog, the CLI listing and the status page share it.

    An operator alias wins over the default, and nothing else about the launch changes when one is
    set. It is a name and not an identity: the model id, the service IDs a client sends, the Compose
    project and the container names are untouched by it, so renaming never drains, restarts or
    re-renders anything. A rename replaces the name the whole way, machine included: it is the operator
    saying what to call this model, and not a prefix handed to the default.
    """
    return (aliases or {}).get(model_id(definition)) or default_name(definition, host)


# Operator names live with the rest of the machine's private state, not in a profile file: a profile
# says what a model launches, and a name is what this house chose to call it. It is the same line
# AGENTS.md draws for host.json, and it is why `aliases.json` is Git-excluded with the UUIDs. Version
# 2 keys the names by model id; the version that keyed them by configuration id is read as a file of
# names for models this machine does not serve, which is what it is.
ALIASES_FILE = 'aliases.json'
ALIASES_VERSION = 2
ALIAS_MAX = NAME_MAX


def nameable_models(config_dir, host=None):
    """{model id: [configurations of this host that load it]} -- what may be given a name.

    Models are nameable and configurations are not, because a configuration is a placement decision
    and a model is the thing a person, a client, and a router all refer to. A model two configurations
    share gets one name, said the same way whichever of them is loaded.
    """
    return {model: summary['configurations']
        for model, summary in registry_summaries(config_dir, host).items()}


def clean_alias(model, value, nameable=None):
    """One validated operator alias: a trimmed printable line, or None to say "use the default".

    A refusal rather than a silent truncation, because the thing being typed is the label an
    operator will later read in a hurry to decide what a machine is running.
    """
    if nameable is not None:
        require(model in nameable, 'Unknown model ' + repr(model) + '; nameable: '
            + ', '.join(sorted(nameable)))
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    require(isinstance(value, str), 'An alias must be text or null, not ' + type(value).__name__)
    alias = value.strip()
    require(len(alias) <= ALIAS_MAX, 'An alias is at most ' + str(ALIAS_MAX) + ' characters')
    require(alias.isprintable() and '\n' not in alias and '\t' not in alias,
        'An alias is one printable line, ' + repr(alias))
    return alias


def read_aliases(path, nameable=None):
    """This machine's aliases, tolerantly: the page and the health poll read it every few seconds.

    A missing file is the ordinary state and says nothing. A file that cannot be read or parsed, or
    that names a model this machine does not serve, is reported as a one-line note and the rest is
    used: a typo in an operator preference must not turn a serving machine into a page that does not
    load, and it must not be silently ignored either.
    """
    path = Path(path)
    if not path.exists():
        return {}, None
    try:
        stored = read(path)
    except Exception as error:
        return {}, 'Saved aliases could not be read and are not in use: ' + str(error)
    if not isinstance(stored, dict) or not isinstance(stored.get('aliases'), dict):
        return {}, 'Saved aliases are malformed and are not in use'
    keys = list(stored['aliases'])
    unknown = sorted(key for key in keys if nameable is not None and key not in nameable)
    aliases = {}
    for key, value in stored['aliases'].items():
        if (nameable is not None and key not in nameable) or not isinstance(value, str) or not value.strip():
            continue
        aliases[key] = value.strip()[:ALIAS_MAX]
    note = None
    if unknown:
        note = 'Ignored aliases for models this machine does not serve: ' + ', '.join(unknown)
    return aliases, note


def catalog_names(config_dir, host=None, aliases=None, profile=None):
    """{model id: display name} as this machine publishes its catalog right now.

    Read back at publish time rather than baked into a stored bundle, because a restoration and a
    re-announcement publish the bundle they were handed and a name is not part of what they restore:
    a machine that comes back from a failed switch would otherwise republish a name it stopped using
    and stay that way, since publication is an event and a stale-but-successful record is not retried.

    `profile` says *which* configuration is being published, and it is not a nicety. One model id can
    be launched by several configurations of one machine that describe it differently -- the same
    weights at two session lengths, or on two cards instead of four -- and a document about the launch
    that is running should call it what the launch that is running calls it. Without a profile there is
    no launch in hand to ask, so the registry answers and the first definition of each model stands.
    """
    config_dir = Path(config_dir)
    if profile is not None:
        try:
            definition = read(config_dir / 'profiles' / (resolve_profile(profile) + '.json'))
            launched = {model_id(model): display_name(model, aliases, host)
                for model in models_of(config_dir, definition)}
        except Exception:
            # A release can restore a bundle whose configuration has since left the registry, and a
            # machine coming back is not the moment to refuse a name because a name went missing.
            launched = {}
        if launched:
            return launched
    return {model: summary['display_name']
        for model, summary in registry_summaries(config_dir, host, aliases).items()}


def registry_summaries(config_dir, host=None, aliases=None):
    """Every model this host serves or could serve, with the names it is published by.

    One pass for the whole registry, because three surfaces -- the page, the CLI listing, and the
    catalog about to be published -- must not each walk the profile files and reach their own opinion.
    """
    config_dir = Path(config_dir)
    shared = read(config_dir / 'shared.json')
    models = {}
    for name in configurations_for(config_dir, host):
        definition = read(config_dir / 'profiles' / (name + '.json'))
        for model in models_of(config_dir, definition):
            entry = models.setdefault(model_id(model), {'configurations': [], 'definition': model})
            entry['configurations'].append(name)
    return {model_id(entry['definition']): {**model_summary(config_dir, entry['definition'], shared, host,
            aliases), 'configurations': sorted(entry['configurations'])}
        for entry in models.values()}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def service_engine(bundle, service):
    """Read both current per-service pins and previous single-engine release state."""
    manifest = bundle['manifest']
    entry = next(s for s in manifest['services'] if s['service'] == service)
    return entry.get('engine') or manifest['engine']


def is_exclusive(definition):
    """Whether this model owns every text GPU this host declares.

    It is a fact about one model and its memory, not about a configuration: a model that needs the
    whole machine cannot share it, and the configuration that loads it loads nothing else.
    """
    exclusive = definition.get('exclusive', False)
    require(exclusive is True or exclusive is False, definition['id'] + ': exclusive must be true or false')
    if exclusive:
        require(definition['gpu_group'] == EXCLUSIVE_GROUP,
            definition['id'] + ': an exclusive model must name the "' + EXCLUSIVE_GROUP + '" GPU group')
    else:
        require(definition['gpu_group'] in PAIR_GROUPS, definition['id'] + ': unknown GPU group')
    return exclusive


def parallel_slots(definition):
    """Backend slots: one by default; a solo configuration, which has every GPU, may run two."""
    slots = definition.get('parallel_slots', 1)
    require(type(slots) is int and slots in (1, 2), definition['id'] + ': parallel_slots must be 1 or 2')
    require(slots == 1 or is_exclusive(definition), definition['id'] + ': only a solo configuration may run two slots')
    return slots


def text_groups(host):
    """The text GPU groups this host declares, in the order a solo configuration reserves them.

    A machine with two text cards declares one pair and a machine with four declares two. `all` has
    always meant "every text GPU there is"; this is that fact read from the host instead of from a
    constant that assumed every machine looked like the first one.
    """
    groups = host['gpu_ids']
    require(isinstance(groups, dict) and groups and not set(groups) - set(PAIR_GROUPS),
        'host.json declares its text GPUs in groups named ' + ' and '.join(PAIR_GROUPS) +
        '; this host declares ' + (', '.join(sorted(groups)) if groups else 'nothing'))
    return [group for group in PAIR_GROUPS if group in groups]


def gpu_pair(host, group):
    """The two text GPU UUIDs of one group this host actually declares.

    The refusal matters more than the lookup: a one-pair host handed a configuration of the second
    pair used to fail on a missing key or on a total that named four GPUs, and neither said that the
    machine simply does not have that many cards.
    """
    groups = host['gpu_ids']
    require(group in groups, 'host.json declares no "' + str(group) + '" GPU group; this host declares '
        + ', '.join(text_groups(host)) + '. A configuration places its GPUs on groups this host states, '
        'and a host with fewer cards runs the configurations of a host with fewer cards')
    pair = groups[group]
    require(len(pair) == 2 and all(x.startswith('GPU-') and 'REPLACE' not in x for x in pair),
        'Configure two real GPU UUIDs per group in host.json')
    return list(pair)


def text_gpus(host, definition):
    """Text GPU UUIDs in CUDA order for a configuration that owns the host, or for its own pair."""
    groups = text_groups(host) if is_exclusive(definition) else (definition['gpu_group'],)
    return [gpu for group in groups for gpu in gpu_pair(host, group)]


def gpu_names(host, definition):
    """Card names in the same order as text_gpus(); empty when the host does not name them."""
    names = (host or {}).get('gpu_names') or {}
    if not names:
        return []
    groups = text_groups(host) if is_exclusive(definition) else (definition['gpu_group'],)
    return [name for group in groups for name in names.get(group, [])]


# ------------------------------------------------------------------------------------ card power
# A card's power state is the one hardware fact this runtime writes as well as reads, and it is the
# one that comes from a campaign on a specific board rather than from a configuration. The split
# follows the house rule: the rungs, their numbers, the envelope they have to sit inside and the
# evidence behind them are source (config/undervolt.json), and which physical card runs which plan is
# operational state (`.state/host.json` -> `power_plans`). A card with no plan is offered no rung --
# `330 W / 1905 MHz / +75` is a GA102 result and means nothing for an Ada board -- but it can still be
# reverted to stock or moved with numbers of your own, because those need no measurement.

POWER_FILE = 'undervolt.json'
POWER_RUNGS = ('safe', 'sensible', 'assertive')


def read_power_registry(config_dir):
    """The rung table, validated. A malformed table is a refusal that names the key to fix.

    Validation is strict because these numbers are written to a live GPU: a rung whose offset is off
    the driver's step, or whose default names a rung that is not there, is a bug that only shows up
    as a card that will not boost during a serving day.
    """
    path = Path(config_dir) / POWER_FILE
    registry = read(path)
    require(isinstance(registry, dict) and isinstance(registry.get('plans'), dict),
        POWER_FILE + ': expected an object with a "plans" object inside it')
    envelope = registry.get('envelope') or {}
    require(all(isinstance(envelope.get(key), int) and envelope[key] > 0 for key in
            ('offset_step', 'offset_max', 'clock_min')),
        POWER_FILE + ': envelope needs integer offset_step, offset_max and clock_min, all positive')
    for plan_id, plan in registry['plans'].items():
        require(isinstance(plan_id, str) and re.fullmatch(r'[a-z0-9][a-z0-9-]*', plan_id),
            POWER_FILE + ': plan ids are lowercase slugs, not ' + repr(plan_id))
        require(isinstance(plan, dict) and isinstance(plan.get('name'), str) and plan['name'],
            POWER_FILE + ': plan ' + plan_id + ' needs a name')
        rungs = plan.get('rungs')
        require(isinstance(rungs, dict) and rungs, 'plan ' + plan_id + ' needs at least one rung')
        offsets = []
        for name, rung in rungs.items():
            require(name in POWER_RUNGS, 'plan ' + plan_id + ': unknown rung name ' + repr(name) +
                '; a rung is one of ' + ', '.join(POWER_RUNGS) + ' so that "the middle one" means one thing')
            require(isinstance(rung, dict) and all(isinstance(rung.get(key), (int, float))
                    and rung[key] > 0 for key in ('power', 'clock_max', 'offset')),
                'plan ' + plan_id + ' rung ' + name + ': power, clock_max and offset are positive numbers')
            require(float(rung['offset']) % envelope['offset_step'] == 0,
                'plan ' + plan_id + ' rung ' + name + ': offset must sit on the driver\'s '
                + str(envelope['offset_step']) + ' MHz step')
            require(-envelope['offset_max'] <= rung['offset'] <= envelope['offset_max'],
                'plan ' + plan_id + ' rung ' + name + ': offset outside +/-' + str(envelope['offset_max']) +
                ' MHz proven on this driver')
            require(rung['clock_max'] > envelope['clock_min'],
                'plan ' + plan_id + ' rung ' + name + ': a max clock at or below the floor locks the card up')
            offsets.append(rung['offset'])
        require(len(set(offsets)) == len(offsets), 'plan ' + plan_id + ': two rungs share an offset')
        # The ladder is a ladder: the more a rung undervolts, the fewer watts it may ask for. A table
        # that violates this is a rung that would raise the watt limit while lowering the voltage,
        # which is not what any of these campaigns measured.
        ladder = sorted(rungs.items(), key=lambda item: item[1]['offset'])
        watts = [rung['power'] for _, rung in ladder]
        require(all(watts[i] >= watts[i + 1] for i in range(len(watts) - 1)),
            'plan ' + plan_id + ': rungs must get no more watts as their offset grows')
        require(plan.get('default') in rungs, 'plan ' + plan_id + ': default must name one of its rungs ('
            + ', '.join(sorted(rungs)) + ')')
    return registry


def power_plans(host):
    """Which card runs which plan, from the deployment's private settings: {GPU-uuid: plan id}.

    Absent is a real answer: it means nobody has measured this card. A malformed entry is reported
    rather than dropped, because a typo'd UUID silently un-measures a card that was measured.
    """
    value = (host or {}).get('power_plans') or {}
    require(isinstance(value, dict), 'host.json power_plans is an object of GPU UUID -> plan id')
    for gpu, plan_id in value.items():
        require(re.fullmatch(GPU_UUID, gpu or ''),
            'host.json power_plans has a key that is not a full GPU UUID: ' + repr(gpu))
        require(isinstance(plan_id, str) and plan_id,
            'host.json power_plans entry for ' + gpu + ' must name a plan')
    return dict(value)


def power_plan(registry, plan_id):
    """One plan by id, or a refusal that lists the plans this revision actually has."""
    plans = (registry or {}).get('plans') or {}
    known = (', '.join(sorted(plans)) if plans else 'none yet')
    require(plan_id is not None, 'This card has no undervolt plan, so it has no rung to apply. A plan is '
        'a measurement, and this card has not been through the ladder; this revision has plans for '
        + known + '. It can still be reverted to stock or given values of your own')
    require(plan_id in plans, 'no undervolt plan named ' + repr(plan_id) + '; this revision knows '
        + known + '. A plan is a measurement, so a card that has not been through the ladder has no '
        'rung to apply')
    return plans[plan_id]


def power_rung(plan, name):
    """The three numbers of one rung, as the card agent takes them."""
    rungs = plan['rungs']
    require(name in rungs, 'plan ' + plan['name'] + ' has no rung ' + repr(name) + '; it has '
        + ', '.join(rung for rung, _ in sorted(rungs.items(), key=lambda item: item[1]['offset']))
        + ', in the order the ladder runs. Choose one of those, or set values of your own')
    rung = dict(rungs[name])
    rung['clock_min'] = None
    return rung


def power_request(body, envelope, limits=None):
    """Validate and normalise a write request against the envelope and the card's own watt bounds.

    `limits` is the card's reported (min, max) power limit from the last read: the driver is the only
    authority on what a particular board accepts, and it is read rather than assumed because a card
    swapped for another model of the same name changes the answer.
    """
    require(isinstance(body, dict), 'a JSON object naming the card and the action')
    allowed = {'gpu', 'action', 'rung', 'power', 'clock_max', 'offset', 'override'}
    require(not set(body) - allowed, 'power accepts only ' + ', '.join(sorted(allowed)) +
        '; this request also carries ' + ', '.join(sorted(set(body) - allowed)))
    gpu, action = body.get('gpu'), body.get('action')
    require(isinstance(gpu, str) and re.fullmatch(GPU_UUID, gpu),
        'gpu must be a full GPU UUID this machine reports')
    require(action in ('apply', 'revert', 'custom', 'read'),
        "action is 'apply' a named rung, 'revert' to stock, 'custom' with your own three numbers, or "
        "'read' this card again right now")
    require(isinstance(body.get('override', False), bool), 'override is true or false')
    step, ceiling = envelope['offset_step'], envelope['offset_max']
    request = {'gpu': gpu, 'action': action, 'override': bool(body.get('override')),
        'rung': None, 'power': None, 'clock_max': None, 'offset': None, 'clock_min': envelope['clock_min']}
    if action == 'read':
        require(not {'rung', 'power', 'clock_max', 'offset', 'override'} & set(body),
            'a read takes the card and nothing else')
    if action == 'apply':
        require(isinstance(body.get('rung'), str) and body['rung'],
            'apply names the rung to put the card at')
        request['rung'] = body['rung']
        require(not {'power', 'clock_max', 'offset'} & set(body),
            'apply takes a rung name, not numbers: use custom for numbers of your own')
    if action == 'apply' or action == 'custom':
        if action == 'custom':
            require(all(isinstance(body.get(key), (int, float))
                    for key in ('power', 'clock_max', 'offset')),
                'custom needs power in watts, clock_max in MHz and offset in MHz')
            request['power'] = body['power']
            request['clock_max'] = body['clock_max']
            request['offset'] = body['offset']
            require(request['clock_max'] > envelope['clock_min'],
                'a max clock at or below ' + str(envelope['clock_min']) +
                ' MHz leaves the card no P-state to idle in, which is what makes it crash at rest')
    raw = request['offset']
    if raw is not None:
        rounded = int(round(raw / step)) * step
        require(-ceiling <= rounded <= ceiling, 'offset must sit within +/-' + str(ceiling) +
            ' MHz: ' + str(ceiling) + ' is what this driver family has been held at without faulting, and'
            ' acceptance is not serving safety')
        request['offset'], request['offset_rounded'] = rounded, rounded != raw
    if request['power'] is not None and limits:
        low, high = limits
        require(low is None or request['power'] >= low,
            'this card reports a minimum of %.0f W' % low)
        require(high is None or request['power'] <= high,
            'this card reports a maximum of %.0f W' % high)
    return request


def service_key(definition):
    """The key this model's Compose service and manifest entry are filed under.

    It is the model id, because a launch is identified by what it loads: two models on one machine are
    two services, and a machine that later loads a different model has a different service rather than
    the same slot refilled. Container names, ports and GPU reservations come from the profile and are
    unaffected by any of this.
    """
    return model_id(definition)


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

    Nothing else is loaded to share the vision GPU, so it carries the projector and
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
    engine refuses to start on a sampling value it does not understand, and for an exclusive model
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
    key is a JSON boolean: for an exclusive model, a value the engine will not read is nothing
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
    the standing reason: an exclusive model whose engine refuses to start is nothing serving, and
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
    not exceed the trained one, is an engine that refuses to start -- and for an exclusive model
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


def require_container_contract(service, cfg):
    missing = [key for key in CONTAINER_CONTRACT if cfg.get(key) is None]
    require(not missing, 'Service {} is missing container settings: {}'.format(service, ', '.join(missing)))


def environment_service(name, definition, engine, shared, host, root, ctx, slots, exclusive,
        aliases=None):
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
    if host.get('backend_address') and 'STRATA_ALLOWED_HOSTS' in environment:
        # The published LAN address is also the HTTP Host header the remote router sends.
        # Keep the engine's DNS-rebinding protection, adding only this deployment's address.
        allowed = [h.strip() for h in environment['STRATA_ALLOWED_HOSTS'].split(',') if h.strip()]
        address = backend_bind_address(host)
        if address not in allowed:
            allowed.append(address)
        environment['STRATA_ALLOWED_HOSTS'] = ','.join(allowed)
    reserved, vision_index = engine_vision_gpu(name, host, engine, options, environment, text_devices)
    # The engine numbers the cards it is given, so two statements in a profile are statements about
    # how many cards this machine has: which ones the language model runs on (`--gpus`) and which ones
    # the container can see. Both are checked against what the host can actually place, because a
    # four-card launch rendered onto a two-card host is otherwise a container of two cards whose engine
    # was told to use four -- and that boot fails after the drain, on a machine that was serving.
    text_indices = [str(index) for index in range(len(text_devices))]
    if '--gpus' in options:
        require([item.strip() for item in str(options['--gpus']).split(',')] == text_indices, name +
            ': --gpus must name the text cards in the order this host can place them, as '
            + ','.join(text_indices) + '. A configuration places its model on the cards the host has; '
            'a host with fewer cards runs the configurations written for fewer cards')
    visible = environment.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        require([item.strip() for item in str(visible).split(',')] == text_indices, name +
            ': the container-visible GPU list must state the text cards as ' + ','.join(text_indices) +
            '; the shared vision GPU is appended by the renderer, being a fact of the host')
    if vision_index is not None:
        # Indices rather than UUIDs, because that is what this engine is addressed in: the encoder is
        # placed by the same numbering, so `--gpus 0,1` keeps meaning the text cards.
        environment['CUDA_VISIBLE_DEVICES'] = ','.join(str(index) for index in range(len(reserved)))
    cfg.update(image=engine['tag'], container_name=definition['container_name'], command=[],
        environment=environment, volumes=mounts,
        ports=[f"{backend_bind_address(host)}:{definition['host_port']}:8080"],
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
    entry = {**copy.deepcopy(shared['catalog_defaults']), **copy.deepcopy(definition['catalog']),
        'backend_url': advertised_backend_url(host, definition)}
    # Same rule as the llama.cpp path: the router never infers this from a model name.
    require(type(entry['capability_profile'].get('nsfw')) is bool,
        name + ': capability_profile.nsfw must be declared true or false')
    declares_vision = list(entry['input_modalities']) == ['text', 'image']
    require(declares_vision == (entry['capability_profile'].get('vision') is True),
        name + ': input_modalities and capability_profile.vision must agree')
    entry.update(model=options['--alias'], context_length=ctx,
        total_context_length=ctx * slots, max_active_requests=slots, gpu_uuids=text_devices,
        model_path=targets[model_target], backend_revision=engine['revision'],
        display_name=display_name(definition, aliases, host), gpu_names=gpu_names(host, definition),
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


def profile_summary(config_dir, name, shared, host=None, aliases=None):
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
    models = [model_summary(config_dir, model, shared, host, aliases)
        for model in models_of(config_dir, definition)]
    return {'profile': definition['id'],
        'models': models,
        'service_ids': sorted({sid for model in models for sid in model['service_ids']}),
        'display_name': display_name(definition, aliases, host),
        'default_name': default_name(definition, host),
        **({'alias': aliases[model_id(definition)]}
            if (aliases or {}).get(model_id(definition)) else {}),
        'model': model_id(definition), 'context_tokens': tokens, 'engine': definition['engine'],
        'engine_tag': engine['tag'], 'backend_revision': engine['revision'],
        'gpu_group': definition['gpu_group'], 'gpu_names': gpu_names(host, definition),
        # The service IDs of the model this configuration is named after. `service_ids` above is the
        # whole configuration; this is the one half of it, and both are published because "what would
        # answer" is a different question from "what is loaded" and neither answers the other.
        'model_service_ids': sorted(definition['catalog'].get('aliases', [])),
        # Whether a switch to this configuration would put anything on the shared vision card, which
        # is what the page needs to draw that card's owner before the switch happens. Declared twice
        # on purpose, the way `render` reads it: a llama.cpp profile says so by shipping a projector,
        # an environment-configured engine by claiming images as an input modality. And a host with no
        # vision card has none, however the profile feels about it.
        'vision_gpu': bool(host and host.get('vision_gpu_id') and (
            options.get('--mmproj') or 'image' in definition['catalog'].get('input_modalities', []))),
        'exclusive': is_exclusive(definition), 'parallel_slots': definition.get('parallel_slots', 1),
        **({'variant_label': definition['variant_label']} if definition.get('variant_label') else {})}


def available_profiles(config_dir, host=None, aliases=None):
    """The configurations this host can select, each with the models it loads.

    The host is not a display nicety. A row on the status page is read as a decision the operator
    could make, and on a machine with one GPU pair the configurations of a four-GPU machine are not
    decisions it can make at all -- they would refuse their own selection. `host=None` asks for the
    whole registry, which is what a revision-wide check wants and what a host whose private settings
    name no machine still sees.
    """
    config_dir = Path(config_dir)
    shared = read(config_dir / 'shared.json')
    require(shared['schema_version'] == 2, 'Unsupported configuration version')
    return {'selectable': [profile_summary(config_dir, name, shared, host, aliases)
        for name in configurations_for(config_dir, host)]}


def models_of(config_dir, definition):
    """The models a configuration loads, in the order it names them.

    A configuration is the decision about which models run on this machine and which of its cards they
    run on, so it says so in one list: one entry for a machine that runs one model, two or more when
    the cards are divided between them. Nothing in this function knows what a configuration is
    *supposed* to load -- there is no default second half of a pair, and a file that names one model
    means one model.

    A profile with no list is a model on its own, which is what the files named by a list are.
    """
    config_dir = Path(config_dir)
    names = definition.get('models')
    if names is None:
        require(definition['id'] not in CONFIGURATIONS, definition['id']
            + ': a configuration must name the models it loads ("models": ["…"])')
        return [definition]
    require(isinstance(names, list) and names and all(isinstance(n, str) and n.strip() for n in names),
        definition['id'] + ': "models" must be a non-empty list of profile ids')
    require(len(set(names)) == len(names), definition['id'] + ': loads '
        + ', '.join(sorted(n for n in names if list(names).count(n) > 1)) + ' twice')
    require(definition['id'] in names, definition['id'] + ': a configuration must name itself among the '
        'models it loads -- this file is both the decision and one of the things decided')
    found = []
    for name in names:
        model = read(config_dir / 'profiles' / (name + '.json'))
        require(model.get('id') == name, name + ': profile file id differs from its filename')
        if name != definition['id']:
            require('models' not in model, name + ' is a configuration and cannot be loaded as a model; '
                + definition['id'] + ' must name the models themselves')
        found.append(model)
    alone = [model['id'] for model in found if is_exclusive(model)]
    if len(found) > 1 and alone:
        require(False, definition['id'] + ': ' + alone[0]
            + ' owns every text GPU of this host, so no other model can be loaded beside it')
    return found


def model_summary(config_dir, definition, shared, host=None, aliases=None):
    """One model of a configuration: what it is called, where it runs, and how a client asks for it.

    These are the only facts a surface needs to draw a model, and they are read the same way for the
    model a configuration is named after and any other it loads beside it -- which is the whole point:
    a machine that serves two models shows two of these, and neither of them is a footnote.
    """
    engine = shared['engines'][definition['engine']]
    options = {**shared['arguments'], **definition['arguments']} if engine_interface(engine) == 'argv' \
        else dict(definition['arguments'])
    return {'model': model_id(definition), 'display_name': display_name(definition, aliases, host),
        'default_name': default_name(definition, host),
        **({'alias': aliases[model_id(definition)]}
            if (aliases or {}).get(model_id(definition)) else {}),
        'context_tokens': definition['context_tokens'],
        'service_ids': sorted(definition['catalog'].get('aliases', [])),
        'gpu_group': definition['gpu_group'], 'gpu_names': gpu_names(host, definition),
        'exclusive': is_exclusive(definition),
        'vision_gpu': bool(host and host.get('vision_gpu_id') and (
            options.get('--mmproj') or 'image' in definition['catalog'].get('input_modalities', [])))}


def offline_services(config_dir, shared, host, loaded, operator_aliases=None):
    """Service IDs this configuration deliberately does not run, so the router can report them as
    temporarily offline rather than unknown.

    The rule is about service IDs: an ID is offline here when some configuration of *this* machine
    publishes it and the models the running configuration loads do not. A configuration that answers
    `nighttime` from a model it loads therefore declares nothing offline, and one that stopped
    offering it does. Which models are loaded is a separate fact, and the services list states it;
    conflating the two is how a status page comes to tell you a model is off while it is answering.

    A model that belongs to another machine publishes nothing here at all, so it makes nothing
    offline: the honest answer for a service this host never offered is that it does not know it,
    not that it is temporarily down.
    """
    config_dir = Path(config_dir)
    shared = read(config_dir / 'shared.json')
    published = {alias for definition in loaded for alias in definition['catalog'].get('aliases', [])}
    # Every service ID this machine's registry offers, whether or not the configuration that offers it
    # is the one running: `nighttime` is offline while a configuration that does not load its model
    # runs, and unknown to a router that never saw it otherwise.
    #
    # Collected per model across every configuration of this host that loads it, because one model id
    # can be loaded by several configurations that publish different IDs (the 256K launch answers
    # `nighttime`; the two-card launch of the same weights does not). The names come from the model's
    # own definition -- an operator's alias above the name the profile states -- and the IDs are the
    # union, so an ID offered only by a configuration that is not running is still reported.
    offline = {}
    for name in configurations_for(config_dir, host):
        definition = read(config_dir / 'profiles' / (name + '.json'))
        for model in models_of(config_dir, definition):
            aliases = [alias for alias in model['catalog'].get('aliases', []) if alias not in published]
            entry = offline.setdefault(model_id(model),
                {**model_summary(config_dir, model, shared, host, operator_aliases), 'aliases': []})
            entry['aliases'] = sorted(set(entry['aliases']) | set(aliases))
    return [{'model': model, 'aliases': entry['aliases'], 'display_name': entry['display_name'],
            'default_name': entry['default_name'],
            **({'alias': entry['alias']} if 'alias' in entry else {}),
            'reason': 'not_loaded_by_this_configuration'}
        for model, entry in offline.items() if entry['aliases']]


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


def render(config_dir, host, profile, aliases=None):
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
    # Another machine's configuration is refused here and not offered upstream: the registry the
    # status page reads filters the same way, but the switch API, the CLI and a hand-written
    # transaction can all name an id directly, and rendering one onto the wrong hardware is a drain
    # spent on a launch that was measured somewhere else.
    require(belongs_to_host(selected, host), profile + ' is a configuration of '
        + ' or '.join(repr(name) for name in profile_hosts(selected)) + ', and this host is '
        + repr(host_id(host)) + '. Its cards, its memory sizing and its artifacts are that machine\'s; '
        'select a configuration of this host, or state this host on the profile if it really runs here')
    loaded = models_of(config_dir, selected)
    for definition in loaded:
        require(belongs_to_host(definition, host), 'Configuration ' + profile + ' loads '
            + definition['id'] + ', and that model belongs to '
            + ' or '.join(repr(name) for name in profile_hosts(definition))
            + ' -- a configuration cannot bring another machine\'s model onto these cards')
    exclusive_profile = is_exclusive(selected)
    for definition in loaded:
        name = definition['id']
        service = service_key(definition)
        engine = shared['engines'][definition['engine']]
        require_engine_architecture(name, host, engine)
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
                host, root, ctx, slots, is_exclusive(definition), aliases)
            for artifact in definition['artifacts']:
                source = str(root / Path(artifact['path']))
                artifacts[source] = {**artifact, 'source': source}
            gpu_ids.extend(text)
            require_container_contract(service, cfg)
            compose['services'][service] = cfg
            models.append(entry)
            manifests.append({'service': service, 'container_name': cfg['container_name'],
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
            ports=[f"{backend_bind_address(host)}:{definition['host_port']}:8080"],
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
        require_container_contract(service, cfg)
        compose['services'][service] = cfg
        entry = {**copy.deepcopy(shared['catalog_defaults']), **copy.deepcopy(definition['catalog']),
            'backend_url': advertised_backend_url(host, definition)}
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
            display_name=display_name(definition, aliases, host),
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
        manifests.append({'service': service, 'container_name': cfg['container_name'],
            'engine': copy.deepcopy(engine),
            'model_alias': entry['model'], 'context_tokens': ctx, 'parallel_slots': slots,
            'display_name': entry['display_name'], 'recommended_argv': argv,
            'gpu_device_ids': devices, 'mounts': mounts})
    # What "every text GPU this host declares, once each" used to be written as: four, in a house
    # with four. The number was the machine, and the machine is now a setting -- a host with one
    # pair renders the configurations of one pair, and a configuration that left a declared card
    # unplugged would be a card nothing can be placed on.
    declared = {gpu for group in text_groups(host) for gpu in gpu_pair(host, group)}
    require(len(gpu_ids) == len(set(gpu_ids)),
        'Backends require one reservation per text GPU; ' + ', '.join(gpu_ids) + ' reserves a card twice')
    require(set(gpu_ids) == declared, 'The selected configuration reserves ' + str(len(set(gpu_ids)))
        + ' text GPUs and this host declares ' + str(len(declared)) +
        '. A configuration reserves every text GPU this host states, across one model or several; a card '
        'that belongs to neither is a card the operator believes is serving and nothing can be placed on')
    catalog = {**copy.deepcopy(models[0]), 'schema_version': 3,
        'default_model': models[0]['model'], 'models': models,
        'configuration': {'id': profile, 'exclusive': exclusive_profile},
        'offline_services': offline_services(config_dir, shared, host, loaded, aliases)}
    bundle = {'schema_version': 1, 'profile': profile, 'compose': compose, 'catalog': catalog,
        'manifest': {'schema_version': 2, 'services': manifests},
        'artifacts': list(artifacts.values())}
    bundle['config_sha256'] = digest(bundle)
    return bundle

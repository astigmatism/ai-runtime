#!/usr/bin/env python3
"""Look at the page without a GPU, a model, or a machine that is willing to be drained.

    scripts/preview-ui.py                      # a machine serving the solo configuration
    scripts/preview-ui.py --state switching    # a switch caught mid-flight, with the stepper
    scripts/preview-ui.py --state restoration  # nothing serving, restoration pending
    scripts/preview-ui.py --port 8899          # then open http://127.0.0.1:8899/

The controller, the registry, the publication and the switch are real: this serves the same
`runtime/static/` files the container serves, against `runtime/system.py`'s fake Docker host from
the test suite, in a temporary state directory with synthetic GPU UUIDs. Nothing is faked below the
status payload, which is why the map, the configuration list and the aliases behave instead of
looking like a drawing of themselves. Clicking Switch really runs a transition.

The `restoration` state is the exception: an honest one is a controller whose last-known-good
release has been established and then broken, which is a longer story to tell than this file needs,
so the payload is edited after the fact and this file says which parts are a look rather than a
fact. It is not a substitute for tests/test_restoration.py, which is the real argument.

This lives in `scripts/` because it is a workbench and not part of the product: the image copies
`runtime/`, `config/` and `tests/`, and never this file. It leans on the test fixtures for the same
reason — they already know how to be a machine that is not this one.
"""
import argparse
from http.server import ThreadingHTTPServer
import copy
import hashlib
import json
import re
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

from test_controller import FakeSystem, vision_host  # noqa: E402
from runtime.config import ALIASES_VERSION, render  # noqa: E402
from runtime.controller import Controller  # noqa: E402
from runtime.operations import Operations  # noqa: E402
from runtime.system import atomic_json  # noqa: E402
from runtime.web import handler, refresh  # noqa: E402

# What the household's router is told about this machine, so the publication line in the status strip
# reads like the one on a real host. A synthetic token, and no secret anywhere in this file.
ROUTER = 'http://192.168.1.23:11435'


def machine(state, profile, arguments):
    """A host that is serving `profile`, with a saved name for each model it loads.

    A one-model configuration therefore comes up with one name and `qwen27b-q8-pair` with two: the
    names belong to models, so how many this machine has to say is a fact about the configuration.

    The default host is the four-text-GPU machine with a shared vision card; `--host-json` says which
    machine to pretend to be instead, which is how a second machine's shape gets looked at before it
    is asked to draw itself."""
    config = state / 'config'
    shutil.copytree(ROOT / 'config', config)
    # `--host-json` takes another machine's `host.json` so the page can be looked at as that machine:
    # two cards instead of four, no vision card, its own service IDs. Only what that machine *is* is
    # taken from the file — its identity, its cards and their order, its architectures, its model root.
    # The plumbing the fake host has to have to answer at all (a router to publish to, paths beside
    # this temporary state) comes from the fixture, because a file copied off a live machine names
    # directories that are not here.
    host = vision_host()
    if arguments.host_json:
        declared = json.loads(Path(arguments.host_json).read_text())
        for key in ('vision_gpu_id', 'vision_gpu_name', 'vision_cuda_device'):
            if key not in declared:
                host.pop(key, None)
        host.update({key: value for key, value in declared.items()
            if key in ('host_id', 'model_root', 'gpu_ids', 'gpu_names', 'gpu_architectures',
                'vision_gpu_id', 'vision_gpu_name', 'vision_cuda_device', 'cuda_order')})
    # The catalog goes to a directory beside this temporary state, the way it does on a machine that
    # runs its own router, because readiness asks that directory whether the models are there. Point
    # `--router` at the household's router instead and the page shows what a real deployment shows
    # when that router is unreachable, which is also worth looking at.
    host.update(router_state_dir=str(state / 'router'))
    # The cards get driver-shaped names before anything reads them, so the power panel and the map are
    # talking about the same four pieces of silicon.
    power_uuids(host)
    host.setdefault('host_id', 'daisy' if arguments.host_json else 'rosalina')
    if arguments.router:
        host['router_target'] = {'url': arguments.router, 'token_env': 'PUBLISHER_TOKEN'}
        os.environ.setdefault('PUBLISHER_TOKEN', 'synthetic-publisher-token')
    atomic_json(state / 'host.json', host)
    (state / 'router-token').write_text('synthetic-token')
    bundle = render(config, host, profile)
    system = FakeSystem(bundle)
    controller = Controller(config, state, 'a' * 40, system)
    atomic_json(state / 'active.json', {'revision': controller.revision, 'image': 'sha256:preview',
        'bundle': bundle})
    # A name for every model this configuration loads — the first, and the second when there is one —
    # because the point of a name is that it is said everywhere, and a preview where only one surface
    # has it proves nothing. They are written before the first publication so the catalog on disk
    # carries them too, and a name given to a model two configurations share is read by both of them.
    named = {model['model']: name for model, name in
        zip(bundle['catalog']['models'], ('House standard', 'Night owl'))}
    atomic_json(state / 'aliases.json', {'version': ALIASES_VERSION, 'aliases': named})
    controller.publish(bundle)
    atomic_json(state / 'previous.json', {'revision': 'b' * 40, 'profile': profile, 'bundle': bundle})
    return controller, system, bundle


# --------------------------------------------------------------------------------- the cards' power
# A preview without a GPU has to answer the card agent the way a GPU does, or every tile says
# "power unreadable" and the page is a lie in the other direction. This is that answer: a handful of
# cards with a driver behind them, at states worth looking at — one at a rung, one at stock, one
# hand-set between rungs, one that this revision has never measured, and one nobody has read yet.
CARDS = {
    # name: the plan this revision has for it, or None when nobody has run the ladder on it.
    'RTX 3090': 'rtx3090', 'RTX 4080 SUPER': 'rtx4080-super', 'RTX 3080 Ti': 'rtx3080-ti',
    'RTX 4080': None, 'RTX 2080 Ti': None}
# What each card is doing when the preview opens, by position in the map: the states a page has to
# draw, in the order a person reading them expects to find them. The last card is deliberately left
# unseeded, so the driver answers it with an error and the page has to say `power unreadable` — and
# a card nobody has polled yet, which shows `no reading yet`, is what every card looks like for the
# first half minute of a fresh preview.
OPENING = [('sensible', None), ('stock', None), ('custom', None), ('rung', 'assertive')]


def power_uuids(host):
    """The host's own cards, given UUIDs a power request can name.

    The fixture's host calls its cards `GPU-day-1`, and a power request names a card the way the
    driver does, so the preview renames the ones that are not already shaped like a card — everywhere
    the old name appears, `cuda_order` included, because a host whose `gpu_ids` say one thing and
    whose ordering says another refuses to render for a reason that has nothing to do with what is
    being looked at. Done before the bundle is rendered, so the map, the services and the power block
    all speak one set of names, which is the whole point of a preview instead of a drawing.
    """
    def shaped(value):
        return re.fullmatch(r'GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}', value or '')

    def invented(value):
        digest = hashlib.sha1(value.encode()).hexdigest()
        return 'GPU-%s-%s-%s-%s-%s' % (digest[:8], digest[8:12], digest[12:16], digest[16:20],
            digest[20:32])

    renamed = {}
    snapshot = copy.deepcopy(host)

    def rename(value):
        if isinstance(value, list):
            return [rename(item) for item in value]
        if isinstance(value, dict):
            return {rename(key): rename(item) for key, item in value.items()}
        if isinstance(value, str) and value.startswith('GPU-') and not shaped(value):
            return renamed.setdefault(value, invented(value))
        return value

    # The names are collected first, so one card gets one UUID however many places name it.
    collected = []

    def find(value):
        if isinstance(value, list):
            for item in value:
                find(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                if isinstance(key, str):
                    collected.append(key)
                find(item)
        elif isinstance(value, str):
            collected.append(value)

    find(host)
    for name in collected:
        if name.startswith('GPU-') and not shaped(name):
            renamed.setdefault(name, invented(name))
    host.clear()
    host.update(rename(snapshot))

    # Nothing written down here on purpose. A card is offered the plan measured for the model its own
    # name says it is, which is every machine in the house: `power_plans` is the override an operator
    # reaches for only to point one card at other numbers, or at none, and a preview that set it for
    # every card would draw a machine that does not exist.
    host['power_plans'] = {}
    return host


class PreviewDriver:
    """The card agent, answered in this process: reads report, writes move the two readable knobs."""

    def __init__(self, controller):
        self.controller = controller
        self.cards = {}
        self.log = []

    def seed(self, uuid, name, state, rung=None):
        plan = CARDS.get(name)
        table = self.controller.power.registry()['plans'].get(plan or '', {})
        default = table.get('rungs', {}).get(table.get('default', ''), {})
        card = {'uuid': uuid, 'name': 'NVIDIA GeForce ' + name, 'driver': '560.35.03',
            'pci': '0000:01:00.0', 'power_limit': default.get('power', 350),
            'power_default': 350, 'power_min': 100, 'power_max': 350, 'draw': 71.0,
            'draw_average': 64.0, 'clock_sm': 1500, 'clock_max_sm': 2100, 'clock_mem': 9750,
            'pstate': 'P0', 'temperature': 44, 'fan': 42, 'utilization': 88, 'memory_used': 21_500,
            'memory_utilization': 23, 'power_capped_hours': 6.4, 'offset': default.get('offset', 0)}
        if state == 'stock':
            card.update(power_limit=card['power_default'], offset=0, utilization=4, pstate='P5',
                clock_sm=210, temperature=31)
        elif state == 'custom':
            card.update({'offset': -30, 'power_limit': 255, 'power_default': 320, 'power_max': 320,
                'clock_max_sm': 2955, 'clock_sm': 2520, 'power_capped_hours': 61.2})
        elif rung:
            asked = table['rungs'][rung]
            card.update(power_limit=asked['power'], offset=asked['offset'],
                clock_sm=asked['clock_max'])
        self.cards[uuid] = card

    def answer(self, argv):
        verb = argv[argv.index('runtime.gpucard') + 1]
        uuids = [argv[i + 1] for i, arg in enumerate(argv) if arg == '--uuid']
        if verb == 'read':
            return {'ok': True, 'verb': 'read', 'cards': {uuid: self.cards.get(uuid) or {
                # A card the driver can see but cannot answer is one card's failure, not the pass's:
                # this is the shape `runtime.gpucard` returns, and the tile says `power unreadable`.
                'error': 'the V/F offset query is not supported on this device'} for uuid in uuids}}
        uuid = uuids[0]
        asked = {'power': int(argv[argv.index('--power') + 1]),
            'offset': int(argv[argv.index('--offset') + 1]),
            'clock_max': None if '--clock-max' not in argv else int(argv[argv.index('--clock-max') + 1]),
            'clock_min': 210 if '--clock-min' not in argv else int(argv[argv.index('--clock-min') + 1])}
        before = self.cards.get(uuid) or {'power_default': 350, 'clock_max_sm': 2100}
        after = dict(before, power_limit=asked['power'], offset=asked['offset'],
            clock_sm=asked['clock_max'] or before['clock_max'], power_capped_hours=0.0)
        self.cards[uuid] = after
        self.log.append((uuid, asked))
        return {'ok': True, 'verb': 'apply', 'apply': {'before': before, 'after': after,
            'attempts': 1, 'asked': asked, 'commands': ['nvidia-smi -i %s -pl %d' % (uuid,
                asked['power'])], 'readback': {'power': 'ok', 'offset': 'ok',
                'clock_max': 'unverifiable'}, 'errors': [], 'ok': True}}


def power_preview(controller, system, scene):
    """Wire the fake driver in, and seed the states the page is being looked at for."""
    driver = PreviewDriver(controller)
    docker = system.docker
    def fake(*args, **kwargs):
        if args[0] == 'run' and 'runtime.gpucard' in args:
            return json.dumps(driver.answer(list(args)))
        return docker(*args, **kwargs)
    system.docker = fake
    cards = controller.power.cards()
    for position, (uuid, name) in enumerate(cards):
        if position == len(cards) - 1 and len(cards) > 1:
            continue  # the last card is the one the driver cannot answer: `power unreadable`
        state, rung = OPENING[position % len(OPENING)]
        if CARDS.get(name) is None and state == 'rung':
            state, rung = 'stock', None
        driver.seed(uuid, name, state, rung)
    controller.power.merge({uuid: {'read_at': time.strftime('%Y-%m-%dT%H:%M:%S+00:00',
        time.gmtime(time.time() - 12)), 'read': card} for uuid, card in driver.cards.items()})
    controller.power.merge({'swept_at': time.strftime('%Y-%m-%dT%H:%M:%S+00:00',
        time.gmtime(time.time() - 12))})
    if scene == 'gate':
        # One engine unwell on a card with a plan: the panel has to refuse, and offer the tick.
        for service in controller.load('active.json')['bundle']['manifest']['services']:
            system.containers[service['container_name']]['State']['Running'] = False
            break
    return driver


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--state', choices=('steady', 'switching', 'restoration'), default='steady')
    parser.add_argument('--profile', default='flash-next-strata-solo-thinking-recovery-cache')
    parser.add_argument('--port', type=int, default=8899)
    parser.add_argument('--router', help='Publish to this router URL instead of a directory beside the '
        'temporary state, so the publication line can be looked at as it looks on a real host.')
    parser.add_argument('--host-json', help='Path to another machine\'s host.json, so the page is '
        'drawn for that machine (its cards, its service IDs, its one configuration) instead of the '
        'four-GPU host this harness otherwise invents.')
    parser.add_argument('--power', choices=('mixed', 'gate'), default='mixed',
        help='What the cards are doing when the page opens: `mixed` is one card at a rung, one at '
        'stock, one hand-set between rungs, one the driver cannot answer, and (for the first half '
        'minute) one nobody has polled yet; `gate` also stops an engine, so the panel has to refuse '
        'a write and offer the tick that means "apply anyway".')
    arguments = parser.parse_args()

    temporary = tempfile.TemporaryDirectory()
    state = Path(temporary.name)
    controller, system, bundle = machine(state, arguments.profile, arguments)
    power_preview(controller, system, arguments.power)
    web_state = {'started': True}
    operations = Operations(controller, web_state)
    # One refresh before anything else, so the machine is already read as ready when the preview
    # below asks it to switch: the gate on a switch is live health, and a page that opens on a
    # machine nobody has polled yet is not a machine that is ready.
    refresh(controller, web_state, operations)

    if arguments.state == 'switching':
        # Two things come off the real path here, and both are the model rather than the machine: the
        # artifact check, which on a host with this preview's model root would refuse the switch over
        # weights that are not there, and one request kept in flight so the drain has a reason to
        # wait. The gate, the drain, the phases, the record and the publication are the controller's
        # own, which is the part worth looking at. After five minutes the drain times out by itself,
        # exactly as it would on a machine with a request that will not end.
        controller.prepare = lambda bundle: system.events.append(('prepare',))
        system.active_count = 1
        atomic_json(state / 'router' / 'runtime-state.json',
            {'draining': False, 'active_count': 1, 'queued_count': 0})
        target = 'qwen27b-q8-pair'
        operations.start({'profile': target, 'request_id': str(uuid.uuid4()),
            'expected_profile': bundle['profile'],
            'expected_revision': controller.revision})
    elif arguments.state == 'restoration':
        # A look, not a fact: the controller really is serving, and the payload below is what the
        # page has to make sense of when it is not. See the module docstring.
        system.containers.clear()
        system.model_loaded = False

    def supervise():
        while True:
            time.sleep(1)
            try:
                refresh(controller, web_state, operations)
            except Exception as error:
                print('preview refresh failed: ' + str(error), flush=True)
            if arguments.state == 'restoration':
                # A look, not a fact — see the module docstring. The controller is serving; this is
                # the payload the page has to make sense of when it is not.
                status = web_state.get('status') or {}
                status.update(ready=False, services=[], offline_services=[])
                status['service'] = {'serving': False, 'restorable': True, 'pending': True,
                    'profile': bundle['profile']}
                status['restoration'] = {'attempts': 3, 'reason': 'Model failed to load',
                    'next_attempt_at': time.strftime('%Y-%m-%dT%H:%M:%S%z',
                        time.localtime(time.time() + 45))}
                status['switching'] = {'available': False,
                    'reason': 'Nothing is serving; restore the last-known-good configuration first.'}

    threading.Thread(target=supervise, daemon=True).start()
    print('Preview of the AI Runtime page, state ' + arguments.state + ', on http://127.0.0.1:'
        + str(arguments.port) + '/ — Ctrl-C stops it.', flush=True)
    ThreadingHTTPServer(('127.0.0.1', arguments.port),
        handler(controller, web_state, operations)).serve_forever()


if __name__ == '__main__':
    main()

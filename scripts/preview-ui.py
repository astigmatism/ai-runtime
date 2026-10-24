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
import json
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
    arguments = parser.parse_args()

    temporary = tempfile.TemporaryDirectory()
    state = Path(temporary.name)
    controller, system, bundle = machine(state, arguments.profile, arguments)
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

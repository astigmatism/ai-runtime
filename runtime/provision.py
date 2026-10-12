"""Measure a self-provisioning engine here, so a switch to it is given a window that is not a guess.

An engine that builds its own state on first start has no honest readiness window until that build
has been timed on this host: the first attempt at this trial sat inside a window nobody had ever
measured and was abandoned with the router drained. This renders the real bundle, runs the real
container once against the real workspace bind, and records what it actually took.

Two measurements, because they cost different things:

* ``provision`` — the workspace build. It runs alongside the serving backend on purpose, so the same
  run also answers whether packing competes with a model that is loaded and answering. The
  container is removed as soon as the workspace exists; nothing about the serving pair is touched.
* ``warm`` — the boot everyone waits for: the same container, from the now-provisioned workspace,
  to its first healthy answer. This needs the text cards to itself, so it refuses to run while a
  backend is serving, and it never stops one either. A warm measurement is taken between turns.

The receipt they leave behind is what ``Controller.switchable`` insists on before a switch.
"""
import copy
import time
from pathlib import Path

from .config import require
from .system import atomic_json, now

# How often the workspace and the container are asked whether they are done, and the ceiling on
# asking. The ceiling is generous because the point of a run is to find out what the real number is;
# the measurement is what gets read afterwards, not a timeout anyone has to tune.
POLL_SECONDS = 15
LIMIT_SECONDS = 7200
# Where a provisioning container is published while the serving backend still holds its own port.
PORT_OFFSET = 1
# Its own compose project, so a measurement is torn down by name and never reaches the serving
# stack by accident. Shutting a loaded engine down is slow, so the default is its own grace period.
PROJECT = 'ai-runtime-provision'
STOP_SECONDS = 30


def workspace_target(cfg, provision):
    """The host directory the engine builds into, taken from the bundle's own bind."""
    for volume in cfg['volumes']:
        if volume['target'] == provision.get('workspace_target'):
            return Path(volume['source'])
    return None


def ready_files(workspace, provision):
    return [workspace / name for name in provision.get('ready_files', [])]


def published_port(cfg, offset=0):
    """The loopback port the real bundle publishes, shifted when something else has it."""
    require(cfg.get('ports'), 'The backend publishes no port to measure against')
    return int(cfg['ports'][0].split(':')[1]) + offset


def throwaway(bundle, role, name, port):
    """The one-off compose project the measurement runs in.

    It is the profile's own rendered service, so what gets measured is the container the switch
    would start — same image, entrypoint, arguments, environment, user, binds and card
    reservations. Three things change, all of them because a measurement runs beside a serving
    machine: the container name, so it cannot collide with the backend that is up; the published
    port, for the same reason; and the network, because a throwaway must not join the router's
    network or take the profile's service aliases. ``restart`` also comes back to ``no`` — a
    measurement that fails is a finding, not a container that keeps trying.

    This is written as a compose file rather than translated to ``docker run`` on purpose: the card
    reservations have to be asked for exactly the way the switch will ask for them, and the switch
    asks for them through compose.
    """
    compose = copy.deepcopy(bundle['compose'])
    compose['name'] = PROJECT
    compose['services'] = {role: compose['services'][role]}
    compose['services'][role].update(container_name=name, restart='no',
        ports=[f'127.0.0.1:{port}:8080'],
        labels={**(compose['services'][role].get('labels') or {}), 'local-ai.provisioning': 'throwaway'})
    compose['networks'] = {'router': {'name': PROJECT + '_router'}}
    return compose


def stop_seconds(cfg):
    """How long the engine gets to shut down, from its own grace period."""
    return int(str(cfg.get('stop_grace_period') or '30s').rstrip('s'))


def probe(system, port):
    """The container's own health answer, or None while it is not answering yet."""
    try:
        answer = system.http(f'http://127.0.0.1:{port}/health', timeout=5)
    except Exception:
        return None
    return answer if isinstance(answer, dict) else None


def neighbour(system, name):
    """What the serving backend is doing, sampled each poll of a run that shares the machine.

    This is the other half of the provisioning measurement: packing a model competes for CPU, disk
    and page cache with a model that is loaded and answering, and that is an assumption worth
    testing rather than one to find out about on a switch.
    """
    state = system.inspect(name)
    if not state:
        return None
    health = (state.get('State') or {}).get('Health') or {}
    return {'running': bool((state.get('State') or {}).get('Running')),
        'health': health.get('Status') or 'none'}


def watch(system, path, role, name, port, files, neighbour_name=None, goal='healthy',
          limit=LIMIT_SECONDS, poll=POLL_SECONDS, stop=STOP_SECONDS, log=print):
    """Start the container and poll until the workspace is built, or it is answering, or time is up.

    ``goal`` is ``workspace`` when only the build is wanted — beside a loaded model the engine may
    never reach health, and waiting for it would prove nothing — and ``healthy`` for the boot a
    switch actually waits for. The project is torn down in every outcome, and the samples always
    come back: a run that failed to reach anything is still a measurement, and is written down
    rather than retried until it is forgotten.
    """
    project = ['-p', PROJECT, '-f', str(path)]
    # Anything left by a run that was interrupted is cleared before this one starts, so the seconds
    # measured below belong to this container.
    system.docker('compose', *project, 'down', '--remove-orphans', '-t', '5', check=False)
    started = time.monotonic()
    samples = []
    reached = 'timed out'
    try:
        # No --wait: the wait is the thing being measured, and it is measured against the workspace
        # and the engine's own answer rather than against Docker's opinion of when to give up.
        system.docker('compose', *project, 'up', '-d', '--no-deps', '--pull', 'never', role)
        while True:
            elapsed = time.monotonic() - started
            answer = probe(system, port)
            workspace = all(target.is_file() for target in files)
            sample = {'seconds': round(elapsed, 1), 'workspace': workspace, 'health': answer is not None}
            if neighbour_name:
                sample['neighbour'] = neighbour(system, neighbour_name)
            samples.append(sample)
            log(f"provision {name}: {sample['seconds']:>7.1f}s workspace={workspace} "
                f"health={sample['health']}")
            if workspace and (goal == 'workspace' or answer is not None):
                reached = 'workspace' if goal == 'workspace' else 'healthy'
                break
            if elapsed >= limit:
                break
            time.sleep(poll)
    finally:
        system.docker('compose', *project, 'down', '--remove-orphans', '-t', str(stop), check=False)
    return {'seconds': round(time.monotonic() - started, 1), 'reached': reached,
        'workspace_ready': any(s['workspace'] for s in samples),
        'samples': samples}


def measurement(controller, profile, phase='provision', limit=LIMIT_SECONDS, poll=POLL_SECONDS, log=print):
    """Take one measurement for a profile and merge it into the receipt the switch reads."""
    require(phase in ('provision', 'warm'), 'Unknown provisioning phase: ' + str(phase))
    bundle = controller.desired(profile)
    role, engine, provision = controller.provisioning(bundle)
    require(role is not None, profile + ': this profile does not provision itself; there is nothing to measure')
    cfg = bundle['compose']['services'][role]
    workspace = workspace_target(cfg, provision)
    require(workspace is not None, profile + ': the engine workspace bind is missing from the bundle')
    system = controller.system
    require(not system.readonly, 'Provisioning starts a container and is refused in inspection mode')
    pending = controller.load('transaction.json', {}).get('phase')
    require(not pending or pending in ('succeeded', 'recovered'),
        'A runtime transaction is in progress; finish or recover it before provisioning')
    if phase == 'warm':
        require(not controller.serving(), 'A warm boot needs the text cards to itself; take this '
            'measurement between turns, when nothing is serving')
    port = published_port(cfg, PORT_OFFSET if phase == 'provision' else 0)
    name = cfg['container_name'] + '-provision'
    spec = controller.state / 'provisioning' / (profile + '-compose.json')
    # The exact file this run started from stays on disk: if a measurement is questioned later, or a
    # later engine revision changes what it asks for, the difference is readable.
    atomic_json(spec, throwaway(bundle, role, name, port))
    result = watch(system, spec, role, name, port, ready_files(workspace, provision),
        neighbour_name=cfg['container_name'], goal='workspace' if phase == 'provision' else 'healthy',
        limit=limit, poll=poll, stop=stop_seconds(cfg), log=log)
    receipt = controller.load('provisioning/' + profile + '.json') or {'profile': profile,
        'engine_tag': engine['tag'], 'image_id': engine['image_id'], 'workspace': str(workspace),
        'ready_files': list(provision.get('ready_files', [])), 'runs': []}
    receipt['measured_at'] = now()
    receipt['runs'].append({'phase': phase, 'reached': result['reached'], 'seconds': result['seconds'],
        'container': name, 'port': port})
    if phase == 'provision':
        receipt['provision_seconds'] = result['seconds']
        receipt['workspace_ready'] = result['workspace_ready']
        nearby = [s['neighbour'] for s in result['samples'] if s.get('neighbour')]
        receipt['neighbour_healthy'] = bool(nearby) and all(s['health'] == 'healthy' for s in nearby)
        receipt['neighbour_samples'] = nearby[-8:]
    else:
        receipt['warm_boot_seconds'] = result['seconds']
        receipt['healthy_warm_boot'] = result['reached'] == 'healthy'
        if result['reached'] != 'healthy':
            receipt['error'] = 'The container never answered its health check from the provisioned workspace'
    atomic_json(controller.state / 'provisioning' / (profile + '.json'), receipt)
    return receipt

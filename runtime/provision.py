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
import glob
import os
import shutil
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
    """The declared paths, or patterns, whose existence says the engine has finished building.

    A pattern is not a shortcut here: the engine names the file it finishes with after the family it
    was told to use — ``config/strata-unsloth-ud-iq4_xs.json`` for the Unsloth quant, ``strata-qwen-``
    for the other family — and a literal that quietly stops matching is worse than no check at all.
    The trial's build run waited for a file the engine had already written under a name nobody had
    written down, and then kept watching while the engine went on to load a model its cards had no
    room for. The quant used to stay literal for the same reason. It cannot any more: provisioning is
    read off the engine (controller.provisioning), so a marker naming a quant stops saying what the
    workspace is building the moment one engine serves two of them -- as it did on 2026-10-15, for the
    XL quant trial that ran on this engine and was rejected the same day. Listing one marker per quant
    is not the fix either -- workspace_ready() wants every declared pattern to match, so the first
    profile would then wait forever for the second's file, which is the same silent wait this function
    exists to avoid. What carries the guarantee now is that the path is workspace-relative and every
    provisioning profile has a workspace of its own; that invariant is asserted by
    ProvisioningWorkspaceTests, because two profiles sharing one workspace would let the first to
    finish declare the second's half-built workspace ready.
    """
    return [workspace / name for name in provision.get('ready_files', [])]


def matching(files):
    """The paths that satisfy the declaration, patterns resolved against what is on disk."""
    found = []
    for entry in files:
        text = str(entry)
        if any(mark in text for mark in '*?'):
            found.extend(sorted(glob.glob(text)))
        elif entry.is_file():
            found.append(text)
    return found


def workspace_ready(files):
    """True when every declared file, or pattern, has something behind it."""
    return len(matching(files)) >= len(files)


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
    port, for the same reason; and the service aliases, which are dropped because on the router
    network the profile's names belong to the container that is serving and nothing may resolve the
    trial to this one. The container's own name still resolves, which is how its health answer is
    read from here. ``restart`` also comes back to ``no`` — a measurement that fails is a finding,
    not a container that keeps trying.

    This is written as a compose file rather than translated to ``docker run`` on purpose: the card
    reservations have to be asked for exactly the way the switch will ask for them, and the switch
    asks for them through compose.
    """
    compose = copy.deepcopy(bundle['compose'])
    compose['name'] = PROJECT
    compose['services'] = {role: compose['services'][role]}
    service = compose['services'][role]
    service.update(container_name=name, restart='no', ports=[f'127.0.0.1:{port}:8080'],
        # No aliases: on this network the profile's names belong to the container that is serving,
        # and nothing may resolve the trial to this one. The container's own name still resolves,
        # which is how the health answer below gets read.
        networks={'router': {}},
        labels={**(service.get('labels') or {}), 'local-ai.provisioning': 'throwaway'})
    return compose


def stop_seconds(cfg):
    """How long the engine gets to shut down, from its own grace period."""
    return int(str(cfg.get('stop_grace_period') or '30s').rstrip('s'))


def health_routes(name, port):
    """How the container's health answer can be reached, in the order the controller can use it.

    The controller sits on the router network, not in the host's network namespace, so a port
    published on the host's loopback is reachable by a human on that host and not from here. The
    container's name on that network is the route the controller uses for every backend, so it is
    tried first and the published port is kept as a fallback. Which route answered is written into
    the samples, so a measurement never rests on an unstated assumption about the network again.
    """
    return [('container', 'http://' + name + ':8080/health'),
        ('host loopback', f'http://127.0.0.1:{port}/health')]


def probe(system, name, port):
    """The container's own health answer and the route it arrived on, or None while not answering.

    An engine may answer its health check before the model is resident: Strata's /health says
    status ok from server start and carries the model state as loaded. An answer that says
    loaded=false is a web page, not a boot, and counting it would measure the time to the page
    instead of the time to the model - the number the switch waits for and the receipt records.
    """
    for route, url in health_routes(name, port):
        try:
            answer = system.http(url, timeout=5)
        except Exception:
            continue
        if isinstance(answer, dict) and answer.get('loaded') is not False:
            return route, answer
    return None


def exited(system, name):
    """The exit code of a container that ran and stopped, or None while it is still running.

    A container that refuses to start does not come back: without asking, the poll loop below watches
    a corpse for the whole limit, and a two-second refusal reads as a two-hour build. That is not a
    hypothetical — the first two runs on the trial host were exactly that, a setup error printed once
    and then ninety minutes of ``workspace=False``.
    """
    state = (system.inspect(name) or {}).get('State') or {}
    if not state or state.get('Running') or state.get('Status') in ('created', 'running', 'restarting'):
        return None
    if not state.get('FinishedAt') or str(state['FinishedAt']).startswith('0001'):
        return None
    return state.get('ExitCode')


def last_words(system, name, tail=25):
    """The end of what the container printed, for a run that died at startup.

    The engine's own reason for refusing is the useful part of a failed measurement, and the person
    watching a run should not have to know to ask for it after the teardown has taken the container.
    """
    try:
        return system.docker('logs', '--tail', str(tail), name, check=False)
    except Exception:
        return None


def serving_containers(controller):
    """The containers actually serving, read from the active record rather than from the target.

    During a build run the target is not up, so watching the target's own container name would
    sample nothing; the machine's other half is whatever the active profile is running.
    """
    active = controller.load('active.json') or {}
    services = ((active.get('bundle') or {}).get('compose') or {}).get('services') or {}
    return sorted({cfg['container_name'] for cfg in services.values() if cfg.get('container_name')})


def neighbours(system, names):
    """What the serving backends are doing, sampled each poll of a run that shares the machine.

    This is the other half of the provisioning measurement: packing a model competes for CPU, disk
    and page cache with a model that is loaded and answering, and that is an assumption worth
    testing rather than one to find out about on a switch. The time the serving backend takes to
    answer the router is recorded beside it, because a backend that has stopped answering promptly
    is the contention to write down and stop for, not to discover on a switch.
    """
    found = []
    for name in names:
        state = system.inspect(name)
        if not state:
            continue
        health = (state.get('State') or {}).get('Health') or {}
        sample = {'name': name, 'running': bool((state.get('State') or {}).get('Running')),
            'health': health.get('Status') or 'none'}
        started = time.monotonic()
        try:
            system.http('http://' + name + ':8080/slots', timeout=10)
            sample['slots_seconds'] = round(time.monotonic() - started, 3)
        except Exception:
            sample['slots_seconds'] = None
        found.append(sample)
    return found


def footprint(workspace):
    """Bytes the workspace holds. Tolerant of a read-only mount and of files moving under it."""
    total = 0
    for root, _directories, entries in os.walk(workspace):
        for entry in entries:
            try:
                total += os.stat(os.path.join(root, entry), follow_symlinks=False).st_size
            except OSError:
                pass
    return total


def free_bytes(workspace):
    """Free space on the filesystem the workspace lives on, from the nearest path that exists."""
    target = Path(workspace)
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free


def watch(system, path, role, name, port, files, serving=(), goal='healthy',
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
    words = None
    try:
        # No --wait: the wait is the thing being measured, and it is measured against the workspace
        # and the engine's own answer rather than against Docker's opinion of when to give up.
        system.docker('compose', *project, 'up', '-d', '--no-deps', '--pull', 'never', role)
        while True:
            elapsed = time.monotonic() - started
            answer = probe(system, name, port)
            workspace = workspace_ready(files)
            sample = {'seconds': round(elapsed, 1), 'workspace': workspace, 'health': answer is not None}
            if answer:
                sample['health_via'] = answer[0]
            if serving:
                sample['neighbours'] = neighbours(system, serving)
            samples.append(sample)
            log(f"provision {name}: {sample['seconds']:>7.1f}s workspace={workspace} "
                f"health={sample['health']}")
            if workspace and (goal == 'workspace' or answer is not None):
                reached = 'workspace' if goal == 'workspace' else 'healthy'
                break
            code = exited(system, name)
            if code is not None:
                reached = 'exited (' + str(code) + ')'
                words = last_words(system, name) or ''
                log(f"provision {name}: the container stopped by itself with code {code} after "
                    f"{sample['seconds']:.1f}s; its last words are —")
                for line in words.splitlines()[-12:]:
                    log('  | ' + line)
                break
            if elapsed >= limit:
                break
            time.sleep(poll)
    finally:
        system.docker('compose', *project, 'down', '--remove-orphans', '-t', str(stop), check=False)
    found = {'seconds': round(time.monotonic() - started, 1), 'reached': reached,
        'workspace_ready': any(s['workspace'] for s in samples),
        # Which file the engine actually finished with, in its own words: the name is the evidence
        # that the declaration still describes what the engine does.
        'ready_paths': matching(files), 'samples': samples}
    if reached.startswith('exited'):
        found['container_log'] = words
    return found


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
    # Space before, space after: a pack that consumes a meaningful chunk of the model filesystem is
    # worth knowing the size of, and so is the free space it left behind.
    free_before = free_bytes(workspace)
    spec = controller.state / 'provisioning' / (profile + '-compose.json')
    # The exact file this run started from stays on disk: if a measurement is questioned later, or a
    # later engine revision changes what it asks for, the difference is readable.
    atomic_json(spec, throwaway(bundle, role, name, port))
    # A build run shares the machine, so the serving backends are watched while it packs. A warm run
    # has no serving backend by construction — that is what its refusal enforces — so there is nothing
    # beside it to sample.
    result = watch(system, spec, role, name, port, ready_files(workspace, provision),
        serving=serving_containers(controller) if phase == 'provision' else (),
        goal='workspace' if phase == 'provision' else 'healthy',
        limit=limit, poll=poll, stop=stop_seconds(cfg), log=log)
    receipt = controller.load('provisioning/' + profile + '.json') or {'profile': profile,
        'engine_tag': engine['tag'], 'image_id': engine['image_id'], 'workspace': str(workspace),
        'ready_files': list(provision.get('ready_files', [])), 'runs': []}
    receipt['measured_at'] = now()
    run = {'phase': phase, 'reached': result['reached'], 'seconds': result['seconds'], 'container': name,
        'port': port, 'via': 'cli', 'pack_bytes': footprint(workspace), 'free_bytes_before': free_before,
        'free_bytes_after': free_bytes(workspace), 'ready_paths': result['ready_paths'],
        'samples': result['samples']}
    if result.get('container_log') is not None:
        # The engine's own reason for refusing to start goes into the receipt with the run it came
        # from: the container that printed it is torn down before anyone reads the file, and this is
        # the difference between a finding and a mystery.
        run['container_log'] = result['container_log']
    receipt['runs'].append(run)
    receipt['pack_bytes'] = run['pack_bytes']
    receipt['disk_free_bytes'] = {'before': free_before, 'after': run['free_bytes_after']}
    if phase == 'provision':
        receipt['provision_seconds'] = result['seconds']
        receipt['workspace_ready'] = result['workspace_ready']
        nearby = [s['neighbours'] for s in result['samples'] if s.get('neighbours')]
        receipt['neighbour_healthy'] = bool(nearby) and all(n['health'] == 'healthy'
            for sample in nearby for n in sample)
        answered = [n['slots_seconds'] for sample in nearby for n in sample
            if n.get('slots_seconds') is not None]
        receipt['neighbour_slots_max_seconds'] = max(answered) if answered else None
        receipt['neighbour_samples'] = nearby[-8:]
    else:
        receipt['warm_boot_seconds'] = result['seconds']
        receipt['healthy_warm_boot'] = result['reached'] == 'healthy'
        if result['reached'] != 'healthy':
            receipt['error'] = 'The container never answered its health check from the provisioned workspace'
    atomic_json(controller.state / 'provisioning' / (profile + '.json'), receipt)
    return receipt

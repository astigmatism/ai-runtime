"""Serialized, recoverable reconciliation of the existing backend containers.

Paired profiles run Daytime and Nighttime side by side. An exclusive Daytime profile owns
all four text GPUs, so selecting it stops and removes Nighttime, and selecting a paired
profile again recreates it.
"""
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import uuid

from . import provision as provision_tool
from .config import (DAYTIME_PROFILES, DEFAULT_PROFILE, NIGHTTIME_PROFILE, available_profiles, digest, read, render,
    require, service_engine)
from .system import System, atomic_json, future, lock, now

# A transaction that has not reached a terminal phase. While one exists the runtime holds its
# drain or its outcome is unknown, so nothing else may change backends.
PENDING_PHASES = ('prepared', 'draining', 'applying', 'verifying', 'restoring', 'needs-attention')
# How many consecutive quiet passes a drain must see before work is assumed finished.
DRAIN_SETTLE_ROUNDS = 3
# How often the switch re-observes a backend that is still reading its model. Docker's health
# state is the engine's own /health, which an engine like Strata answers from server start, so
# the waiting has to happen against the controller's own view, and this is the cadence of it.
READINESS_POLL_SECONDS = 15
# A failed restoration is retried on this timer until the service is back, and the wait stops
# growing at the last entry. The host keeps trying rather than giving up: an operator may be
# asleep, and deciding to stop trying is the same as nobody being there.
RESTORATION_BACKOFF = (30, 60, 120, 300, 600, 900)
# How many steps of a restoration pass are kept in its record, so a long retry history cannot
# outgrow the state file. The failure that started it is kept separately in `reason`.
RESTORATION_PHASE_LOG = 50


def services_of(bundle):
    return bundle['compose']['services'] if bundle else {}


class Controller:
    def __init__(self, config_dir=None, state_dir=None, revision=None, system=None):
        self.config_dir = Path(config_dir or os.environ.get('RUNTIME_CONFIG_DIR', '/app/config'))
        self.state = Path(state_dir or os.environ['RUNTIME_STATE_DIR'])
        self.host = read(self.state / 'host.json')
        self.system = system or System()
        self.revision = revision or Path('/app/REVISION').read_text().strip()
        self.image = os.environ.get('RUNTIME_IMAGE', 'local/ai-runtime:git-' + self.revision)

    def load(self, filename, default=None):
        path = self.state / filename
        return read(path) if path.exists() else default

    def save(self, filename, value):
        require(not self.system.readonly, 'State mutation is forbidden in inspection mode')
        atomic_json(self.state / filename, value)

    def admin(self, path, body=None):
        token = (self.state / 'router-token').read_text().strip()
        require(bool(token), 'Router credential is missing')
        return self.system.http(self.host['router_admin_url'].rstrip('/') + '/admin/api/' + path,
            body, {self.host['router_admin_header']: token})

    def admin_healing(self, path, body=None):
        """The admin call a mutating operation makes, with the one repair it is allowed.

        A router whose active catalog is unusable (a backend kind it cannot handle) refuses every
        admin call, including the read a recovery needs, and the state is persisted: a restart
        comes back wedged exactly as it was. The runtime is the only writer of that catalog file,
        so it is the only component that can converge it. If the catalog on disk is not the
        last-known-good release, it is the catalog of a switch the router refused: the good one is
        written back and the router is reloaded, and the call is retried once. If the router was
        not wedged that way, the error is re-raised and the caller's retry schedule is untouched.
        """
        try:
            return self.admin(path, body)
        except urllib.error.HTTPError as error:
            if not self.heal_wedged_router(error):
                raise
            return self.admin(path, body)

    def heal_wedged_router(self, error):
        """Write the last-known-good catalog back to the router and reload it.

        Returns True only when something was actually changed. If the router was not wedged on a
        catalog this runtime refused - the error says otherwise, no last-known-good release is
        recorded, or the file on disk holds a catalog of the same backend kind as the release it
        is supposed to serve - it is left alone so the caller's retry surfaces the failure
        again. Only a foreign kind in that file is a wedge this runtime owns, because only this
        runtime writes that file, and it never published a catalog of a kind the router refused.
        The decorated fields a healthy publish adds to the file are deliberately ignored: a
        transient 503 against an otherwise-healthy file must not rewrite it. The router re-reads
        its catalog file on every call, so the reload is enough and no restart is involved.
        """
        body = {}
        try:
            body = json.loads(error.read() or b'{}')
        except Exception:
            pass
        code = (body.get('error') or {}).get('code')
        if error.code != 503 or code not in (None, 'INVALID_BACKEND_KIND'):
            return False
        good = self.last_known_good()
        if good is None:
            return False
        catalog = copy.deepcopy(good['bundle']['catalog'])
        path = Path(self.host['router_state_dir']) / 'active-model.json'
        try:
            on_disk = json.loads(path.read_text())
        except Exception:
            return False
        if on_disk.get('backend_kind') == catalog.get('backend_kind'):
            return False
        atomic_json(path, catalog)
        self.admin('reload-config', {})
        return True

    def desired(self, profile=None):
        current = self.load('active.json', {})
        initial = self.host.get('initial_profile')
        # A host's initial_profile may name a configuration later retired; start the default instead.
        selected = profile or current.get('bundle', {}).get('profile') or (
            initial if initial in DAYTIME_PROFILES else DEFAULT_PROFILE)
        return render(self.config_dir, self.host, selected)

    def backend_url(self, cfg):
        return 'http://' + cfg['container_name'] + ':8080'

    def nighttime_container(self):
        return read(self.config_dir / 'profiles' / (NIGHTTIME_PROFILE + '.json'))['container_name']

    def removed_backends(self, bundle, previous_bundle):
        """Existing containers that must not run beside `bundle`, so they release their GPUs first.

        These are the previous release's services that the target omits, plus any leftover
        Nighttime container when the target is exclusive.
        """
        services = services_of(bundle)
        names = [cfg['container_name'] for role, cfg in services_of(previous_bundle).items() if role not in services]
        if 'everyday' not in services:
            names.append(self.nighttime_container())
        return [name for name in dict.fromkeys(names) if self.system.inspect(name)]

    def release(self, names):
        """Stop, then remove, so a restart policy can never bring a backend back onto reassigned GPUs."""
        for name in names:
            if self.system.inspect(name):
                self.system.docker('stop', '--time', '60', name, timeout=180)
                self.system.docker('rm', name)
            require(not self.system.inspect(name), 'Backend could not be removed: ' + name)

    def provisioning(self, bundle):
        """Return (role, engine, contract) for a service that builds its own state on first start.

        The switch now prepares such a workspace itself before it drains, so the readiness window
        only has to cover the load, and the load is waited on for the engine's actual answer.
        """
        for role, cfg in bundle['compose']['services'].items():
            engine = service_engine(bundle, role)
            if engine.get('provisioning'):
                return role, engine, engine['provisioning']
        return None, None, None

    def provisioned(self, bundle):
        """The measured receipt for this bundle's workspace, or None if it has never been measured.

        A receipt only speaks for the engine image and the workspace directory it was taken with, so
        both have to match before it counts as evidence about the thing about to be switched to.
        """
        role, engine, provision = self.provisioning(bundle)
        if engine is None:
            return None
        receipt = self.load('provisioning/' + bundle['profile'] + '.json')
        if not receipt:
            return None
        source = next((v['source'] for v in bundle['compose']['services'][role]['volumes']
                       if v['target'] == provision.get('workspace_target')), None)
        if (receipt.get('image_id') != engine['image_id'] or receipt.get('workspace') != source
                or receipt.get('engine_tag') != engine['tag']):
            return None
        return receipt

    def preparation_state(self, bundle):
        """(role, engine, contract, workspace, cfg) for the service that builds its own state,
        read without touching anything, or None when the bundle has no such service."""
        role, engine, contract = self.provisioning(bundle)
        if role is None:
            return None
        cfg = bundle['compose']['services'][role]
        return role, engine, contract, provision_tool.workspace_target(cfg, contract), cfg

    def preparation_needed(self, bundle):
        """Whether the switch must build this bundle's workspace before it takes the drain.

        A workspace that already holds everything the engine needs is skipped without ceremony:
        the common case after the first switch is a comparison, not a build.
        """
        found = self.preparation_state(bundle)
        if found is None or found[3] is None:
            return False
        role, engine, contract, workspace, cfg = found
        return not provision_tool.workspace_ready(provision_tool.ready_files(workspace, contract))

    def ensure_provisioned(self, bundle, log=lambda message: None):
        """The half of a first start that is safe beside a loaded model, done before the drain.

        The trial host measured the split: while 94 GB was verified and packed beside a serving
        model, its slots answered in 0.003 s against a 0.002 s baseline and it never left health;
        the load that follows the pack is the half that cannot run beside the model it replaces,
        and it waits for the drain. So a switch to a self-provisioning engine does the safe half
        here, in plain sight, before it has asked for anything. If the engine cannot build its
        workspace, the switch stops at the edge of the transaction: no drain, no touched backend,
        and the reason the engine gave is the reason the operator gets.
        """
        found = self.preparation_state(bundle)
        require(found is not None and found[3] is not None,
            'This bundle declares no provisionable workspace')
        role, engine, contract, workspace, cfg = found
        name = cfg['container_name'] + '-provision'
        port = provision_tool.published_port(cfg, provision_tool.PORT_OFFSET)
        files = provision_tool.ready_files(workspace, contract)
        free_before = provision_tool.free_bytes(workspace)
        spec = self.state / 'provisioning' / (bundle['profile'] + '-compose.json')
        provision_tool.atomic_json(spec, provision_tool.throwaway(bundle, role, name, port))
        result = provision_tool.watch(self.system, spec, role, name, port, files,
            serving=provision_tool.serving_containers(self), goal='workspace',
            stop=provision_tool.stop_seconds(cfg), log=log)
        if result['reached'] != 'workspace':
            words = (result.get('container_log') or '').strip().splitlines()
            last_line = words[-1] if words else 'no output was captured from the container'
            raise RuntimeError(bundle['profile'] + ': the engine could not build its workspace '
                '(' + result['reached'] + ' after ' + str(result['seconds']) + 's): ' + last_line)
        self._record_provision_run(bundle, role, engine, contract, workspace, name, port, result, free_before)
        return result

    def _record_provision_run(self, bundle, role, engine, contract, workspace, name, port, result, free_before):
        """The build the switch just did, in the same receipt the read-only measurement writes, so
        one file is the whole story of the workspace on this machine."""
        receipt = self.load('provisioning/' + bundle['profile'] + '.json') or {
            'profile': bundle['profile'], 'engine_tag': engine['tag'], 'image_id': engine['image_id'],
            'workspace': str(workspace), 'ready_files': list(contract.get('ready_files', [])), 'runs': []}
        receipt['measured_at'] = now()
        run = {'phase': 'provision', 'reached': result['reached'], 'seconds': result['seconds'],
            'container': name, 'port': port, 'via': 'switch',
            'pack_bytes': provision_tool.footprint(workspace),
            'free_bytes_before': free_before, 'free_bytes_after': provision_tool.free_bytes(workspace),
            'ready_paths': result['ready_paths'], 'samples': result['samples']}
        if result.get('container_log') is not None:
            run['container_log'] = result['container_log']
        receipt['runs'].append(run)
        receipt['pack_bytes'] = run['pack_bytes']
        receipt['provision_seconds'] = result['seconds']
        receipt['workspace_ready'] = result['workspace_ready']
        receipt['disk_free_bytes'] = {'before': free_before, 'after': run['free_bytes_after']}
        nearby = [sample['neighbours'] for sample in result['samples'] if sample.get('neighbours')]
        receipt['neighbour_healthy'] = bool(nearby) and all(n['health'] == 'healthy'
            for sample in nearby for n in sample)
        answered = [n['slots_seconds'] for sample in nearby for n in sample
                    if n.get('slots_seconds') is not None]
        receipt['neighbour_slots_max_seconds'] = max(answered) if answered else None
        receipt['neighbour_samples'] = nearby[-8:]
        atomic_json(self.state / 'provisioning' / (bundle['profile'] + '.json'), receipt)

    def _record_warm_boot(self, bundle, roles, seconds):
        """The boot the switch just measured, where the next refusal can read it as evidence.

        A successful switch is the warm boot: the engine loaded from its own workspace with no
        other model on the cards and answered the bounded generation, so the number the switch
        waited is the number the window has to cover.
        """
        found = self.preparation_state(bundle)
        if found is None or found[0] not in roles:
            return
        role, engine, contract, workspace, cfg = found
        receipt = self.load('provisioning/' + bundle['profile'] + '.json') or {
            'profile': bundle['profile'], 'engine_tag': engine['tag'], 'image_id': engine['image_id'],
            'workspace': str(workspace), 'ready_files': list(contract.get('ready_files', [])), 'runs': []}
        receipt['measured_at'] = now()
        receipt['runs'].append({'phase': 'warm', 'reached': 'healthy', 'seconds': seconds,
            'via': 'switch', 'container': cfg['container_name']})
        receipt['warm_boot_seconds'] = seconds
        receipt['healthy_warm_boot'] = True
        receipt['workspace_ready'] = True
        atomic_json(self.state / 'provisioning' / (bundle['profile'] + '.json'), receipt)

    def verdict(self, bundle):
        """The hard reasons a switch to this bundle is refused, and nothing softer.

        A refusal is evidence the machine cannot do the job: a warm boot that was measured and
        failed, a container that stopped itself at startup, or a backend kind the deployed router
        cannot serve. Everything else — a workspace that still needs building, a window that is
        thin against a measurement — is a note the gate shows, because the switch now builds the
        workspace itself and waits for the engine's actual answer, and the recovery path stands
        behind a switch that turns out to be wrong.
        """
        reasons = []
        kind = bundle['catalog'].get('backend_kind')
        kinds = self.router_supported_kinds()
        if kind and kinds is not None and kind not in kinds:
            reasons.append('The deployed router does not support backend kind ' + kind +
                '; a switch to this profile is refused until the router is updated to a release that does')
        if self.preparation_state(bundle) is None:
            return reasons
        receipt = self.provisioned(bundle)
        if receipt is None:
            return reasons
        last = (receipt.get('runs') or [{}])[-1]
        phase = str(last.get('phase') or '')
        reached = str(last.get('reached') or '')
        if phase == 'warm' and reached != 'healthy':
            detail = str(receipt.get('error') or 'see the receipt')
            reasons.append('This engine has never completed a warm boot from its own workspace on this '
                'host (' + detail + '); the switch is refused until it does, or the profile is retired')
        if phase == 'provision' and reached.startswith('exited'):
            role, engine, contract, workspace, cfg = self.preparation_state(bundle)
            reasons.append('The ' + engine['tag'] + ' container stopped by itself at startup ' + reached +
                '; its output is in the receipt. A switch would refuse to build the workspace for '
                'this engine until that is fixed, and the profile is better retired if it cannot be')
        return reasons

    def router_supported_kinds(self):
        """The backend kinds the deployed router can serve, or None if it cannot be asked.

        Cached briefly because the status page asks about every selectable profile; the switch
        itself verifies fresh, so this is the display layer and a stale entry costs at most a
        retry after a router update.
        """
        # Wall-clock, not monotonic: the switch's drain accounting owns the monotonic clock,
        # and this cache must not steal a reading from it.
        cached = getattr(self, '_kinds_cache', None)
        if cached is not None and (
                datetime.datetime.now(datetime.timezone.utc) - cached[0]).total_seconds() < 30:
            return cached[1]
        kinds = None
        try:
            state = self.admin('runtime-state')
            reported = (state.get('router') or {}).get('supported_backend_kinds')
            # A router old enough not to report its kinds supports exactly the two kinds it has
            # always served; a catalog of anything else is refused below, as it should be.
            kinds = list(reported) if reported else ['ollama', 'llama_cpp']
        except Exception:
            kinds = None
        self._kinds_cache = (datetime.datetime.now(datetime.timezone.utc), kinds)
        return kinds

    def require_router_supports(self, bundle):
        """Refuse, before the drain, a catalog the deployed router cannot serve.

        Publishing a kind the router does not implement wedges the router's own health and admin
        paths, and because the wedged catalog is a file the router reads back on boot, a restart
        cannot clear it: the host would come up serving nothing, and only a hand-written file
        would put it back. Refusing at the edge is the only way a switch cannot leave that state
        behind, so the check fails closed - a router that cannot be asked cannot be switched.
        """
        kind = bundle['catalog'].get('backend_kind')
        if not kind:
            return
        try:
            state = self.admin_healing('runtime-state')
        except Exception as error:
            require(False, 'The router cannot be reached, so this switch cannot verify that its backend '
                'kind ' + kind + ' is supported: ' + str(error))
        reported = (state.get('router') or {}).get('supported_backend_kinds')
        supported = tuple(reported) if reported else ('ollama', 'llama_cpp')
        require(kind in supported,
            'The deployed router does not support backend kind ' + kind + ': publishing this catalog would '
            'leave the router unable to serve the configuration or to take administration, and a restart '
            'would not clear it. The switch is refused before the drain; update the router to a release '
            'that supports the kind, or the profile must name a supported kind')

    def verdict_notes(self, bundle):
        """What a switch to this bundle will do that is worth knowing before it is selected.

        These never refuse; they are the difference between selecting a profile and knowing what
        the selection costs.
        """
        found = self.preparation_state(bundle)
        if found is None:
            return []
        role, engine, contract, workspace, cfg = found
        notes = []
        receipt = self.provisioned(bundle)
        if workspace is None or not provision_tool.workspace_ready(
                provision_tool.ready_files(workspace, contract)):
            note = ('The ' + engine['tag'] + ' workspace has not been built on this host; the '
                    'switch will build it first, beside the model that is serving, before it '
                    'takes the drain')
            if receipt:
                built = [run for run in reversed(receipt.get('runs', []))
                         if run.get('phase') == 'provision' and run.get('reached') == 'workspace']
                if built:
                    note += ' (the last build here took ' + str(built[0]['seconds']) + 's)'
            notes.append(note)
        if receipt and receipt.get('healthy_warm_boot') is True:
            window = engine.get('readiness_wait_seconds') or 0
            measured = float(receipt.get('warm_boot_seconds') or 0)
            if measured and window < 3 * measured:
                notes.append('The readiness window for ' + engine['tag'] + ' is ' + str(window) +
                    's, under three times the measured warm boot of '
                    + str(receipt['warm_boot_seconds']) + 's; raise readiness_wait_seconds to at '
                    + 'least ' + str(int(3 * measured + 1)) + 's')
        return notes

    def switchable(self, bundle):
        """Refuse a switch the evidence says will not work; let the rest proceed and prepare.

        The refusal used to be paperwork — a window no one had measured. After the switch learned
        to build the workspace itself and wait for the engine's actual answer, the refusals are:
        a warm boot that was measured and failed, an engine that stops itself at startup, and a
        backend kind the deployed router cannot serve - the one refusal the evidence cannot catch
        afterwards, because publishing it wedges the router in a state a restart cannot clear.
        """
        reasons = self.verdict(bundle)
        if reasons:
            raise RuntimeError(reasons[0])
        # The verdict's kind check is the display layer and may be a second stale; the switch
        # itself verifies against the router fresh, and fails closed if it cannot.
        self.require_router_supports(bundle)

    def validate(self, bundle, full_hash=False):
        images = {}
        for role, cfg in bundle['compose']['services'].items():
            engine = service_engine(bundle, role)
            require(cfg['image'] == engine['tag'], role + ': inference engine tag differs from manifest')
            image = json.loads(self.system.docker('image', 'inspect', engine['tag']))[0]
            require(image['Id'] == engine['image_id'] and (image['Config'].get('Labels') or {}).get(
                'org.opencontainers.image.revision') == engine['revision'], role + ': pinned inference engine identity changed')
            images[role] = image['Id']
        network = bundle['compose']['networks']['router']['name']
        self.system.docker('network', 'inspect', network)
        cache = self.load('artifact-receipts.json', {})
        receipts = {}
        for artifact in bundle['artifacts']:
            path = Path(artifact['source'])
            require(path.is_file(), 'Required model artifact is missing: ' + path.name)
            require(path.resolve().is_relative_to(Path(self.host['model_root']).resolve()), 'Artifact symlink escapes model root')
            stat = path.stat()
            require(stat.st_size == artifact['bytes'], 'Model artifact size changed: ' + path.name)
            fingerprint = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                'ctime_ns': stat.st_ctime_ns, 'inode': stat.st_ino}
            prior = cache.get(str(path), {})
            verified = prior.get('verified') is True and prior.get('fingerprint') == fingerprint and prior.get('sha256') == artifact['sha256']
            if full_hash or (not self.system.readonly and not verified):
                sha = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        sha.update(chunk)
                require(sha.hexdigest() == artifact['sha256'], 'Model artifact checksum mismatch: ' + path.name)
                after = path.stat()
                require((stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino) ==
                    (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino), 'Artifact changed while hashing')
                verified = True
            receipts[str(path)] = {'fingerprint': fingerprint, 'sha256': artifact['sha256'], 'verified': verified}
        if not self.system.readonly:
            self.save('artifact-receipts.json', {**cache, **receipts})
        return {'image_ids': images, 'artifacts': len(receipts),
            'all_checksums_verified': all(x['verified'] for x in receipts.values()), 'receipts': receipts}

    def observe(self, bundle):
        observed = {}
        network = bundle['compose']['networks']['router']['name']
        for role, cfg in bundle['compose']['services'].items():
            engine = service_engine(bundle, role)
            ci = self.system.inspect(cfg['container_name'])
            reasons = []
            result = {'container_name': cfg['container_name'], 'healthy': False,
                'running': bool(ci and ci['State']['Running']), 'differences': reasons}
            if not ci:
                reasons.append('container missing')
                observed[role] = result
                continue
            result.update(id=ci['Id'], started_at=ci['State']['StartedAt'],
                restart_count=ci['RestartCount'], image_id=ci['Image'])
            if not result['running']:
                # A stopped container answers nothing, and every identity check above passes for it:
                # without this line the observer would report it unhealthy with no reason at all,
                # which is exactly the record the 18:47 trial left — an engine that died mid-load
                # reads as "nothing wrong", and the switch's window is never the answer to "why".
                reasons.append('container not running')
            # Every expectation is read with .get(): a service dictionary that lost a key is a
            # configuration fault, and the operator needs to see which setting drifted, not a crash in
            # the middle of a switch. A missing key never matches, so it falls through as a reason.
            for valid, reason in [
                (ci['Image'] == engine['image_id'], 'image'),
                ((ci['Config'].get('Cmd') or []) == (cfg.get('command') or []), 'arguments'),
                (ci['Config'].get('Entrypoint') == cfg.get('entrypoint'), 'entrypoint'),
                # A service that runs as the image's own user renders no user at all and the daemon
                # reports that as an empty string; comparing them literally would call such a
                # container drifted on every check, which would schedule pointless restarts.
                ((ci['Config'].get('User') or None) == cfg.get('user'), 'user'),
                (ci['HostConfig'].get('ReadonlyRootfs') == cfg.get('read_only'), 'read-only root'),
                (ci['HostConfig'].get('RestartPolicy', {}).get('Name') == cfg.get('restart'), 'restart policy'),
                (ci['HostConfig'].get('Init') == cfg.get('init'), 'init'),
                (ci['HostConfig'].get('Privileged', False) is False, 'privileged mode')]:
                if not valid:
                    reasons.append(reason)
            requests = ci['HostConfig'].get('DeviceRequests') or []
            reserved = cfg.get('deploy', {}).get('resources', {}).get('reservations', {})
            expected_ids = (reserved.get('devices') or [{}])[0].get('device_ids')
            if len(requests) != 1 or requests[0].get('DeviceIDs') != expected_ids:
                reasons.append('GPU assignment')
            expected_cuda = cfg.get('environment', {}).get('CUDA_VISIBLE_DEVICES')
            live_env = dict(item.split('=', 1) for item in ci['Config'].get('Env', []) if '=' in item)
            if live_env.get('CUDA_VISIBLE_DEVICES') != expected_cuda:
                reasons.append('CUDA device order')
            if expected_cuda and (len(requests) != 1 or requests[0].get('Driver') != 'nvidia'
                    or requests[0].get('Count') != 0 or requests[0].get('Capabilities') != [['gpu']]):
                reasons.append('GPU reservation contract')
            mounts = {(x.get('Source'), x.get('Destination'), x.get('RW'))
                for x in ci.get('Mounts') or [] if x.get('Type') == 'bind'}
            if mounts != {(x.get('source'), x.get('target'), not x.get('read_only', True))
                    for x in cfg.get('volumes') or []}:
                reasons.append('model mounts')
            ports = cfg.get('ports') or []
            host_port = ports[0].split(':')[1] if ports else None
            if not host_port or ci['HostConfig'].get('PortBindings') != {
                    '8080/tcp': [{'HostIp': '127.0.0.1', 'HostPort': host_port}]}:
                reasons.append('port bindings')
            connection = (ci.get('NetworkSettings') or {}).get('Networks', {}).get(network, {})
            aliases = ((cfg.get('networks') or {}).get('router') or {}).get('aliases') or []
            if not aliases or not connection or not set(aliases).issubset(connection.get('Aliases') or []):
                reasons.append('router network')
            if result['running']:
                try:
                    health = self.system.http(self.backend_url(cfg) + '/health', timeout=3)
                    slots = self.system.http(self.backend_url(cfg) + '/slots', timeout=3)
                    service = next(x for x in bundle['manifest']['services'] if x['role'] == role)
                    if engine.get('interface') == 'environment':
                        # Both /health and /slots answer in llama.cpp's shape, but there is no argument
                        # vector to read the window from: the published configuration is the reference.
                        expected_ctx, expected_slots = service['context_tokens'], service['parallel_slots']
                    else:
                        argv = cfg['command']
                        expected_ctx = int(argv[argv.index('--kv-unified-per-slot') + 1])
                        expected_slots = int(argv[argv.index('--parallel') + 1])
                    reported = [x.get('n_ctx') for x in slots if isinstance(x, dict)] if isinstance(slots, list) else None
                    if (not isinstance(slots, list) or len(slots) != expected_slots
                            or len(reported) != expected_slots
                            or any(x != expected_ctx for x in reported)):
                        # What the backend answered goes in the record: a window smaller than the one the
                        # profile contracted for is permanent and needs reading, while an empty slot list
                        # is a backend that is still loading. Both look the same as a waiting engine
                        # otherwise, and the switch would sit out the whole health timeout to find out.
                        result['slot_window'] = {'contracted': expected_ctx, 'slots': expected_slots,
                            'reported_context': reported, 'loaded': health.get('loaded')
                                if isinstance(health, dict) else None}
                        reasons.append('slot context')
                    result['healthy'] = isinstance(health, dict) and health.get('status') == 'ok' and not reasons
                    result['processing'] = isinstance(slots, list) and any(
                        x.get('is_processing', False) for x in slots if isinstance(x, dict))
                except Exception:
                    result['health_error'] = 'Backend readiness unavailable'
            observed[role] = result
        if 'everyday' not in bundle['compose']['services'] and 'coding' in observed:
            night = self.system.inspect(self.nighttime_container())
            if night and night['State']['Running']:
                observed['coding']['differences'].append('competing Nighttime backend running')
                observed['coding']['healthy'] = False
        legacy = self.system.inspect('local-ai-llama-cpp')
        require(not legacy or not legacy['State']['Running'], 'Legacy inference backend is running; refusing competing GPU ownership')
        return observed

    def inspect(self, profile=None, full_hash=False):
        bundle = self.desired(profile)
        validation = self.validate(bundle, full_hash=full_hash)
        services = self.observe(bundle)
        router = self.admin('runtime-state')
        return {'mode': 'inspection', 'revision': self.revision, 'profile': bundle['profile'],
            'config_sha256': bundle['config_sha256'], 'validation': validation, 'services': services,
            'matches_live': all(s['healthy'] and not s['differences'] for s in services.values()),
            'router_draining': router['runtime']['draining'],
            'active_requests': router['runtime']['active_count'],
            'queued_requests': router['runtime'].get('queued_count', 0)}

    def drain_pass(self, configs):
        """One look at the router and every running backend: (work the router relays, notes).

        The router's own counters are the authority on whether a generation is being served to
        anyone. A backend is only evidence of work when it answers `/slots` and a slot says it is
        processing: a backend whose `/slots` cannot be fetched is serving nothing, and a response
        this code cannot read is not evidence either. The trial's second failure was here — the
        engine that replaced llama.cpp answered no port at all, `except` pinned the pass noisy
        forever, and the drain check could never pass when that failed backend was the only
        container left standing. That is the exact case restoration exists for, so an unreachable
        backend is recorded and never holds the drain open.
        """
        state = self.admin('runtime-state')['runtime']
        relayed = int(state['active_count'] or 0) + int(state.get('queued_count', 0) or 0)
        notes = {'unreachable': [], 'busy': [], 'malformed': []}
        for name, cfg in configs.items():
            ci = self.system.inspect(name)
            if not ci or not ci['State']['Running']:
                continue
            try:
                slots = self.system.http(self.backend_url(cfg) + '/slots', timeout=5)
            except Exception:
                notes['unreachable'].append(name)
                continue
            if not isinstance(slots, list):
                notes['malformed'].append(name)
                continue
            if any(x.get('is_processing', False) for x in slots if isinstance(x, dict)):
                notes['busy'].append(name)
        return relayed, notes

    def wait_idle(self, bundles, timeout=300, force_on_deadline=False):
        """Wait until nothing is in flight, and return what the last pass saw.

        `force_on_deadline` is for a caller already committed to bringing a backend back. The
        point of a drain is to not interrupt work in flight; once the deadline passes and the
        router is relaying nothing, there is nobody left to be polite to and the router already
        rejects everything, so the only remaining outcome is to continue. Aborting there would
        leave the host drained with nothing serving — which is what happened at 01:12 on
        2026-10-12 and stayed that way until someone rebooted the machine.

        Without it, a deadline is a refusal before anything is touched: the router still reports
        work, or a reachable backend is generating for a client of its own (a direct request to a
        backend port is a legitimate thing on this host), and the caller has not destroyed
        anything yet, so backing away is the polite and correct answer.
        """
        configs = {cfg['container_name']: cfg for b in bundles if b for cfg in b['compose']['services'].values()}
        deadline, stable = time.monotonic() + timeout, 0
        while time.monotonic() < deadline:
            relayed, notes = self.drain_pass(configs)
            quiet = relayed == 0 and not notes['busy']
            stable = stable + 1 if quiet else 0
            if stable >= DRAIN_SETTLE_ROUNDS:
                return notes
            time.sleep(1)
        if force_on_deadline:
            relayed, notes = self.drain_pass(configs)
            if relayed == 0:
                return {**notes, 'forced': True}
        raise RuntimeError('Drain timed out; no active generation was stopped')

    def compose_path(self, bundle):
        folder = self.state / 'generated' / bundle['config_sha256']
        self.save(str(folder.relative_to(self.state) / 'compose.json'), bundle['compose'])
        return folder / 'compose.json'

    def prepare(self, bundle):
        self.validate(bundle)
        path = self.compose_path(bundle)
        self.system.docker('compose', '-p', bundle['compose']['name'], '-f', str(path), 'config', '--quiet')

    def reconcile(self, bundle, roles, sequential=False):
        """Start the changed roles; across a topology change Daytime starts first, then Nighttime,
        so the outgoing Daytime has released every GPU before Nighttime loads."""
        if roles:
            path = self.compose_path(bundle)
            batches = [[role] for role in sorted(roles, key=lambda role: role != 'coding')] if sequential else [roles]
            for batch in batches:
                # Engines that prepare their own state before they answer the first health check
                # (Strata builds its expert pack and installs its draft layer on a fresh workspace)
                # name their own bounded readiness window; everything else keeps the usual one.
                wait = max([service_engine(bundle, role).get('readiness_wait_seconds', 450)
                    for role in batch])
                self.system.docker('compose', '-p', bundle['compose']['name'], '-f', str(path),
                    'up', '-d', '--no-deps', '--pull', 'never', '--wait', '--wait-timeout', str(wait), *batch,
                    timeout=wait + 60)
        observed = self.observe(bundle)
        # Docker's health state is the engine's own /health, and an engine whose server answers
        # before the model is resident (Strata's /health says status ok from server start, carries
        # the model state as loaded, and /slots stays empty until the read completes) is healthy
        # for Docker while it is still reading: up --wait came back long before the model did, and
        # the one-shot observe above rejected it while the read was mid-flight. For those, the
        # declared readiness window is the bound on waiting for the controller's own view. A
        # backend that loaded with the wrong window, or drifted in any other way, is not waited
        # out: the window would only turn the fault into a delay.
        if roles:
            wait = max([service_engine(bundle, role).get('readiness_wait_seconds', 450) for role in roles])
            deadline = time.monotonic() + wait
            while not all(s['healthy'] for s in observed.values()) and self.still_loading(observed) \
                    and time.monotonic() < deadline:
                time.sleep(READINESS_POLL_SECONDS)
                observed = self.observe(bundle)
        bad = [(role, s) for role, s in observed.items() if not s['healthy']]
        for role, s in bad:
            # The record must carry the witness while the container still exists: restoration
            # removes it, and an engine that stopped by itself has no health and no slots left to
            # explain why. Its exit code and its own last words are the only account of the
            # failure, so they are read here, before anything can delete them.
            ci = self.system.inspect(s.get('container_name') or '')
            if ci:
                if not ci['State']['Running']:
                    s['exit_code'] = ci['State'].get('ExitCode')
                try:
                    words = self.system.docker('logs', '--tail', '120', s['container_name'], check=False)
                    lines = [line.strip() for line in words.splitlines() if line.strip()]
                    s['last_words'] = [line[:180] for line in lines[-10:]]
                except Exception:
                    pass
        detail = '; '.join(role + ': ' + ', '.join(
            s['differences']
            + (['slot window ' + json.dumps(s['slot_window'])] if s.get('slot_window') else [])
            + (['health: ' + s['health_error']] if s.get('health_error') else [])
            + (['exit ' + str(s['exit_code'])] if s.get('exit_code') is not None else [])
            + (['last words: ' + ' | '.join(s['last_words'])] if s.get('last_words') else []))
            for role, s in bad)
        require(not bad, 'Backend identity/readiness verification failed' + (': ' + detail if detail else ''))
        return observed

    @staticmethod
    def still_loading(observed):
        """The one unhealthy state a readiness window can fix: an engine that answers its own
        health check before the model is resident. Strata's /health says status ok from server
        start and carries the model state as loaded; /slots stays empty until the read completes,
        so the observer records slot context with loaded=false. That is a backend mid-load, and
        waiting within the declared window is the fix. The same empty list with loaded=true, or
        any other difference at all, is a fault the window will not fix: saying so here is what
        keeps the transition from spending the window on it."""
        for service in observed.values():
            if service['healthy']:
                continue
            window = service.get('slot_window')
            if window is None or 'slot context' not in service['differences']:
                return False
            if any(reason != 'slot context' for reason in service['differences']):
                return False
            if window.get('loaded') is not False:
                return False
        return bool(observed)

    def acceptance(self, bundle, roles):
        for role in roles:
            cfg = bundle['compose']['services'][role]
            if service_engine(bundle, role).get('interface') == 'environment':
                # The native /completion probe belongs to llama.cpp; the same bounded-generation check
                # runs against the OpenAI chat dialect every engine in this house has to serve. A
                # reasoning model may spend its whole budget on thinking, so either channel counts.
                answer = self.system.http(self.backend_url(cfg) + '/v1/chat/completions', {
                    'messages': [{'role': 'user', 'content': 'Complete the word: O'}],
                    'max_tokens': 8, 'temperature': 0, 'stream': False}, timeout=120)
                choices = answer.get('choices') or []
                message = (choices[0].get('message') or {}) if choices else {}
                said = ' '.join(message.get(key) or '' for key in ('content', 'reasoning_content')).strip()
                require(bool(said), 'Bounded generation acceptance failed for ' + role)
                continue
            answer = self.system.http(self.backend_url(cfg) + '/completion', {
                'prompt': 'Complete the word: O', 'n_predict': 8, 'temperature': 0,
                'stream': False, 'cache_prompt': False}, timeout=90)
            require(isinstance(answer.get('content'), str) and bool(answer['content'].strip()),
                'Bounded generation acceptance failed for ' + role)
        self.wait_idle([bundle])

    def publish(self, bundle):
        observed = self.observe(bundle)
        require(all(x['healthy'] for x in observed.values()), 'Refusing to publish an unverified backend catalog')
        catalog = copy.deepcopy(bundle['catalog'])
        for entry in catalog['models']:
            service = next(x for x in bundle['manifest']['services'] if x['model_alias'] == entry['model'])
            ci = self.system.inspect(service['container_name'])
            entry['updated_at'] = now()
            if service_engine(bundle, service['role']).get('interface') == 'environment':
                # No argument vector to read an output ceiling from: this engine has none, and what
                # was inspected is its environment, so the record says so instead of naming argv.
                entry['runtime_output_policy'] = {'verification': 'docker-inspect-env',
                    'engine_sets_no_output_ceiling': True, 'verified_at': now(), 'container_id': ci['Id'],
                    'started_at': ci['State']['StartedAt'],
                    'env_sha256': hashlib.sha256(json.dumps(sorted(ci['Config'].get('Env') or [])).encode()).hexdigest()}
                continue
            entry['runtime_output_policy'] = {'n_predict': -1, 'reasoning_budget': -1,
                'reasoning_effort': 'default', 'verification': 'docker-inspect-argv',
                'verified_at': now(), 'container_id': ci['Id'], 'started_at': ci['State']['StartedAt'],
                'argv_sha256': hashlib.sha256(json.dumps(ci['Config']['Cmd']).encode()).hexdigest()}
        catalog.update(catalog['models'][0])
        # Identify the live configuration for router clients. Bundles rendered before this field
        # existed (a recovered previous release) still get an accurate id and exclusivity.
        configuration = catalog.get('configuration') or {'id': bundle['profile'],
            'exclusive': 'everyday' not in bundle['compose']['services']}
        catalog['configuration'] = {**configuration, 'runtime_revision': self.revision, 'published_at': now()}
        require(not self.system.readonly, 'Publication forbidden in inspection mode')
        catalog_path = Path(self.host['router_state_dir']) / 'active-model.json'
        previous_catalog = read(catalog_path) if catalog_path.exists() else None
        try:
            atomic_json(catalog_path, catalog)
            self.admin('reload-config', {})
            state = self.admin('runtime-state')
            models = state.get('models', [])
            expected = {m['model']: m['context_length'] for m in catalog['models']}
            require({m['id'] for m in models} == set(expected) and all(
                m['x_ollama_router']['health']['available'] and m['x_ollama_router']['output_policy'] == 'unrestricted'
                and m['x_ollama_router']['context_window'] == expected[m['id']]
                and m['x_ollama_router'].get('default_output_tokens') is None
                and m['x_ollama_router'].get('max_output_tokens') is None for m in models),
                'Router catalog/readiness verification failed')
        except BaseException:
            # An unsupported catalog can break runtime-state itself. Restore the exact prior
            # catalog before recovery asks who owns the drain; leave admissions and release
            # receipts alone until backend recovery verifies the previous release.
            if previous_catalog is not None:
                atomic_json(catalog_path, previous_catalog)
                try:
                    self.admin('reload-config', {})
                except Exception:
                    pass  # The original failure remains the diagnostic; recovery will retry.
            else:
                catalog_path.unlink(missing_ok=True)
            raise

    def set_drain(self, enabled, transaction):
        return self.admin_healing('runtime-drain', {'enabled': enabled,
            'reason': 'local-ai-runtime:' + transaction['id'] if enabled else 'runtime ready'})

    def owns_drain(self, transaction):
        state = self.admin_healing('runtime-state')['runtime']
        return state['draining'] and state.get('drain_reason', state.get('reason')) == 'local-ai-runtime:' + transaction['id']

    def mark(self, transaction, phase, step=None):
        """Write the phase a pass has reached into the transaction before doing anything else.

        A restoration is the one path whose record has to be readable afterwards without a
        journal: it runs while the machine is not serving, so every step it reached is recorded
        beside the phase, and the transaction is the evidence of what was attempted.
        """
        transaction['phase'] = phase
        if step is not None:
            transaction.setdefault('restoration', {}).setdefault('phases', []).append({'step': step, 'at': now()})
        self.save('transaction.json', transaction)

    def finish(self, transaction, state, error=None, serving_profile=None):
        transaction.update(phase=state, finished_at=now(), error=error)
        self.save('transaction.json', transaction)
        # `phase` and `error` are what a person reads first, and `intent` says which kind of attempt
        # this was: a failed switch and the restoration that put the machine back afterwards
        # otherwise look identical in this file.
        record = {k: transaction.get(k) for k in
            ('id', 'intent', 'phase', 'profile', 'started_at', 'finished_at', 'error', 'changed_roles', 'revision', 'config_sha256')}
        record['intent'] = transaction.get('intent') or 'deployment'
        self.save('last-deployment.json', record)
        if state in ('succeeded', 'recovered'):
            # A verified release is serving, so nothing further is owed. The restoration record
            # stays on disk as history; it simply stops being an action waiting to happen.
            # A recovered switch restored its previous release, not the target the transaction
            # records, so the note names the release that is actually serving.
            serving = serving_profile or transaction.get('profile')
            self.clear_restoration('The runtime is serving ' + str(serving) + ' and requests are open.')

    def last_known_good(self):
        """The release this host was last known serving, which is what a restoration brings back.

        `active.json` is written only after a deployment has verified its backends and reopened
        admissions, so a failed switch has not moved it: what served is still what the record
        says. That is why the boot-time recovery brought llama.cpp back on its own on
        2026-10-12 - after the reboot the controller started what the record named - and it is the
        same lever the in-process path, the CLI and the one-action button pull, not a second
        implementation of it.
        """
        return self.load('active.json') or self.load('previous.json')

    def serving(self, bundle=None):
        """Whether a healthy backend answers and admissions are open.

        This is the state the runtime must come to rest in. It is deliberately conservative: an
        unreadable router or a missing record is not serving.
        """
        try:
            bundle = (bundle if bundle is not None else (self.last_known_good() or {}).get('bundle'))
            if not bundle:
                return False
            if self.admin('runtime-state')['runtime']['draining']:
                return False
            return bool(self.observe(bundle)) and all(s['healthy'] for s in self.observe(bundle).values())
        except Exception:
            return False

    def restoration(self):
        """The durable record that service has to be brought back, or None."""
        return self.load('restoration.json')

    def restoration_pending(self):
        record = self.restoration()
        return bool(record) and not record.get('finished_at')

    def restoration_due(self):
        record = self.restoration()
        if not record or record.get('finished_at'):
            return False
        # Same-format UTC timestamps, so a schedule compares correctly as text.
        return record.get('next_attempt_at', '') <= now()

    def schedule_restoration(self, error, transaction=None, attempted=True):
        """Record that the runtime is not serving and say when the next attempt runs.

        This is what makes a failed recovery an action instead of a condition: the host is left
        with a pending restoration and a time, so nobody has to be there for it. It is retried
        until the service comes back, with the wait capped, because stopping to try is the same
        as nobody being there.
        """
        current = self.restoration() or {}
        if current.get('finished_at'):
            current = {}
        attempts = int(current.get('attempts', 0) or 0) + (1 if attempted else 0)
        delay = RESTORATION_BACKOFF[min(attempts - 1, len(RESTORATION_BACKOFF) - 1)] if attempts else 0
        good = self.last_known_good() or {}
        phases = list(current.get('phases', []))
        phases.append({'step': 'scheduled', 'at': now(), 'after_seconds': delay, 'error': str(error)})
        record = {'id': current.get('id') or str(uuid.uuid4()), 'attempts': attempts,
            'requested_at': current.get('requested_at') or now(), 'updated_at': now(),
            'next_attempt_at': future(delay), 'profile': good.get('bundle', {}).get('profile'),
            'transaction': (transaction or {}).get('id') or current.get('transaction'),
            'reason': str((transaction or {}).get('error') or error), 'last_error': str(error),
            'phases': phases[-RESTORATION_PHASE_LOG:]}
        self.save('restoration.json', record)
        return record

    def clear_restoration(self, note=None):
        record = self.restoration()
        if not record or record.get('finished_at'):
            return record
        record.update(finished_at=now(), phase='restored', note=note)
        self.save('restoration.json', record)
        return record

    def restore_last_known_good(self, operation_id=None, progress=lambda phase: None):
        """Bring the last-known-good release back and reopen admissions, whatever the failure left.

        One action, no reboot, and no lever a person is without: it is the boot-time recovery
        path, callable at any time by the button, the CLI or the supervision loop. It uses the
        interrupted transaction when that transaction owns the router drain - it knows which
        backend it replaced - and otherwise opens its own against the release on record.
        """
        require(not self.system.readonly, 'Mutation forbidden in inspection mode')
        with lock(self.state / 'runtime.lock'):
            return self._restore_locked(operation_id, progress)

    def _restore_locked(self, operation_id=None, progress=lambda phase: None):
        """Caller must hold runtime.lock for the whole restoration, as every transition does."""
        require(not self.system.readonly, 'Mutation forbidden in inspection mode')
        before = self.restoration()
        try:
            return self._restore_attempt_locked(operation_id, progress)
        except Exception as error:
            # Failures before recover_locked (including an unreadable router admin API) must
            # advance the retry timer too. A recovery failure already scheduled its own retry.
            if self.restoration() == before and self.last_known_good() is not None:
                self.schedule_restoration(error, self.load('transaction.json', {}))
            raise

    def _restore_attempt_locked(self, operation_id=None, progress=lambda phase: None):
        transaction = self.load('transaction.json', {})
        if transaction.get('phase') in PENDING_PHASES:
            if self.owns_drain(transaction):
                self.recover_locked(transaction, restoring=True, progress=progress)
                return {'profile': transaction['previous']['bundle']['profile'], 'restored': True,
                    'serving': self.serving()}
            # It stopped before the drain, so no backend was replaced. Close the attempt; the
            # release on record is still the one that served, and that is what is brought up.
            self.finish(transaction, 'failed',
                transaction.get('error') or 'Interrupted before draining; prior runtime preserved')
        good = self.last_known_good()
        require(good is not None, 'No last-known-good release is recorded; operator attention required')
        # A router release reached through the admin API is the one owner of the drain this path
        # must respect. Everything else in this process holds runtime.lock for its whole pass, and
        # the caller holds it now, so a drain that is still set here belongs to an attempt that
        # stopped - which is exactly the flag restoration exists to clear, not a reason to refuse.
        require(not self.load('router-maintenance.json', {}).get('reserved'),
            'A router release reserves the runtime; finish or recover it first')
        if self.serving(good['bundle']):
            self.clear_restoration('A healthy backend is serving and requests are open.')
            return {'profile': good['bundle']['profile'], 'restored': True, 'already_serving': True, 'serving': True}
        transaction = {'id': operation_id or str(uuid.uuid4()), 'started_at': now(), 'phase': 'prepared',
            'intent': 'restore-last-known-good', 'previous': good, 'target': copy.deepcopy(good),
            'changed_roles': [], 'revision': self.revision, 'profile': good['bundle']['profile'],
            'config_sha256': good['bundle']['config_sha256'],
            'restoration': {'operation': operation_id, 'phases': []}}
        self.save('transaction.json', transaction)
        self.set_drain(True, transaction)
        self.recover_locked(transaction, restoring=True, progress=progress)
        return {'profile': good['bundle']['profile'], 'restored': True, 'serving': self.serving()}

    def _recover_locked(self, transaction, progress, force_on_deadline):
        previous = transaction.get('previous')
        require(previous is not None, 'No previous release exists; recovery needs operator attention')
        require(self.owns_drain(transaction), 'Recovery does not own the router drain; refusing to change it')
        old = previous['bundle']
        target = transaction['target']['bundle']
        progress('restoring')
        self.mark(transaction, 'restoring', 'preparing')
        self.prepare(old)
        # A drain deadline must never strand the host: this pass exists to bring service back, and
        # the router already rejects everything, so there is nothing left to be polite to.
        self.wait_idle([old, target], force_on_deadline=force_on_deadline)
        self.mark(transaction, 'restoring', 'draining')
        protected = transaction.get('protected_services', {})
        topology = set(services_of(old)) != set(services_of(target))
        # Leaving an exclusive target frees the GPUs it took before the paired release returns;
        # restoring an exclusive release first removes a Nighttime the failed target started.
        extra = self.removed_backends(old, target)
        require(not extra or not protected, 'Recovery would replace a protected backend; operator attention required')
        self.release(extra)
        self.mark(transaction, 'restoring', 'recreating')
        observed = self.observe(old)
        roles = [role for role, s in observed.items() if not s['healthy']]
        require(not set(roles).intersection(protected), 'Recovery would replace a protected backend; operator attention required')
        after = self.reconcile(old, roles, sequential=topology)
        require(all(after[r].get('id') == cid for r, cid in protected.items()),
            'A protected backend changed during recovery')
        self.acceptance(old, roles)
        self.mark(transaction, 'restoring', 'verifying')
        self.publish(old)
        self.save('active.json', previous)
        self.set_drain(False, transaction)
        self.mark(transaction, 'restoring', 'publishing')
        self.finish(transaction, 'recovered', transaction.get('error'), serving_profile=old['profile'])

    def recover_locked(self, transaction, restoring=False, progress=lambda phase: None):
        """Restore the release a transaction recorded, through the path a healthy boot uses.

        Called by the recovery leg of a failed deployment, by `recover` and the boot supervisor,
        and by the one-action restore. `restoring` marks a pass whose purpose is to bring service
        back: it then refuses nothing the drain can be used to argue about, logs every step into
        the transaction, and schedules the next attempt instead of resting when it fails. Without
        it the pass is the undo a caller may still walk away from.
        """
        if not restoring:
            return self._recover_locked(transaction, progress, False)
        try:
            return self._recover_locked(transaction, progress, True)
        except Exception as error:
            self.schedule_restoration(error, transaction)
            raise

    def transition(self, profile=None, deploy=False, adopt=False):
        require(not self.system.readonly, 'Mutation forbidden in inspection mode')
        with lock(self.state / 'runtime.lock'):
            return self._transition_locked(profile, deploy=deploy, adopt=adopt)

    def _transition_locked(self, profile=None, deploy=False, adopt=False, *, operation_id=None,
                           progress=lambda phase: None, daytime_only=False):
        """Caller must hold runtime.lock for the entire transaction, including recovery."""
        require(not self.system.readonly, 'Mutation forbidden in inspection mode')
        require(not self.load('router-maintenance.json', {}).get('reserved'),
            'A router release reserves the runtime; finish or recover it first')
        pending = self.load('transaction.json', {})
        require(pending.get('phase') not in PENDING_PHASES,
            'An interrupted deployment needs explicit runtime recover before another operation')
        previous = self.load('active.json')
        require(previous is not None or adopt, 'Runtime is not adopted; use the reviewed migration procedure')
        require(not previous or previous['revision'] == self.revision or deploy,
            'This controller has been superseded; it cannot reapply an older release')
        progress('checking')
        bundle = self.desired(profile)
        self.switchable(bundle)
        self.prepare(bundle)
        observed = self.observe(bundle)
        prior = services_of(previous['bundle']) if previous else {}
        roles = [role for role, s in observed.items() if not s['healthy'] or (previous and
            prior.get(role) != bundle['compose']['services'][role])]
        # Selecting or leaving an exclusive profile deliberately removes or restores Nighttime.
        topology = bool(previous) and set(prior) != set(bundle['compose']['services'])
        # So does choosing a paired configuration that names a different Nighttime variant.
        night_change = bool(previous) and not topology and 'everyday' in bundle['compose']['services'] and (
            previous['bundle'].get('nighttime_profile', NIGHTTIME_PROFILE) != bundle.get('nighttime_profile', NIGHTTIME_PROFILE))
        removed = self.removed_backends(bundle, previous['bundle'] if previous else None)
        require(not adopt or not (roles or removed), 'Adoption requires an exact match to healthy existing backends')
        require(not daytime_only or topology or night_change or 'everyday' not in roles,
            'Profile switch would replace Nighttime; operator attention required')
        state = self.admin_healing('runtime-state')['runtime']
        require(not state['draining'], 'Another operation owns the router drain')
        if previous and previous['revision'] == self.revision and previous['bundle'] == bundle and not roles and not removed:
            return {'already_active': True, 'profile': bundle['profile']}
        unchanged = {role: s['id'] for role, s in observed.items() if role not in roles}
        target = {'revision': self.revision, 'image': self.image, 'bundle': bundle, 'applied_at': now()}
        # Adoption's prior runtime is already known healthy and can be restored by this image.
        if adopt:
            previous = copy.deepcopy(target)
        tx = {'id': operation_id or str(uuid.uuid4()), 'started_at': now(), 'phase': 'prepared', 'previous': previous,
            'target': target, 'changed_roles': roles, 'revision': self.revision, 'profile': bundle['profile'],
            'config_sha256': bundle['config_sha256']}
        if removed:
            tx['removed_backends'] = removed
        if daytime_only and not topology and not night_change and 'everyday' in observed:
            tx['protected_services'] = {'everyday': observed['everyday']['id']}
        self.save('transaction.json', tx)
        changed = False
        try:
            if self.preparation_needed(bundle):
                # The workspace build is safe beside the serving model, so it happens before the
                # drain: a failure here stops the switch at its edge, and nothing has been taken.
                self.mark(tx, 'preparing')
                progress('preparing')
                self.ensure_provisioned(bundle)
            self.set_drain(True, tx)
            self.mark(tx, 'draining')
            progress('draining')
            self.wait_idle([bundle, previous['bundle'] if previous else None])
            self.mark(tx, 'applying')
            progress('loading')
            changed = True
            self.release(removed)
            load_started = time.monotonic()
            after = self.reconcile(bundle, roles, sequential=topology)
            require(all(after[r]['id'] == cid for r, cid in unchanged.items()), 'An unchanged backend was unexpectedly recreated')
            self.mark(tx, 'verifying')
            progress('verifying')
            self.acceptance(bundle, roles)
            self._record_warm_boot(bundle, roles, round(time.monotonic() - load_started, 1))
            self.publish(bundle)
            self.save('previous.json', previous)
            self.save('active.json', target)
            self.set_drain(False, tx)
            self.finish(tx, 'succeeded')
            return {'profile': bundle['profile'], 'changed_roles': roles, 'revision': self.revision}
        except BaseException as error:
            tx['error'] = str(error)
            if changed:
                # A backend has already been replaced, so this is no longer an undo the caller can
                # walk away from: it is the restoration, and it runs as one. If it cannot finish,
                # it schedules the next attempt rather than resting with the router drained.
                try:
                    progress('restoring')
                    self.recover_locked(tx, restoring=True)
                except BaseException as recovery:
                    self.finish(tx, 'needs-attention', f'Update failed: {error}; recovery failed: {recovery}')
                    raise RuntimeError('Deployment failed; router remains drained and recovery needs attention') from recovery
            else:
                if self.owns_drain(tx):
                    self.set_drain(False, tx)
                self.finish(tx, 'failed', str(error))
            raise RuntimeError('Deployment failed: ' + str(error)) from error

    def recover(self):
        require(not self.system.readonly, 'Mutation forbidden in inspection mode')
        with lock(self.state / 'runtime.lock'):
            tx = self.load('transaction.json')
            require(tx and tx['phase'] in PENDING_PHASES,
                'There is no interrupted deployment to recover')
            if tx['phase'] == 'prepared' and not self.admin_healing('runtime-state')['runtime']['draining']:
                self.finish(tx, 'failed', 'Interrupted before draining; prior runtime preserved')
            else:
                self.recover_locked(tx, restoring=True)

    def rollback_release(self):
        """Recover a successful runtime apply if the new controller cannot become healthy."""
        require(not self.system.readonly, 'Mutation forbidden in inspection mode')
        with lock(self.state / 'runtime.lock'):
            previous = self.load('previous.json')
            active = self.load('active.json')
            require(previous and active, 'No prior release is available')
            require(active['revision'] == self.revision, 'Refusing rollback from a superseded updater')
            require(not self.admin_healing('runtime-state')['runtime']['draining'], 'Another operation owns the router drain')
            tx = {'id': str(uuid.uuid4()), 'phase': 'prepared', 'started_at': now(), 'previous': previous,
                'target': active, 'revision': self.revision, 'profile': previous['bundle']['profile'],
                'error': 'Replacement controller failed readiness'}
            self.save('transaction.json', tx)
            self.set_drain(True, tx)
            try:
                self.recover_locked(tx, restoring=True)
            except BaseException as error:
                self.finish(tx, 'needs-attention', str(error))
                raise

    def republish(self):
        """Router-only releases may republish, but may not replace runtime source/configuration."""
        with lock(self.state / 'runtime.lock'):
            active = self.load('active.json')
            require(active and active['revision'] == self.revision, 'Controller does not own the deployed revision')
            require(self.load('transaction.json', {}).get('phase') not in PENDING_PHASES,
                'Interrupted runtime transaction blocks publication')
            self.publish(active['bundle'])

    def router_maintenance(self, begin):
        """Reserve runtime configuration across a router-only deployment's process boundary."""
        require(not self.system.readonly, 'Mutation forbidden in inspection mode')
        with lock(self.state / 'runtime.lock'):
            active = self.load('active.json')
            require(active and active['revision'] == self.revision, 'Controller does not own the deployed revision')
            reservation = self.load('router-maintenance.json', {})
            if begin:
                require(not reservation.get('reserved') and not self.admin_healing('runtime-state')['runtime']['draining'],
                    'Another maintenance operation is active')
                require(self.load('transaction.json', {}).get('phase') not in PENDING_PHASES,
                    'Interrupted runtime transaction')
                reservation = {'id': str(uuid.uuid4()), 'reserved': True, 'started_at': now()}
                self.save('router-maintenance.json', reservation)
                try:
                    self.set_drain(True, reservation)
                    self.wait_idle([active['bundle']])
                except BaseException:
                    if self.owns_drain(reservation):
                        self.set_drain(False, reservation)
                    self.save('router-maintenance.json', {**reservation, 'reserved': False, 'failed_at': now()})
                    raise
            else:
                require(reservation.get('reserved') and self.owns_drain(reservation), 'No owned router maintenance reservation')
                self.wait_idle([active['bundle']])
                self.publish(active['bundle'])
                self.set_drain(False, reservation)
                self.save('router-maintenance.json', {**reservation, 'reserved': False, 'finished_at': now()})

    def status(self):
        active = self.load('active.json')
        bundle = active['bundle'] if active else self.desired()
        result = {'revision': self.revision, 'deployed_revision': active['revision'] if active else None,
            'profile': bundle['profile'], 'ready': False, 'services': [],
            'offline_roles': [role for role in ('coding', 'everyday') if role not in services_of(bundle)],
            'last_deployment': self.load('last-deployment.json'), 'updated_at': now()}
        # Read-only presentation of the profile registry. It is reported before the live
        # checks below so an unreadable definition never hides the health of the pair. Each
        # selectable configuration carries the same verdict the read-only gate gives: what a
        # switch to it would do (notes) and what would refuse it (reasons), so the page can
        # show the cost of a selection before the selection is made.
        try:
            configurations = available_profiles(self.config_dir, self.host)
            for entry in configurations.get('selectable', []):
                try:
                    candidate = render(self.config_dir, self.host, entry['profile'])
                    entry['reasons'] = self.verdict(candidate)
                    entry['notes'] = self.verdict_notes(candidate)
                except Exception as error:
                    entry['reasons'] = ['This configuration cannot be rendered: ' + str(error)]
                    entry['notes'] = []
            result['configurations'] = {'active': bundle['profile'], **configurations}
        except Exception as error:
            result['configurations'] = {'active': bundle['profile'], 'selectable': [],
                'always_included': [], 'error': str(error)}
        update = self.load('update-job.json')
        result['source_deployment'] = {k: update.get(k) for k in
            ('phase', 'started_at', 'finished_at', 'error', 'revision')} if update else None
        if update and (update.get('finished_at') or update['started_at']) > (
                (result['last_deployment'] or {}).get('finished_at') or ''):
            result['last_deployment'] = {k: update.get(k) for k in
                ('phase', 'started_at', 'finished_at', 'error', 'revision')}
        # The service-restoration contract, where a page or a person can read it: either a backend
        # serves, or a restoration is running or scheduled. Reported ahead of the live checks
        # below, so an unreadable router can never hide the fact that nothing is serving.
        good = self.last_known_good()
        result['restoration'] = self.restoration()
        result['service'] = {'serving': self.serving(bundle), 'restorable': good is not None,
            'pending': self.restoration_pending(),
            'profile': (good or {}).get('bundle', {}).get('profile')}
        try:
            observed = self.observe(bundle)
            names = {gpu: name for group, ids in self.host.get('gpu_ids', {}).items()
                for gpu, name in zip(ids, self.host.get('gpu_names', {}).get(group, []))}
            for model, service in zip(bundle['catalog']['models'], bundle['manifest']['services']):
                live = observed[service['role']]
                text = model.get('text_gpu_uuids', model['gpu_uuids'])
                result['services'].append({'role': service['role'], 'name': model['display_name'], 'model': model['model'],
                    'context_tokens': model['context_length'],
                    'gpu_ids': text,
                    'vision_gpu_id': model.get('vision_gpu_uuid'),
                    'vision_gpu_name': self.host.get('vision_gpu_name') if model.get('vision_gpu_uuid') else None,
                    'vision_device': model.get('vision_device', 'CPU'),
                    'vision_gpu_shared': model.get('vision_gpu_shared', False),
                    'exclusive': model.get('exclusive', False),
                    'gpu_names': [names[gpu] for gpu in text if gpu in names],
                    **live})
            router = self.admin('runtime-state')
            state = router['runtime']
            result['maintenance'] = {'draining': state['draining'], 'active_requests': state['active_count'],
                'queued_requests': state.get('queued_count', 0)}
            tx = self.load('transaction.json', {})
            expected = {m['model']: m['context_length'] for m in bundle['catalog']['models']}
            catalog_ready = {m['id'] for m in router.get('models', [])} == set(expected) and all(
                m['x_ollama_router']['health']['available'] and
                m['x_ollama_router']['context_window'] == expected[m['id']] for m in router.get('models', []))
            result['ready'] = bool(active and active['revision'] == self.revision and
                all(x['healthy'] for x in observed.values()) and catalog_ready and not state['draining'] and
                not self.load('router-maintenance.json', {}).get('reserved') and
                tx.get('phase') not in PENDING_PHASES)
        except Exception as error:
            result['error'] = str(error)
        return result

"""Serialized, recoverable reconciliation of the existing backend containers.

Paired profiles run Daytime and Nighttime side by side. An exclusive Daytime profile owns
all four text GPUs, so selecting it stops and removes Nighttime, and selecting a paired
profile again recreates it.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from .config import (DAYTIME_PROFILES, DEFAULT_PROFILE, NIGHTTIME_PROFILE, available_profiles, digest, read, render,
    require, service_engine)
from .system import System, atomic_json, future, lock, now

# A transaction that has not reached a terminal phase. While one exists the runtime holds its
# drain or its outcome is unknown, so nothing else may change backends.
PENDING_PHASES = ('prepared', 'draining', 'applying', 'verifying', 'restoring', 'needs-attention')
# How many consecutive quiet passes a drain must see before work is assumed finished.
DRAIN_SETTLE_ROUNDS = 3
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

        Such an engine has no honest readiness window until a warm boot from that state has been
        measured on this host; the window that a switch is given has to come from the measurement.
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

    def verdict(self, bundle):
        """Why a switch to this bundle would be refused, worked out without touching anything.

        An empty list is the go-ahead. ``switchable`` raises on these and the read-only ``gate``
        command reports them, so the reason a profile will not be switched to is readable without
        attempting a switch — a refusal should never require provoking one to be understood.
        """
        role, engine, provision = self.provisioning(bundle)
        if engine is None:
            return []
        command = 'python3 -m runtime provision --profile ' + bundle['profile']
        receipt = self.provisioned(bundle)
        if receipt is None:
            return ['The ' + engine['tag'] + ' workspace has never been pre-provisioned '
                'on this host for this release; run ' + command + ' and let it record a healthy warm boot']
        if receipt.get('healthy_warm_boot') is not True:
            return ['Pre-provisioning ' + engine['tag'] + ' did not '
                'reach a healthy warm boot (' + str(receipt.get('error') or 'see the receipt') + '); the trial '
                'is not switchable until it does, or the profile is retired']
        window = engine.get('readiness_wait_seconds') or 0
        measured = float(receipt.get('warm_boot_seconds') or 0)
        if window < 3 * measured:
            return ['The readiness window for ' + engine['tag'] + ' is ' + str(window) +
                's, under three times the measured warm boot of ' + str(receipt['warm_boot_seconds']) +
                's; set readiness_wait_seconds to at least ' + str(int(3 * measured + 1)) + 's and publish']
        return []

    def switchable(self, bundle):
        """Refuse to switch onto a workspace whose first boot has never been measured here.

        A trial whose engine provisions on first start can sit inside the readiness window for
        hours; the first attempt at this one did exactly that and was abandoned with the router
        drained. The window is only trustworthy once a warm boot from the provisioned workspace has
        been measured on this host, with room to spare, so a blind switch is refused rather than
        run into a timeout.
        """
        reasons = self.verdict(bundle)
        if reasons:
            raise RuntimeError(reasons[0])

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
            # Every expectation is read with .get(): a service dictionary that lost a key is a
            # configuration fault, and the operator needs to see which setting drifted, not a crash in
            # the middle of a switch. A missing key never matches, so it falls through as a reason.
            for valid, reason in [
                (ci['Image'] == engine['image_id'], 'image'),
                ((ci['Config'].get('Cmd') or []) == (cfg.get('command') or []), 'arguments'),
                (ci['Config'].get('Entrypoint') == cfg.get('entrypoint'), 'entrypoint'),
                (ci['Config'].get('User') == cfg.get('user'), 'user'),
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
        require(all(s['healthy'] for s in observed.values()), 'Backend identity/readiness verification failed')
        return observed

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
        atomic_json(Path(self.host['router_state_dir']) / 'active-model.json', catalog)
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

    def set_drain(self, enabled, transaction):
        return self.admin('runtime-drain', {'enabled': enabled,
            'reason': 'local-ai-runtime:' + transaction['id'] if enabled else 'runtime ready'})

    def owns_drain(self, transaction):
        state = self.admin('runtime-state')['runtime']
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

    def finish(self, transaction, state, error=None):
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
            self.clear_restoration('The runtime is serving ' + str(transaction.get('profile')) + ' and requests are open.')

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
        self.finish(transaction, 'recovered', transaction.get('error'))

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
        state = self.admin('runtime-state')['runtime']
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
            self.set_drain(True, tx)
            self.mark(tx, 'draining')
            progress('draining')
            self.wait_idle([bundle, previous['bundle'] if previous else None])
            self.mark(tx, 'applying')
            progress('loading')
            changed = True
            self.release(removed)
            after = self.reconcile(bundle, roles, sequential=topology)
            require(all(after[r]['id'] == cid for r, cid in unchanged.items()), 'An unchanged backend was unexpectedly recreated')
            self.mark(tx, 'verifying')
            progress('verifying')
            self.acceptance(bundle, roles)
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
            if tx['phase'] == 'prepared' and not self.admin('runtime-state')['runtime']['draining']:
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
            require(not self.admin('runtime-state')['runtime']['draining'], 'Another operation owns the router drain')
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
                require(not reservation.get('reserved') and not self.admin('runtime-state')['runtime']['draining'],
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
        # checks below so an unreadable definition never hides the health of the pair.
        try:
            result['configurations'] = {'active': bundle['profile'], **available_profiles(self.config_dir, self.host)}
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

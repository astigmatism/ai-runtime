"""Serialized, recoverable reconciliation of the existing two backend containers."""
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from .config import available_profiles, digest, read, render, require, service_engine
from .system import System, atomic_json, lock, now


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
        selected = profile or current.get('bundle', {}).get('profile') or self.host.get('initial_profile', 'daytime')
        return render(self.config_dir, self.host, selected)

    def backend_url(self, cfg):
        return 'http://' + cfg['container_name'] + ':8080'

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
            for valid, reason in [
                (ci['Image'] == engine['image_id'], 'image'),
                (ci['Config']['Cmd'] == cfg['command'], 'arguments'),
                (ci['Config']['Entrypoint'] == cfg['entrypoint'], 'entrypoint'),
                (ci['Config']['User'] == cfg['user'], 'user'),
                (ci['HostConfig']['ReadonlyRootfs'] == cfg['read_only'], 'read-only root'),
                (ci['HostConfig']['RestartPolicy']['Name'] == cfg['restart'], 'restart policy'),
                (ci['HostConfig'].get('Init') == cfg['init'], 'init'),
                (ci['HostConfig'].get('Privileged', False) is False, 'privileged mode')]:
                if not valid:
                    reasons.append(reason)
            requests = ci['HostConfig']['DeviceRequests'] or []
            expected_ids = cfg['deploy']['resources']['reservations']['devices'][0]['device_ids']
            if len(requests) != 1 or requests[0]['DeviceIDs'] != expected_ids:
                reasons.append('GPU assignment')
            expected_cuda = cfg.get('environment', {}).get('CUDA_VISIBLE_DEVICES')
            live_env = dict(item.split('=', 1) for item in ci['Config'].get('Env', []) if '=' in item)
            if live_env.get('CUDA_VISIBLE_DEVICES') != expected_cuda:
                reasons.append('CUDA device order')
            if expected_cuda and (len(requests) != 1 or requests[0].get('Driver') != 'nvidia'
                    or requests[0].get('Count') != 0 or requests[0].get('Capabilities') != [['gpu']]):
                reasons.append('GPU reservation contract')
            mounts = {(x['Source'], x['Destination'], x['RW']) for x in ci['Mounts'] if x['Type'] == 'bind'}
            if mounts != {(x['source'], x['target'], False) for x in cfg['volumes']}:
                reasons.append('model mounts')
            host_port = cfg['ports'][0].split(':')[1]
            if ci['HostConfig']['PortBindings'] != {'8080/tcp': [{'HostIp': '127.0.0.1', 'HostPort': host_port}]}:
                reasons.append('port bindings')
            connection = ci['NetworkSettings']['Networks'].get(network, {})
            if not connection or not set(cfg['networks']['router']['aliases']).issubset(connection.get('Aliases') or []):
                reasons.append('router network')
            if result['running']:
                try:
                    health = self.system.http(self.backend_url(cfg) + '/health', timeout=3)
                    slots = self.system.http(self.backend_url(cfg) + '/slots', timeout=3)
                    expected_ctx = int(cfg['command'][cfg['command'].index('--ctx-size') + 1])
                    if len(slots) != 1 or slots[0]['n_ctx'] != expected_ctx:
                        reasons.append('slot context')
                    result['healthy'] = health.get('status') == 'ok' and not reasons
                    result['processing'] = any(x.get('is_processing', False) for x in slots)
                except Exception:
                    result['health_error'] = 'Backend readiness unavailable'
            observed[role] = result
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

    def wait_idle(self, bundles, timeout=300):
        configs = {cfg['container_name']: cfg for b in bundles if b for cfg in b['compose']['services'].values()}
        deadline, stable = time.monotonic() + timeout, 0
        while time.monotonic() < deadline:
            state = self.admin('runtime-state')['runtime']
            quiet = state['active_count'] == 0 and state.get('queued_count', 0) == 0
            for name, cfg in configs.items():
                ci = self.system.inspect(name)
                if not ci or not ci['State']['Running']:
                    continue
                try:
                    slots = self.system.http(self.backend_url(cfg) + '/slots', timeout=5)
                    quiet = quiet and bool(slots) and all(not x.get('is_processing', False) for x in slots)
                except Exception:
                    quiet = False  # An unknown or initializing backend is never assumed idle.
            stable = stable + 1 if quiet else 0
            if stable >= 3:
                return
            time.sleep(1)
        raise RuntimeError('Drain timed out; no active generation was stopped')

    def compose_path(self, bundle):
        folder = self.state / 'generated' / bundle['config_sha256']
        self.save(str(folder.relative_to(self.state) / 'compose.json'), bundle['compose'])
        return folder / 'compose.json'

    def prepare(self, bundle):
        self.validate(bundle)
        path = self.compose_path(bundle)
        self.system.docker('compose', '-p', bundle['compose']['name'], '-f', str(path), 'config', '--quiet')

    def reconcile(self, bundle, roles):
        if roles:
            path = self.compose_path(bundle)
            self.system.docker('compose', '-p', bundle['compose']['name'], '-f', str(path),
                'up', '-d', '--no-deps', '--pull', 'never', '--wait', '--wait-timeout', '450', *roles,
                timeout=510)
        observed = self.observe(bundle)
        require(all(s['healthy'] for s in observed.values()), 'Backend identity/readiness verification failed')
        return observed

    def acceptance(self, bundle, roles):
        for role in roles:
            cfg = bundle['compose']['services'][role]
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
            entry['runtime_output_policy'] = {'n_predict': -1, 'reasoning_budget': -1,
                'reasoning_effort': 'default', 'verification': 'docker-inspect-argv',
                'verified_at': now(), 'container_id': ci['Id'], 'started_at': ci['State']['StartedAt'],
                'argv_sha256': hashlib.sha256(json.dumps(ci['Config']['Cmd']).encode()).hexdigest()}
        catalog.update(catalog['models'][0])
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

    def finish(self, transaction, state, error=None):
        transaction.update(phase=state, finished_at=now(), error=error)
        self.save('transaction.json', transaction)
        self.save('last-deployment.json', {k: transaction.get(k) for k in
            ('id', 'phase', 'started_at', 'finished_at', 'error', 'changed_roles', 'revision', 'config_sha256')})

    def recover_locked(self, transaction):
        previous = transaction.get('previous')
        require(previous is not None, 'No previous release exists; recovery needs operator attention')
        require(self.owns_drain(transaction), 'Recovery does not own the router drain; refusing to change it')
        old = previous['bundle']
        self.prepare(old)
        self.wait_idle([old, transaction['target']['bundle']])
        observed = self.observe(old)
        roles = [role for role, s in observed.items() if not s['healthy']]
        self.reconcile(old, roles)
        self.acceptance(old, roles)
        self.publish(old)
        self.save('active.json', previous)
        self.set_drain(False, transaction)
        self.finish(transaction, 'recovered', transaction.get('error'))

    def transition(self, profile=None, deploy=False, adopt=False):
        require(not self.system.readonly, 'Mutation forbidden in inspection mode')
        with lock(self.state / 'runtime.lock'):
            require(not self.load('router-maintenance.json', {}).get('reserved'),
                'A router release reserves the runtime; finish or recover it first')
            pending = self.load('transaction.json', {})
            require(pending.get('phase') not in ('prepared', 'draining', 'applying', 'verifying', 'needs-attention'),
                'An interrupted deployment needs explicit runtime recover before another operation')
            previous = self.load('active.json')
            require(previous is not None or adopt, 'Runtime is not adopted; use the reviewed migration procedure')
            require(not previous or previous['revision'] == self.revision or deploy,
                'This controller has been superseded; it cannot reapply an older release')
            bundle = self.desired(profile)
            self.prepare(bundle)
            observed = self.observe(bundle)
            roles = [role for role, s in observed.items() if not s['healthy'] or (previous and
                previous['bundle']['compose']['services'][role] != bundle['compose']['services'][role])]
            require(not adopt or not roles, 'Adoption requires an exact match to healthy existing backends')
            state = self.admin('runtime-state')['runtime']
            require(not state['draining'], 'Another operation owns the router drain')
            if previous and previous['revision'] == self.revision and previous['bundle'] == bundle and not roles:
                return {'already_active': True, 'profile': bundle['profile']}
            unchanged = {role: s['id'] for role, s in observed.items() if role not in roles}
            target = {'revision': self.revision, 'image': self.image, 'bundle': bundle, 'applied_at': now()}
            # Adoption's prior runtime is already known healthy and can be restored by this image.
            if adopt:
                previous = copy.deepcopy(target)
            tx = {'id': str(uuid.uuid4()), 'started_at': now(), 'phase': 'prepared', 'previous': previous,
                'target': target, 'changed_roles': roles, 'revision': self.revision,
                'config_sha256': bundle['config_sha256']}
            self.save('transaction.json', tx)
            changed = False
            try:
                self.set_drain(True, tx)
                tx['phase'] = 'draining'
                self.save('transaction.json', tx)
                self.wait_idle([bundle, previous['bundle'] if previous else None])
                tx['phase'] = 'applying'
                self.save('transaction.json', tx)
                changed = True
                after = self.reconcile(bundle, roles)
                require(all(after[r]['id'] == cid for r, cid in unchanged.items()), 'An unchanged backend was unexpectedly recreated')
                self.acceptance(bundle, roles)
                tx['phase'] = 'verifying'
                self.save('transaction.json', tx)
                self.publish(bundle)
                self.save('previous.json', previous)
                self.save('active.json', target)
                self.set_drain(False, tx)
                self.finish(tx, 'succeeded')
                return {'profile': bundle['profile'], 'changed_roles': roles, 'revision': self.revision}
            except BaseException as error:
                tx['error'] = str(error)
                if changed:
                    try:
                        self.recover_locked(tx)
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
            require(tx and tx['phase'] in ('prepared', 'draining', 'applying', 'verifying', 'needs-attention'),
                'There is no interrupted deployment to recover')
            if tx['phase'] == 'prepared' and not self.admin('runtime-state')['runtime']['draining']:
                self.finish(tx, 'failed', 'Interrupted before draining; prior runtime preserved')
            else:
                self.recover_locked(tx)

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
                'target': active, 'revision': self.revision, 'error': 'Replacement controller failed readiness'}
            self.save('transaction.json', tx)
            self.set_drain(True, tx)
            try:
                self.recover_locked(tx)
            except BaseException as error:
                self.finish(tx, 'needs-attention', str(error))
                raise

    def republish(self):
        """Router-only releases may republish, but may not replace runtime source/configuration."""
        with lock(self.state / 'runtime.lock'):
            active = self.load('active.json')
            require(active and active['revision'] == self.revision, 'Controller does not own the deployed revision')
            require(self.load('transaction.json', {}).get('phase') not in
                ('prepared', 'draining', 'applying', 'verifying', 'needs-attention'), 'Interrupted runtime transaction blocks publication')
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
                require(self.load('transaction.json', {}).get('phase') not in
                    ('prepared', 'draining', 'applying', 'verifying', 'needs-attention'), 'Interrupted runtime transaction')
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
            'last_deployment': self.load('last-deployment.json'), 'updated_at': now()}
        # Read-only presentation of the profile registry. It is reported before the live
        # checks below so an unreadable definition never hides the health of the pair.
        try:
            result['configurations'] = {'active': bundle['profile'], **available_profiles(self.config_dir, self.host)}
        except Exception as error:
            result['configurations'] = {'active': bundle['profile'], 'selectable': [],
                'always_included': [], 'error': str(error)}
        update = self.load('update-job.json')
        if update and (update.get('finished_at') or update['started_at']) > (
                (result['last_deployment'] or {}).get('finished_at') or ''):
            result['last_deployment'] = {k: update.get(k) for k in
                ('phase', 'started_at', 'finished_at', 'error', 'revision')}
        try:
            observed = self.observe(bundle)
            for model, service in zip(bundle['catalog']['models'], bundle['manifest']['services']):
                live = observed[service['role']]
                result['services'].append({'name': model['display_name'], 'model': model['model'],
                    'context_tokens': model['context_length'],
                    'gpu_ids': model.get('text_gpu_uuids', model['gpu_uuids']),
                    'vision_gpu_id': model.get('vision_gpu_uuid'),
                    'vision_gpu_name': self.host.get('vision_gpu_name') if model.get('vision_gpu_uuid') else None,
                    'vision_device': model.get('vision_device', 'CPU'),
                    'vision_gpu_shared': model.get('vision_gpu_shared', False),
                    'gpu_names': self.host.get('gpu_names', {}).get('daytime' if service['role'] == 'coding' else 'nighttime', []),
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
                tx.get('phase') not in ('prepared', 'draining', 'applying', 'verifying', 'needs-attention'))
        except Exception as error:
            result['error'] = str(error)
        return result

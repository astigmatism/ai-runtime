"""Explicit staging and cutover commands. Preparing/inspecting never starts the controller."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys

from .config import read, render, require
from .system import atomic_json, lock, now
from .update import Updater

WRAPPERS = ['primary', 'daytime', 'daytime-swift', 'nighttime', 'daytime-256', 'nighttime-256', 'local-ai-config.sh']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shell_wrapper(name):
    prefix = '#!/bin/sh\nset -eu\n'
    if name == 'local-ai-config.sh':
        prefix += 'if [ "${1:-}" = gpus ]; then exec nvidia-smi; fi\n'
    return prefix + 'exec docker exec local-ai-runtime python3 -m runtime.compat ' + shlex.quote(name) + ' "$@"\n'


def legacy_bridge(kind):
    return '''#!/usr/bin/env python3
"""Compatibility entrypoint installed by the versioned runtime migration."""
import os
import sys
if __name__ != '__main__':
    raise RuntimeError('Runtime ownership moved to local-ai-runtime; use its container CLI, not the retired host module')
''' + ("args = ['primary', *sys.argv[1:]]\n" if kind == 'primary.py' else
        "args = sys.argv[1:] or ['daytime']\n") + \
        "os.execvp('docker', ['docker', 'exec', 'local-ai-runtime', 'python3', '-m', 'runtime.compat', *args])\n"


class Migration:
    def __init__(self, root, home):
        self.root = Path(root).resolve(); self.home = Path(home).resolve()
        self.primary = self.home / 'apps/local-ai-primary'
        self.state = self.root / '.state'; self.updater = Updater(self.root)

    def tracked_sources(self):
        return [self.primary / name for name in ['primary.py', 'daytime-profile.py', 'manager.sh',
            'compose.json', 'manifest.json', 'model-catalog.json', 'profiles/selected.json',
            'evidence/qualified.json', 'evidence/artifacts.json']] + [self.home / name for name in WRAPPERS] + [
            self.home / '.config/systemd/user/local-ai-primary.service', self.home / 'local-ai-configs.json',
            self.home / '.local-ai-selected-profile.json']

    def prepare(self):
        revision = self.updater.source_preflight()
        require(not (self.primary / 'runtime-owner.json').exists(), 'Runtime migration is already installed')
        selected = read(self.primary / 'profiles/selected.json')['selected']
        compose = read(self.primary / 'compose.json')
        stack = self.home / 'apps/local-ai-ollama-stack'
        environment = {}
        for line in (stack / '.env').read_text().splitlines():
            if line.strip() and not line.lstrip().startswith('#') and '=' in line:
                key, value = line.split('=', 1); parsed = shlex.split(value, comments=True)
                environment[key.strip()] = parsed[0] if parsed else ''
        require(bool(environment.get('ADMIN_TOKEN')), 'Existing router credential is unavailable')
        host = {'model_root': str(self.home / 'ai/models'), 'router_state_dir': str(stack / 'runtime/router'),
            'router_admin_url': 'http://local-ai-ollama-router:11435',
            'router_admin_header': environment.get('ADMIN_SESSION_HEADER', 'X-Admin-Token'),
            'uid': os.getuid(), 'gid': os.getgid(), 'initial_profile': selected,
            'gpu_ids': {group: compose['services'][role]['deploy']['resources']['reservations']['devices'][0]['device_ids']
                for group, role in [('daytime', 'coding'), ('nighttime', 'everyday')]},
            'gpu_names': {'daytime': ['RTX 3090', 'RTX 4080 SUPER'], 'nighttime': ['RTX 4080', 'RTX 3080 Ti']}}
        desired = render(self.root / 'config', host, selected)
        require(desired['compose'] == compose, 'Published configuration differs from the live source; merge changes before migration')
        self.state.mkdir(mode=0o700, exist_ok=True); self.state.chmod(0o700)
        atomic_json(self.state / 'host.json', host)
        token = self.state / 'router-token'; token.write_text(environment['ADMIN_TOKEN'] + '\n'); token.chmod(0o600)
        image = 'local/ai-runtime:git-' + revision
        env = {'RUNTIME_IMAGE': image, 'RUNTIME_STATE_DIR': str(self.state),
            'RUNTIME_USER': f"{host['uid']}:{host['gid']}", 'DOCKER_GID': str(os.stat('/var/run/docker.sock').st_gid),
            'RUNTIME_BIND_IP': '192.168.1.21', 'MODEL_ROOT': host['model_root'], 'ROUTER_STATE_DIR': host['router_state_dir']}
        (self.root / '.env').write_text(''.join(key + '=' + shlex.quote(value) + '\n' for key, value in env.items()))
        (self.root / '.env').chmod(0o600)
        baseline = {'revision': revision, 'profile': selected, 'prepared_at': now(),
            'files': {str(p.relative_to(self.home)): sha(p) for p in self.tracked_sources()},
            'containers': {cfg['container_name']: json.loads(self.updater.run('docker', 'inspect', cfg['container_name']))[0]['Id']
                for cfg in compose['services'].values()},
            'legacy_unit_enabled': self.updater.run('systemctl', '--user', 'is-enabled', 'local-ai-primary.service', check=False),
            'legacy_unit_active': self.updater.run('systemctl', '--user', 'is-active', 'local-ai-primary.service', check=False)}
        atomic_json(self.state / 'migration-baseline.json', baseline)
        print('Prepared private deployment state for ' + selected + '; no service was changed.')

    def inspection(self):
        revision = self.updater.source_preflight()
        image = 'local/ai-runtime:git-' + revision
        metadata = json.loads(self.updater.run('docker', 'image', 'inspect', image))[0]
        require(metadata['Config']['Labels'].get('org.opencontainers.image.revision') == revision,
            'Inspection image does not match the published source')
        result = json.loads(self.updater.candidate(image, 'inspect', full_hash=True, capture=True))
        require(result['matches_live'] and result['validation']['all_checksums_verified'],
            'Live runtime comparison or model checksum verification failed')
        atomic_json(self.state / 'inspection.json', result)
        atomic_json(self.state / 'artifact-receipts.json', result['validation']['receipts'])
        summary = {k: result[k] for k in ['mode', 'revision', 'profile', 'config_sha256', 'matches_live',
            'router_draining', 'active_requests', 'queued_requests']}
        summary['all_checksums_verified'] = True
        print(json.dumps(summary, indent=2))

    def cutover(self):
        revision = self.updater.source_preflight()
        baseline = read(self.state / 'migration-baseline.json')
        inspection = read(self.state / 'inspection.json')
        require(revision == baseline['revision'] == inspection['revision'], 'Migration source changed; prepare and inspect again')
        require(not (self.primary / 'runtime-owner.json').exists(), 'Runtime migration is already installed')
        with lock(self.home / '.local-ai-profile-switch.lock'):
            for relative, expected in baseline['files'].items():
                require(sha(self.home / relative) == expected, 'Legacy configuration changed since inspection: ' + relative)
            for name, expected in baseline['containers'].items():
                require(json.loads(self.updater.run('docker', 'inspect', name))[0]['Id'] == expected,
                    'A backend changed since inspection; re-inspect before cutover')
            with socket.socket() as probe:
                probe.bind(('192.168.1.21', 11436))
            backup = self.state / 'migration-backup'
            require(not backup.exists(), 'A migration backup already exists; review recovery before retrying')
            for relative in baseline['files']:
                target = backup / relative; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.home / relative, target)
            image = 'local/ai-runtime:git-' + revision
            self.updater.candidate(image, 'adopt')
            self.updater.compose(self.root, image, 'up', '-d', '--pull', 'never', '--wait', '--wait-timeout', '150', 'controller')
            for name, expected in baseline['containers'].items():
                require(json.loads(self.updater.run('docker', 'inspect', name))[0]['Id'] == expected,
                    'Migration unexpectedly recreated a backend')
            # The existing controller is disabled only after the new controller is healthy.
            self.updater.run('systemctl', '--user', 'disable', '--now', 'local-ai-primary.service')
            for name in WRAPPERS:
                path = self.home / name; path.write_text(shell_wrapper(name)); path.chmod(0o755)
            for name in ['primary.py', 'daytime-profile.py']:
                path = self.primary / name; path.write_text(legacy_bridge(name)); path.chmod(0o755)
            (self.primary / 'manager.sh').write_text(shell_wrapper('local-ai-config.sh'))
            (self.primary / 'manager.sh').chmod(0o755)
            atomic_json(self.primary / 'runtime-owner.json', {'repository': 'https://github.com/astigmatism/local-ai-runtime',
                'container': 'local-ai-runtime', 'state_dir': str(self.state), 'installed_at': now()})
            atomic_json(self.state / 'migration-result.json', {'status': 'succeeded', 'revision': revision,
                'backend_ids_unchanged': baseline['containers'], 'completed_at': now()})
            print('Cutover complete; backend container identities are unchanged. Status: http://192.168.1.21:11436')

    def restore(self):
        """Undo only the initial migration while the original backend configuration still matches."""
        baseline = read(self.state / 'migration-baseline.json')
        active = read(self.state / 'active.json')
        require(active['revision'] == baseline['revision'] and active['bundle']['profile'] == baseline['profile'],
            'Initial migration restore cannot undo subsequent runtime changes; use release recovery')
        image = active['image']
        check = json.loads(self.updater.candidate(image, 'inspect', capture=True))
        require(check['matches_live'] and not check['router_draining'], 'Runtime is not healthy and idle for host-controller restoration')
        with lock(self.home / '.local-ai-profile-switch.lock'), lock(self.state / 'runtime.lock'):
            self.updater.run('docker', 'stop', '-t', '15', 'local-ai-runtime', check=False)
            for relative in baseline['files']:
                shutil.copy2(self.state / 'migration-backup' / relative, self.home / relative)
            owner = self.primary / 'runtime-owner.json'
            if owner.exists(): owner.unlink()
            self.updater.run('systemctl', '--user', 'daemon-reload')
            if baseline['legacy_unit_enabled'] == 'enabled':
                self.updater.run('systemctl', '--user', 'enable', 'local-ai-primary.service')
            # Start after releasing the old controller's lock.
        if baseline['legacy_unit_active'] == 'active':
            self.updater.run('systemctl', '--user', 'start', 'local-ai-primary.service')
        atomic_json(self.state / 'migration-result.json', {'status': 'restored-host-controller', 'completed_at': now()})
        print('Original host controller restored; resident backends preserved.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'inspect', 'cutover', 'restore'])
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--host-home', default=str(Path.home()))
    args = parser.parse_args()
    migration = Migration(args.root, args.host_home)
    try:
        getattr(migration, {'inspect': 'inspection'}.get(args.action, args.action))()
    except Exception as error:
        print('Error: ' + str(error) + '; production cutover failures require the documented recovery procedure.', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

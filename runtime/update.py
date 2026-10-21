"""Portal-compatible public-Git updater; production applies immutable image content."""
import argparse
import ast
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

from .config import read, require
from .controller import PENDING_PHASES
from .system import atomic_json, lock, now

REMOTE = 'https://github.com/astigmatism/ai-runtime.git'
# Older deployed updaters still use the former URL. Accept both checkout
# identities during the rename, while fetching only from the canonical remote.
LEGACY_REMOTE = 'https://github.com/astigmatism/local-ai-runtime.git'
ACCEPTED_REMOTES = (
    REMOTE, REMOTE[:-4], 'git@github.com:astigmatism/ai-runtime.git',
    LEGACY_REMOTE, LEGACY_REMOTE[:-4], 'git@github.com:astigmatism/local-ai-runtime.git',
)


class Updater:
    def __init__(self, root, execute=None):
        self.root = Path(root).resolve()
        self.state = self.root / '.state'
        self.execute = execute or subprocess.run

    def run(self, *args, check=True, capture=True, timeout=1800, env=None):
        result = self.execute(list(args), cwd=self.root, text=True,
            capture_output=capture, timeout=timeout,
            env={**os.environ, 'GIT_TERMINAL_PROMPT': '0', **(env or {})})
        if check and result.returncode:
            raise RuntimeError(' '.join(args[:3]) + ' failed: ' + (result.stderr or '')[-2000:])
        return (result.stdout or '').strip()

    def git(self, *args, **kw):
        return self.run('git', *args, **kw)

    def source_preflight(self):
        require(self.root.is_dir() and self.root != Path('/'), 'Invalid checkout directory')
        require(Path(self.git('rev-parse', '--show-toplevel')).resolve() == self.root, 'Checkout root mismatch')
        require(self.git('symbolic-ref', '--quiet', '--short', 'HEAD', check=False) == 'main',
            'Refusing detached HEAD or non-main branch')
        require(self.git('rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{upstream}', check=False) == 'origin/main',
            'Refusing unexpected upstream')
        require(self.git('remote', 'get-url', 'origin') in ACCEPTED_REMOTES,
            'Refusing unexpected source repository')
        require(not self.git('status', '--porcelain', '--untracked-files=normal'),
            'Refusing a dirty checkout; local changes have been preserved')
        return self.git('rev-parse', 'HEAD')

    def fetch(self, before):
        prior_upstream = self.git('rev-parse', 'refs/remotes/origin/main')
        self.git('fetch', '--no-tags', REMOTE, 'refs/heads/main')
        target = self.git('rev-parse', 'FETCH_HEAD')
        require(re.fullmatch('[0-9a-f]{40}', target), 'Invalid release revision')
        self.git('merge-base', '--is-ancestor', prior_upstream, target)
        self.git('merge-base', '--is-ancestor', before, target)
        self.git('update-ref', 'refs/remotes/origin/main', target, prior_upstream)
        return target

    def export(self, target):
        source = self.state / 'releases' / target / 'source'
        if source.exists():
            # A fresh export avoids reusing a tampered or partial candidate directory.
            import shutil
            shutil.rmtree(source)
        source.mkdir(parents=True)
        archive = subprocess.run(['git', 'archive', '--format=tar', target], cwd=self.root,
            check=True, capture_output=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            for member in tar.getmembers():
                require(not member.issym() and not member.islnk() and
                    (member.isfile() or member.isdir()), 'Release archive contains unsupported file types')
                require(not Path(member.name).is_absolute() and '..' not in Path(member.name).parts,
                    'Release archive path escapes source directory')
            tar.extractall(source, filter='data')
        return source

    # The names a revision keeps its selectable configurations under. Both are read because the
    # revision doing this work is the one already deployed, and a rename of the constant itself is
    # only deployable if the old revision can still read the new one.
    PROFILE_CONSTANTS = ('CONFIGURATIONS', 'DAYTIME_PROFILES')

    def target_profiles(self, target):
        """Configurations the target revision offers, read from its published source.

        Returns what is selectable there and what the active release may be recorded as. Those differ
        by PROFILE_ALIASES, and the difference is the whole point: a configuration that was *renamed*
        is still the configuration running, so a release that renames one must not refuse itself,
        while a configuration that was *retired* has no backends left to launch and still refuses.
        """
        code = self.git('show', target + ':runtime/config.py', check=False)
        require(code, 'Cannot read runtime/config.py from the target revision')
        literals, bound = {}, {}
        for node in ast.walk(ast.parse(code)):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                continue
            key = node.targets[0].id
            if isinstance(node.value, ast.Name):
                # `NEW = OLD`, the transitional spelling a rename leaves behind.
                bound[key] = node.value.id
                continue
            try:
                literals[key] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                pass

        def named(key, seen=()):
            require(key not in seen, 'Target revision defines ' + key + ' through a cycle')
            if key in literals:
                return literals[key]
            if key in bound:
                return named(bound[key], (*seen, key))
            return None

        selected = next((named(key) for key in self.PROFILE_CONSTANTS
            if isinstance(named(key), tuple) and named(key) and all(isinstance(x, str) for x in named(key))), None)
        require(selected is not None, 'Target revision defines no selectable configurations as '
            + ' or '.join(self.PROFILE_CONSTANTS) + ' (a non-empty tuple of ids)')
        aliases = named('PROFILE_ALIASES') or {}
        require(isinstance(aliases, dict) and all(isinstance(k, str) and isinstance(v, str)
            for k, v in aliases.items()), 'Target revision defines PROFILE_ALIASES as something '
            'other than a mapping of retired id to current id')
        return tuple(selected), tuple(selected) + tuple(old for old, new in aliases.items() if new in selected)

    def candidate(self, image, action, full_hash=False, capture=False):
        host = read(self.state / 'host.json')
        gid = str(os.stat('/var/run/docker.sock').st_gid)
        readonly = action in ('inspect', 'validate', 'status', 'check', 'render')
        suffix = ',readonly' if readonly else ''
        args = ['docker', 'run', '--rm', '--init', '--read-only', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges', '--tmpfs', '/tmp:rw,nosuid,nodev,size=128m',
            '--label', 'io.service-portal.hidden=true',
            '--user', f"{host['uid']}:{host['gid']}", '--group-add', gid,
            '--network', 'local-ai-ollama_default']
        if (self.root / '.env').exists():
            # The deployment's own environment, which is the same file `compose` is handed for the
            # long-lived service. A one-off action started without it can render, verify and check
            # health, but it cannot publish to a router elsewhere: the credential a push needs is a
            # secret, so it lives here and not in the image or the repository, and an action run by
            # the updater or by hand would otherwise fail with "the token is not in this container
            # environment" — true, and useless when the cause is only how the container was started.
            args += ['--env-file', str(self.root / '.env')]
        args += ['--env', 'RUNTIME_STATE_DIR=' + str(self.state), '--env', 'RUNTIME_IMAGE=' + image,
            '--mount', 'type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock',
            '--mount', f'type=bind,source={self.state},target={self.state}' + suffix,
            '--mount', f"type=bind,source={host['model_root']},target={host['model_root']},readonly"]
        if host.get('router_state_dir'):
            # The directory belongs to a router on this machine. A machine that publishes to a router
            # elsewhere has no such directory, and Docker fails a container whose bind source does not
            # exist — so the mount travels with the setting that says this machine writes a file.
            args += ['--mount',
                f"type=bind,source={host['router_state_dir']},target={host['router_state_dir']}" + suffix]
        # Everything the container needs has to be said before the image: after it, Docker stops reading
        # options and starts reading a command, and the runtime receives `--mount` as an argument of its
        # own. The image and the action are therefore appended last, once, here.
        args += [image, 'python3', '-m', 'runtime', action]
        if full_hash:
            args.append('--full-hash')
        return self.run(*args, capture=capture)

    def compose(self, source, image, *args):
        return self.run('docker', 'compose', '--project-directory', str(self.root),
            '--env-file', str(self.root / '.env'), '-p', 'local-ai-runtime',
            '-f', str(source / 'compose.yaml'), *args, env={'RUNTIME_IMAGE': image}, capture=False)

    def persist_image(self, image):
        path = self.root / '.env'
        lines = path.read_text().splitlines()
        lines = [line for line in lines if not line.startswith('RUNTIME_IMAGE=')]
        temporary = path.with_name('.env.next')
        temporary.write_text('\n'.join([*lines, 'RUNTIME_IMAGE=' + image]) + '\n')
        temporary.chmod(0o600)
        os.replace(temporary, path)

    def update(self):
        # Refuse dirty source before fetch, build, container commands, or state mutations.
        before = self.source_preflight()
        require(self.state.is_dir() and (self.root / '.env').is_file(), 'Run the documented migration setup first')
        with lock(self.state / 'update.lock'):
            require(self.source_preflight() == before, 'Source changed while acquiring the update lock')
            target = self.fetch(before)
            previous = read(self.state / 'active.json')
            last = read(self.state / 'update-job.json') if (self.state / 'update-job.json').exists() else {}
            known_recovery = (last.get('phase') == 'failed' and last.get('target') == before
                and last.get('previous') == previous['revision'])
            require(previous['revision'] == before or known_recovery,
                'Checkout and deployed revision differ; use update recovery first')
            if target == before and previous['revision'] == target:
                print('Published revision is already deployed; checking runtime health.', flush=True)
                self.candidate(previous['image'], 'check')
                return
            active_profile = previous.get('bundle', {}).get('profile')
            if active_profile:
                selectable, accepted = self.target_profiles(target)
                require(active_profile in accepted,
                    f"Active configuration '{active_profile}' is retired in target revision {target}; on the "
                    f"deployed revision, switch to a configuration selectable there ({', '.join(selectable)}) "
                    f'before re-running the update')
            source = self.export(target)
            image = 'local/ai-runtime:git-' + target
            print('Building published revision ' + target, flush=True)
            self.run('docker', 'build', '--build-arg', 'SOURCE_REVISION=' + target,
                '-t', image, str(source), capture=False)
            actual = json.loads(self.run('docker', 'image', 'inspect', image))[0]
            require(actual['Config']['Labels'].get('org.opencontainers.image.revision') == target,
                'Candidate image does not match the published revision')
            self.run('docker', 'run', '--rm', image, 'python3', '-m', 'unittest', 'discover', '-s', 'tests', capture=False)
            self.compose(source, image, 'config', '--quiet')
            self.candidate(image, 'validate')
            require(self.source_preflight() == before, 'Source changed during release preparation')
            old_source = self.export(previous['revision']) if previous['revision'] != target else source
            receipt = {'previous': previous['revision'], 'target': target, 'revision': target, 'image': image,
                'started_at': now(), 'phase': 'prepared'}
            atomic_json(self.state / 'update-job.json', receipt)
            applied = False
            try:
                print('Applying runtime configuration after workload drain.', flush=True)
                self.candidate(image, 'deploy')
                applied = True
                receipt['phase'] = 'controller-replacement'
                atomic_json(self.state / 'update-job.json', receipt)
                self.compose(source, image, 'up', '-d', '--no-deps', '--pull', 'never',
                    '--wait', '--wait-timeout', '150', 'controller')
                require(self.source_preflight() == before, 'Checkout changed during deployment; refusing to overwrite local work')
                self.persist_image(image)
                self.git('merge', '--ff-only', target)
                receipt.update(phase='succeeded', finished_at=now())
                atomic_json(self.state / 'update-job.json', receipt)
                print('Updated and healthy: ' + target, flush=True)
            except BaseException as error:
                receipt.update(phase='failed', error=str(error), finished_at=now())
                atomic_json(self.state / 'update-job.json', receipt)
                if applied:
                    try:
                        self.candidate(image, 'rollback-release')
                        # The still-running old controller may carry an unhealthy
                        # check from the temporary revision mismatch. Reset only
                        # that container so Compose waits through fresh startup.
                        self.compose(old_source, previous['image'], 'up', '-d', '--no-deps', '--force-recreate', '--pull', 'never',
                            '--wait', '--wait-timeout', '150', 'controller')
                        self.candidate(previous['image'], 'check')
                        self.persist_image(previous['image'])
                    except BaseException as recovery:
                        raise RuntimeError(f'Update failed: {error}; recovery needs attention: {recovery}') from recovery
                raise

    def recover_update(self):
        before = self.source_preflight()
        with lock(self.state / 'update.lock'):
            job = read(self.state / 'update-job.json')
            require(job['phase'] != 'succeeded', 'The last update already completed')
            require(before in (job['previous'], job['target']), 'Checkout changed since the interrupted update')
            tx = read(self.state / 'transaction.json')
            if tx['phase'] in PENDING_PHASES:
                self.candidate(job['image'], 'recover')
            active = read(self.state / 'active.json')
            require(active['revision'] in (job['previous'], job['target']), 'Unknown active revision; refusing recovery')
            source = self.export(active['revision'])
            self.compose(source, active['image'], 'up', '-d', '--no-deps', '--force-recreate', '--pull', 'never',
                '--wait', '--wait-timeout', '150', 'controller')
            self.candidate(active['image'], 'check')
            self.persist_image(active['image'])
            if active['revision'] == job['target']:
                require(self.source_preflight() == before, 'Checkout changed during recovery')
                self.git('merge', '--ff-only', job['target'])
                job['phase'] = 'succeeded'
            else:
                job['phase'] = 'failed'
                job['error'] = 'Previous runtime restored after interrupted update; source preserved for explicit retry'
            job['finished_at'] = now()
            atomic_json(self.state / 'update-job.json', job)
            print('Update recovery complete: ' + active['revision'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--recover', action='store_true')
    args = parser.parse_args()
    try:
        updater = Updater(args.root)
        updater.recover_update() if args.recover else updater.update()
    except Exception as error:
        print('Error: ' + str(error), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

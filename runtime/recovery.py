"""Host-side recovery export. Never stops containers or downloads model weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

from .config import read, require, service_engine
from .migration import WRAPPERS, shell_wrapper
from .system import atomic_json, lock, now
from .update import Updater


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def stable_state(state):
    for name, key, allowed in [
        ('transaction.json', 'phase', {'succeeded', 'recovered'}),
        ('update-job.json', 'phase', {'succeeded', 'recovered', 'failed'}),
    ]:
        path = state / name
        require(not path.exists() or read(path).get(key) in allowed,
                'Finish or recover maintenance before exporting: ' + name)
    reservation = state / 'router-maintenance.json'
    require(not reservation.exists() or not read(reservation).get('reserved'),
            'Router maintenance is reserved')


def model_inventory(root):
    artifacts = {}
    for profile in sorted((root / 'config/profiles').glob('*.json')):
        for item in read(profile)['artifacts']:
            entry = {k: item[k] for k in ('path', 'bytes', 'sha256')}
            require(item['path'] not in artifacts or artifacts[item['path']] == entry,
                    'Conflicting artifact definitions')
            artifacts[item['path']] = entry
    return list(artifacts.values())


def export(root, destination, include_images=True):
    root = Path(root).resolve(); destination = Path(destination).resolve()
    updater = Updater(root); state = root / '.state'
    require(state.is_dir(), 'Installed runtime state is required')
    require(not destination.exists(), 'Use a new, empty recovery destination')
    require(not destination.is_relative_to(root), 'Keep recovery exports outside the source checkout')
    updater.source_preflight()
    with lock(state / 'update.lock'), lock(state / 'runtime.lock'):
        stable_state(state)
        active = read(state / 'active.json')
        status = json.loads(updater.run('docker', 'exec', 'local-ai-runtime', 'python3', '-m', 'runtime', 'check'))
        require(status['ready'] and status['deployed_revision'] == active['revision'], 'Runtime is not ready')
        require(updater.git('rev-parse', 'HEAD') == active['revision'], 'Checkout and deployed release differ')
        engines = read(root / 'config/shared.json')['engines']
        expected = {e['tag']: e['image_id'] for e in engines.values()}
        controller_revisions = {active['image']: active['revision']}
        tags = set(expected) | {active['image']}
        previous = state / 'previous.json'
        if previous.exists():
            prior = read(previous)
            tags.add(prior['image'])
            controller_revisions[prior['image']] = prior['revision']
            for role in prior['bundle']['compose']['services']:
                engine = service_engine(prior['bundle'], role)
                tags.add(engine['tag']); expected[engine['tag']] = engine['image_id']
        images = []
        for tag in sorted(tags):
            image = json.loads(updater.run('docker', 'image', 'inspect', tag))[0]
            require(tag not in expected or image['Id'] == expected[tag], 'Engine identity drift: ' + tag)
            revision = (image['Config'].get('Labels') or {}).get('org.opencontainers.image.revision')
            require(tag not in controller_revisions or revision == controller_revisions[tag],
                    'Controller source identity drift: ' + tag)
            images.append({'tag': tag, 'id': image['Id'], 'bytes': image['Size'],
                           'revision': revision})
        destination.parent.mkdir(parents=True, exist_ok=True)
        require(not include_images or shutil.disk_usage(destination.parent).free >
                sum(i['bytes'] for i in images) + 1024**3, 'Insufficient space for image export')
        destination.mkdir(mode=0o700)
        os.chmod(destination, 0o700)
        # Secrets belong only in this private archive, never in the source bundle.
        with tarfile.open(destination / 'private-state.tar.gz', 'w:gz') as archive:
            for name in ['.env', '.state']:
                archive.add(root / name, arcname=name)
        updater.git('bundle', 'create', str(destination / 'runtime.git.bundle'), '--all')
        if include_images:
            updater.run('docker', 'image', 'save', '-o', str(destination / 'images.tar'), *sorted(tags), timeout=3600)
        inventory = {'created_at': now(), 'root': str(root), 'revision': active['revision'],
                     'profile': active['bundle']['profile'], 'host': read(state / 'host.json'),
                     'images_included': include_images, 'images': images,
                     'models_included': False, 'models': model_inventory(root),
                     'scope': 'Runtime only. Restore router, its credential/catalog path, drivers, network and model files separately.'}
        atomic_json(destination / 'inventory.json', inventory)
        for name in ['model-downloads.json', 'rebuild.md']:
            source = root / 'docs' / name
            if source.exists(): shutil.copy2(source, destination / name)
        checksums = {p.name: checksum(p) for p in sorted(destination.iterdir()) if p.is_file()}
        atomic_json(destination / 'checksums.json', checksums)
        for path in destination.iterdir(): path.chmod(0o600)
        print(json.dumps({'destination': str(destination), 'profile': inventory['profile'],
                          'images': len(images), 'models_included': False, 'complete': True}))


def install_wrappers(home):
    home = Path(home).resolve()
    require(home.is_dir(), 'Home directory does not exist')
    for name in WRAPPERS:
        path = home / name
        require(not path.is_symlink(), 'Refusing a symlink: ' + str(path))
        require(not path.exists() or path.read_text() == shell_wrapper(name),
                'Preserve the existing command before replacing it: ' + str(path))
    for name in WRAPPERS:
        path = home / name
        path.write_text(shell_wrapper(name)); path.chmod(0o755)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['export', 'install-wrappers'])
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--home', type=Path, default=Path.home())
    parser.add_argument('--without-images', action='store_true', help='Metadata-only export; not an image backup')
    args = parser.parse_args()
    if args.action == 'export':
        require(args.destination is not None, '--destination is required')
        export(args.root, args.destination, not args.without_images)
    else:
        install_wrappers(args.home)


if __name__ == '__main__':
    main()

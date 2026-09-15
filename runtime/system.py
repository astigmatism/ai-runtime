"""Small, mockable boundaries for Docker, HTTP, atomic state, and process locks."""
import contextlib
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile
import urllib.request


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another runtime operation holds the maintenance lock') from None
        yield


class System:
    def __init__(self, readonly=False):
        self.readonly = readonly

    def docker(self, *args, check=True, timeout=900):
        if self.readonly and not (args[:1] in [('inspect',), ('info',)]
                or args[:2] in [('image', 'inspect'), ('network', 'inspect')]):
            raise RuntimeError('Docker mutation is forbidden in inspection mode')
        result = subprocess.run(['docker', *args], text=True, capture_output=True, timeout=timeout)
        if check and result.returncode:
            raise RuntimeError('Docker ' + ' '.join(args[:3]) + ' failed: ' + result.stderr[-1800:])
        return result.stdout.strip()

    def inspect(self, name):
        raw = self.docker('inspect', name, check=False)
        return json.loads(raw)[0] if raw and raw != '[]' else None

    def http(self, url, body=None, headers=None, timeout=10):
        if self.readonly and body is not None:
            raise RuntimeError('HTTP mutation is forbidden in inspection mode')
        request = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', **(headers or {})})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

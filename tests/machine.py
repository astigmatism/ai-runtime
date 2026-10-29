"""The machine a test is asking.

Which configurations exist, what each one launches, and which names have been retired are facts about a
machine, so a test that renders has to say which machine it means. This is the one these suites mean:
the source's `config/` directory, which holds the engines a body may boot and nothing else, and the
configuration bodies themselves in `tests/fixtures/profiles` -- a machine's home in the only sense that
matters to a test, which is that a host states it, reads it, and edits it only where a test says so.
Both fixture machines read that one home, because what separates a machine from another is each body's
own `hosts` line and not a directory per box.

`tests/fixtures/legacy-fingerprints.json` records the host settings a real machine stated at the revision
its pins were cut on. `HOST` is that host with one line added: today a machine also names the directory
it keeps its own configurations in. That line is the whole difference between asking a checkout and
asking a machine, and it is why nothing in these suites renders out of the source tree.

`tree(destination)` puts a machine in a temporary directory, because a refusal worth reaching is a
refusal reached through a file on disk: the same edit applied to a dictionary in memory skips the part of
a rule that lives in how a body is found and read. The copy is a whole machine -- source half and home
half, record included -- so a test never has to remember which of the two it meant to copy.
"""
import copy
import shutil
from pathlib import Path

from runtime.config import machine_registry, read

ROOT = Path(__file__).resolve().parents[1]
# The source's config directory: shared.json with the engines, host.example.json, and no configuration
# bodies, because a configuration is a measurement of one machine and this checkout is not a machine.
CONFIG = ROOT / 'config'
HOME = ROOT / 'tests/fixtures/profiles'


def stating(host, home, **changes):
    """The same machine, keeping its configurations in `home`, with `changes` said about it.

    A host that states no directory of its own is asked a question it cannot answer; every fixture host
    here states one, and a test that copied a machine into a temporary directory re-points it at the copy.
    """
    return {**copy.deepcopy(host), 'profiles_dir': str(Path(home)), **changes}


HOST = stating(read(ROOT / 'tests/fixtures/legacy-fingerprints.json')['host'], HOME)


def tree(destination, **host_changes):
    """A machine under `destination`: the source's config directory, its own home beside it, and the
    host that reads them out. Returns the config directory to render against, its home, and the host.

    An existing tree is replaced, because re-copying over the same directory is how a test writes one
    refusal into a body, reaches it, and then starts again from what the machine actually says -- the
    same reset every one of these sites used to do with a `copytree` of its own.
    """
    destination = Path(destination)
    for directory in (destination / 'config', destination / 'profiles'):
        shutil.rmtree(directory, ignore_errors=True)
    shutil.copytree(CONFIG, destination / 'config')
    shutil.copytree(HOME, destination / 'profiles')
    return (destination / 'config', destination / 'profiles', stating(HOST, destination / 'profiles',
        **host_changes))


def home(destination):
    """This machine's own home, copied to `destination`, record and all.

    For a test that acts *on* a machine -- taking a configuration out of service -- where the bodies are
    the thing being edited and the source's config directory is only where an engine is defined.
    """
    destination = Path(destination)
    shutil.copytree(HOME, destination)
    return destination


def record(config_dir=CONFIG, host=HOST):
    """What the fixture machine says about itself: its order, its renames, its retirements, its default."""
    return machine_registry(config_dir, host)

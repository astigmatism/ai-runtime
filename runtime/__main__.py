import argparse
import json
import os
from pathlib import Path
import signal
import sys

from .controller import Controller
from .system import System
from .config import (CONFIGURATIONS, PROFILE_ALIASES, available_profiles, read, require,
    resolve_profile, unknown_profile)


def describe(entry, note=''):
    return f"{entry['profile']}: {entry['display_name']} - {entry['model']}{note}"


def host_settings():
    """This machine's settings, for the surfaces that show only this machine's share of the registry.

    `list` is also run against a bare source tree, where there is no host to ask, and then it lists
    every configuration the revision has. With a state directory it must read: a deployment whose
    host.json is missing or unreadable gets that said out loud rather than a listing of another
    machine's configurations, which is what a silent fallback would hand back.
    """
    state = os.environ.get('RUNTIME_STATE_DIR')
    if not state:
        return None
    return read(Path(state) / 'host.json')


def listed(config_dir, host=None):
    """The supported configurations straight from the profile registry, so the CLI, the
    container CLI consumers, and the status page can never disagree. With a host, the registry is
    that host's: a configuration of another machine is not a thing this host supports."""
    registry = available_profiles(config_dir, host)
    second = registry['always_included']
    # `always_included` is empty on a host whose every configuration runs alone; the listing has to
    # say that rather than reach for a second resident that no configuration loads.
    rows = ['primary: the selected configuration' +
        (f" + {second[0]['display_name']}" if second else ' (no second resident in any configuration)')]
    rows += [describe(entry, ' (owns every text GPU of this host; runs without a second resident)'
        if entry['exclusive'] else '') for entry in registry['selectable']]
    rows += [describe(entry, ' (the second resident a paired configuration loads beside it)') for entry in second]
    return '\n'.join(rows)


def main():
    parser = argparse.ArgumentParser(prog='ai-runtime')
    parser.add_argument('action', choices=['inspect', 'render', 'validate', 'status', 'apply',
        'deploy', 'adopt', 'install', 'recover', 'restore', 'rollback-release', 'publish', 'retire',
        'serve',
        'list', 'check', 'provision', 'gate',
        'router-maintenance-begin', 'router-maintenance-end'])
    # Not `choices`: argparse prints "invalid choice" and exits 2, which tells an operator that a
    # name they typed for three years is simply wrong. resolve_profile says what it means -- a
    # renamed configuration is what they typed, a retired one names its replacement.
    parser.add_argument('--profile')
    parser.add_argument('--phase', choices=['provision', 'warm'], default='provision')
    parser.add_argument('--full-hash', action='store_true')
    args = parser.parse_args()
    if args.profile:
        require(args.profile in CONFIGURATIONS or args.profile in PROFILE_ALIASES,
            unknown_profile(args.profile))
        args.profile = resolve_profile(args.profile)
    if args.action == 'list':
        print(listed(os.environ.get('RUNTIME_CONFIG_DIR', '/app/config'), host_settings()))
        return
    readonly = args.action in ('inspect', 'render', 'validate', 'status', 'check', 'gate')
    controller = Controller(system=System(readonly=readonly))
    if args.action == 'serve':
        from .web import serve
        serve(controller)
        return
    def interrupted(signum, frame):
        raise RuntimeError('Runtime operation interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    if args.action == 'inspect':
        result = controller.inspect(args.profile, full_hash=args.full_hash)
    elif args.action == 'render':
        result = controller.desired(args.profile)
    elif args.action == 'validate':
        result = controller.validate(controller.desired(args.profile), full_hash=args.full_hash)
    elif args.action in ('status', 'check'):
        result = controller.status()
        if args.profile:
            result.update(requested_profile=args.profile, active=result['profile'] == args.profile)
        if args.action == 'check' and not result['ready']:
            raise RuntimeError('Runtime is not ready: ' + json.dumps(result))
    elif args.action in ('apply', 'adopt', 'deploy'):
        result = controller.transition(args.profile, deploy=args.action == 'deploy',
            adopt=args.action == 'adopt', install=args.action == 'install')
    elif args.action == 'recover':
        result = controller.recover()
    elif args.action == 'restore':
        # The one action that brings the last-known-good release back; the same path a healthy
        # boot takes, so nobody has to reboot a machine to get the model serving again.
        result = controller.restore_last_known_good()
    elif args.action == 'provision':
        # An engine that builds its own state before it answers is not switched to on a guessed
        # window: this measures the build and the warm boot here, and the switch insists on it.
        from .provision import measurement
        if args.profile is None:
            raise RuntimeError('provision needs --profile')
        result = measurement(controller, args.profile, phase=args.phase)
    elif args.action == 'gate':
        # What a switch would do, given without switching: read-only, nothing locked, nothing
        # drained, no container asked to do anything. `reasons` are the refusals; `notes` are the
        # things the selection will do that are worth knowing first (a build, a thin window).
        if args.profile is None:
            raise RuntimeError('gate needs --profile')
        bundle = controller.desired(args.profile)
        reasons = controller.verdict(bundle)
        notes = controller.verdict_notes(bundle)
        result = {'profile': args.profile, 'switchable': not reasons, 'reasons': reasons, 'notes': notes}
    elif args.action == 'rollback-release':
        result = controller.rollback_release()
    elif args.action == 'publish':
        result = controller.republish()
    elif args.action == 'retire':
        result = controller.retire()
    elif args.action.startswith('router-maintenance-'):
        result = controller.router_maintenance(args.action.endswith('begin'))
    print(json.dumps(result, indent=2))
    if args.action == 'inspect' and not result['matches_live']:
        raise RuntimeError('Requested profile does not match the healthy live runtime')
    if args.action == 'gate' and not result['switchable']:
        raise SystemExit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('Error: ' + str(error), file=sys.stderr)
        sys.exit(1)

import argparse
import json
import os
import signal
import sys

from .controller import Controller
from .system import System
from .config import DAYTIME_PROFILES, available_profiles


def describe(entry, note=''):
    return f"{entry['profile']}: {entry['display_name']} - {entry['model']}{note}"


def listed(config_dir):
    """The supported configurations straight from the profile registry, so the CLI, the
    container CLI consumers, and the status page can never disagree."""
    registry = available_profiles(config_dir)
    night = registry['always_included'][0]
    rows = [f"primary: the selected Daytime configuration + {night['display_name']}"]
    rows += [describe(entry, ' (all four text GPUs; stops Nighttime)' if entry['exclusive'] else '')
        for entry in registry['selectable']]
    rows += [describe(entry, ' (paired with every non-exclusive Daytime configuration)') for entry in registry['always_included']]
    return '\n'.join(rows)


def main():
    parser = argparse.ArgumentParser(prog='ai-runtime')
    parser.add_argument('action', choices=['inspect', 'render', 'validate', 'status', 'apply',
        'deploy', 'adopt', 'recover', 'restore', 'rollback-release', 'publish', 'serve', 'list', 'check',
        'provision', 'gate',
        'router-maintenance-begin', 'router-maintenance-end'])
    parser.add_argument('--profile', choices=DAYTIME_PROFILES)
    parser.add_argument('--phase', choices=['provision', 'warm'], default='provision')
    parser.add_argument('--full-hash', action='store_true')
    args = parser.parse_args()
    if args.action == 'list':
        print(listed(os.environ.get('RUNTIME_CONFIG_DIR', '/app/config')))
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
        result = controller.transition(args.profile, deploy=args.action == 'deploy', adopt=args.action == 'adopt')
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
        # The refusal a switch would hand back, given without switching: read-only, nothing locked,
        # nothing drained, no container asked to do anything.
        if args.profile is None:
            raise RuntimeError('gate needs --profile')
        reasons = controller.verdict(controller.desired(args.profile))
        result = {'profile': args.profile, 'switchable': not reasons, 'reasons': reasons}
    elif args.action == 'rollback-release':
        result = controller.rollback_release()
    elif args.action == 'publish':
        result = controller.republish()
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

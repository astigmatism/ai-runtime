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
    rows += [describe(entry) for entry in registry['selectable']]
    rows += [describe(entry, ' (paired with every Daytime configuration)') for entry in registry['always_included']]
    return '\n'.join(rows)


def main():
    parser = argparse.ArgumentParser(prog='ai-runtime')
    parser.add_argument('action', choices=['inspect', 'render', 'validate', 'status', 'apply',
        'deploy', 'adopt', 'recover', 'rollback-release', 'publish', 'serve', 'list', 'check',
        'router-maintenance-begin', 'router-maintenance-end'])
    parser.add_argument('--profile', choices=DAYTIME_PROFILES)
    parser.add_argument('--full-hash', action='store_true')
    args = parser.parse_args()
    if args.action == 'list':
        print(listed(os.environ.get('RUNTIME_CONFIG_DIR', '/app/config')))
        return
    readonly = args.action in ('inspect', 'render', 'validate', 'status', 'check')
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
    elif args.action == 'rollback-release':
        result = controller.rollback_release()
    elif args.action == 'publish':
        result = controller.republish()
    elif args.action.startswith('router-maintenance-'):
        result = controller.router_maintenance(args.action.endswith('begin'))
    print(json.dumps(result, indent=2))
    if args.action == 'inspect' and not result['matches_live']:
        raise RuntimeError('Requested profile does not match the healthy live runtime')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('Error: ' + str(error), file=sys.stderr)
        sys.exit(1)

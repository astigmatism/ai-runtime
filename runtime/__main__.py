import argparse
import json
import os
from pathlib import Path
import signal
import sys

from .controller import Controller
from .system import System
from .config import (ALIASES_FILE, CONFIGURATIONS, PROFILE_ALIASES, available_profiles, read,
    nameable_models, read_aliases, require, resolve_profile, unknown_profile)


def describe(entry, note=''):
    return f"{entry['profile']}: {entry['display_name']} - {entry['model']}{note}"


def host_settings(config_dir):
    """This machine's settings and its operator names, for the surfaces that show only this machine's
    share of the registry.

    `list` is also run against a bare source tree, where there is no host to ask, and then it lists
    every configuration the revision has, each by its default name. With a state directory it must
    read: a deployment whose host.json is missing or unreadable gets that said out loud rather than a
    listing of another machine's configurations, which is what a silent fallback would hand back.
    Names travel with the host because the listing and the page must never disagree about what a
    model is called, which is the same fact stated in two places that people trust.
    """
    state = os.environ.get('RUNTIME_STATE_DIR')
    if not state:
        return None, {}
    host = read(Path(state) / 'host.json')
    return host, read_aliases(Path(state) / ALIASES_FILE, nameable_models(config_dir, host))[0]


def listed(config_dir, host=None, aliases=None):
    """The supported configurations straight from the profile registry, so the CLI, the
    container CLI consumers, and the status page can never disagree. With a host, the registry is
    that host's: a configuration of another machine is not a thing this host supports."""
    registry = available_profiles(config_dir, host, aliases)
    # Each configuration is listed with the models it loads, indented under it, because that is what
    # it is. `list` is the command an operator runs to ask what this machine could be doing, and the
    # answer is models per configuration -- one, two, or as many as its cards hold -- not a
    # configuration with a footnote about whatever runs beside it.
    rows = []
    for entry in registry['selectable']:
        rows.append(describe(entry))
        for model in entry['models']:
            rows.append('    %s · %s · %s' % (model['display_name'],
                'every text GPU of this host' if model['exclusive']
                else ' + '.join(model['gpu_names']) or 'cards unknown',
                ', '.join(model['service_ids']) or 'no service ID'))
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
        config_dir = os.environ.get('RUNTIME_CONFIG_DIR', '/app/config')
        host, aliases = host_settings(config_dir)
        print(listed(config_dir, host, aliases))
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
    elif args.action in ('apply', 'adopt', 'deploy', 'install'):
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

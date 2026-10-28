import argparse
import json
import os
from pathlib import Path
import signal
import sys

from .controller import Controller
from .system import System
from .config import (ALIASES_FILE, available_profiles, machine_registry, nameable_models, read, read_aliases,
    require, resolve_profile, unknown_profile)


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


def power_card(power, wanted):
    """A card from `--gpu` or `--omit`, named by UUID or by enough of a name to be one card.

    A full UUID is 40 characters and nobody types one from memory on a terminal; the script this
    command replaces took a moniker like `3090`. What it must never take is an ambiguous one, because
    the whole point of a UUID here is that it is the card and not a card of that model. The rule is the
    runtime's own, so the page, `--gpu` and `--omit` cannot disagree about which card a word names.
    """
    try:
        return power.card(wanted)
    except Exception as error:
        require(False, str(error).replace(
            'name one card, or give its full UUID',
            'name one card, or give its full UUID for --gpu'))


def power_command(controller, args):
    """The card power state from a terminal: what every card is doing, one rung to one, or a set to all.

    The same table, the same refusals and the same read-back as the page, because both read
    config/undervolt.json and one cache. `power` with no card reports what the driver last said about
    each card; `--gpu … --read` asks the driver again right now; `--rung`, `--revert` and `--set`
    write. `--save-set NAME` records the whole machine as it stands, `--use-set NAME` puts that back,
    `--use-set stock` reverts every card it reports, and `--sets` says how each set fits the cards
    there are now. Nothing here survives a reboot, by design: the reboot is the recovery method, and
    this command does not arrange one.
    """
    power = controller.power
    if args.plans:
        registry = power.registry()
        require(registry, 'The rung table could not be read: ' + str(power.registry_error()))
        # `this_machine` is what an operator wrote down; `for_cards` is what each card is actually
        # offered, which is that table with the plan measured for the card's own model applied where
        # nothing was written down. Printing both is how a card with no rungs gets explained from a
        # terminal: "nobody has measured this model" and "this machine turned it off" are answers, and
        # "the plan a card is offered is invisible" is the bug that started this.
        resolved = power.plans_for_cards()
        return {'envelope': registry['envelope'], 'plans': registry['plans'],
            'this_machine': power.host_plans(), 'for_cards': resolved}
    if args.sets:
        # The same block the page draws: `stock` first, then each saved set, every line resolved against
        # the cards this machine reports right now — which is how a set with a card that left the
        # machine reads here exactly as it reads there.
        registry = power.registry()
        require(registry, 'The rung table could not be read: ' + str(power.registry_error()))
        return power.sets_block(registry)
    name = args.save_set or args.use_set or args.delete_set
    if name:
        action = ('save' if args.save_set else 'apply' if args.use_set else 'delete')
        if args.omit:
            require(action == 'save', '--omit leaves cards out of a *save*. An apply writes the cards '
                'its set names and skips only cards that are no longer in the machine.')
        return controller.power.write_sets({'action': action, 'name': name, 'omit': args.omit,
            'override': bool(args.override)})
    if not args.gpu:
        return power.status()
    body = {'gpu': power_card(power, args.gpu)}
    if args.rung:
        body.update(action='apply', rung=args.rung)
    elif args.revert:
        body.update(action='revert')
    elif getattr(args, 'set'):
        parts = [part.strip() for part in getattr(args, 'set').split(',')]
        require(len(parts) == 3, '--set takes POWER,CLOCK_MAX,OFFSET in watts and MHz, '
            'for example --set 330,1905,75')
        watts, clock_max, offset = parts
        body.update(action='custom', power=int(watts), clock_max=int(clock_max), offset=int(offset))
    else:
        body.update(action='read')
    body['override'] = bool(args.override)
    return power.write(body)


def main():
    parser = argparse.ArgumentParser(prog='ai-runtime')
    parser.add_argument('action', choices=['inspect', 'render', 'validate', 'status', 'apply',
        'deploy', 'adopt', 'install', 'recover', 'restore', 'rollback-release', 'publish', 'retire',
        'serve', 'power',
        'list', 'check', 'provision', 'gate',
        'router-maintenance-begin', 'router-maintenance-end'])
    # Not `choices`: argparse prints "invalid choice" and exits 2, which tells an operator that a
    # name they typed for three years is simply wrong. resolve_profile says what it means -- a
    # renamed configuration is what they typed, a retired one names its replacement.
    parser.add_argument('--profile')
    parser.add_argument('--phase', choices=['provision', 'warm'], default='provision')
    parser.add_argument('--full-hash', action='store_true')
    # Only `power` reads these. A card is named by UUID or by enough of its name to be one card;
    # `--plans` prints the rung table without touching a driver.
    parser.add_argument('--gpu')
    parser.add_argument('--rung')
    parser.add_argument('--set', default=None, metavar='POWER,CLOCK_MAX,OFFSET')
    parser.add_argument('--revert', action='store_true')
    parser.add_argument('--plans', action='store_true')
    # Power sets: the whole machine's card states, saved under a name and recalled by it. `--use-set
    # stock` is the one that always exists — every card this machine reports, back at its own numbers.
    parser.add_argument('--sets', action='store_true', help='with power: list the power sets and how '
        'each one fits the cards this machine has now')
    parser.add_argument('--save-set', default=None, metavar='NAME',
        help='with power: record what every card is sitting at now, as a set named NAME')
    parser.add_argument('--use-set', default=None, metavar='NAME',
        help='with power: put the named set back on the cards it still fits (`stock` reverts them all)')
    parser.add_argument('--delete-set', default=None, metavar='NAME', help='with power: forget a saved set')
    parser.add_argument('--omit', action='append', default=[], metavar='CARD',
        help='with power --save-set: leave this card out of the set, by UUID or by enough of a name '
            'to be one card')
    parser.add_argument('--override', action='store_true',
        help='with power: write even when the model on that card is not healthy')
    args = parser.parse_args()
    config_dir = os.environ.get('RUNTIME_CONFIG_DIR', '/app/config')
    host = aliases = None
    if args.profile or args.action == 'list':
        # A name typed at a keyboard is asked about on the machine that keeps the configurations, not
        # against this revision's own list: whether a name is one of *this* machine's configurations,
        # and what one of its own older names resolves to, is in the record it keeps beside their
        # bodies. With no state directory there is no machine to ask, and the answer is this revision's
        # list -- which is what a bare source tree, and a deployment that has not moved its
        # configurations into its own home, has always been given.
        host, aliases = host_settings(config_dir)
    if args.profile:
        profile = resolve_profile(args.profile, config_dir, host)
        require(profile in machine_registry(config_dir, host)['selectable'],
            unknown_profile(args.profile, config_dir, host))
        args.profile = profile
    if args.action == 'list':
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
    elif args.action == 'power':
        result = power_command(controller, args)
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

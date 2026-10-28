"""Compatibility with the existing home-folder commands; executed inside the image."""
import os
import sys
from .__main__ import host_settings, main
from .config import machine_registry, resolve_profile, unknown_profile

# Names the home wrappers have accepted for years. They name a resident, not a configuration, and
# were never passed through as one -- which is exactly why they outlive a rename of the
# configurations that happen to load it.
RESIDENT_NAMES = ('primary', 'nighttime', 'daytime-256', 'nighttime-256')


def dispatch(argv):
    command, *arguments = argv
    if command == 'local-ai-config.sh':
        action = arguments[0] if arguments else 'list'
        profile = arguments[1] if len(arguments) > 1 else 'primary'
    else:
        profile = command
        action = arguments[0] if arguments else 'apply'
    action = {'names': 'list', 'plan': 'render', 'show': 'render',
        'render-runtime': 'render', 'active-check': 'inspect'}.get(action, action)
    if action not in ('list', 'render', 'inspect', 'validate', 'status', 'apply'):
        raise RuntimeError('Use list, status, apply, validate, plan, or active-check; historical restore commands are retired')
    # A renamed configuration is applied under the id this machine knows; a retired one is refused, and
    # the refusal names what to use instead. The wrapper runs on a machine, so it asks the machine: the
    # names it accepts are the ones this box keeps in the record beside its own configuration bodies,
    # which is what lets a rename made on the machine be honoured by a command written before the
    # machine had either.
    config_dir = os.environ.get('RUNTIME_CONFIG_DIR', '/app/config')
    host, _ = host_settings(config_dir)
    selectable = machine_registry(config_dir, host)['selectable']
    selected = resolve_profile(profile, config_dir, host)
    if selected not in selectable and profile not in RESIDENT_NAMES:
        raise RuntimeError(unknown_profile(profile, config_dir, host))
    return [action, *( ['--profile', selected] if selected in selectable else [] )]


if __name__ == '__main__':
    try:
        sys.argv = ['ai-runtime', *dispatch(sys.argv[1:])]
        main()
    except Exception as error:
        print('Error: ' + str(error), file=sys.stderr)
        sys.exit(1)

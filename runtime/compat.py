"""Compatibility with the existing home-folder commands; executed inside the image."""
import sys
from .__main__ import main
from .config import CONFIGURATIONS, resolve_profile, unknown_profile

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
    # A renamed configuration is applied under the id this revision knows; a retired one is refused,
    # and the refusal names what to use instead.
    selected = resolve_profile(profile)
    if selected not in CONFIGURATIONS and profile not in RESIDENT_NAMES:
        raise RuntimeError(unknown_profile(profile))
    return [action, *( ['--profile', selected] if selected in CONFIGURATIONS else [] )]


if __name__ == '__main__':
    try:
        sys.argv = ['ai-runtime', *dispatch(sys.argv[1:])]
        main()
    except Exception as error:
        print('Error: ' + str(error), file=sys.stderr)
        sys.exit(1)

"""Compatibility with the existing home-folder commands; executed inside the image."""
import sys
from .__main__ import main


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
    profile_args = ['--profile', profile] if profile in ('daytime', 'daytime-swift') else []
    if profile not in ('primary', 'daytime', 'daytime-swift', 'nighttime', 'daytime-256', 'nighttime-256'):
        raise RuntimeError('Unknown or retired profile')
    return [action, *profile_args]


if __name__ == '__main__':
    try:
        sys.argv = ['ai-runtime', *dispatch(sys.argv[1:])]
        main()
    except Exception as error:
        print('Error: ' + str(error), file=sys.stderr)
        sys.exit(1)

"""A machine's own levers over the configurations it keeps: read its record, take one out of service.

`runtime registry` answers "which configurations does this machine offer" out of the record it keeps
beside their bodies, and `runtime retire-profile` performs the retirement there: the body leaves the
machine's own directory, the name leaves its own record, and the record gains the row that says what
replaced it. No source change is involved and no other machine is asked, which is the whole point -- the
updater reading a target revision, the status page listing a machine's share, and the refusal that names a
replacement all read this same record, so a retirement is in force the moment it is written and costs
another machine nothing at all.

A retirement edits the one file every selection path reads, so it takes the locks the switch and the
updater take, it refuses instead of repairing, and it puts the record and the body it deleted back if what
is left no longer validates. The record is the receipt of the act: it is where the machine states what it
offers, and a retirement is a fact about what it no longer offers, so there is no second log to keep in
step with it. See docs/deployment.md, "A machine's own configurations".
"""
from pathlib import Path

from .config import (PROFILES_VERSION, REGISTRY_FILE, available_profiles, configurations_for,
    default_profile, machine_registry, profile_body, profiles_home, read, render, require, unknown_profile)
from .recovery import stable_state
from .system import atomic_json, lock


def active_release(state):
    """The bundle of the release a machine has recorded as serving, or nothing when it never served.

    A release is written as its `revision`, its `image`, and the `bundle` it published; the configuration it
    is running and the digest that launch renders to live *inside* that bundle, because they are facts of the
    launch and not of the record that points at it. A retirement has to know which launch is running to refuse
    to drain it, and has to be able to compare what that launch renders now against the digest its own receipt
    holds, so both are read from the place the updater, the restoration path and the status page read them.
    """
    if state is None:
        return {}
    path = Path(state) / 'active.json'
    return (read(path).get('bundle') or {}) if path.is_file() else {}


def audit(config_dir, host, state=None):
    """What a machine states about its own configurations, read from where it states it.

    Everything here comes from the machine's own record and the bodies beside it, in the order the machine
    gave them, which is what makes the answer reviewable on the box instead of something a person has to
    ask a repository about. `available_profiles()` is the same call the status page and `runtime list` use,
    so a retirement cannot leave the audit and the listing disagreeing about what is offered.
    """
    home = profiles_home(config_dir, host)
    record = machine_registry(config_dir, host)
    offered = configurations_for(config_dir, host)
    summaries = {entry['profile']: entry for entry in available_profiles(config_dir, host)['selectable']}
    active = active_release(state)
    # Bodies the record does not offer are not leftovers: a configuration names the models it loads, and a
    # model it loads has a body of its own in the same directory. `machine_registry()` refuses anything
    # that is neither, so naming which configurations load each one is the whole of what this has to say.
    kept = sorted(path.stem for path in home.glob('*.json')
        if path.name != REGISTRY_FILE and path.stem not in set(record['selectable']))
    return {'home': str(home), 'record': str(home / REGISTRY_FILE), 'schema_version': PROFILES_VERSION,
        'offers': [{'profile': name, 'display_name': summaries[name]['display_name'],
            'service_ids': summaries[name]['service_ids'],
            'models': [model['model'] for model in summaries[name]['models']]} for name in offered],
        # A record is one machine's statement; a host asks, and a body built for another machine's cards is
        # refused rather than offered. Both facts are worth seeing at once, because the first is what is on
        # disk and the second is what an operator can actually select.
        'not_for_this_host': [name for name in record['selectable'] if name not in offered],
        'model_bodies': [{'body': name, 'loaded_by': [profile for profile in offered
            if name in (profile_body(config_dir, host, profile).get('models') or [])]} for name in kept],
        'renamed': record['renamed'], 'retired': record['retired'],
        'starts_on': default_profile(config_dir, host),
        'starts_on_stated': record['default'] is not None,
        'running': active.get('profile')}


def retire(config_dir, state, host, profile, replacement):
    """Take one of this machine's configurations out of service, on the machine that keeps it.

    The refusal cases are the point of the function. It refuses while the configuration is what the
    machine is running -- an operator switches first, and a retirement never drains a workload by
    surprise -- while maintenance is mid-flight, and when the record would be left stating something
    nothing can select: a replacement the machine does not offer, a rename that promised this
    configuration's own backends, or a configuration still loading it as a model. What it does accept, it
    does in one step under both maintenance locks, and it undoes if what remains no longer renders.
    """
    state = Path(state)
    require(host is not None, 'Retiring a configuration is an act on the machine that keeps it, and no '
        'machine is being asked: run this against a deployment, where .state/host.json says which machine '
        'this is and profiles_dir says where it keeps its configurations')
    home = profiles_home(config_dir, host)
    record = machine_registry(config_dir, host)
    require(home.is_relative_to(state), 'This machine keeps its configurations at ' + str(home) + ', outside '
        'the state directory at ' + str(state) + '. A retirement writes there, and the state directory is '
        'the part of a deployment the container is allowed to write in')
    require(profile in record['selectable'], unknown_profile(profile, config_dir, host))
    offered = configurations_for(config_dir, host)
    require(replacement in offered, 'This machine does not offer ' + replacement + ', so it cannot be what '
        + profile + ' is retired in favour of. Retiring a configuration leaves a refusal that names a '
        'replacement, and a refusal that names something nobody can select is a dead end.')
    require(profile != replacement, profile + ' cannot be retired in favour of itself')

    active = active_release(state)
    require(active.get('profile') != profile, profile + ' is what this machine is running now. Switch it '
        'off it first -- runtime apply --profile ' + replacement + ' -- and then retire ' + profile +
        '; a retirement that drained a serving workload to take a definition out of a directory would be '
        'the machine acting on the operator instead of the other way round')
    with lock(state / 'update.lock'), lock(state / 'runtime.lock'):
        stable_state(state, 'retiring a configuration')
        # A configuration id is also a model id, and a model body is what another configuration mounts. A
        # retirement deletes a body, so it refuses while a configuration the machine still offers loads it.
        loaders = sorted(name for name in offered if name != profile
            and profile in (profile_body(config_dir, host, name).get('models') or []))
        require(not loaders, profile + ' is a model ' + ', '.join(loaders) + ' loads. Retiring it deletes '
            'the definition those configurations mount, so take them out of service first, or restate them '
            'to load something else.')
        # A rename promises the same backends under the name a client used to type. A retirement says the
        # backends are gone, which is why a retired name refuses instead of resolving. A rename pointing at
        # a configuration being retired is the two promises on the same row, so the row is the operator's
        # to move or drop -- this function does not guess which one was meant.
        renamed = sorted(old for old, new in record['renamed'].items() if new == profile)
        require(not renamed, 'This machine resolves ' + ', '.join(renamed) + ' to ' + profile + ' as a '
            'rename, and a rename promises the backends that configuration ran. ' + profile + ' is being '
            'retired, where the backends change; move those rows onto ' + replacement + ', or drop them, '
            'and retire ' + profile + ' again.')
        # A retired name is a pointer to a successor, so a successor's successor is the answer the row
        # exists to give: the rows pointing at the retiring configuration are carried onto its replacement.
        retired = {old: (replacement if new == profile else new) for old, new in record['retired'].items()}
        retired[profile] = replacement
        carried = sorted(old for old, new in record['retired'].items() if new == profile)
        default = replacement if record['default'] == profile else record['default']
        active_before = None
        if active.get('profile'):
            # The receipt is compared before anything is written. A retirement is the wrong moment to find
            # out that the running launch no longer renders to what was published: that is a drift to
            # settle on its own, not one to bury under a record edit.
            active_before = render(config_dir, host, active['profile'])['config_sha256']
            require(active_before == active.get('config_sha256'), 'What this machine has recorded as active '
                'no longer renders to the digest its receipt holds: ' + str(active_before) + ' now, '
                + str(active.get('config_sha256')) + ' recorded. Settle that before retiring ' + profile +
                ', so a retirement is not what appears to have changed a running launch.')
        body = home / (profile + '.json')
        before = read(home / REGISTRY_FILE)
        kept_body = body.read_bytes()
        after = {'schema_version': PROFILES_VERSION,
            'profiles': [name for name in record['selectable'] if name != profile],
            'renamed': record['renamed'], 'retired': retired}
        if default is not None:
            after['default_profile'] = default
        atomic_json(home / REGISTRY_FILE, after)
        body.unlink()
        try:
            # What is left offered is validated as a whole: every remaining configuration is read through
            # the same path a selection goes through, so a record edit cannot leave a machine offering
            # something that no longer composes.
            remaining = configurations_for(config_dir, host)
            rendered = {name: render(config_dir, host, name)['config_sha256'] for name in remaining}
            if active_before is not None:
                # The record sits outside what a launch renders, and nothing this machine still offers
                # mounts the body being deleted -- both were refused above. This proves it instead of
                # asserting it: the running launch renders to the bytes it already published, so taking a
                # definition out of service drained nobody.
                require(rendered[active['profile']] == active_before, 'Retiring ' + profile + ' would change '
                    'what this machine is running: ' + active['profile'] + ' now renders to ' +
                    str(rendered[active['profile']]) + ' and not to ' + str(active_before))
        except Exception as error:
            atomic_json(home / REGISTRY_FILE, before)
            body.write_bytes(kept_body)
            raise RuntimeError('Retiring ' + profile + ' is refused: ' + str(error) + '. ' + REGISTRY_FILE +
                ' and ' + profile + '.json are back where they were, and this machine is still offering ' +
                ', '.join(configurations_for(config_dir, host))) from None
        # The answer is the audit of the machine as it now stands -- one vocabulary, so a listing read
        # after a retirement cannot disagree with the listing read before it -- with the act itself in its
        # own field. Naming the configuration retired and the table of retired names with the same key is
        # how one of the two quietly disappears, which is why the act is a thing of its own here.
        audited = audit(config_dir, host, state)
        return {**audited, 'act': {'retired': profile, 'replacement': replacement,
            'carried_retirements': carried, 'default_followed_replacement': record['default'] == profile},
            'drain': 'none'}

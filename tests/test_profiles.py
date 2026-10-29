"""A machine's own levers over the configurations it keeps, exercised on a machine.

`runtime registry` and `runtime retire-profile` are the two acts a machine takes over its own
configurations, and both of them live or die on files: the record beside the bodies, the body being taken
out of service, the receipt of what is running. So every case here is run against a deployment written to
a temporary directory -- a checkout half and a state half, the home named by the host's own settings --
because the interesting failures are not in the arithmetic of a dictionary. A retirement that deleted a
body a remaining configuration mounts, or left a record offering a name with nothing beside it, or left the
machine offering something that no longer composes, each shows up as a file in the wrong place.

The refusal cases get as much room as the success case, because a refusal is what protects a serving
workload and a rename an operator wrote years ago. The success case is checked the way an operator would
check it on the box: every configuration the machine still offers renders to what it rendered before, so
nothing was drained to take a definition out of a directory.
"""
import shutil
import tempfile
import unittest
from pathlib import Path

import machine
from runtime import profiles
from runtime.config import (PROFILES_VERSION, REGISTRY_FILE, available_profiles, configurations_for,
    default_profile, read, render, resolve_profile, unknown_profile)
from runtime.system import atomic_json

YARN = 'flash-next-strata-solo-thinking-recovery-cache-yarn'
YARN_REPLACEMENT = 'flash-next-strata-solo-thinking-recovery-cache'
RECOVERY = 'flash-next-strata-solo-thinking-recovery'
ROSALINA, DAISY = 'rosalina', 'daisy'
CODING = 'qwen27b-q8-pair'
CODING_Q6K = 'qwen27b-q6k-pair'
EVERYDAY = 'everyday'
TWO_CARD_SOLO = 'flash-next-strata-two-card-iq3-s'
TWO_CARD_QUIET = 'flash-next-strata-two-card-iq4-xs'


class MachineKeepingItsConfigurations(unittest.TestCase):
    def deployment(self, host_id=ROSALINA, **host_changes):
        """A machine on disk: the source it was built from, its state directory, and the home it states.

        Shaped like a real deployment -- `config/` beside `.state/`, the configuration bodies inside
        `.state/` because that is the part of a deployment the container is mounted to write in -- so a
        rule that only exists on disk is in the fixture instead of in a test's head.
        """
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        state = root / '.state'
        state.mkdir()
        shutil.copytree(machine.CONFIG, root / 'config')
        home = machine.home(state / 'profiles')
        host = machine.stating(machine.HOST, home, **({} if host_id is None else {'host_id': host_id}),
            **host_changes)
        atomic_json(state / 'host.json', host)
        return root / 'config', state, home, host

    def running(self, state, config, host, profile):
        """Say this machine is running that configuration, and mean it: write the release record a switch
        writes. `active.json` is a revision, an image, and the bundle it published -- the configuration and
        the digest it renders to live *inside* that bundle -- so that is where this puts them, and a
        retirement that read them from anywhere else would be reading a shape no machine writes."""
        bundle = render(config, host, profile)
        atomic_json(state / 'active.json', {'revision': 'b' * 40, 'image': 'local/ai-runtime:git-' + 'b' * 40,
            'applied_at': '2026-10-28T04:00:00Z', 'bundle': bundle})
        return bundle['config_sha256']

    def digests(self, config, host):
        return {name: render(config, host, name)['config_sha256'] for name in configurations_for(config, host)}

    def test_the_record_and_the_bodies_beside_it_are_what_the_audit_reports(self):
        config, state, home, host = self.deployment()
        audited = profiles.audit(config, host, state)
        record = machine.record(config, host)
        own = [name for name in record['selectable']
            if ROSALINA in read(machine.HOME / (name + '.json'))['hosts']]
        self.assertEqual(audited['home'], str(home))
        self.assertEqual(audited['record'], str(home / REGISTRY_FILE))
        self.assertEqual(audited['schema_version'], PROFILES_VERSION)
        # In the order the machine wrote them down, because that order is a decision: it is what the
        # machine starts on when it states no default, and the order a page or a listing walks its
        # configurations in. An audit that sorted them would be a listing of the household's.
        self.assertEqual([entry['profile'] for entry in audited['offers']], own)
        self.assertEqual(audited['not_for_this_host'], [name for name in record['selectable'] if name not in own],
            "the bodies of the other machine's configurations are in the record too, and an audit that "
            "pretended they were not would be a machine describing a household")
        listed = {entry['profile']: entry for entry in available_profiles(config, host)['selectable']}
        for entry in audited['offers']:
            with self.subTest(profile=entry['profile']):
                self.assertEqual(entry['display_name'], listed[entry['profile']]['display_name'])
                self.assertEqual(entry['service_ids'], listed[entry['profile']]['service_ids'])
                self.assertEqual(entry['models'], [model['model'] for model in
                    listed[entry['profile']]['models']])
        # A body the record does not offer is not a leftover to clean up. A configuration names the models
        # it loads, and those models have bodies here too, so the only thing this can honestly say about
        # each one is which configurations mount it -- which is also everything the record read insists on.
        self.assertEqual(audited['model_bodies'], [{'body': EVERYDAY, 'loaded_by': [name for name in
            record['selectable'] if EVERYDAY in read(home / (name + '.json'))['models']]}])
        self.assertEqual(audited['renamed'], record['renamed'])
        self.assertEqual(audited['retired'], record['retired'])
        self.assertEqual(audited['starts_on'], default_profile(config, host))
        self.assertFalse(audited['starts_on_stated'], 'this machine wrote no default, so its own first '
            'configuration is its default, and an audit that did not say which of the two it was would '
            'leave an operator guessing about a fact the machine has stated')
        self.assertIsNone(audited['running'], 'a machine that has never switched has nothing recorded as '
            'active, and the audit says nothing rather than naming its default as if it were serving')

    def test_a_machine_is_asked_about_its_own_configurations_and_not_the_households(self):
        config, state, home, host = self.deployment(host_id=DAISY, display_name='Daisy')
        audited = profiles.audit(config, host, state)
        daisy = [name for name in machine.record()['selectable']
            if DAISY in read(machine.HOME / (name + '.json'))['hosts']]
        self.assertEqual([entry['profile'] for entry in audited['offers']], daisy)
        # Both facts are on the record and both are worth seeing together: what the machine has written
        # down, and what an operator of this box can actually select out of it.
        self.assertEqual(audited['not_for_this_host'], [name for name in machine.record()['selectable']
            if name not in daisy])
        self.assertEqual(audited['model_bodies'], [{'body': EVERYDAY, 'loaded_by': []}],
            "Daisy offers no configuration that mounts the model this fixture's home keeps beside its "
            "record, and an audit that named a loader for it would be describing a machine it was not "
            "asked about")
        self.assertEqual(audited['starts_on'], daisy[0], 'a machine that runs none of the first '
            'configurations of its record starts on the first of its own')
        self.assertEqual(audited['renamed'], machine.record()['renamed'], 'a rename a machine never made '
            'is not this machine\'s to answer for, and a fixture that invented one would hide that')

    def test_the_audit_names_the_configuration_the_machine_is_running(self):
        config, state, home, host = self.deployment()
        self.running(state, config, host, CODING)
        self.assertEqual(profiles.audit(config, host, state)['running'], CODING)

    def test_a_retirement_refuses_when_the_running_launch_no_longer_renders_to_its_own_receipt(self):
        # The comparison a retirement makes is against the digest the machine published, held inside the
        # bundle its release record carries. A machine whose running launch has already drifted is a thing to
        # settle on its own, and a retirement that wrote its record over the drift would leave the machine
        # blaming the retirement for a launch that had already moved.
        config, state, home, host = self.deployment()
        active = self.running(state, config, host, CODING)
        drift = {**read(machine.HOME / (CODING + '.json')), 'context_tokens': 131072}
        atomic_json(home / (CODING + '.json'), drift)
        self.assertNotEqual(render(config, host, CODING)['config_sha256'], active,
            'the fixture is meant to move the running launch, and a fixture that did not would test nothing')
        record = read(home / REGISTRY_FILE)
        body = (home / (YARN + '.json')).read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'no longer renders to the digest its receipt holds.*'
                'Settle that before retiring'):
            profiles.retire(config, state, host, YARN, YARN_REPLACEMENT)
        self.assertEqual(read(home / REGISTRY_FILE), record, 'the drift was the machine\'s to settle, not for '
            'this act to edit around')
        self.assertEqual((home / (YARN + '.json')).read_bytes(), body)
        self.assertIn(YARN, configurations_for(config, host))

    def test_a_machine_that_has_never_written_down_its_configurations_has_no_answer(self):
        config, state, home, host = self.deployment()
        (home / REGISTRY_FILE).unlink()
        for act, expected in ((lambda: profiles.audit(config, host, state), 'keeps no list of configurations'),
            (lambda: profiles.retire(config, state, host, YARN, YARN_REPLACEMENT),
                'keeps no list of configurations')):
            with self.subTest(act=act), self.assertRaisesRegex(RuntimeError, expected):
                act()
        self.assertTrue((home / (YARN + '.json')).exists())

    def test_a_retirement_takes_a_configuration_out_of_service_and_costs_the_running_launch_nothing(self):
        config, state, home, host = self.deployment()
        before = self.digests(config, host)
        offered = configurations_for(config, host)
        started_on = audited_starts_on(config, host)
        active = self.running(state, config, host, CODING)
        body = home / (YARN + '.json')
        kept_body = body.read_bytes()
        result = profiles.retire(config, state, host, YARN, YARN_REPLACEMENT)
        self.assertEqual(result['act'], {'retired': YARN, 'replacement': YARN_REPLACEMENT,
            'carried_retirements': [], 'default_followed_replacement': False}, 'this machine stated no '
            "default of its own and its first configuration is not the one being retired, so nothing had "
            "to move; nothing was retired into what was taken out of service, so no row had to be carried")
        self.assertEqual(result['running'], CODING)
        self.assertEqual(result['drain'], 'none')
        self.assertEqual(result['starts_on'], started_on, 'a retirement that had no default to move left '
            'the machine starting on what it said it started on')
        self.assertFalse(body.exists(), 'the body of a configuration taken out of service is not kept')
        after = read(home / REGISTRY_FILE)
        self.assertEqual(after['profiles'], [name for name in machine.record()['selectable'] if name != YARN])
        self.assertEqual(after['retired'][YARN], YARN_REPLACEMENT)
        self.assertEqual(after['renamed'], machine.record()['renamed'], 'a retirement is about the backends '
            'being taken out of service, not about the older names other configurations were given')
        # The name still refuses, and refuses by pointing at where a client can still go. That is the whole
        # difference between a retirement and a deletion, and it is in force the moment the record is written.
        refused = unknown_profile(YARN, config, host)
        self.assertIn('was retired', refused)
        self.assertIn(YARN_REPLACEMENT, refused)
        for act in ((lambda: render(config, host, YARN)),
            (lambda: profiles.retire(config, state, host, YARN, YARN_REPLACEMENT)),
            (lambda: render(config, host, resolve_profile(YARN, config, host)))):
            with self.subTest(act=act), self.assertRaisesRegex(RuntimeError, 'Unknown or retired configuration'):
                act()
        # And the proof an operator asks for on the box: every configuration this machine still offers
        # renders to what it rendered before, including the one that is serving. The record sits outside
        # what a launch renders, and nothing left mounted the body that was deleted -- both were refused
        # above, and this is what it looks like when they hold.
        remaining = self.digests(config, host)
        self.assertEqual(remaining, {name: sha for name, sha in before.items() if name != YARN})
        self.assertEqual(render(config, host, CODING)['config_sha256'], active)
        # The listing read out of the machine after the act is the listing it gave before, with the retired
        # configuration taken out of it, in the order the machine wrote the record: the names of other
        # machines a shared record carries are not this machine's to offer, and a listing that showed them
        # would be a machine describing a household.
        self.assertEqual([entry['profile'] for entry in profiles.audit(config, host, state)['offers']],
            [name for name in offered if name != YARN],
            'the listing an operator reads after a retirement is the record it just wrote')

    def test_the_rows_that_pointed_at_a_retired_configuration_follow_it_to_its_successor(self):
        config, state, home, host = self.deployment()
        self.running(state, config, host, CODING)
        pointed_at = sorted(old for old, new in machine.record()['retired'].items() if new == RECOVERY)
        self.assertTrue(pointed_at, 'the fixture machine retired six configurations into this one')
        result = profiles.retire(config, state, host, RECOVERY, YARN_REPLACEMENT)
        self.assertEqual(result['act']['carried_retirements'], pointed_at)
        retired = read(home / REGISTRY_FILE)['retired']
        self.assertEqual([retired[old] for old in pointed_at], [YARN_REPLACEMENT] * len(pointed_at))
        self.assertEqual(retired[RECOVERY], YARN_REPLACEMENT)
        self.assertFalse((home / (RECOVERY + '.json')).exists())
        # A name retired twice answers with the configuration that exists now, which is what carrying a
        # row onto a successor is for: these were retired into a configuration this machine has now
        # retired in its own turn, and an operator typing the old habit gets one answer, not a chain.
        for old in pointed_at:
            with self.subTest(name=old):
                self.assertIn('was retired; select ' + YARN_REPLACEMENT, unknown_profile(old, config, host))

    def test_a_stated_default_follows_the_configuration_it_was_a_default_of(self):
        config, state, home, host = self.deployment()
        atomic_json(home / REGISTRY_FILE, {**read(home / REGISTRY_FILE), 'default_profile': YARN})
        self.running(state, config, host, CODING)
        result = profiles.retire(config, state, host, YARN, YARN_REPLACEMENT)
        self.assertTrue(result['act']['default_followed_replacement'])
        self.assertEqual(result['act']['retired'], YARN)
        self.assertEqual(result['starts_on'], YARN_REPLACEMENT)
        self.assertTrue(result['starts_on_stated'], 'the audit facts of the answer are the machine as it '
            'now stands, so an operator reads the same shape out of a retirement as out of a listing')
        self.assertEqual(default_profile(config, host), YARN_REPLACEMENT, 'a machine that said it starts on '
            'a configuration it has just taken out of service has to say what it starts on now, and '
            'guessing the next one over would be this revision deciding a machine\'s business for it')

    def test_every_way_a_retirement_refuses_before_it_writes_anything(self):
        cases = [
            ('while the machine is running it', YARN, lambda *m: None, YARN, YARN_REPLACEMENT,
                'what this machine is running now.*Switch it off it first', True),
            ('in favour of itself', None, lambda *m: None, YARN, YARN, 'retired in favour of itself', True),
            ('for a name this machine does not offer', None, lambda *m: None, YARN, 'daytime',
                'does not offer.*dead end', True),
            ('for a configuration nobody can select', None, lambda *m: None, YARN, 'flash-next-strata-solo',
                'does not offer.*dead end', True),
            ('a configuration this machine has never offered', None, lambda *m: None, 'flash-next-strata-solo',
                YARN_REPLACEMENT, 'Unknown or retired configuration', False),
            # A configuration id is also a model id, and the model body is what other configurations mount.
            # Retiring a configuration that is mounted by what the machine still offers deletes a mount.
            ('while configurations it still offers load it as a model', None,
                lambda c, h, hm: atomic_json(hm / REGISTRY_FILE,
                    {**read(hm / REGISTRY_FILE), 'profiles': read(hm / REGISTRY_FILE)['profiles'] + [EVERYDAY]}),
                EVERYDAY, YARN_REPLACEMENT, 'is a model.*loads.*take them out of service first', False),
            # A rename promises the same backends under the name a client used to type; a retirement says
            # the backends are gone. The two promises cannot share one row, and which was meant is the
            # operator's call to make, so the rows are left where they are.
            ('while a rename promises this configuration its own backends', None, lambda *m: None, CODING,
                CODING_Q6K, 'resolves qwen27b-q8-with-nighttime.*as a rename.*move those rows', True),
            ('while a record offers it and no body is beside it', None,
                lambda c, h, hm: (hm / (RECOVERY + '.json')).unlink(), RECOVERY, YARN_REPLACEMENT,
                'is not there.*a decision nothing can launch', False),
        ]
        for label, active, mutate, profile, replacement, expected, left_selectable in cases:
            with self.subTest(case=label):
                config, state, home, host = self.deployment()
                body = (home / (YARN + '.json')).read_bytes()
                if active: self.running(state, config, host, active)
                mutate(config, host, home)
                # The listing a refusal has to leave behind is the one the machine was giving at the moment
                # it was asked; asked for it, rather than assumed from the record, because the point of these
                # cases is that the machine's own answer did not change.
                offered_before = configurations_for(config, host) if left_selectable else None
                rendered_before = self.digests(config, host) if left_selectable else None
                # Snapshot after the case has been arranged: what a refusal has to leave behind is what the
                # machine was saying at the moment it was asked to retire something.
                record = read(home / REGISTRY_FILE)
                with self.assertRaisesRegex(RuntimeError, expected):
                    profiles.retire(config, state, host, profile, replacement)
                self.assertEqual(read(home / REGISTRY_FILE), record, label + ': the refusal left the record edited')
                self.assertEqual((home / (YARN + '.json')).read_bytes(), body, label + ': the refusal left '
                    'a body moved')
                # A refusal is only safe if the machine is still offering what it was offering, in the same
                # order, and the configuration the operator meant to retire is still there to be retired.
                if left_selectable:
                    self.assertEqual(configurations_for(config, host), offered_before,
                        label + ': the refusal left the machine offering something else')
                    self.assertEqual(self.digests(config, host), rendered_before,
                        label + ': the refusal left a configuration rendering differently')

    def test_a_retirement_refuses_while_maintenance_is_mid_flight(self):
        config, state, home, host = self.deployment()
        self.running(state, config, host, CODING)
        for name, body, expected in (
            ('update-job.json', {'phase': 'draining'}, 'Finish or recover maintenance before retiring.*update-job'),
            ('restoration.json', {'started_at': 'now'}, 'A restoration is bringing the service back'),
            ('router-maintenance.json', {'reserved': True}, 'Router maintenance is reserved')):
            with self.subTest(maintenance=name):
                atomic_json(state / name, body)
                with self.assertRaisesRegex(RuntimeError, expected):
                    profiles.retire(config, state, host, YARN, YARN_REPLACEMENT)
                (state / name).unlink()
        self.assertTrue((home / (YARN + '.json')).exists(), 'a refusal while maintenance was mid-flight '
            'left the machine offering what it offered, which is the only safe thing for it to do')

    def test_a_retirement_that_leaves_behind_a_body_it_cannot_render_puts_everything_back(self):
        # The damage is in a body this retirement does not touch, and it is only visible once the record is
        # rewritten: what a machine guarantees after taking a configuration out of service is that what it
        # still offers it can still launch. Nothing had been recorded as active here, which is the case
        # where the check that catches it is the one that runs after the write.
        config, state, home, host = self.deployment()
        before = read(home / REGISTRY_FILE)
        before_digests = self.digests(config, host)
        kept_body = (home / (YARN + '.json')).read_bytes()
        damaged = home / (TWO_CARD_SOLO + '.json')
        atomic_json(damaged, {**read(damaged), 'engine': 'an-engine-this-machine-has-never-had'})
        with self.assertRaisesRegex(RuntimeError, 'Retiring ' + YARN + ' is refused: ' + TWO_CARD_SOLO +
                ': unknown engine an-engine-this-machine-has-never-had.*back where they were.*still offering'):
            profiles.retire(config, state, host, YARN, YARN_REPLACEMENT)
        self.assertEqual(read(home / REGISTRY_FILE), before, 'the record was left edited by a failure')
        self.assertEqual((home / (YARN + '.json')).read_bytes(), kept_body, 'the body was left deleted')
        # The damage is still in place here, and it is a file the machine wrote, so the answer about it names
        # the body and the engine the body names. A KeyError out of a renderer would name neither, and an
        # operator reading a refusal on their own machine is the whole reason a body is a machine's file.
        with self.assertRaisesRegex(RuntimeError, TWO_CARD_SOLO + ': unknown engine '
                'an-engine-this-machine-has-never-had'):
            render(config, host, CODING)
        atomic_json(damaged, read(machine.HOME / (TWO_CARD_SOLO + '.json')))
        self.assertEqual(self.digests(config, host), before_digests, 'with the machine\'s own file put back '
            'to rights, the machine after a refused retirement is the machine before it was asked, launch '
            'for launch')

    def test_a_retirement_is_an_act_on_the_machine_that_keeps_its_configurations(self):
        config, state, home, host = self.deployment()
        with self.assertRaisesRegex(RuntimeError, 'no machine is being asked'):
            profiles.retire(config, state, None, YARN, YARN_REPLACEMENT)
        # An audit of no machine is the same question with no machine to answer it, and it refuses the way
        # every other surface does when a checkout is asked what configurations exist.
        with self.assertRaisesRegex(RuntimeError, 'keeps no list of configurations'):
            profiles.audit(config, None, state)
        # The container is mounted to write in its state directory and nowhere else, so a home outside it
        # is refused where the act begins, rather than half written and found unwritable.
        outside = machine.stating(host, config.parent / 'machine-profiles')
        shutil.copytree(home, Path(outside['profiles_dir']))
        atomic_json(state / 'host.json', outside)
        with self.assertRaisesRegex(RuntimeError, 'outside the state directory.*the part of a deployment '
            'the container is allowed to write in'):
            profiles.retire(config, state, outside, YARN, YARN_REPLACEMENT)
        self.assertEqual(read(home / REGISTRY_FILE), read(machine.HOME / REGISTRY_FILE))

    def test_the_two_homes_of_one_checkout_answer_for_themselves(self):
        # `retire-profile` writes where the host it was given says its home is, which is the whole point of
        # the machine owning its own configurations: two machines in one checkout, one retirement, and the
        # other one has not been asked about anything and does not know it happened.
        config, state, home, host = self.deployment()
        second = config.parent / 'second-machine'
        second_home = machine.home(second)
        atomic_json(state / 'second-host.json', machine.stating(host, second_home, host_id=DAISY,
            display_name='Daisy'))
        self.running(state, config, host, CODING)
        profiles.retire(config, state, host, YARN, YARN_REPLACEMENT)
        self.assertFalse((home / (YARN + '.json')).exists(), 'the machine being acted on kept its own body')
        self.assertTrue((second_home / (YARN + '.json')).exists(), 'a retirement is written in the home of '
            'the machine being acted on, and this one was copied from the same fixture home')
        second_host = read(state / 'second-host.json')
        with self.assertRaisesRegex(RuntimeError, 'outside the state directory'):
            profiles.retire(config, state, second_host, YARN, YARN_REPLACEMENT)
        self.assertEqual([entry['profile'] for entry in profiles.audit(config, second_host, state)['offers']],
            [name for name in machine.record()['selectable'] if DAISY in
                read(machine.HOME / (name + '.json'))['hosts']])


def audited_starts_on(config, host):
    """What the machine says it starts on, read the way the audit reads it.

    A retirement that changed nothing about the default has to answer with the same default the machine
    gave before it was asked, so the expectation is the machine's own answer and not a name written here.
    """
    return profiles.audit(config, host)['starts_on']


if __name__ == '__main__':
    unittest.main()

import copy
import json
import threading
import unittest
import uuid
from unittest.mock import patch

try:
    import machine
    import test_controller as controller_fixture
except ModuleNotFoundError:
    # Run as `python3 -m unittest tests.test_operations` from the repository root, the fixtures are
    # reached through the package instead of the tests directory, which is not on the import path.
    from tests import machine
    from tests import test_controller as controller_fixture
from models import CODING, CODING_Q6K, EVERYDAY, STRATA_SOLO
from runtime.operations import Operations, OperationError, public_operation
from runtime.system import atomic_json, lock, now

DAY, NIGHT = 'qwen38-daytime', 'qwen38-nighttime'


class OperationFixture:
    controller_setup = controller_fixture.ControllerTests.setUp
    HOST = controller_fixture.HOST

    def setUp(self):
        self.controller_setup()
        self.web_state = {'started': True, 'status': self.c.status()}
        self.ops = Operations(self.c, self.web_state)
        self.release = threading.Event()
        self.addCleanup(self.stop_worker)

    def stop_worker(self):
        self.release.set()
        if self.ops.worker:
            self.ops.worker.join(3)

    def request(self, profile='qwen27b-q6k-pair'):
        return {'profile': profile, 'request_id': str(uuid.uuid4()),
            'expected_profile': self.c.load('active.json')['bundle']['profile'], 'expected_revision': self.c.revision}

    def run_switch(self, request=None):
        operation, created = self.ops.start(request or self.request())
        self.assertTrue(created)
        self.ops.worker.join(3)
        self.assertFalse(self.ops.worker.is_alive())
        return self.ops.read(operation['id'])

    def block_worker(self):
        entered = threading.Event()
        def prepare(bundle):
            entered.set()
            if not self.release.wait(3):
                raise RuntimeError('Test worker timed out')
        self.c.prepare.side_effect = prepare
        return entered


class OperationTests(OperationFixture, unittest.TestCase):
    def test_switches_both_directions_and_noop_changes_nothing(self):
        # Nothing survives a switch to a different configuration -- not even a model both sides
        # name -- so the only preservation left is the no-op: selecting the configuration already
        # running must not move a container.
        for profile in ('qwen27b-q6k-pair', 'qwen27b-q8-pair'):
            result = self.run_switch(self.request(profile))
            self.assertEqual(result['status'], 'succeeded')
            self.assertEqual(self.c.desired()['profile'], profile)
            self.assertFalse(result['already_active'])
            for name in (DAY, NIGHT):
                self.assertTrue(self.system.inspect(name)['State']['Running'])
            self.assertFalse(self.system.draining)
        served = {name: self.system.inspect(name)['Id'] for name in (DAY, NIGHT)}
        self.assertTrue(self.run_switch(self.request('qwen27b-q8-pair'))['already_active'])
        self.assertEqual({name: self.system.inspect(name)['Id'] for name in served}, served)

    def test_flash_f16_browser_switch_and_failed_load_restore_original(self):
        original = self.system.inspect(DAY)['Config']['Cmd']
        day, night = self.system.inspect(DAY)['Id'], self.system.inspect(NIGHT)['Id']
        result = self.run_switch(self.request('qwen27b-q6k-pair'))
        self.assertEqual(result['status'], 'succeeded')
        self.assertEqual(self.c.desired()['profile'], 'qwen27b-q6k-pair')
        self.assertEqual(self.web_state['status']['profile'], 'qwen27b-q6k-pair')
        # The switch asked for every model of the target and got what it asked for: the model both
        # configurations name is byte-identical, and it is reloaded anyway. That is the promise the
        # page makes when it says a switch runs the models the target names and nothing else -- the
        # pause is the target's full load time, every time, rather than something that depends on
        # which launch happens to match. The container a failed load has to put back is therefore
        # every container of the release being restored, which is what the next half of this test
        # checks.
        self.assertNotEqual(self.system.inspect(DAY)['Id'], day)
        self.assertNotEqual(self.system.inspect(NIGHT)['Id'], night)
        self.assertEqual(self.run_switch(self.request('qwen27b-q8-pair'))['status'], 'succeeded')
        self.assertEqual(self.system.inspect(DAY)['Config']['Cmd'], original)

        # One launch fails; the restoration's launches work. The old knob would have failed the
        # restoration too, because recovery now recreates every model of the previous release.
        self.system.failed_ups = 1
        failure = self.run_switch(self.request('qwen27b-q6k-pair'))
        self.assertEqual(failure['status'], 'recovered')
        self.assertEqual(self.c.desired()['profile'], 'qwen27b-q8-pair')
        self.assertEqual(self.system.inspect(DAY)['Config']['Cmd'], original)
        self.assertTrue(self.system.inspect(NIGHT)['State']['Running'])
        self.assertFalse(self.system.draining)

    def test_stages_wrap_real_work_and_verification(self):
        seen = []
        original = self.ops.save
        def save(operation):
            seen.append(operation['phase'])
            original(operation)
        with patch.object(self.ops, 'save', side_effect=save):
            result = self.run_switch()
        self.assertEqual(result['status'], 'succeeded')
        self.assertEqual(list(dict.fromkeys(seen)), ['checking', 'draining', 'loading', 'verifying', 'finished'])
        self.assertEqual(self.c.load('transaction.json')['id'], result['id'])

    def test_locks_cover_validation_and_block_cli_portal_and_other_clients(self):
        entered = self.block_worker()
        request = self.request()
        self.ops.start(request)
        self.assertTrue(entered.wait(1))
        self.assertFalse(self.ops.availability()['available'])
        for filename in ('runtime.lock', 'update.lock'):
            with self.assertRaises(RuntimeError):
                with lock(self.state / filename): pass
        with self.assertRaises(RuntimeError): self.c.transition('qwen27b-q8-pair')
        with self.assertRaises(OperationError) as error: self.ops.start(self.request())
        self.assertEqual(error.exception.status, 409)
        self.assertEqual(self.system.events, [])
        self.release.set(); self.ops.worker.join(3)
        self.assertTrue(self.ops.availability()['available'])

    def test_existing_cli_or_updater_lock_prevents_acceptance(self):
        for filename in ('runtime.lock', 'update.lock'):
            with lock(self.state / filename):
                with self.assertRaises(OperationError) as error: self.ops.start(self.request())
                self.assertEqual(error.exception.code, 'busy')
        self.assertIsNone(self.ops.latest())

    def test_duplicate_requests_survive_response_loss_and_manager_restart(self):
        request = self.request(); result = self.run_switch(request)
        events = list(self.system.events)
        restarted = Operations(self.c, self.web_state)
        duplicate, created = restarted.start(request)
        self.assertFalse(created); self.assertEqual(duplicate, result)
        self.assertEqual(events, self.system.events)
        request['profile'] = 'qwen27b-q8-pair'
        with self.assertRaises(OperationError): restarted.start(request)

    def test_duplicate_while_running_returns_same_receipt(self):
        entered = self.block_worker(); request = self.request()
        first, _ = self.ops.start(request); self.assertTrue(entered.wait(1))
        second, created = self.ops.start(request)
        self.assertEqual(first['id'], second['id']); self.assertFalse(created)

    def test_invalid_and_stale_requests_do_not_create_receipts(self):
        variants = [None, [], {}, {**self.request(), 'profile': 'everyday'},
            {**self.request(), 'profile': '../daytime'}, {**self.request(), 'profile': []},
            {**self.request(), 'request_id': '../../host'}, {**self.request(), 'force': True}]
        for request in variants:
            with self.subTest(request=request), self.assertRaises(OperationError) as error:
                self.ops.start(request)
            self.assertEqual(error.exception.status, 400)
        for key, value in [('expected_profile', 'qwen27b-q6k-pair'), ('expected_revision', 'b'*40)]:
            with self.assertRaises(OperationError) as error: self.ops.start({**self.request(), key: value})
            self.assertEqual(error.exception.code, 'stale-selection')
        self.assertIsNone(self.ops.latest())

    def test_a_configuration_of_another_machine_is_refused_before_anything_moves(self):
        # The page shows a person only what this machine can run, and the API is the same promise: a
        # request for another machine's configuration is a bad request, not a drain that closes the
        # router and refuses afterwards. `expected_profile` is checked the other way round on purpose
        # -- it reports what is already running, and a host that was given the wrong host_id has to be
        # able to switch itself back to a configuration this revision knows.
        self.c.host = {**self.c.host, 'host_id': 'daisy'}
        with self.assertRaises(OperationError) as error:
            self.ops.start(self.request())
        self.assertEqual((error.exception.status, error.exception.code), (400, 'invalid-request'))
        self.assertIsNone(self.ops.latest())
        self.assertEqual(self.system.events, [])
        self.assertFalse(self.system.draining)
        # What the API refused is exactly what this machine's page does not show, and what it does show
        # is that machine's own share of the registry -- not an empty list, which a host that names
        # itself a machine with nothing built for it is entitled to and this one is not.
        offered = [x['profile'] for x in self.c.status()['configurations']['selectable']]
        self.assertNotIn(self.request()['profile'], offered)
        self.assertEqual(offered, ['flash-next-strata-iq3-xxs-daisy',
            'flash-next-strata-iq3-xxs-daisy-tc', 'flash-next-strata-iq3-xxs-daisy-128k',
            'flash-next-strata-iq3-s-daisy'])

    def test_startup_maintenance_recovery_and_unhealthy_runtime_block_switches(self):
        for filename, record in [('router-maintenance.json', {'reserved': True}),
                ('transaction.json', {'phase': 'needs-attention'}), ('update-job.json', {'phase': 'controller-replacement'})]:
            self.c.save(filename, record)
            with self.assertRaises(OperationError): self.ops.start(self.request())
            (self.state / filename).unlink()
        self.web_state['started'] = False
        with self.assertRaises(OperationError): self.ops.start(self.request())
        self.web_state['started'] = True; self.web_state['status']['ready'] = False
        with self.assertRaises(OperationError): self.ops.start(self.request())

    def test_validation_failure_exposes_no_private_exception(self):
        self.c.prepare.side_effect = RuntimeError('secret-token /private/model/path')
        result = self.run_switch()
        self.assertEqual(result['status'], 'failed')
        self.assertIn('secret-token', result['error'])
        self.assertNotIn('secret-token', json.dumps(public_operation(result)))
        self.assertNotIn('/private', json.dumps(public_operation(result)))
        self.assertFalse(self.system.draining)
        self.assertFalse(any(e[0] == 'docker' for e in self.system.events))

    def test_busy_requests_timeout_without_stopping_generation(self):
        for attribute in ('active_count', 'queued_count', 'direct_busy'):
            with self.subTest(attribute=attribute):
                setattr(self.system, attribute, 1)
                before = copy.deepcopy(self.system.containers)
                with patch('runtime.controller.time.monotonic', side_effect=[0, 0, 301]):
                    result = self.run_switch()
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['error_code'], 'drain-timeout')
                self.assertEqual(before, self.system.containers)
                self.assertFalse(self.system.draining)
                setattr(self.system, attribute, 0)

    def test_load_failure_restores_previous_and_reports_recovered(self):
        # A switch stops what it replaces and recovery asks for every model of the previous
        # release, so "the load failed" means exactly one failed launch: this plays the target's
        # first launch failing and the restoration's launches working, and the failure it protects
        # is a recovered switch that came back serving only half of what it restored.
        self.system.failed_ups = 1
        result = self.run_switch()
        self.assertEqual(result['status'], 'recovered')
        self.assertEqual(self.c.desired()['profile'], 'qwen27b-q8-pair')
        for name in (DAY, NIGHT):
            self.assertTrue(self.system.inspect(name)['State']['Running'])
        self.assertFalse(self.system.draining)

    def test_failed_recovery_stays_drained_until_explicit_cli_recovery(self):
        self.system.generation_fails = True
        result = self.run_switch()
        self.assertEqual(result['status'], 'needs-attention')
        self.assertTrue(self.system.draining)
        with self.assertRaises(OperationError): self.ops.start(self.request())
        self.system.generation_fails = False
        self.c.recover()
        self.ops.reconcile_interrupted()
        self.assertEqual(self.ops.latest()['status'], 'recovered')
        self.assertFalse(self.system.draining)

    def test_a_switch_repairs_a_drifted_backend_and_leaves_the_healthy_neighbour_alone(self):
        # The refusal this test replaces protected the second resident from being recreated under a
        # browser switch. A switch now recreates everything its target changed, so the promise left
        # standing is the repair: re-selecting the running configuration must restart the backend
        # that drifted from the release and leave its healthy neighbour alone, rather than report
        # a machine with a wrong container standing as already_active. The browser still keeps its
        # hands off a not-ready machine, so this is the door the CLI and the updater walk through.
        self.system.containers[NIGHT]['Image'] = 'unexpected'
        day, night = self.system.inspect(DAY)['Id'], self.system.inspect(NIGHT)['Id']
        result = self.c.transition('qwen27b-q8-pair')
        self.assertEqual(result['changed_services'], [EVERYDAY])
        self.assertNotEqual(self.system.inspect(NIGHT)['Id'], night)
        self.assertEqual(self.system.inspect(DAY)['Id'], day)
        self.assertFalse(self.system.draining)

    def test_a_failed_recovery_stays_drained_with_every_previous_model_restarted(self):
        # Protection for the case where recovery itself fails: nothing may be left half-loaded
        # behind it, and nothing may be called served. The switch's acceptance check fails, and the
        # pass that puts the previous release back has to ask for every model of it -- the failed
        # switch stopped them -- while admission stays closed until the next attempt.
        def fail_acceptance(bundle, roles):
            self.system.containers[NIGHT]['Image'] = 'unexpected'
            raise RuntimeError('Simulated verification failure')
        with patch.object(self.c, 'acceptance', side_effect=fail_acceptance): result = self.run_switch()
        self.assertEqual(result['status'], 'needs-attention')
        for name in (DAY, NIGHT):
            self.assertTrue(self.system.inspect(name)['State']['Running'])
        self.assertTrue(self.system.draining)

    def receipt(self, phase='checking'):
        request = self.request()
        op = {'id': request['request_id'], 'request': request, 'from_profile': 'qwen27b-q8-pair',
            'target_profile': 'qwen27b-q6k-pair', 'revision': self.c.revision, 'status': 'running', 'phase': phase, 'started_at': now()}
        self.ops.save(op); self.c.save('latest-operation.json', {'id': op['id']})
        return op

    def test_restart_before_transaction_marks_interrupted_without_retry(self):
        self.receipt(); self.ops.reconcile_interrupted()
        self.assertEqual(self.ops.latest()['status'], 'failed')
        self.assertEqual(self.ops.latest()['error_code'], 'interrupted')
        self.assertEqual(self.system.events, [])

    def test_restart_reads_matching_terminal_journal_or_requires_recovery(self):
        for phase in ('prepared', 'draining', 'applying', 'verifying', 'needs-attention', 'succeeded', 'failed', 'recovered'):
            op = self.receipt()
            tx = {'id': op['id'], 'phase': phase}
            self.c.save('transaction.json', tx)
            self.ops.reconcile_interrupted()
            expected = phase if phase in ('succeeded', 'failed', 'recovered') else 'needs-attention'
            self.assertEqual(self.ops.latest()['status'], expected)
            self.assertEqual(self.c.load('transaction.json'), tx)
        self.assertEqual(self.system.events, [])

    def test_live_worker_is_not_classified_as_interrupted(self):
        entered = self.block_worker(); self.ops.start(self.request()); self.assertTrue(entered.wait(1))
        self.ops.reconcile_interrupted()
        self.assertEqual(self.ops.latest()['status'], 'running')

    def test_restart_recovers_receipt_written_before_latest_pointer(self):
        op = self.receipt()
        (self.state / 'latest-operation.json').unlink()
        restarted = Operations(self.c, self.web_state)
        restarted.reconcile_interrupted()
        self.assertEqual(restarted.latest()['id'], op['id'])
        self.assertEqual(restarted.latest()['status'], 'failed')
        self.assertEqual(self.system.events, [])

    def test_older_health_poll_cannot_overwrite_newer_snapshot(self):
        self.run_switch()
        old = {'ready': True, 'profile': 'qwen27b-q8-pair', 'updated_at': '2020-01-01T00:00:00Z'}
        self.ops.publish_status(old)
        self.assertEqual(self.web_state['status']['profile'], 'qwen27b-q6k-pair')

    def test_live_health_is_rechecked_after_cached_admission(self):
        self.system.containers[NIGHT]['Image'] = 'unexpected'
        result = self.run_switch()
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(self.system.draining)
        self.assertFalse(any(e[0] == 'docker' for e in self.system.events))

    def test_replacement_controller_can_reconcile_while_updater_waits_for_health(self):
        op = self.receipt('verifying')
        self.c.save('transaction.json', {'id': op['id'], 'phase': 'succeeded', 'finished_at': now()})
        with lock(self.state / 'update.lock'):
            self.ops.reconcile_interrupted()
        self.assertEqual(self.ops.latest()['status'], 'succeeded')
        self.assertEqual(self.system.events, [])


class ExclusiveOperationTests(OperationFixture, unittest.TestCase):
    """A browser switch to the configuration that owns all four text GPUs stops every model the
    target does not name; a switch back loads all of the target's models, one launch at a time."""
    controller_setup = controller_fixture.ExclusiveProfileTests.setUp
    HOST = controller_fixture.ExclusiveProfileTests.HOST
    SOLO = 'flash-next-strata-solo-thinking-recovery'  # the one configuration that owns all four text GPUs

    def test_browser_switch_to_solo_and_back(self):
        result = self.run_switch(self.request(self.SOLO))
        self.assertEqual(result['status'], 'succeeded')
        # The operation records, by model id, the running launches the target leaves nothing of --
        # both of the pair's models -- and says the same thing below the one loading sentence.
        self.assertEqual(result['stops_models'], [CODING, EVERYDAY])
        loading = {**result, 'status': 'running', 'phase': 'loading'}
        message = public_operation(loading)['message']
        self.assertIn('Loading the models of the selected configuration; anything it does not name stops.', message)
        self.assertIn(' Stops: ' + CODING + ', ' + EVERYDAY + '.', message)
        self.assertIsNone(self.system.inspect(NIGHT))
        # The placement fact ("the everyday role is offline") is gone with the role; what remains is
        # what a client cannot ask for, and `nighttime` is not it: the solo backend answers it.
        offline = sorted({alias for entry in self.web_state['status']['offline_services'] for alias in entry['aliases']})
        self.assertEqual(offline, ['daisy'])
        self.assertTrue(self.web_state['status']['ready'])
        # Selecting the configuration that is already running is a no-op, and a no-op stops nothing:
        # the message has no Stops clause because `stops_models` is empty.
        again = self.run_switch(self.request(self.SOLO))
        self.assertTrue(again['already_active'])
        self.assertEqual(again['stops_models'], [])
        self.assertNotIn(' Stops: ', public_operation({**again, 'status': 'running', 'phase': 'loading'})['message'])
        self.assertIsNone(self.system.inspect(NIGHT))
        back = self.run_switch(self.request('qwen27b-q8-pair'))
        self.assertEqual(back['status'], 'succeeded')
        self.assertEqual(back['stops_models'], [STRATA_SOLO])
        self.assertTrue(self.system.inspect(NIGHT)['State']['Running'])
        self.assertFalse(self.system.draining)

    def variant_registry(self):
        """A copy of the registry with a second Everyday variant: the former draft-free one, now a
        model of its own that a configuration has to name to load."""
        import shutil, tempfile
        from pathlib import Path
        from runtime import config as runtime_config
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        config = Path(tmp.name) / 'config'; shutil.copytree(controller_fixture.ROOT / 'config', config)
        # The variant lives in a home of its own, because that is where a machine keeps the bodies it
        # offers; the copy of the checkout beside it holds the engines and the rules, and no body.
        home = self.state / 'variant-profiles'; shutil.copytree(machine.HOME, home)
        night = json.loads((home / 'everyday.json').read_text())
        night.update(id='everyday-plain', display_name='Qwen3.8 27B Abliterated Q6_K, no draft',
            variant_label='no draft', context_tokens=131072)
        night.pop('draft_on_vision_gpu')
        night['argument_order'] = [k for k in night['argument_order'] if not k.startswith('--spec-')]
        night['arguments'] = {k: v for k, v in night['arguments'].items() if not k.startswith('--spec-')}
        night['arguments']['--alias'] += '-plain'
        night['artifacts'] = [a for a in night['artifacts'] if a['target'] != '/weights/mtp.gguf']
        night['catalog']['mtp'] = {'enabled': False}
        (home / 'everyday-plain.json').write_text(json.dumps(night))
        day = json.loads((home / 'qwen27b-q6k-pair.json').read_text())
        day.update(id='qwen27b-q6k-pair-plain', models=['qwen27b-q6k-pair-plain', 'everyday-plain'])
        (home / 'qwen27b-q6k-pair-plain.json').write_text(json.dumps(day))
        # The variant is added the way a machine adds one: a body, plus the record that says this
        # machine offers it. Nothing in the revision is patched -- a configuration is a decision of
        # this box, and the box says so where it lives.
        offered = [path.stem for path in sorted(home.glob('*.json'))
            if path.name != runtime_config.REGISTRY_FILE and json.loads(path.read_text()).get('models')]
        atomic_json(home / runtime_config.REGISTRY_FILE,
            {'schema_version': runtime_config.PROFILES_VERSION, 'profiles': offered})
        # And it says where that home is, in the settings the controller reads it out of. A body copied
        # into a directory the machine has not named is a body nothing can find.
        self.c.host = {**self.c.host, 'profiles_dir': str(home)}
        atomic_json(self.state / 'host.json', self.c.host)
        self.c.config_dir = config

    def test_browser_switch_between_configurations_that_share_a_model_stops_the_one_it_drops(self):
        self.variant_registry()
        self.assertEqual(self.run_switch(self.request('qwen27b-q6k-pair'))['status'], 'succeeded')
        day, night = self.system.inspect(DAY)['Id'], self.system.inspect(NIGHT)['Id']
        plain = next(x for x in self.c.status()['configurations']['selectable']
            if x['profile'] == 'qwen27b-q6k-pair-plain')
        self.assertEqual([x['model'] for x in plain['models']], [CODING_Q6K, EVERYDAY + '-plain'])
        result = self.run_switch(self.request('qwen27b-q6k-pair-plain'))
        self.assertEqual(result['status'], 'succeeded')
        # A configuration is a decision, not a patch: what the old variant beside the shared
        # coding model guarantees now is what the new list says about the model it drops. The
        # plain target does not name the draft-carrying Everyday, so that model's memory is not
        # left held beside it -- the container comes back running the named variant -- and the
        # coding launch, identical in both variants, is reloaded with it: switching to a different
        # configuration is not a patch applied to the running one.
        self.assertEqual(result['stops_models'], [EVERYDAY])
        self.assertNotEqual(self.system.inspect(DAY)['Id'], day)
        self.assertNotEqual(self.system.inspect(NIGHT)['Id'], night)
        self.assertNotIn('--spec-type', self.system.inspect(NIGHT)['Config']['Cmd'])
        back = self.run_switch(self.request('qwen27b-q6k-pair'))
        self.assertEqual(back['status'], 'succeeded')
        self.assertEqual(back['stops_models'], [EVERYDAY + '-plain'])
        argv = self.system.inspect(NIGHT)['Config']['Cmd']
        self.assertEqual(argv[argv.index('--spec-draft-device') + 1], 'CUDA2')
        self.assertFalse(self.system.draining)

    def test_paired_browser_switch_replaces_the_models_its_target_names(self):
        # A paired switch used to refuse a target whose resident service had changed after
        # admission, because that resident was untouchable; a paired switch now carries out
        # whatever its target says. What this case keeps is that a change reaching the launch of
        # the shared model is part of the switch -- it is asked for and it lands -- and that the
        # operation still says honestly, by model id, which running model the target stops.
        night = self.system.inspect(NIGHT)['Id']
        desired = self.c.desired('qwen27b-q6k-pair')
        argv = desired['compose']['services'][EVERYDAY]['command']
        argv[argv.index('--threads') + 1] = '17'
        with patch.object(self.c, 'desired', return_value=desired):
            result = self.run_switch(self.request('qwen27b-q6k-pair'))
        self.assertEqual(result['status'], 'succeeded')
        self.assertEqual(result['stops_models'], [CODING])
        self.assertNotEqual(self.system.inspect(NIGHT)['Id'], night)
        self.assertEqual(self.system.inspect(NIGHT)['Config']['Cmd'][
            self.system.inspect(NIGHT)['Config']['Cmd'].index('--threads') + 1], '17')
        self.assertTrue(any(e[0] == 'docker' for e in self.system.events))
        self.assertFalse(self.system.draining)

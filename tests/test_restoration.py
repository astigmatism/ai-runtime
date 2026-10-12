"""The service-restoration invariant, and the failures it exists for.

On 2026-10-12 a switch to the Strata trial engine drained the router, the new backend came up and
never answered, the drain check could not pass because that dead container was the only thing
left, and recovery aborted with `Drain timed out; no active generation was stopped`. The host
stayed drained with no backend for three hours and forty-four minutes until somebody rebooted it,
and the machine came back on llama.cpp by itself - boot recovery did in ten minutes what the
in-process path had refused to do at all.

These tests are what stops that from being the end of a story again. They are release gates: CI
runs the whole suite, so the exact incident is re-fought on every change.
"""
import json
from pathlib import Path
import unittest
import uuid
from unittest.mock import patch

import test_controller as fixture
from runtime.config import render
from runtime.operations import Operations, OperationError, public_operation
from runtime.system import atomic_json
from runtime.web import public_status

ROOT = Path(__file__).resolve().parents[1]
SERVED = 'The last-known-good configuration is serving and requests are open.'
BOTH = ['qwen38-daytime', 'qwen38-nighttime']


class StubbornSystem(fixture.FakeSystem):
    """A fake that fails the ways the trial actually failed.

    Each knob is one observed behaviour of a real container, not a general-purpose injector.
    """

    def __init__(self, bundle):
        super().__init__(bundle)
        self.never_healthy_alias = None  # running, but /health never answers for this launch
        self.refuse_slots = ()  # answers no port at all, as the abandoned container did
        self.slots_shape = None  # answers something that is not a slot list
        self.busy_names = None  # which backends are generating, rather than all of them
        self.slot_polls = 0

    def http(self, url, body=None, headers=None, timeout=None):
        name = url.split('/')[2].split(':')[0]
        if url.endswith('/slots'):
            self.slot_polls += 1
            if name in self.refuse_slots:
                raise ConnectionError('Simulated connection refused on ' + name)
            if self.slots_shape is not None:
                return self.slots_shape(name)
            slots = super().http(url, body, headers, timeout)
            if self.busy_names is not None:
                return [dict(slot, is_processing=name in self.busy_names) for slot in slots]
            return slots
        if url.endswith('/health'):
            container = self.containers.get(name)
            if self.never_healthy_alias and container:
                if self.never_healthy_alias in (container['Config']['Cmd'] or []):
                    raise RuntimeError('Simulated readiness timeout: ' + name + ' never became healthy')
        return super().http(url, body, headers, timeout)


class RestorationTests(unittest.TestCase):
    """Shares the controller fixture, then bends its fake in the ways the deploy paths bend."""

    HOST = fixture.HOST

    def setUp(self):
        fixture.ControllerTests.setUp(self)
        # The same fake the controller fixture built, with the trial's failure modes added to it.
        # The controller keeps the instance it was handed, so the knobs are set on it in place.
        self.system.__class__ = StubbornSystem
        self.system.never_healthy_alias = None
        self.system.refuse_slots = ()
        self.system.slots_shape = None
        self.system.busy_names = None
        self.system.slot_polls = 0
        self.web_state = {'started': True, 'status': self.c.status()}
        self.ops = Operations(self.c, self.web_state)

    def switch_to_other_daytime(self):
        """The alias the other Daytime launches on, to make one backend of it refuse readiness."""
        target = render(ROOT / 'config', fixture.HOST, 'qwen27b-q6k-with-nighttime')
        self.assertNotEqual(target['config_sha256'], self.bundle['config_sha256'])
        argv = target['compose']['services']['coding']['command']
        return argv[argv.index('--alias') + 1]

    def strand_the_backend(self, drained=True):
        """No backend running, and the router shut: what a failed first boot leaves behind."""
        self.system.containers['qwen38-daytime']['State']['Running'] = False
        if drained:
            self.c.set_drain(True, {'id': 'stranded'})
        self.assertFalse(self.c.serving())

    def make_restoration_due(self):
        """Pull the scheduled retry forward without waiting for it or pretending time stopped."""
        record = dict(self.c.restoration())
        record['next_attempt_at'] = '2000-01-01T00:00:00+00:00'
        atomic_json(self.state / 'restoration.json', record)

    def wait_for_worker(self):
        self.assertIsNotNone(self.ops.worker)
        self.ops.worker.join(10)
        self.assertFalse(self.ops.worker.is_alive(), 'the restoration worker did not finish')

    # 1e (i). The incident itself: a target that starts but never becomes healthy.
    def test_a_target_that_starts_but_never_becomes_healthy_is_handled(self):
        """Previous profile restored hands-off, router undrained, an active backend healthy.

        Nothing in this test reaches for an operator, a CLI, or a power button, and no deadline is
        extended: the deployment call comes back having already put the machine back.
        """
        self.system.never_healthy_alias = self.switch_to_other_daytime()
        with self.assertRaisesRegex(RuntimeError, 'Deployment failed'):
            self.c.transition('qwen27b-q6k-with-nighttime')
        transaction = self.c.load('transaction.json')
        self.assertNotIn('recovery needs attention', str(transaction.get('error') or ''),
            'the restoration was supposed to finish, not stop for an operator')
        self.assertFalse(self.system.draining, 'the router was left drained with nothing serving')
        self.assertTrue(self.c.serving(), 'no healthy backend is serving after the failure')
        self.assertEqual(transaction['phase'], 'recovered')
        self.assertEqual(self.c.load('active.json')['bundle']['profile'], 'qwen27b-q8-with-nighttime')
        self.assertFalse(self.c.restoration_pending(), 'a finished restoration is still an open action')
        # Every phase the restoration reached is in the record, because the record is the evidence.
        steps = [x['step'] for x in transaction.get('restoration', {}).get('phases', [])]
        for step in ('preparing', 'draining', 'recreating', 'verifying', 'publishing'):
            self.assertIn(step, steps)
        # And it is readable without anyone opening a private journal.
        status = self.c.status()
        self.assertTrue(status['service']['serving'])
        self.assertFalse(status['service']['pending'])
        # `last-deployment` is the attempt, not the outcome: it names the profile that was tried
        # and says, in phase and error, that the machine was put back afterwards.
        deployment = self.c.load('last-deployment.json')
        self.assertEqual(deployment['profile'], 'qwen27b-q6k-with-nighttime')
        self.assertEqual(deployment['phase'], 'recovered')
        self.assertEqual(deployment['intent'], 'deployment')
        self.assertIn('readiness', str(deployment['error']))

    # 1c/1e (ii). The recovery pass itself fails, then re-enters and finishes on its own.
    def test_a_failed_attempt_schedules_the_next_one_and_the_next_one_finishes_the_job(self):
        self.system.never_healthy_alias = self.switch_to_other_daytime()
        self.system.failed_generations = 1  # the restoration trips over its own acceptance check
        with self.assertRaisesRegex(RuntimeError, 'recovery needs attention'):
            self.c.transition('qwen27b-q6k-with-nighttime')
        self.assertTrue(self.system.draining)
        self.assertFalse(self.c.serving())
        self.assertTrue(self.c.restoration_pending())
        record = self.c.restoration()
        self.assertEqual(record['attempts'], 1)
        self.assertIn('acceptance', record['last_error'])
        self.assertEqual(record['phases'][-1]['step'], 'scheduled')
        self.assertFalse(self.c.restoration_due(), 'the retry waits its backoff rather than hammering')

        # Nobody has to be there when the timer comes round: the retry is taken up on its own.
        self.make_restoration_due()
        self.assertTrue(self.c.restoration_due())
        started = self.ops.restore_if_due()
        self.assertIsNotNone(started)
        self.assertEqual(started['kind'], 'restore')
        self.wait_for_worker()
        operation = self.ops.read(started['id'])
        self.assertEqual(operation['status'], 'recovered')
        self.assertEqual(public_operation(operation)['message'], SERVED)
        self.assertTrue(self.c.serving())
        self.assertFalse(self.system.draining)
        self.assertFalse(self.c.restoration_pending())
        self.assertEqual(self.ops.latest()['id'], started['id'])

    # 1c. A failed attempt always leaves a next one, with a longer wait, until service returns.
    def test_restoration_keeps_trying_until_the_service_comes_back(self):
        self.strand_the_backend(drained=False)
        self.system.fail_reconcile = True  # nothing can be brought up, so no attempt may rest
        with self.assertRaisesRegex(RuntimeError, 'recovery needs attention'):
            self.c.transition('qwen27b-q6k-with-nighttime')
        first = self.c.restoration()
        self.assertEqual(first['attempts'], 1)
        self.assertFalse(self.c.serving())

        self.make_restoration_due()
        with self.assertRaisesRegex(RuntimeError, 'Simulated load failure'):
            self.c.restore_last_known_good()
        second = self.c.restoration()
        self.assertEqual(second['attempts'], 2)
        self.assertEqual(second['id'], first['id'], 'a retry continues one restoration, not a new one')
        self.assertTrue(second['next_attempt_at'] > first['next_attempt_at'], 'the wait keeps growing')
        self.assertTrue(self.c.restoration_pending())

        # The cause clears; the next scheduled attempt is enough, with nobody touching anything.
        self.system.fail_reconcile = False
        self.make_restoration_due()
        self.assertTrue(self.c.restoration_due())
        result = self.c.restore_last_known_good()
        self.assertTrue(result['restored'])
        self.assertTrue(self.c.serving())
        self.assertFalse(self.c.restoration_pending())
        self.assertFalse(self.system.draining)

    # 1d. The one action, in the state the button exists to end.
    def test_the_one_action_restores_from_a_state_where_nothing_is_serving(self):
        self.strand_the_backend()
        result = self.c.restore_last_known_good()
        self.assertTrue(result['restored'])
        self.assertEqual(result['profile'], 'qwen27b-q8-with-nighttime')
        self.assertTrue(self.c.serving())
        self.assertFalse(self.system.draining)
        self.assertEqual(self.c.load('active.json')['bundle']['profile'], 'qwen27b-q8-with-nighttime')
        self.assertFalse(self.c.restoration_pending())

    def test_the_one_action_is_enough_when_the_record_is_the_only_thing_left(self):
        """The 04:12 case: the transaction is gone, the drain is set, only the record knows."""
        self.strand_the_backend()
        (self.state / 'transaction.json').unlink(missing_ok=True)
        result = self.c.restore_last_known_good()
        self.assertTrue(result['restored'])
        self.assertTrue(self.c.serving())
        self.assertFalse(self.system.draining)
        transaction = self.c.load('transaction.json')
        self.assertEqual(transaction['intent'], 'restore-last-known-good')
        # The receipt a person reads afterwards says this was a restoration, not a switch.
        self.assertEqual(self.c.load('last-deployment.json')['intent'], 'restore-last-known-good')

    def test_status_reports_the_service_contract_and_hides_the_raw_failure(self):
        self.strand_the_backend(drained=False)
        self.system.fail_reconcile = True
        with self.assertRaisesRegex(RuntimeError, 'recovery needs attention'):
            self.c.transition('qwen27b-q6k-with-nighttime')
        status = self.c.status()
        self.assertFalse(status['service']['serving'])
        self.assertTrue(status['service']['restorable'])
        self.assertTrue(status['service']['pending'])
        self.assertEqual(status['service']['profile'], 'qwen27b-q8-with-nighttime')
        self.assertIn('Simulated load failure', status['restoration']['last_error'])
        browser = public_status({'status': status})
        text = json.dumps(browser)
        self.assertNotIn('Simulated load failure', text)
        self.assertNotIn(str(self.state), text)
        self.assertTrue(browser['service']['pending'])
        self.assertTrue(browser['restoration']['attempts'])
        # What each attempt reached stays readable; only its raw reason is replaced.
        self.assertEqual([x['step'] for x in browser['restoration']['phases']], ['scheduled'])

    # 1a. A backend whose /slots cannot be fetched is serving nothing.
    def test_a_backend_that_answers_no_port_is_idle_not_busy(self):
        configs = {cfg['container_name']: cfg for cfg in self.bundle['compose']['services'].values()}
        self.system.refuse_slots = ('qwen38-daytime',)
        relayed, notes = self.c.drain_pass(configs)
        self.assertEqual(relayed, 0)
        self.assertEqual(notes['unreachable'], ['qwen38-daytime'])
        self.assertEqual(notes['busy'], [])
        # Before, this pass could never quiet down and the drain check ran out its whole deadline.
        notes = self.c.wait_idle([self.bundle], timeout=300)
        self.assertEqual(notes['unreachable'], ['qwen38-daytime'])

    def test_a_healthy_backend_with_no_work_is_idle_after_a_few_settled_rounds(self):
        self.system.slot_polls = 0
        notes = self.c.wait_idle([self.bundle], timeout=300)
        self.assertEqual(notes['busy'], [])
        self.assertEqual(self.system.slot_polls, 6, 'three settled rounds across two backends')

    def test_work_behind_the_router_keeps_the_drain_open_until_it_finishes(self):
        self.system.active_count = 1
        with patch('runtime.controller.time.monotonic', side_effect=[0, 0, 0, 0, 301]):
            with self.assertRaisesRegex(RuntimeError, 'Drain timed out'):
                self.c.wait_idle([self.bundle], timeout=300)

    # 1e (iii). A /slots response this code cannot read is not evidence of work, and must not hang.
    def test_an_unreadable_slots_response_is_recorded_and_the_wait_stays_bounded(self):
        self.system.slots_shape = lambda name: {'status': 'ok'}
        self.system.slot_polls = 0
        notes = self.c.wait_idle([self.bundle], timeout=300)
        self.assertEqual(notes['malformed'], BOTH)
        self.assertEqual(notes['busy'], [])
        self.assertEqual(self.system.slot_polls, 6, 'a wrong shape must not make the wait unbounded')
        self.system.slots_shape = lambda name: []
        self.assertEqual(self.c.wait_idle([self.bundle], timeout=300)['busy'], [])

    # 1b. A drain deadline in a restoration means continue, never abort into darkness.
    def test_a_restoration_does_not_abort_at_the_drain_deadline(self):
        self.system.busy_names = ('qwen38-daytime',)  # generating on its own port, outside the router
        with patch('runtime.controller.time.monotonic', side_effect=[0, 0, 301]):
            notes = self.c.wait_idle([self.bundle], timeout=300, force_on_deadline=True)
        self.assertTrue(notes['forced'])
        self.assertEqual(notes['busy'], ['qwen38-daytime'])
        # Without that commitment the same pass still backs away, having touched nothing.
        with patch('runtime.controller.time.monotonic', side_effect=[0, 0, 301]):
            with self.assertRaisesRegex(RuntimeError, 'Drain timed out'):
                self.c.wait_idle([self.bundle], timeout=300)

    def test_only_the_backend_that_says_it_is_generating_holds_the_drain_open(self):
        self.system.busy_names = ('qwen38-nighttime',)
        relayed, notes = self.c.drain_pass(
            {cfg['container_name']: cfg for cfg in self.bundle['compose']['services'].values()})
        self.assertEqual(relayed, 0)
        self.assertEqual(notes['busy'], ['qwen38-nighttime'])

    def test_the_button_offers_the_restoration_and_refuses_only_a_competing_owner(self):
        self.assertEqual(self.ops.restore_blockers(), None)
        self.assertTrue(self.ops.restore_availability()['available'])
        request = {'request_id': str(uuid.uuid4())}
        operation, created = self.ops.start_restore(request)
        self.assertTrue(created)
        self.assertEqual(operation['kind'], 'restore')
        self.wait_for_worker()
        finished = self.ops.read(operation['id'])
        self.assertEqual(finished['status'], 'recovered')
        self.assertIn('serving', public_operation(finished)['message'])

        again, created = self.ops.start_restore(request)
        self.assertFalse(created, 'a repeated request returns the receipt instead of restoring twice')
        self.assertEqual(again['id'], request['request_id'])

        for bad in ({}, {'request_id': 'not-a-uuid'}, {'request_id': str(uuid.uuid4()), 'profile': 'x'}):
            with self.subTest(request=bad), self.assertRaises(OperationError) as error:
                self.ops.start_restore(bad)
            self.assertEqual(error.exception.status, 400)

        other = str(uuid.uuid4())
        atomic_json(self.state / 'operations' / (other + '.json'), {'id': other, 'kind': 'switch'})
        with self.assertRaises(OperationError) as error:
            self.ops.start_restore({'request_id': other})
        self.assertEqual(error.exception.code, 'request-id-reused')

    def test_a_restoration_is_offered_when_a_switch_would_refuse_it(self):
        """A host that refuses a switch for being mid-transaction needs this action, not a reboot."""
        self.strand_the_backend(drained=False)
        self.system.fail_reconcile = True
        with self.assertRaisesRegex(RuntimeError, 'recovery needs attention'):
            self.c.transition('qwen27b-q6k-with-nighttime')
        with self.assertRaisesRegex(RuntimeError, 'interrupted deployment'):
            self.c.transition('qwen27b-q8-with-nighttime')
        self.assertEqual(self.ops.restore_blockers(), None)
        self.make_restoration_due()
        self.assertIsNotNone(self.ops.restore_if_due())
        self.wait_for_worker()
        # Still stranded: the cause has not gone away, and an attempt that could not finish is not
        # quietly counted as one that did.
        self.assertFalse(self.c.serving())
        self.assertTrue(self.c.restoration_pending())
        self.assertEqual(self.c.restoration()['attempts'], 2)
        self.system.fail_reconcile = False
        self.make_restoration_due()
        self.assertIsNotNone(self.ops.restore_if_due())
        self.wait_for_worker()
        self.assertTrue(self.c.serving())
        self.assertFalse(self.c.restoration_pending())
        self.assertFalse(self.system.draining)

    def test_nothing_to_restore_to_is_said_plainly_rather_than_tried(self):
        (self.state / 'active.json').unlink(missing_ok=True)
        (self.state / 'previous.json').unlink(missing_ok=True)
        self.assertIsNone(self.c.last_known_good())
        self.assertIn('No last-known-good', self.ops.restore_blockers())
        with self.assertRaisesRegex(RuntimeError, 'No last-known-good'):
            self.c.restore_last_known_good()
        self.assertFalse(self.c.restoration_pending())

    def test_rejected_catalog_cannot_disable_the_admin_api_needed_for_recovery(self):
        """The router accepts reload but rejects runtime-state for backend_kind=strata.

        Its admin API reads the published file on every call. Leaving that file in place
        after rejection prevents recovery from even checking ownership of the drain.
        """
        self.c.publish(self.bundle)
        target = render(ROOT / 'config', self.c.host, 'flash-next-strata-160k')
        for service in target['manifest']['services']:
            self.system.services[service['container_name']] = service
        catalog_path = Path(self.c.host['router_state_dir']) / 'active-model.json'
        http = self.system.http
        rejected = []

        def router(url, *args, **kwargs):
            if url.endswith('/runtime-state'):
                catalog = json.loads(catalog_path.read_text())
                if catalog['backend_kind'] == 'strata':
                    rejected.append(True)
                    raise RuntimeError('Unsupported active backend kind: strata')
            return http(url, *args, **kwargs)

        with patch.object(self.system, 'http', side_effect=router), \
                patch.object(self.c, 'preparation_needed', return_value=False), \
                patch.object(self.c, '_record_warm_boot'):
            with self.assertRaisesRegex(RuntimeError, 'Unsupported active backend kind: strata'):
                self.c.transition(target['profile'])
            self.assertTrue(self.c.serving())
        self.assertTrue(rejected, 'the candidate catalog must actually reach the router')
        self.assertFalse(self.system.draining)
        self.assertEqual(set(self.system.containers), set(BOTH))
        self.assertEqual(self.c.load('transaction.json')['phase'], 'recovered')
        self.assertEqual(self.c.load('active.json'), self.active)
        self.assertEqual(json.loads(catalog_path.read_text())['backend_kind'], 'llama_cpp')
        self.assertFalse(self.c.restoration_pending())

    def test_router_failure_before_recovery_advances_the_retry_schedule(self):
        self.strand_the_backend()
        self.c.schedule_restoration('Initial failure')
        first = self.c.restoration()
        self.c.save('transaction.json', {'id': 'stranded', 'phase': 'needs-attention'})
        http = self.system.http

        def unavailable(url, *args, **kwargs):
            if url.endswith('/runtime-state'):
                raise RuntimeError('Router admin unavailable')
            return http(url, *args, **kwargs)

        with patch.object(self.system, 'http', side_effect=unavailable):
            self.make_restoration_due()
            started = self.ops.restore_if_due()
            self.wait_for_worker()
        retry = self.c.restoration()
        self.assertEqual(retry['attempts'], first['attempts'] + 1)
        self.assertEqual(retry['last_error'], 'Router admin unavailable')
        self.assertFalse(self.c.restoration_due())
        self.assertEqual(self.ops.read(started['id'])['status'], 'needs-attention')

    def test_catalog_reload_failure_preserves_the_prior_catalog(self):
        self.c.publish(self.bundle)
        catalog_path = Path(self.c.host['router_state_dir']) / 'active-model.json'
        previous = json.loads(catalog_path.read_text())
        with patch.object(self.c, 'admin', side_effect=RuntimeError('Reload unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'Reload unavailable'):
                self.c.publish(self.bundle)
        self.assertEqual(json.loads(catalog_path.read_text()), previous)

    def test_a_reserved_router_or_pending_release_blocks_the_restoration(self):
        atomic_json(self.state / 'router-maintenance.json', {'reserved': True})
        self.assertIn('router deployment', self.ops.restore_blockers())
        atomic_json(self.state / 'router-maintenance.json', {'reserved': False})
        atomic_json(self.state / 'update-job.json', {'phase': 'prepared'})
        self.assertIn('source deployment', self.ops.restore_blockers())


if __name__ == '__main__':
    unittest.main()

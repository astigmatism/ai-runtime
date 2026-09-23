import copy
import json
import threading
import unittest
import uuid
from unittest.mock import patch

import test_controller as controller_fixture
from runtime.operations import Operations, OperationError, public_operation
from runtime.system import lock, now


class OperationFixture:
    def setUp(self):
        controller_fixture.ControllerTests.setUp(self)
        self.web_state = {'started': True, 'status': self.c.status()}
        self.ops = Operations(self.c, self.web_state)
        self.release = threading.Event()
        self.addCleanup(self.stop_worker)

    def stop_worker(self):
        self.release.set()
        if self.ops.worker:
            self.ops.worker.join(3)

    def request(self, profile='daytime-27b'):
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
    def test_switches_both_directions_and_noop_preserve_nighttime(self):
        night = self.system.inspect('qwen38-nighttime')['Id']
        for profile in ('daytime-27b', 'daytime', 'daytime'):
            result = self.run_switch(self.request(profile))
            self.assertEqual(result['status'], 'succeeded')
            self.assertEqual(self.c.desired()['profile'], profile)
            self.assertEqual(self.system.inspect('qwen38-nighttime')['Id'], night)
            self.assertFalse(self.system.draining)
        self.assertTrue(result['already_active'])

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
        with self.assertRaises(RuntimeError): self.c.transition('daytime')
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
        request['profile'] = 'daytime'
        with self.assertRaises(OperationError): restarted.start(request)

    def test_duplicate_while_running_returns_same_receipt(self):
        entered = self.block_worker(); request = self.request()
        first, _ = self.ops.start(request); self.assertTrue(entered.wait(1))
        second, created = self.ops.start(request)
        self.assertEqual(first['id'], second['id']); self.assertFalse(created)

    def test_invalid_and_stale_requests_do_not_create_receipts(self):
        variants = [None, [], {}, {**self.request(), 'profile': 'nighttime'},
            {**self.request(), 'profile': '../daytime'}, {**self.request(), 'profile': []},
            {**self.request(), 'request_id': '../../host'}, {**self.request(), 'force': True}]
        for request in variants:
            with self.subTest(request=request), self.assertRaises(OperationError) as error:
                self.ops.start(request)
            self.assertEqual(error.exception.status, 400)
        for key, value in [('expected_profile', 'daytime-27b'), ('expected_revision', 'b'*40)]:
            with self.assertRaises(OperationError) as error: self.ops.start({**self.request(), key: value})
            self.assertEqual(error.exception.code, 'stale-selection')
        self.assertIsNone(self.ops.latest())

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
        self.system.fail_reconcile = True
        result = self.run_switch()
        self.assertEqual(result['status'], 'recovered')
        self.assertEqual(self.c.desired()['profile'], 'daytime')
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

    def test_browser_switch_refuses_nighttime_changes_even_after_admission(self):
        desired = self.c.desired('daytime-27b')
        desired['compose']['services']['everyday']['logging']['options']['max-size'] = '31m'
        with patch.object(self.c, 'desired', return_value=desired):
            result = self.run_switch()
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(self.system.draining)
        self.assertFalse(any(e[0] == 'docker' for e in self.system.events))

    def test_recovery_refuses_to_replace_nighttime_if_it_drifts(self):
        night = self.system.inspect('qwen38-nighttime')['Id']
        def fail_acceptance(bundle, roles):
            self.system.containers['qwen38-nighttime']['Image'] = 'unexpected'
            raise RuntimeError('Simulated verification failure')
        with patch.object(self.c, 'acceptance', side_effect=fail_acceptance): result = self.run_switch()
        self.assertEqual(result['status'], 'needs-attention')
        self.assertEqual(self.system.inspect('qwen38-nighttime')['Id'], night)
        self.assertTrue(self.system.draining)

    def receipt(self, phase='checking'):
        request = self.request()
        op = {'id': request['request_id'], 'request': request, 'from_profile': 'daytime',
            'target_profile': 'daytime-27b', 'revision': self.c.revision, 'status': 'running', 'phase': phase, 'started_at': now()}
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
        old = {'ready': True, 'profile': 'daytime', 'updated_at': '2020-01-01T00:00:00Z'}
        self.ops.publish_status(old)
        self.assertEqual(self.web_state['status']['profile'], 'daytime-27b')

    def test_live_health_is_rechecked_after_cached_admission(self):
        self.system.containers['qwen38-nighttime']['Image'] = 'unexpected'
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

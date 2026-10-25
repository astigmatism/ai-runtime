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
import copy
import json
from io import BytesIO
from pathlib import Path
import shutil
import tempfile
import unittest
import urllib.error
import uuid
from unittest.mock import patch

import test_controller as fixture
from models import CODING_Q6K, STRATA_DAISY, STRATA_SOLO
from runtime.config import read, render
from runtime.controller import Controller
from runtime.operations import Operations, OperationError, public_operation
from runtime.system import atomic_json
from runtime.web import public_status
import test_publication as publication_fixture

ROOT = Path(__file__).resolve().parents[1]
SERVED = 'The last-known-good configuration is serving and requests are open.'
BOTH = ['qwen38-daytime', 'qwen38-nighttime']


class RemoteRouterRestorationTests(unittest.TestCase):
    # Reuse the fixture, not its tests. Its router reports the model it serves and the
    # configuration separately, as the production router did during the October 21 outage.
    build = publication_fixture.PublisherTests.build
    PROFILE = publication_fixture.PublisherTests.PROFILE

    def setUp(self):
        self.c = self.build()
        self.c.validate = lambda bundle, **kw: None
        self.c.preparation_needed = lambda bundle: False
        self.c.transition(self.PROFILE, adopt=True)

    def assert_serving(self):
        self.assertTrue(self.c.serving())
        self.assertFalse(self.system.draining)
        self.assertEqual(self.router.drain, {})
        self.assertFalse(self.c.restoration_pending())

    def test_controller_deployment_verifies_the_configuration_apart_from_the_models_it_loads(self):
        self.c.revision = 'b' * 40
        result = self.c.transition(deploy=True)
        self.assertEqual(result['changed_services'], [])
        self.assertEqual(self.c.load('transaction.json')['phase'], 'succeeded')
        self.assert_serving()

    def test_failed_lan_migration_restores_the_previous_local_backend_and_opens_admission(self):
        previous = self.c.load('active.json')
        self.c.host['backend_address'] = '192.168.1.4'
        self.router.available = False
        with self.assertRaisesRegex(RuntimeError, 'Deployment failed:.*cannot reach'):
            self.c.transition()
        self.assertEqual(self.c.load('active.json'), previous)
        self.assertEqual(self.c.load('transaction.json')['phase'], 'recovered')
        self.assert_serving()

    def test_lan_migration_verifies_the_requested_binding_and_reopens_both_routers(self):
        self.c.host['backend_address'] = '192.168.1.4'
        result = self.c.transition()
        self.assertEqual(result['changed_services'], [STRATA_SOLO])
        active = self.c.load('active.json')['bundle']
        name = active['compose']['services'][STRATA_SOLO]['container_name']
        self.assertEqual(self.system.containers[name]['HostConfig']['PortBindings'],
            {'8080/tcp': [{'HostIp': '192.168.1.4', 'HostPort': '18080'}]})
        self.assertEqual(self.router.catalogs['rosalina']['models'][0]['backend_url'],
            'http://192.168.1.4:18080')
        self.assert_serving()

    def test_wrong_lan_binding_recovers_the_recorded_loopback_binding(self):
        previous = self.c.load('active.json')
        self.c.host['backend_address'] = '192.168.1.4'
        install = self.system.install
        def wrong_binding(role, cfg):
            install(role, cfg)
            if cfg['ports'][0].startswith('192.168.1.4:'):
                self.system.containers[cfg['container_name']]['HostConfig']['PortBindings'][
                    '8080/tcp'][0]['HostIp'] = '0.0.0.0'
        self.system.install = wrong_binding
        with self.assertRaisesRegex(RuntimeError, 'Deployment failed:.*port bindings'):
            self.c.transition()
        self.assertEqual(self.c.load('active.json'), previous)
        self.assertEqual(self.c.load('transaction.json')['phase'], 'recovered')
        name = previous['bundle']['compose']['services'][STRATA_SOLO]['container_name']
        self.assertEqual(self.system.containers[name]['HostConfig']['PortBindings'][
            '8080/tcp'][0]['HostIp'], '127.0.0.1')
        self.assert_serving()

    def test_admin_failure_during_restoration_retries_and_reopens_both_routers(self):
        self.c.set_drain(True, {'id': 'interrupted'})
        real = self.router.handle
        def fail_once(path, *args, **kwargs):
            if path.endswith('/runtime-state'):
                raise urllib.error.HTTPError(path, 401, 'Unauthorized', {}, BytesIO(b''))
            return real(path, *args, **kwargs)
        self.router.handle = fail_once
        with self.assertRaises(urllib.error.HTTPError):
            self.c.restore_last_known_good()
        pending = self.c.restoration()
        self.assertEqual(pending['attempts'], 1)
        self.router.handle = real
        pending['next_attempt_at'] = '2000-01-01T00:00:00+00:00'
        self.c.save('restoration.json', pending)
        operations = Operations(self.c, {})
        self.assertTrue(operations.restore_if_due())
        operations.worker.join(10)
        self.assertFalse(operations.worker.is_alive())
        self.assert_serving()

    def test_orphan_on_one_router_does_not_release_operator_drain_on_the_other(self):
        self.c.set_drain(True, {'id': 'orphan'})
        self.router.drain['rosalina'] = 'operator: maintenance'
        self.assertFalse(self.c.release_orphan_drain())
        self.assertTrue(self.system.draining)
        self.assertEqual(self.router.drain['rosalina'], 'operator: maintenance')

    def test_orphan_recovery_cannot_race_a_runtime_operation(self):
        self.c.set_drain(True, {'id': 'orphan'})
        with fixture.lock(self.c.state / 'runtime.lock'):
            self.assertFalse(self.c.release_orphan_drain())
        self.assertTrue(self.system.draining)
        self.assertIn('rosalina', self.router.drain)


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
        # A router that cannot serve its active catalog: every admin call it must resolve a
        # backend for is a 503 until the catalog file holds a kind it can serve, and the file
        # is what a restart reads back, so the wedge survives reboots. The 2026-10-12 incident.
        self.router_wedged_kind = None
        self.router_supported_kinds = None  # None: an old router that reports nothing
        self.router_catalog_path = None  # the file the router serves its catalog from
        self.router_503 = None  # a bare 503 with no JSON body, as a router mid-restart gives

    def _wedged_router_check(self, url):
        """Refuse admin calls the way a router wedged on an unsupported kind does.

        The real router resolves its backend adapter for each such request and re-reads its
        catalog file on every call, so a catalog of a kind it does not implement 503s each
        request, and only a rewritten file - which only the runtime, the file's writer, can do
        - clears it. `reload-config` resolves no adapter, so it answers even while wedged,
        exactly as the real endpoint does.
        """
        if self.router_wedged_kind is None or self.router_catalog_path is None:
            return
        if url.endswith('/reload-config'):
            return
        try:
            catalog = json.loads(Path(self.router_catalog_path).read_text())
        except Exception:
            return
        if catalog.get('backend_kind') != self.router_wedged_kind:
            return
        raise urllib.error.HTTPError(url, 503, 'Service Unavailable', {}, BytesIO(json.dumps(
            {'ok': False, 'error': {'code': 'INVALID_BACKEND_KIND',
                'message': 'Unsupported active backend kind: ' + self.router_wedged_kind + '.'}}).encode()))

    def http(self, url, body=None, headers=None, timeout=None, method=None):
        if url.endswith(('/runtime-state', '/runtime-drain', '/reload-config')):
            self._wedged_router_check(url)
            if self.router_503 is not None:
                raise urllib.error.HTTPError(url, 503, 'Service Unavailable', {}, BytesIO(b''))
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
        result = super().http(url, body, headers, timeout)
        if url.endswith('/runtime-state'):
            if self.router_supported_kinds is not None:
                result['router'] = {'supported_backend_kinds': self.router_supported_kinds}
            else:
                result.pop('router', None)  # default: an old router that reports nothing
        return result


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
        self.system.router_wedged_kind = None
        self.system.router_supported_kinds = None
        self.system.router_catalog_path = str(self.state / 'router' / 'active-model.json')
        self.system.router_503 = None
        self.web_state = {'started': True, 'status': self.c.status()}
        self.ops = Operations(self.c, self.web_state)

    def wedge_the_router(self, transaction, catalog=None):
        """The state a refused Strata catalog leaves: the router serves a file of a kind it
        cannot handle, the drain is held for that attempt, and every admin call 503s. The state
        is persisted in the file, which is what a restart reads back."""
        lkg = self.c.last_known_good()
        wedged = copy.deepcopy(catalog if catalog is not None else lkg['bundle']['catalog'])
        wedged['backend_kind'] = 'strata'
        for entry in wedged['models']:
            entry['backend_kind'] = 'strata'
        atomic_json(self.state / 'router' / 'active-model.json', wedged)
        self.system.router_wedged_kind = 'strata'
        self.system.draining = True
        self.system.reason = 'local-ai-runtime:' + transaction['id']

    def failed_switch_transaction(self, target_profile='flash-next-strata-solo-thinking-recovery'):
        """The transaction a refused switch leaves behind: pending, its drain still owned, the
        target recorded - the same shape the 2026-10-12 host carried across its reboot."""
        target = render(ROOT / 'config', fixture.HOST, target_profile)
        tx = {'id': str(uuid.uuid4()), 'started_at': '2000-01-01T00:00:00+00:00', 'phase': 'verifying',
            'intent': 'switch', 'previous': self.active,
            'target': {'revision': 'a' * 40, 'image': 'old-image', 'bundle': target},
            'changed_services': [STRATA_SOLO], 'revision': 'a' * 40, 'profile': target_profile,
            'config_sha256': target['config_sha256'],
            'error': 'Update failed: HTTP Error 503: Service Unavailable; recovery failed: HTTP Error 503: Service Unavailable'}
        atomic_json(self.state / 'transaction.json', tx)
        return tx

    def switch_to_other_daytime(self):
        """The alias the other Daytime launch runs under, to make one backend of it refuse readiness."""
        target = render(ROOT / 'config', fixture.HOST, 'qwen27b-q6k-pair')
        self.assertNotEqual(target['config_sha256'], self.bundle['config_sha256'])
        argv = target['compose']['services'][CODING_Q6K]['command']
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
            self.c.transition('qwen27b-q6k-pair')
        transaction = self.c.load('transaction.json')
        self.assertNotIn('recovery needs attention', str(transaction.get('error') or ''),
            'the restoration was supposed to finish, not stop for an operator')
        self.assertFalse(self.system.draining, 'the router was left drained with nothing serving')
        self.assertTrue(self.c.serving(), 'no healthy backend is serving after the failure')
        self.assertEqual(transaction['phase'], 'recovered')
        self.assertEqual(self.c.load('active.json')['bundle']['profile'], 'qwen27b-q8-pair')
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
        self.assertEqual(deployment['profile'], 'qwen27b-q6k-pair')
        self.assertEqual(deployment['phase'], 'recovered')
        self.assertEqual(deployment['intent'], 'deployment')
        self.assertIn('readiness', str(deployment['error']))

    # 1c/1e (ii). The recovery pass itself fails, then re-enters and finishes on its own.
    def test_a_failed_attempt_schedules_the_next_one_and_the_next_one_finishes_the_job(self):
        self.system.never_healthy_alias = self.switch_to_other_daytime()
        self.system.failed_generations = 1  # the restoration trips over its own acceptance check
        with self.assertRaisesRegex(RuntimeError, 'recovery needs attention'):
            self.c.transition('qwen27b-q6k-pair')
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
            self.c.transition('qwen27b-q6k-pair')
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
        self.assertEqual(result['profile'], 'qwen27b-q8-pair')
        self.assertTrue(self.c.serving())
        self.assertFalse(self.system.draining)
        self.assertEqual(self.c.load('active.json')['bundle']['profile'], 'qwen27b-q8-pair')
        self.assertFalse(self.c.restoration_pending())

    def test_a_restoration_publishes_the_names_saved_while_it_was_broken(self):
        # An alias is not part of what a restoration restores: the release comes back exactly as it
        # was measured. It is part of what the household is told. A catalog rebuilt from the stored
        # bundle would announce a name the operator retired while the machine was down, and a router
        # that is never corrected is a name nobody can unsend.
        self.strand_the_backend()
        self.assertTrue(self.c.set_alias('qwen3.8-27b-q8_0-tensor-next', 'Workhorse')['saved'])
        result = self.c.restore_last_known_good()
        self.assertTrue(result['restored'])
        published = json.loads((self.state / 'router' / 'active-model.json').read_text())
        self.assertEqual(published['configuration']['id'], 'qwen27b-q8-pair')
        self.assertEqual(published['models'][0]['display_name'], 'Workhorse')
        self.assertEqual(published['models'][0]['model'], 'qwen3.8-27b-q8_0-tensor-next')

    def test_a_drain_left_by_a_finished_attempt_is_released(self):
        """The drain outlives the transaction when the pass that would release it cannot reach the
        router: the machine is serving, the router is not handing anything out, and only the reason
        string says who was owed the release."""
        self.strand_the_backend()
        (self.state / 'transaction.json').unlink(missing_ok=True)
        self.assertTrue(self.c.draining_for_me(healing=False), 'parked behind a drain nobody owns')
        self.assertTrue(self.c.release_orphan_drain())
        self.assertFalse(self.system.draining)
        self.assertTrue(self.c.serving(), 'reopening a drain must restore its stopped backend first')
        # Released once, and the same pass is not asked again for a minute: an announcement loop that
        # polled every router it has would be a new kind of noise in a house that has one page.
        self.assertFalse(self.c.release_orphan_drain())

    def test_a_drain_somebody_else_is_holding_is_left_alone(self):
        """An operator holding traffic to look at a card is not an orphan. The reason is the only
        thing that distinguishes the two, and it is treated as the contract it is."""
        self.strand_the_backend()
        (self.state / 'transaction.json').unlink(missing_ok=True)
        self.system.reason = 'operator: swap the card'
        self.assertFalse(self.c.release_orphan_drain())
        self.assertTrue(self.system.draining)

    def test_orphan_recovery_failure_keeps_admission_closed_and_schedules_retry(self):
        self.strand_the_backend()
        self.system.fail_reconcile = True
        self.assertFalse(self.c.release_orphan_drain())
        self.assertTrue(self.system.draining)
        self.assertTrue(self.c.restoration_pending())
        self.system.fail_reconcile = False
        self.make_restoration_due()
        self.assertTrue(self.ops.restore_if_due())
        self.wait_for_worker()
        self.assertTrue(self.c.serving())
        self.assertFalse(self.c.restoration_pending())

    def test_router_maintenance_reservation_blocks_orphan_recovery(self):
        self.strand_the_backend()
        self.c.save('router-maintenance.json', {'reserved': True})
        self.assertFalse(self.c.release_orphan_drain())
        self.assertTrue(self.system.draining)

    def test_the_announcement_loop_is_what_releases_it(self):
        """No new loop, and no waiting for a person to notice: the pass that already reports whether
        this machine was heard from is the one that reopens the door."""
        self.strand_the_backend()
        (self.state / 'transaction.json').unlink(missing_ok=True)
        self.c.announce(self.web_state['status'])
        self.assertFalse(self.system.draining)

    def test_an_in_flight_attempt_keeps_its_own_drain(self):
        self.strand_the_backend()
        transaction = self.c.load('transaction.json', {}) or {}
        transaction['phase'] = 'draining'
        self.c.save('transaction.json', transaction)
        self.assertFalse(self.c.release_orphan_drain())
        self.assertTrue(self.system.draining)

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
            self.c.transition('qwen27b-q6k-pair')
        status = self.c.status()
        self.assertFalse(status['service']['serving'])
        self.assertTrue(status['service']['restorable'])
        self.assertTrue(status['service']['pending'])
        self.assertEqual(status['service']['profile'], 'qwen27b-q8-pair')
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
            self.c.transition('qwen27b-q6k-pair')
        with self.assertRaisesRegex(RuntimeError, 'interrupted deployment'):
            self.c.transition('qwen27b-q8-pair')
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

        This is the backstop behind the pre-drain gate: the router reports the kind as
        supported (so the gate admits the switch) and still rejects the file once it is
        published. A router that reports nothing is refused at the gate instead, which is
        what test_a_switch_to_an_unsupported_kind_is_refused_before_the_drain covers.
        """
        self.system.router_supported_kinds = ['ollama', 'llama_cpp', 'strata']
        self.c._kinds_cache = None
        self.c.publish(self.bundle)
        target = render(ROOT / 'config', self.c.host, 'flash-next-strata-solo-thinking-recovery')
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

    # The 2026-10-12 root cause, end to end: a catalog the router refuses wedges every admin
    # call, and the wedge is a file, so it survives a reboot. The restoration must converge it,
    # because the runtime is the only writer of the file the router is wedged on.
    def test_a_wedged_router_is_converged_by_the_restoration_itself(self):
        """The host is exactly as it was at 22:27: a refused Strata catalog is served, the
        drain is held for a dead attempt, and every admin call is a 503. No person, CLI or
        power button is involved: the scheduled restoration converges the router, then
        completes the recovery."""
        tx = self.failed_switch_transaction()
        self.wedge_the_router(tx)
        self.system.containers['qwen38-daytime']['State']['Running'] = False  # the refused launch is dead
        self.c.schedule_restoration(RuntimeError('HTTP Error 503: Service Unavailable'), tx)
        self.make_restoration_due()
        started = self.ops.restore_if_due()
        self.assertIsNotNone(started, 'the due restoration was not taken up on its own')
        self.wait_for_worker()
        self.assertEqual(self.ops.read(started['id'])['status'], 'recovered')
        # The wedge is cleared at the source: the router's file holds the last-known-good
        # release again, and a reload told the router to serve it.
        served = read(self.state / 'router' / 'active-model.json')
        self.assertEqual(served['backend_kind'], 'llama_cpp')
        self.assertEqual(served['model'], self.c.last_known_good()['bundle']['catalog']['model'])
        self.assertIn(('publish',), self.system.events)
        # And the host is where a restart is promised to land: serving, undrained, recorded.
        self.assertTrue(self.c.serving())
        self.assertFalse(self.system.draining)
        self.assertFalse(self.c.restoration_pending())
        self.assertEqual(self.c.load('active.json')['bundle']['profile'], 'qwen27b-q8-pair')
        record = self.c.restoration()
        self.assertIn('qwen27b-q8-pair', record['note'])
        self.assertNotIn('flash-next', record['note'])

    def test_a_reboot_converges_a_wedged_host_without_human_intervention(self):
        """The governing promise: whatever state the machine is in, a restart lands it on the
        known-working configuration. The 2026-10-12 host rebooted into the wedged state and
        stayed there until a hand wrote a file; the supervisor's first quiet tick after boot
        must now do that work, with nothing else touched."""
        tx = self.failed_switch_transaction()
        self.wedge_the_router(tx)
        self.system.containers['qwen38-daytime']['State']['Running'] = False
        # The failed attempt left a scheduled retry and a pending transaction: the state a
        # reboot carries into the fresh process.
        self.c.schedule_restoration(RuntimeError('HTTP Error 503: Service Unavailable'), tx)
        rebooted = fixture.Controller(ROOT / 'config', self.state, 'a' * 40, self.system)
        prepare = patch.object(rebooted, 'prepare', side_effect=lambda b: None)
        prepare.start(); self.addCleanup(prepare.stop)
        ops = Operations(rebooted, {'started': True, 'status': rebooted.status()})
        self.make_restoration_due()
        started = ops.restore_if_due()
        self.assertIsNotNone(started, 'the boot tick did not take up the scheduled restoration')
        ops.worker.join(10)
        self.assertFalse(ops.worker.is_alive(), 'the restoration worker did not finish')
        served = read(self.state / 'router' / 'active-model.json')
        self.assertEqual(served['backend_kind'], 'llama_cpp')
        self.assertTrue(rebooted.serving())
        self.assertFalse(self.system.draining)
        self.assertFalse(rebooted.restoration_pending())
        self.assertEqual(rebooted.load('active.json')['bundle']['profile'], 'qwen27b-q8-pair')

    def test_a_switch_to_an_unsupported_kind_is_refused_before_the_drain(self):
        """The gate the 22:20 click should have been: a catalog of a kind the deployed router
        cannot serve is refused before the drain is taken, the machine keeps serving what it
        serves, and nothing is left behind that a reboot could then resurrect."""
        with self.assertRaisesRegex(RuntimeError, 'does not support backend kind strata'):
            self.c.transition('flash-next-strata-solo-thinking-recovery')
        self.assertFalse(self.system.draining, 'the drain was taken to reach the refusal')
        self.assertNotIn(('drain', True), self.system.events)
        self.assertEqual(self.c.load('active.json')['bundle']['profile'], 'qwen27b-q8-pair')
        self.assertTrue(self.c.serving())
        # The card says it before the click, because the verdict is the gate.
        bundle = render(ROOT / 'config', fixture.HOST, 'flash-next-strata-solo-thinking-recovery')
        self.assertTrue(any('does not support backend kind strata' in r for r in self.c.verdict(bundle)))

    def test_the_router_gate_passes_a_kind_the_router_reports(self):
        """Once the router update reports the kinds it serves, the same catalog is no longer
        refused at the gate: the refusal was about the deployed router, not the profile."""
        self.system.router_supported_kinds = ['ollama', 'llama_cpp', 'strata']
        # The status poll in setUp cached the old router's answer; the clock moves past it.
        self.c._kinds_cache = None
        bundle = render(ROOT / 'config', fixture.HOST, 'flash-next-strata-solo-thinking-recovery')
        self.c.require_router_supports(bundle)  # must not raise
        self.assertFalse(any('does not support backend kind' in r for r in self.c.verdict(bundle)))

    def test_a_transient_503_leaves_a_matching_catalog_untouched(self):
        """A 503 without the wedge signature is a router in trouble, not a wedged one: the
        repair leaves the file alone - it holds a catalog of the same kind as the release it
        serves, decorations and all - and the caller's retry schedule still sees the error."""
        self.c.publish(self.bundle)  # the file now holds exactly what a healthy publish leaves
        path = self.state / 'router' / 'active-model.json'
        before = path.read_bytes()
        self.system.router_503 = 'bare'
        with self.assertRaises(urllib.error.HTTPError):
            self.c.owns_drain({'id': 't1'})
        self.assertEqual(path.read_bytes(), before, 'a transient failure rewrote the catalog')


if __name__ == '__main__':
    unittest.main()


class OnePairHostRestorationTests(unittest.TestCase):
    """The same promise on a host with one text pair, where one backend is the whole service.

    A second machine is not a Rosalina with two cards taken away. With a single model running, the
    drain check has no second backend to lean on and the restoration has one container to bring back,
    so the 2026-10-12 failures have to be re-fought on that shape as well -- it is the shape a two-card
    host actually has, and it is the shape the release that first runs there will be restored into.
    """

    PAIR = ('flash-next-strata-two-card-iq4-xs', 'flash-next-strata-two-card-iq3-s')
    HOST = {'model_root': '/models', 'router_state_dir': '/router',
        'router_admin_url': 'http://ai-router:11435', 'router_admin_header': 'X-Admin-Token',
        'uid': 1000, 'gid': 1000, 'host_id': 'daisy', 'gpu_architectures': ['86', '89'],
        'gpu_ids': {'daytime': ['GPU-daisy-1', 'GPU-daisy-2']},
        'gpu_names': {'daytime': ['RTX 2080 Ti', 'RTX 2080 Ti']}}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name)
        self.config = self.state / 'config'
        shutil.copytree(ROOT / 'config', self.config)
        # Two configurations of this machine, each running alone across its two cards. As a model
        # that owns every card it loads nothing else, so its list is just itself.
        for name in self.PAIR:
            path = self.config / 'profiles' / (name + '.json')
            definition = read(path)
            definition.update(hosts=['daisy'], exclusive=True, gpu_group='all', models=[name])
            path.write_text(json.dumps(definition))
        host = copy.deepcopy(self.HOST)
        host['router_state_dir'] = str(self.state / 'router')
        atomic_json(self.state / 'host.json', host)
        (self.state / 'router-token').write_text('synthetic-token')
        self.bundle = render(self.config, host, self.PAIR[0])
        self.name = self.bundle['compose']['services'][STRATA_SOLO]['container_name']
        self.assertEqual(list(self.bundle['compose']['services']), [STRATA_SOLO],
            'a host with one text pair runs one backend, which is what these tests are about')
        self.system = fixture.FakeSystem(self.bundle)
        self.c = Controller(self.config, self.state, 'a' * 40, self.system)
        self.active = {'revision': 'a' * 40, 'image': 'old-image', 'bundle': self.bundle}
        atomic_json(self.state / 'active.json', self.active)
        patch.object(self.c, 'prepare',
            side_effect=lambda b: self.system.events.append(('prepare',))).start()
        # Both configurations launch an engine that provisions its own run workspace, and this fake has
        # no model tree to provision one into. Filling it is another test's subject; here it would only
        # say that /models does not exist before the drain this test is about is ever opened.
        patch.object(self.c, 'ensure_provisioned', return_value=False).start()
        patch('runtime.controller.time.sleep').start()
        self.addCleanup(patch.stopall)
        self.system.__class__ = StubbornSystem
        for knob, value in (('never_healthy_alias', None), ('refuse_slots', ()), ('slots_shape', None),
                ('busy_names', None), ('slot_polls', 0), ('router_wedged_kind', None),
                ('router_supported_kinds', None), ('router_503', None)):
            setattr(self.system, knob, value)
        self.system.router_catalog_path = str(self.state / 'router' / 'active-model.json')
        self.web_state = {'started': True, 'status': self.c.status()}
        self.ops = Operations(self.c, self.web_state)

    def test_the_one_action_restores_the_only_backend_this_host_has(self):
        self.system.containers[self.name]['State']['Running'] = False
        self.c.set_drain(True, {'id': 'stranded'})
        self.assertFalse(self.c.serving())
        result = self.c.restore_last_known_good()
        self.assertTrue(result['restored'])
        self.assertEqual(result['profile'], self.PAIR[0])
        self.assertTrue(self.c.serving())
        self.assertFalse(self.system.draining, 'the router was left drained with nothing serving')
        self.assertFalse(self.c.restoration_pending())
        # What the page can honestly say about a host that has no second model: `nighttime` is not
        # a service of this machine at all, and the only offline service is the ID another of this
        # host's configurations offers and the running one does not load the model for.
        status = self.c.status()
        self.assertTrue(status['service']['serving'])
        self.assertEqual([(x['model'], x['aliases'], x['reason']) for x in status['offline_services']],
            [('qwen3.8-flash-next-iq3_s', ['daisy'], 'not_loaded_by_this_configuration'),
                (STRATA_DAISY, ['daisy'], 'not_loaded_by_this_configuration')])
        self.assertEqual(status['host']['text_gpus'], 2)

    def test_a_switch_that_never_becomes_healthy_leaves_this_host_serving(self):
        # The incident, on one pair: the target starts, never answers, and the drain it opened has to
        # be closed by putting the previous release back -- with no second backend to serve meanwhile.
        target = render(self.config, self.c.host, self.PAIR[1])
        # The pre-drain gate, not the failure under test: this fake router has to report the kind these
        # configurations launch, or the switch is refused before it ever reaches the drain.
        self.system.router_supported_kinds = ['ollama', 'llama_cpp', 'strata']
        self.c._kinds_cache = None
        self.system.never_healthy_alias = target['manifest']['services'][0]['model_alias']
        with self.assertRaisesRegex(RuntimeError, 'Deployment failed'):
            self.c.transition(self.PAIR[1])
        self.assertFalse(self.system.draining, 'the router was left drained with nothing serving')
        self.assertTrue(self.c.serving(), 'no healthy backend is serving after the failure')
        self.assertEqual(self.c.load('transaction.json')['phase'], 'recovered')
        self.assertEqual(self.c.load('active.json')['bundle']['profile'], self.PAIR[0])
        self.assertFalse(self.c.restoration_pending())
        self.assertEqual(set(self.system.containers), {self.name})

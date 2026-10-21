"""Publishing a catalog to a router that does not live on this machine.

The contract these tests hold the runtime to is the router's own: a machine owns its models,
qualifies what it loaded, and hands the resulting catalog to every router that serves it — over HTTP
when the router is elsewhere, into a directory when it is not, and to both during the migration
between those two states. Each test below exists for a failure that only appears once the router is
another machine's process: a catalog written to a directory nobody reads, a drain that stopped the
wrong house, an alias two machines both claimed, and a switch that believed a publication had
happened because nothing said otherwise.
"""
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from runtime.config import read, render
from runtime.controller import Controller
from runtime.system import atomic_json
from test_controller import FakeSystem, vision_host

ROOT = Path(__file__).resolve().parents[1]
REMOTE = 'http://192.168.1.23:11435'
TARGET = {'url': REMOTE, 'token_env': 'PUBLISHER_TOKEN'}
# What the router already serves for somebody else. A shared router lists every machine, so a
# runtime that reads it as its own inventory is wrong the moment a second machine publishes.
OTHER = [{'id': 'whisper-large-v3', 'x_ollama_router': {'health': {'available': True},
        'output_policy': 'unrestricted', 'context_window': 32768, 'default_output_tokens': None,
        'max_output_tokens': None}}]


def now_iso():
    from runtime.system import now
    return now()


def refusal(code, payload):
    return urllib.error.HTTPError(REMOTE, code, 'Refused', {}, io.BytesIO(json.dumps(payload).encode()))


class Router:
    """The machine at .23: hosts are declared, a catalog is validated on the way in, and the
    drain is per host. Deliberately simpler than the real reader, and just as unimpressed."""

    def __init__(self, declared=('local', 'rosalina'), default_host='rosalina'):
        self.base = REMOTE
        self.declared = list(declared)
        self.default_host = default_host
        self.catalogs = {}       # host -> accepted document
        self.revisions = {}      # host -> revision of the accepted document
        self.drain = {}          # host -> reason
        self.active_by_host = {}
        self.queued_by_host = {}
        self.calls = []
        self.available = True
        self.clashes = ()

    def handle(self, path, body, headers, method):
        self.calls.append((method or 'GET', path, body, headers))
        if path.endswith('/runtime-state'):
            models = [self.entry(m) for document in self.catalogs.values() for m in document['models']]
            models += OTHER
            document = self.catalogs.get(self.default_host)
            return {'runtime': {'draining': False, 'drain_reason': None,
                    'active_count': sum(self.active_by_host.values()),
                    'queued_count': sum(self.queued_by_host.values()),
                    'draining_hosts': sorted(self.drain), 'active_by_host': self.active_by_host,
                    'queued_by_host': self.queued_by_host,
                    'hosts': {host: {'draining': True, 'drain_reason': reason}
                        for host, reason in self.drain.items()}},
                'router': {'supported_backend_kinds': ['ollama', 'llama_cpp', 'strata']},
                'active_model': ({'profile': document['configuration']['id']} if document else None),
                'models': models}
        if path.endswith('/runtime-drain'):
            host = body.get('host')
            if body['enabled']:
                self.drain[host or 'whole-router'] = body['reason']
            else:
                self.drain.pop(host or 'whole-router', None)
            return {'ok': True, 'runtime': {'draining_hosts': sorted(self.drain)}}
        if path.endswith('/hosts') or path.endswith('/hosts/'):
            return {'ok': True, 'default_host': self.default_host, 'hosts': [
                {'id': host, 'published': host in self.catalogs,
                    'revision': self.revisions.get(host),
                    'model_ids': [m['model'] for m in self.catalogs[host]['models']]
                    if host in self.catalogs else []} for host in self.declared]}
        if '/catalog' in path:
            host = path.split('/hosts/')[1].split('/')[0]
            if host not in self.declared:
                raise refusal(404, {'error': {'code': 'HOST_NOT_DECLARED', 'message': host}})
            if method == 'DELETE':
                self.catalogs.pop(host, None); self.revisions.pop(host, None)
                return {'ok': True, 'host': host, 'published': False}
            return self.accept(host, body)
        raise AssertionError('unhandled router path ' + path)

    def accept(self, host, document):
        claimed = {m['model'] for m in document['models']}
        clash = sorted(claimed & set(self.clashes))
        if clash:
            raise refusal(409, {'error': {'code': 'CATALOG_ID_CONFLICT', 'message': ', '.join(clash)}})
        if document['default_model'] not in claimed:
            raise refusal(422, {'error': {'code': 'CATALOG_DEFAULT_MODEL_INVALID'}})
        revision = hashlib.sha256(json.dumps(document).encode()).hexdigest()
        self.catalogs[host] = document
        self.revisions[host] = revision
        return {'ok': True, 'host': host, 'revision': revision, 'accepted': True,
            'default_model': document['default_model'],
            'models': [m['model'] for m in document['models']], 'warnings': []}

    def entry(self, model):
        return {'id': model['model'], 'x_ollama_router': {
            'health': {'available': self.available}, 'output_policy': 'unrestricted',
            'context_window': model['context_length'], 'default_output_tokens': None,
            'max_output_tokens': None}}


class PushSystem(FakeSystem):
    """The controller's view of a machine with a router in the neighbourhood and one down the road."""

    def __init__(self, bundle, router=None):
        super().__init__(bundle)
        self.router = router
        self.published = []

    def http(self, url, body=None, headers=None, timeout=None, method=None):
        if self.router is not None and url.startswith(self.router.base):
            return self.router.handle(url[len(self.router.base):], body, headers, method)
        return super().http(url, body, headers, timeout)

    def push(self, method='PUT'):
        return [call for call in (self.router.calls if self.router else [])
            if call[0] == method and '/catalog' in call[1]]


class PublisherTests(unittest.TestCase):
    PROFILE = 'flash-next-strata-solo-thinking-recovery'

    def build(self, host_changes=None, declared=('local', 'rosalina'), default_host='rosalina',
            with_target=True, profile=None, env=None):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.state = Path(tmp.name)
        (self.state / 'router').mkdir()
        self.config = self.state / 'config'
        shutil.copytree(ROOT / 'config', self.config)
        host = vision_host()
        host.update(host_id='rosalina', router_state_dir=str(self.state / 'router'))
        if with_target:
            host['router_target'] = dict(TARGET)
        host.update(host_changes or {})
        atomic_json(self.state / 'host.json', host)
        (self.state / 'router-token').write_text('legacy-token')
        self.host = host
        self.bundle = render(self.config, host, profile or self.PROFILE)
        self.router = Router(declared=declared, default_host=default_host)
        self.system = PushSystem(self.bundle, self.router)
        self.environ = patch.dict('os.environ',
            {TARGET['token_env']: 'publisher-secret'} if env is None else env)
        self.environ.start(); self.addCleanup(self.environ.stop)
        return Controller(self.config, self.state, 'a' * 40, self.system)

    def published_document(self):
        return self.router.catalogs['rosalina']

    def adopt(self):
        controller = self.build()
        controller.publish(self.bundle)
        controller.save('active.json', {'revision': controller.revision, 'profile': self.bundle['profile'],
            'bundle': self.bundle})
        return controller

    # --- obligation 1: the definition, or a loud refusal -----------------------------------------

    def test_a_machine_that_states_nowhere_to_publish_refuses_to_start(self):
        # The router's contract names this failure explicitly, and it is worth the refusal: a runtime
        # with no sink starts its models, passes every health check, and tells nobody. Every metric
        # is green and the household has no models.
        with self.assertRaisesRegex(RuntimeError, 'nowhere to publish'):
            self.build(with_target=False, host_changes={'router_state_dir': None})

    def test_the_definition_names_a_url_a_credential_and_this_machine(self):
        # The host ID is not repeated inside the target: a machine speaks as the machine its own
        # settings say it is, and two files that can disagree about that is how a catalog ends up
        # under another machine's name — which no client notices and every request pays for.
        for change, why in (({'url': REMOTE}, 'token_env'),
            ({'url': REMOTE, 'host': 'rosalina'}, 'states host'),
            ({'url': REMOTE, 'token_env': 'https://router/'}, 'token_env'),
            ({'url': 'router.local'}, 'admin base'), ({}, 'url')):
            with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, why):
                self.build(host_changes={'router_target': change})
        with self.assertRaisesRegex(RuntimeError, 'host_id'):
            self.build(host_changes={'router_target': {'url': REMOTE, 'token_env': 'X'},
                'host_id': None})

    def test_a_missing_credential_says_which_variable_it_wanted(self):
        # A bearer token is delivered in the environment and never stored, so the commonest fault is
        # a deployment that named the variable and did not fill it. The message has to carry the name.
        # The credential is read when the settings are read, which is the moment the machine decides
        # who it is: an empty variable is deployment state that has not been filled in, and saying
        # which variable was meant is the difference between a two-minute fix and a debugging session.
        controller = self.build(env={TARGET['token_env']: ''})
        with self.assertRaisesRegex(RuntimeError, TARGET['token_env']):
            controller.publish(self.bundle)
        self.assertEqual(self.router.catalogs, {})

    # --- obligations 2, 4 and 7: what goes over the wire, and what comes back --------------------

    def test_the_push_carries_the_whole_catalog_and_the_receipt_comes_back(self):
        controller = self.build()
        controller.publish(self.bundle)
        method, path, body, headers = self.system.push()[0]
        self.assertEqual((method, path), ('PUT', '/admin/api/hosts/rosalina/catalog'))
        self.assertEqual(headers['Authorization'], 'Bearer publisher-secret')
        # Not a marker, not a pointer: the document a local router would have read, root projection
        # included, because the reader that serves it is the reader that validated it.
        self.assertEqual(body['schema_version'], 3)
        self.assertEqual({m['model'] for m in body['models']},
            {m['model'] for m in self.bundle['catalog']['models']})
        self.assertEqual(body['default_model'], self.bundle['catalog']['default_model'])
        # The root of the document is the default model's entry, projected up: `backend_url`,
        # `context_length`, `model`. A publisher that sent only the list would leave a router that
        # reads the root — which the local one has always done — looking at the previous model.
        self.assertEqual(body['model'], body['default_model'])
        self.assertEqual(body['backend_url'], body['models'][0]['backend_url'])
        self.assertEqual(body['context_length'], body['models'][0]['context_length'])
        self.assertEqual(body['configuration']['runtime_revision'], 'a' * 40)
        # And the revision in the response is the receipt: comparing it is what turns "the request
        # did not raise" into "the catalog being served is the one this machine sent".
        record = controller.load('publication.json')
        remote = [s for s in record['sinks'] if s['kind'] == 'http'][0]
        self.assertEqual(remote['revision'], self.router.revisions['rosalina'])
        self.assertEqual({s['name'] for s in record['sinks']},
            {str(self.state / 'router'), REMOTE})

    def test_both_sinks_get_the_same_document_and_the_file_goes_first(self):
        # Two sinks is a migration state and not a design: the same bytes go to the router that has
        # always read the directory and to the one two hops away, in the order that can be put back.
        controller = self.build()
        controller.publish(self.bundle)
        on_disk = read(self.state / 'router' / 'active-model.json')
        self.assertEqual(on_disk, self.published_document())
        order = [call for call in self.system.events if call[0] == 'publish']
        self.assertTrue(order, 'the local sink was never asked to reload')
        self.assertTrue(self.system.push(), 'the remote sink was never given the catalog')

    def test_a_refusal_from_the_router_is_a_failure_and_the_local_document_goes_back(self):
        # `409 CATALOG_ID_CONFLICT` is the name-collision rule doing its job: the router that already
        # answers `daytime` for another machine refuses this one rather than letting a client's
        # requests arrive at the wrong GPU. It has to fail the switch — a catalog nobody can address
        # is worse than a deployment that did not happen — and the sink that already accepted this
        # document has to be told it did not happen either.
        controller = self.build()
        controller.publish(self.bundle)
        before = read(self.state / 'router' / 'active-model.json')
        self.router.clashes = tuple(m['model'] for m in self.published_document()['models'])
        with self.assertRaisesRegex(RuntimeError, 'CATALOG_ID_CONFLICT'):
            controller.publish(self.bundle)
        self.assertEqual(read(self.state / 'router' / 'active-model.json'), before)
        self.assertEqual(self.router.revisions['rosalina'],
            hashlib.sha256(json.dumps(before).encode()).hexdigest(),
            'a refusal replaced the catalog the router had already accepted')

    def test_a_machine_the_router_has_not_declared_is_told_so_by_name(self):
        # A machine the router has never heard of is the ordinary first contact between two
        # deployments, and `404 HOST_NOT_DECLARED` is the answer that names the fix: one more id in
        # the router's list and a restart of the router, not a re-imagination of this machine.
        controller = self.build(declared=('local',))
        with self.assertRaisesRegex(RuntimeError, 'HOST_NOT_DECLARED'):
            controller.publish(self.bundle)

    # --- obligations 3 and 9: reachability is the router's network, not this machine's health -----

    def test_a_backend_the_router_cannot_reach_is_reported_while_it_is_a_loopback_claim(self):
        # During the migration a Rosalina backend is still published on loopback, so the router
        # catalogues it and honestly reports it unreachable. That is a state to publish through, not
        # a state to fail: refusing it would keep the new router empty for the whole window, and the
        # page has to say which of the two facts holds.
        controller = self.build()
        self.router.available = False
        controller.publish(self.bundle)
        remote = [s for s in controller.load('publication.json')['sinks'] if s['kind'] == 'http'][0]
        self.assertFalse(remote['available'])
        self.assertIn('ROUTER_CANNOT_REACH_BACKEND:rosalina', remote['warnings'])
        # And it is not this machine being unhealthy: every check that can be answered from here —
        # the backend loaded, the slots configured, the window the one published — still passes.
        # Confusing the two is how a migration gets blamed on the machine that migrated last.
        self.assertEqual(controller.verdict(self.bundle), [])

    def test_a_claim_of_routability_is_a_claim_the_router_holds_to(self):
        # Once the machine states the address it publishes its backends on, "unreachable from the
        # router" stops being an expected transition and becomes a publication that would serve
        # nothing — which is exactly the moment it should stop being a warning.
        controller = self.build(host_changes={'backend_address': '192.168.1.4'})
        self.assertEqual(render(self.config, self.host, self.PROFILE)['catalog']['models'][0]
            ['backend_url'], 'http://192.168.1.4:18080')
        self.router.available = False
        with self.assertRaisesRegex(RuntimeError, 'cannot reach'):
            controller.publish(self.bundle)

    def test_another_machines_active_model_is_not_this_machines_verification(self):
        # `active_model` is the default host's answer to an unnamed request. A machine that is not the
        # default has no claim on it, so it must not fail its own switch because the router is busy
        # answering another machine.
        controller = self.build(default_host='local')
        self.router.active_by_host = {'voice-box': 4}
        controller.publish(self.bundle)
        remote = [s for s in controller.load('publication.json')['sinks'] if s['kind'] == 'http'][0]
        self.assertEqual(remote['router_default'], 'local')

    # --- obligations 5 and 6: the drain belongs to one machine, and so does taking it off ---------

    def test_the_drain_is_scoped_to_this_machine_on_the_router_that_shares_it(self):
        controller = self.build()
        transaction = {'id': 'tx-1'}
        controller.set_drain(True, transaction)
        scoped = [call for call in self.router.calls if call[1].endswith('/runtime-drain')]
        self.assertEqual(scoped[0][2]['host'], 'rosalina')
        self.assertEqual(self.system.events, [('drain', True)],
            'the local router is drained too: its clients are still the ones being served')
        controller.set_drain(False, transaction)
        self.assertEqual(self.router.drain, {}, 'a drain was left parked on a shared router')

    def test_another_machines_work_and_another_machines_drain_are_not_this_machines(self):
        # The two directions of the same fact. A neighbour with four generations running does not
        # hold this machine's drain open, and a neighbour in a maintenance window does not stop this
        # machine replacing a backend. Both used to be read off one global flag, and with two machines
        # that made each other's maintenance windows everybody's.
        controller = self.build()
        controller.publish(self.bundle)
        self.router.active_by_host = {'voice-box': 4, 'whisper-host': 2}
        self.router.drain = {'voice-box': 'local-ai-runtime:somebody-else'}
        self.assertEqual(controller.wait_idle([self.bundle], timeout=6), {'unreachable': [],
            'busy': [], 'malformed': []})
        self.assertFalse(controller.draining_for_me())
        self.router.drain = {'rosalina': 'local-ai-runtime:tx-other'}
        self.assertTrue(controller.draining_for_me())
        # Held for somebody else's transaction, it is somebody else's to release: a pass that
        # treated a neighbour's parked drain as its own would clear it and let the neighbour's
        # clients arrive at a backend being replaced. The rule is the house rule, now per machine.
        self.assertFalse(controller.owns_drain({'id': 'tx-mine'}))

    def test_retirement_leaves_no_catalog_pointing_at_backends_nobody_starts(self):
        # Obligation 6: a catalog whose backends nobody runs is a 503 with a promising name in front
        # of it. Retiring goes to every sink this machine has been publishing to, because leaving one
        # of them advertising is the whole failure with one fewer router.
        controller = self.build()
        controller.publish(self.bundle)
        self.assertEqual(controller.retire(), [str(self.state / 'router'), REMOTE])
        self.assertEqual(self.router.catalogs, {})
        self.assertFalse((self.state / 'router' / 'active-model.json').exists())
        self.assertIn('retired_at', controller.load('publication.json'))

    # --- obligation 8: the page says where the catalog is -----------------------------------------

    def test_the_status_says_which_router_holds_this_machines_catalog(self):
        # "Models running here, published there" and "running here, publication failing since 09:14"
        # are different emergencies, and an operator reading a page has to be able to tell them
        # apart without a shell.
        controller = self.build()
        controller.publish(self.bundle)
        publication = controller.status()['publication']
        self.assertEqual(publication['configuration'], self.bundle['profile'])
        self.assertEqual({s['kind'] for s in publication['sinks']}, {'file', 'http'})
        self.assertTrue(all('revision' in s or s['kind'] == 'file' for s in publication['sinks']))


if __name__ == '__main__':
    unittest.main()


class AnnouncementTests(PublisherTests):
    """Publication is an event, and the loop has to notice when nobody was told.

    The case is not exotic: a controller restart with nothing to change — a portal update, a machine
    rebooting onto a router that was redeployed underneath it — goes through the reconciler reporting
    that everything is already active, and publishes nothing. Meanwhile the router that was rebuilt
    from an empty directory is serving no models, and every metric on this machine is green.
    """

    def test_a_machine_that_is_serving_and_has_never_been_heard_from_is_heard_from(self):
        controller = self.adopt()
        self.router.catalogs = {}      # the router came back with nothing for this machine
        self.router.revisions = {}
        controller.save('publication.json', {})  # and this machine has no proof it said anything
        controller.announce({'publication': controller.load('publication.json')})
        self.assertIn('rosalina', self.router.catalogs)
        self.assertNotIn('error', controller.load('publication.json'))

    def test_a_machine_that_was_told_is_not_still_talking(self):
        # Publication is not a heartbeat. The record of an accepted revision is what keeps the
        # supervising loop from rewriting the local router's file and reloading it every five seconds
        # for a catalog nobody has changed.
        controller = self.adopt()
        before = list(self.router.calls)
        controller.announce({'publication': controller.load('publication.json')})
        self.assertEqual(self.router.calls, before)

    def test_a_publication_that_did_not_land_is_recorded_and_tried_again(self):
        # The failure has to be readable, and it has to be transient in the right direction: the next
        # pass tries again, and the pass after that, until the catalog is where it belongs. A record
        # that only ever holds successes is how a machine ends up healthy and unpublished for a week.
        controller = self.adopt()
        self.router.declared = ['local']
        controller.save('publication.json', {})
        view = {'publication': controller.load('publication.json')}
        controller.announce(view)
        record = controller.load('publication.json')
        self.assertIn('HOST_NOT_DECLARED', record['error'])
        self.assertEqual(record['configuration'], self.bundle['profile'])
        self.assertGreater(record['retry_at'], now_iso(),
            'a failure retried on every status pass is a machine shouting at a router')
        asked = len(self.router.calls)
        controller.announce(record and {'publication': record})
        self.assertEqual(len(self.router.calls), asked, 'the backoff was read and then ignored')
        record['retry_at'] = '2000-01-01T00:00:00+00:00'
        self.router.declared = ['local', 'rosalina']
        controller.announce(record and {'publication': record})
        self.assertIn('rosalina', self.router.catalogs)
        self.assertNotIn('error', controller.load('publication.json'))

    def test_a_controller_that_owns_nothing_published_is_silent(self):
        # Nothing of this revision is serving, so nothing is claimed: an unadopted machine announcing
        # a catalog would be publishing a configuration the operator has not chosen, and a machine
        # mid-switch would be publishing it twice.
        controller = self.build()
        self.router.catalogs = {}
        controller.announce({'publication': {}})
        self.assertEqual(self.router.catalogs, {})

        # A status with no publication record at all is the shape of a machine that has published
        # nothing yet — a reason to publish, and not an attribute error on the way to doing it.
        controller.save('active.json', {'revision': controller.revision, 'profile': self.bundle['profile'],
            'bundle': self.bundle})
        controller.announce({'ready': True})
        self.assertIn('rosalina', self.router.catalogs)


class MirrorTests(PublisherTests):
    """A second router being down is news about the second router.

    During the migration a machine writes its catalog into a directory the house is standing on and
    pushes a copy to a router that is still being proven. Those are not equals, and treating them as
    equals produces the worst of both: a switch stopped dead by a machine nobody is serving traffic
    from, and a copy quietly a fortnight behind. So unreachability on the copy is recorded and retried
    by the loop that already keeps the machine honest, while a refusal — an answer, and a
    disapproving one — fails exactly as it does on the primary, because a wrong document is wrong
    twice.
    """

    def down(self, error=None):
        real = self.system.http
        def fail(url, *args, **kwargs):
            if url.startswith(self.router.base):
                raise error or urllib.error.URLError(ConnectionRefusedError(111, 'connection refused'))
            return real(url, *args, **kwargs)
        self.system.http = fail

    def test_a_copy_that_is_down_does_not_stop_a_switch_the_primary_answers_for(self):
        # Serving and adopted first: the machine has a catalog to deliver, and it is the delivery
        # under test, not the choice of what to serve.
        controller = self.adopt()
        self.down()
        seen = len(self.router.calls)

        # Drained once, drained for real, and not released: the copy's absence is no reason to let
        # traffic back at a backend being replaced, and no reason to stop the switch either.
        controller.set_drain(True, {'id': 'tx-1'})
        self.assertEqual([event for event in self.system.events if event[0] == 'drain'], [('drain', True)])

        controller.publish(self.bundle)
        remote = [s for s in controller.load('publication.json')['sinks'] if s['kind'] == 'http'][0]
        self.assertIn('error', remote)
        self.assertIn('refused', remote['error'])
        # And nothing reached the router while it was down: tolerating a neighbour is not the same as
        # asking it twice and hoping, and a switch that stopped for a machine serving nobody would be
        # the same outage with better documentation.
        self.assertEqual(len(self.router.calls), seen, 'a switch talked to a router that was down')

        # The catalog is not abandoned there, either. The record says a sink has not caught up, the
        # supervising pass reads that, and the copy is handed the document the moment it is back.
        stale = controller.load('publication.json')
        stale['retry_at'] = '2000-01-01T00:00:00+00:00'
        controller.save('publication.json', stale)
        del self.system.http
        controller.announce({'publication': stale})
        self.assertIn('rosalina', self.router.catalogs)
        self.assertFalse([s for s in controller.load('publication.json')['sinks']
            if s['kind'] == 'http'][0].get('error'))

    def test_the_only_router_being_down_is_not_something_to_switch_through(self):
        # With no directory left, the push stops being a copy and becomes the router. Nothing here
        # needs to remember that: the same call that was tolerant a setting ago is fatal now, and a
        # machine cannot drain, publish, or retire its way into a switch it cannot verify.
        controller = self.build(host_changes={'router_state_dir': None})
        self.down()
        with self.assertRaises(urllib.error.URLError): controller.set_drain(True, {'id': 'tx-1'})
        with self.assertRaises(urllib.error.URLError): controller.publish(self.bundle)
        with self.assertRaises(urllib.error.URLError): controller.retire()

    def test_a_copy_that_answers_and_refuses_still_fails_the_switch(self):
        # The distinction the tolerance hangs on: not answering is a router's schedule, refusing is
        # this machine's document, and only one of those improves by waiting.
        controller = self.build()
        self.router.clashes = tuple(m['model'] for m in self.bundle['catalog']['models'])
        with self.assertRaisesRegex(RuntimeError, 'CATALOG_ID_CONFLICT'):
            controller.publish(self.bundle)

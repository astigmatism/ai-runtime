import json
from http.server import ThreadingHTTPServer
import threading
import unittest
import urllib.error
import urllib.request

from runtime.web import handler
from runtime.compat import dispatch


class StatusTests(unittest.TestCase):
    def setUp(self):
        self.state = {'status': {'ready': False, 'profile': 'qwen27b-q8-pair'}}
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler(None, self.state))
        thread = threading.Thread(target=self.server.serve_forever, daemon=True); thread.start()
        self.addCleanup(self.server.server_close); self.addCleanup(self.server.shutdown)
        self.base = 'http://127.0.0.1:' + str(self.server.server_address[1])

    def test_health_reflects_readiness_and_status_remains_available(self):
        with self.assertRaises(urllib.error.HTTPError) as error: urllib.request.urlopen(self.base + '/healthz')
        self.assertEqual(error.exception.code, 503)
        error.exception.close()
        with urllib.request.urlopen(self.base + '/api/status') as response:
            self.assertFalse(json.load(response)['ready'])
        self.state['status']['ready'] = True
        with urllib.request.urlopen(self.base + '/healthz') as response: self.assertTrue(json.load(response)['ready'])

    def test_ui_restricts_methods_and_does_not_allow_path_traversal(self):
        for path, method, expected in [('/api/status', 'POST', 405), ('/../config/shared.json', 'GET', 404)]:
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(urllib.request.Request(self.base + path, method=method))
            self.assertEqual(error.exception.code, expected)
            error.exception.close()
        with urllib.request.urlopen(self.base + '/') as response:
            self.assertIn("frame-ancestors 'none'", response.headers['Content-Security-Policy'])
            self.assertIn(b'AI Runtime', response.read())
        with urllib.request.urlopen(self.base + '/') as response:
            page = response.read().decode()
        self.assertIn('id="configurations"', page)
        self.assertIn('id="config-list"', page)
        self.assertNotIn('innerHTML', urllib.request.urlopen(self.base + '/app.js').read().decode())

    def test_status_passes_the_configuration_registry_through_unchanged(self):
        self.state['status']['configurations'] = {'active': 'qwen27b-q8-pair', 'selectable':
            [{'profile': 'qwen27b-q8-pair'}, {'profile': 'qwen27b-q6k-pair'}], 'always_included': [{'profile': 'everyday'}]}
        with urllib.request.urlopen(self.base + '/api/status') as response:
            self.assertEqual(json.load(response)['configurations'], self.state['status']['configurations'])

    def test_the_configuration_list_says_which_of_two_configurations_of_one_backend_it_is(self):
        # `flash-next-strata-solo` and `flash-next-strata-solo-thinking` carry the same display name
        # because it is the same model, so the variant label the registry publishes is the only thing
        # an operator can select knowingly by. There is no JS harness here, so this reads the page's
        # own row builder: the cheapest thing that fails if the label is dropped from it.
        source = urllib.request.urlopen(self.base + '/app.js').read().decode()
        row = source.split('function rowOf(config) {')[1].split('\n}\n')[0]
        self.assertIn('config.variant_label', row,
            'the configuration list names a configuration by its model alone, and two configurations '
            'of one model share that name: the selection would be a coin toss')
        # The same argument applies to a name an operator wrote. A name belongs to a model, and a row
        # offers one editor per model it loads, so the row builder reads the names and the aliases off
        # the models and never off the configuration.
        editor = source.split('function editor(config, model) {')[1].split('\n}\n')[0]
        self.assertIn('nameOf(model)', row)
        self.assertIn('model.alias', row)
        # The editor is where a name is written down, so it is where the name being covered over has
        # to be visible: an operator who cannot see what clearing the field brings back cannot decide.
        self.assertIn('defaultNameOf(model)', editor)
        self.assertIn("saveAlias(model.model", editor)
        # And the list of models comes from the registry, not from a count the page assumes: a machine
        # that loads two models shows two lines and a machine that loads one shows one.
        self.assertIn('modelsOf(config)', row)
        self.assertNotIn('everyday', row)

    def test_an_alias_note_reaches_the_page_without_the_path_that_explained_it(self):
        # The note is a fact an operator needs — the names on this page are not the names they saved —
        # and the reason is a filesystem detail about this machine, which the page cannot act on.
        self.state['status']['aliases'] = {'assignments': {},
            'note': '/home/deploy/.state/aliases.json could not be read: Expecting , delimiter'}
        note = json.load(urllib.request.urlopen(self.base + '/api/status'))['aliases']['note']
        self.assertEqual(note, 'Saved aliases could not be read and are not in use.')
        self.assertNotIn('.state', note)

    def test_compatibility_commands_preserve_profile_selection_and_retire_restore(self):
        # The legacy home commands for retired configurations name the current ones instead of guessing.
        for retired in ('daytime', 'daytime-27b'):
            with self.assertRaisesRegex(RuntimeError, 'retired.*flash-next-strata-solo'): dispatch([retired])
        # A retired id refuses and says what to select; a renamed id is applied under the id this
        # revision has, so an old habit reaches the backend that exists rather than a description of
        # one that does not. flash-next-strata-160k was the second kind while the configuration it was
        # renamed into was selectable; on 2026-10-18 that configuration was retired with the sampling
        # lever it existed to offer, and a name whose backends are described nowhere is a retirement
        # again whatever it used to be. The qwen pair below is still a rename and still resolves.
        for name in ('flash-next-solo-160k', 'flash-next-strata-160k'):
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError,
                    'retired.*select flash-next-strata-solo-thinking-recovery'):
                dispatch([name])
        self.assertEqual(dispatch(['flash-next-strata-solo-thinking-recovery']), ['apply', '--profile', 'flash-next-strata-solo-thinking-recovery'])
        self.assertEqual(dispatch(['qwen27b-q6k-pair']), ['apply', '--profile', 'qwen27b-q6k-pair'])
        self.assertEqual(dispatch(['local-ai-config.sh', 'apply', 'qwen27b-q6k-pair']),
            ['apply', '--profile', 'qwen27b-q6k-pair'])
        self.assertEqual(dispatch(['qwen27b-q6k-pair', 'status']), ['status', '--profile', 'qwen27b-q6k-pair'])
        with self.assertRaisesRegex(RuntimeError, 'retired'): dispatch(['daytime-swift'])
        self.assertEqual(dispatch(['nighttime']), ['apply'])
        self.assertEqual(dispatch(['local-ai-config.sh', 'show', 'qwen27b-q8-pair']), ['render', '--profile', 'qwen27b-q8-pair'])
        with self.assertRaisesRegex(RuntimeError, 'retired'): dispatch(['primary', 'rollback'])


try:
    from test_operations import OperationFixture
except ModuleNotFoundError:
    # Run as `python3 -m unittest tests.test_web` from the repository root, the fixtures are reached
    # through the package instead of the tests directory, which is not on the import path.
    from tests.test_operations import OperationFixture
from runtime.operations import public_operation
from runtime.web import refresh


class SwitchingHttpTests(OperationFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler(self.c, self.web_state, self.ops))
        thread = threading.Thread(target=self.server.serve_forever, daemon=True); thread.start()
        self.addCleanup(self.server.server_close); self.addCleanup(self.server.shutdown)
        self.base = 'http://127.0.0.1:' + str(self.server.server_address[1])
        with urllib.request.urlopen(self.base + '/api/status') as response:
            self.token = json.load(response)['csrf_token']

    def http(self, path, body=None, headers=None, method=None):
        data = json.dumps(body).encode() if body is not None else None
        actual = {'Content-Type': 'application/json', 'Origin': self.base, 'X-Runtime-CSRF': self.token}
        if headers: actual.update(headers)
        request = urllib.request.Request(self.base + path, data=data, headers=actual, method=method)
        try:
            with urllib.request.urlopen(request) as response: return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            with error: return error.code, json.load(error)

    def test_post_returns_operation_and_polling_survives_reload(self):
        entered = self.block_worker()
        code, reply = self.http('/api/profile-switch', self.request())
        self.assertEqual(code, 202); self.assertTrue(entered.wait(1))
        code, receipt = self.http('/api/operations/' + reply['operation_id'])
        self.assertEqual(code, 200); self.assertEqual(receipt['status'], 'running')
        self.assertEqual(self.http('/healthz')[0], 503)
        status = self.http('/api/status')[1]
        self.assertEqual(status['operation']['id'], receipt['id'])
        self.assertFalse(status['switching']['available'])
        self.release.set(); self.ops.worker.join(3)
        receipt = self.http('/api/operations/' + reply['operation_id'])[1]
        self.assertEqual(receipt['status'], 'succeeded')
        self.assertEqual(self.http('/healthz')[0], 200)

    def test_csrf_origin_and_json_boundaries_prevent_mutation(self):
        for headers, expected in [({'Origin': 'http://evil.example'},403),
                ({'X-Runtime-CSRF': ''},403), ({'Origin': 'null'},403),
                ({'Sec-Fetch-Site': 'cross-site'},403), ({'Content-Type':'text/plain'},415),
                ({'Host':'evil.example'},403)]:
            with self.subTest(headers=headers):
                self.assertEqual(self.http('/api/profile-switch', self.request(), headers)[0], expected)
        self.assertIsNone(self.ops.latest())
        self.assertEqual(self.http('/api/status', headers={'Host':'evil.example'})[0],421)

    def test_methods_paths_and_oversized_bodies_remain_restricted(self):
        self.assertEqual(self.http('/api/profile-switch', self.request(), method='PUT')[0],405)
        self.assertEqual(self.http('/api/status', self.request())[0],405)
        self.assertEqual(self.http('/api/operations/../../host.json')[0],404)
        self.assertEqual(self.http('/api/profile-switch', {'padding':'x'*4097})[0],400)
        self.assertIsNone(self.ops.latest())

    def test_private_exceptions_are_redacted_from_all_web_summaries(self):
        secret = 'synthetic-token /private/model/path'
        self.web_state['status'].update(error=secret, source_deployment={'error':secret}, last_deployment={'error':secret})
        self.web_state['startup_error'] = secret
        serialized = json.dumps(self.http('/api/status')[1])
        self.assertNotIn('synthetic-token', serialized)
        self.assertNotIn('/private/model/path', serialized)

    def test_startup_conflict_keeps_diagnostics_and_recovers_after_lock_release(self):
        from runtime.system import lock
        self.web_state['started'] = False
        with lock(self.state / 'runtime.lock'):
            refresh(self.c, self.web_state, self.ops)
            status = self.http('/api/status')[1]
            self.assertFalse(status['ready'])
            self.assertEqual(len(status['services']), 2)
            self.assertIn('startup_error', status)
        refresh(self.c, self.web_state, self.ops)
        self.assertTrue(self.web_state['started'])
        self.assertNotIn('startup_error', self.web_state)
        self.assertEqual(self.http('/healthz')[0], 200)

class AliasHttpTests(OperationFixture, unittest.TestCase):
    """Naming a configuration over the wire: the endpoint the page's editor posts to."""

    def setUp(self):
        super().setUp()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler(self.c, self.web_state, self.ops))
        thread = threading.Thread(target=self.server.serve_forever, daemon=True); thread.start()
        self.addCleanup(self.server.server_close); self.addCleanup(self.server.shutdown)
        self.base = 'http://127.0.0.1:' + str(self.server.server_address[1])
        with urllib.request.urlopen(self.base + '/api/status') as response:
            self.token = json.load(response)['csrf_token']

    def http(self, path, body=None, headers=None):
        actual = {'Content-Type': 'application/json', 'Origin': self.base, 'X-Runtime-CSRF': self.token}
        if headers: actual.update(headers)
        request = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers=actual,
            method='POST')
        try:
            with urllib.request.urlopen(request) as response: return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            with error: return error.code, json.load(error)

    def status(self):
        self.web_state['status'] = self.c.status()
        return self.web_state['status']

    def test_a_name_saved_here_is_the_name_the_page_shows_and_the_router_is_told(self):
        code, reply = self.http('/api/alias', {'model': 'qwen3.8-27b-q8_0-tensor-next', 'alias': 'Workhorse'})
        self.assertEqual((code, reply), (200, {'saved': True, 'published': True}))
        status = self.status()
        self.assertEqual(status['aliases'], {'assignments': {'qwen3.8-27b-q8_0-tensor-next': 'Workhorse'}})
        self.assertEqual([x['name'] for x in status['services']][0], 'Workhorse')
        entry = next(x for x in status['configurations']['selectable'] if x['profile'] == 'qwen27b-q8-pair')
        self.assertEqual((entry['display_name'], entry['alias'], entry['default_name']),
            ('Workhorse', 'Workhorse', 'Qwen3.8 27B Q8 (160K)'))
        self.assertEqual(json.loads((self.state / 'router' / 'active-model.json').read_text())
            ['models'][0]['display_name'], 'Workhorse')
        # Clearing it is the same one call, and it says so rather than refusing an empty string.
        self.assertEqual(self.http('/api/alias', {'model': 'qwen3.8-27b-q8_0-tensor-next', 'alias': ''})[0], 200)
        self.assertEqual(self.status()['aliases'], {'assignments': {}})

    def test_the_name_is_refused_for_the_rule_it_broke_and_says_which(self):
        for body, expected in [({'model': 'qwen3.8-27b-q8_0-tensor-next', 'alias': 'x' * 49}, '48'),
                ({'model': 'qwen3.8-27b-q8_0-tensor-next', 'alias': 'two\nlines'}, 'one printable line'),
                ({'model': 'no-such-model', 'alias': 'Ghost'}, 'Unknown model'),
                ({'model': 'qwen3.8-27b-q8_0-tensor-next', 'alias': 7}, 'text'),
                ({'alias': 'nameless'}, 'model and an alias')]:
            with self.subTest(body=body):
                code, reply = self.http('/api/alias', body)
                self.assertEqual(code, 400)
                self.assertIn(expected, reply['error'])
        self.assertFalse((self.state / 'aliases.json').exists())

    def test_the_guard_that_protects_a_switch_protects_a_name(self):
        # A name is a smaller change than a switch and the same kind of one: it changes what the rest
        # of the household is told about this machine, by a browser, without a shell.
        for headers, expected in [({'Origin': 'http://evil.example'}, 403), ({'X-Runtime-CSRF': ''}, 403),
                ({'Content-Type': 'text/plain'}, 415)]:
            with self.subTest(headers=headers):
                self.assertEqual(self.http('/api/alias', {'model': 'qwen3.8-27b-q8_0-tensor-next', 'alias': 'X'},
                    headers)[0], expected)
        self.assertFalse((self.state / 'aliases.json').exists())

    def test_inspection_mode_refuses_a_name_as_a_conflict_not_a_mistake(self):
        # The browser did nothing wrong and cannot fix it by re-sending, so it is not a 400: the
        # editor keeps the draft open and says the machine is not taking changes.
        self.system.readonly = True
        code, reply = self.http('/api/alias', {'model': 'qwen3.8-27b-q8_0-tensor-next', 'alias': 'Workhorse'})
        self.assertEqual(code, 409)
        self.assertIn('inspection mode', reply['error'])
        self.assertFalse((self.state / 'aliases.json').exists())

    def test_a_name_never_queues_behind_a_switch_and_a_switch_never_waits_for_one(self):
        # A rename during a drain is still just a save: it cannot be the second hand on the catalog,
        # so it returns without an operation, and the switch publishes it when it publishes.
        entered = self.block_worker()
        self.http('/api/profile-switch', self.request())     # switching to qwen27b-q6k-pair
        self.assertTrue(entered.wait(1))
        code, reply = self.http('/api/alias', {'model': 'qwen3.8-27b-ud-q6_k_xl-tensor-next', 'alias': 'Spare'})
        self.assertEqual(code, 200)
        self.assertEqual({'saved': reply['saved'], 'published': reply['published']},
            {'saved': True, 'published': False})
        self.assertIn('published with', reply['note'])
        # The promise the note makes: the run that is holding the lock says the name itself, because
        # it reads the names when it publishes rather than when it started.
        self.release.set(); self.ops.worker.join(5)
        self.assertEqual(self.ops.read(self.ops.latest()['id'])['status'], 'succeeded')
        published = json.loads((self.state / 'router' / 'active-model.json').read_text())
        self.assertEqual(published['configuration']['id'], 'qwen27b-q6k-pair')
        self.assertEqual(published['models'][0]['display_name'], 'Spare')

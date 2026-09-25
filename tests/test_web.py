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
        self.state = {'status': {'ready': False, 'profile': 'daytime'}}
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
        self.state['status']['configurations'] = {'active': 'daytime', 'selectable':
            [{'profile': 'daytime'}, {'profile': 'daytime-27b'}], 'always_included': [{'profile': 'nighttime'}]}
        with urllib.request.urlopen(self.base + '/api/status') as response:
            self.assertEqual(json.load(response)['configurations'], self.state['status']['configurations'])

    def test_compatibility_commands_preserve_profile_selection_and_retire_restore(self):
        self.assertEqual(dispatch(['daytime']), ['apply', '--profile', 'daytime'])
        self.assertEqual(dispatch(['daytime-27b']), ['apply', '--profile', 'daytime-27b'])
        self.assertEqual(dispatch(['local-ai-config.sh', 'apply', 'daytime-flash-f16']),
            ['apply', '--profile', 'daytime-flash-f16'])
        self.assertEqual(dispatch(['daytime-27b', 'status']), ['status', '--profile', 'daytime-27b'])
        with self.assertRaisesRegex(RuntimeError, 'retired'): dispatch(['daytime-swift'])
        self.assertEqual(dispatch(['nighttime']), ['apply'])
        self.assertEqual(dispatch(['local-ai-config.sh', 'show', 'daytime']), ['render', '--profile', 'daytime'])
        with self.assertRaisesRegex(RuntimeError, 'retired'): dispatch(['primary', 'rollback'])


from test_operations import OperationFixture
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

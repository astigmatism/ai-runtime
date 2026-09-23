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

    def test_ui_is_readonly_and_does_not_allow_path_traversal(self):
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
        self.assertIn('id="configurations"', page)  # The available-configurations listing is served, not injected.
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
        self.assertEqual(dispatch(['daytime-27b', 'status']), ['status', '--profile', 'daytime-27b'])
        with self.assertRaisesRegex(RuntimeError, 'retired'): dispatch(['daytime-swift'])
        self.assertEqual(dispatch(['nighttime']), ['apply'])
        self.assertEqual(dispatch(['local-ai-config.sh', 'show', 'daytime']), ['render', '--profile', 'daytime'])
        with self.assertRaisesRegex(RuntimeError, 'retired'): dispatch(['primary', 'rollback'])

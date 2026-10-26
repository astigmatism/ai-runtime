import json
from http.server import ThreadingHTTPServer
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

import test_power
from models import CODING, CODING_Q6K, EVERYDAY
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
        # The page draws the models out of each configuration's own list, so a registry that arrived
        # with its `models` arrays folded away would render a machine that loads nothing.
        self.state['status']['configurations'] = {'active': 'qwen27b-q8-pair', 'selectable':
            [{'profile': 'qwen27b-q8-pair', 'models': [{'model': CODING}, {'model': EVERYDAY}]},
             {'profile': 'qwen27b-q6k-pair', 'models': [{'model': CODING_Q6K}, {'model': EVERYDAY}]}]}
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


class PowerHttpTests(OperationFixture, unittest.TestCase):
    """A card's power state over the wire: the endpoint the panel posts to, and what it may say.

    The route is the smaller sibling of the alias route — it writes hardware and drains nothing — so
    these tests are about the three things only the wire can show: that a browser cannot use it
    without the same-origin guard, that a refusal arrives with the status the page branches on, and
    that a driver's own words stay off the page.
    """
    # The controller fixture's host, with cards that carry the UUIDs a power request has to name.
    HOST = test_power.power_host()

    def setUp(self):
        super().setUp()
        self.agent = test_power.CardAgent()
        for uuid in (test_power.MEASURED, test_power.OTHER, test_power.UNMEASURED, test_power.TI):
            self.agent.read(uuid)
        docker = self.system.docker
        def fake(*args, **kwargs):
            if args[0] == 'run' and 'runtime.gpucard' in args:
                return json.dumps(self.agent.answer_for(list(args)))
            return docker(*args, **kwargs)
        self.system.docker = fake
        self.c.power.merge({uuid: {'read_at': '2026-10-25T00:00:00+00:00',
            'read': test_power.stock_read(uuid, test_power.CardAgent.names[uuid])}
            for uuid in (test_power.MEASURED, test_power.OTHER, test_power.UNMEASURED, test_power.TI)})
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

    def test_a_rung_applied_from_the_page_comes_back_with_what_the_driver_said(self):
        code, reply = self.http('/api/gpu-power', {'gpu': test_power.MEASURED, 'action': 'apply',
            'rung': 'sensible'})
        self.assertEqual(code, 200)
        self.assertTrue(reply['ok'])
        self.assertEqual((reply['to'], reply['rung'], reply['state']), ('sensible', 'sensible', 'rung'))
        self.assertEqual(reply['readback'], {'power': 'ok', 'offset': 'ok', 'clock_max': 'unverifiable'})
        card = self.status()['power']['cards'][test_power.MEASURED]
        self.assertEqual((card['state'], card['rung']), ('rung', 'sensible'))
        self.assertEqual(card['read']['power_limit'], 330)

    def test_the_guard_that_protects_a_switch_protects_a_card(self):
        for headers, expected in [({'Origin': 'http://evil.example'}, 403), ({'X-Runtime-CSRF': ''}, 403),
                ({'Sec-Fetch-Site': 'cross-site'}, 403), ({'Content-Type': 'text/plain'}, 415)]:
            with self.subTest(headers=headers):
                self.assertEqual(self.http('/api/gpu-power', {'gpu': test_power.MEASURED,
                    'action': 'apply', 'rung': 'safe'}, headers)[0], expected)
        self.assertEqual(self.agent.writes, [], 'a refused request must not reach the driver')

    def test_a_refusal_arrives_with_the_status_the_page_branches_on(self):
        # A rung this card has never been given: the browser asked well, and the answer names the
        # rungs that exist. That is a 400, and the page shows it beside the choice.
        code, reply = self.http('/api/gpu-power', {'gpu': test_power.MEASURED, 'action': 'apply',
            'rung': 'extreme'})
        self.assertEqual(code, 400)
        self.assertIn('safe, sensible, assertive', reply['error'],
            'a refusal that lists the rungs lists them in the order the ladder runs')
        # An unhealthy model on the card is not the browser's mistake: it is a 409 the panel answers
        # with the tick that means "apply anyway".
        self.system.containers['qwen38-daytime']['State']['Running'] = False
        code, reply = self.http('/api/gpu-power', {'gpu': test_power.MEASURED, 'action': 'apply',
            'rung': 'safe'})
        self.assertEqual((code, reply['code']), (409, 'engine'))
        self.assertIn('not healthy', reply['error'])
        self.assertEqual(self.http('/api/gpu-power', {'gpu': test_power.MEASURED, 'action': 'apply',
            'rung': 'safe', 'override': True})[0], 200)
        self.system.containers['qwen38-daytime']['State']['Running'] = True
        # A machine in the middle of changing its own shape says so, and the page keeps the draft. The
        # phases are the runtime's own: a transaction that finished stays on disk saying `succeeded`,
        # and a page that read that as busy would never let anybody write a card again.
        self.c.save('transaction.json', {'phase': 'draining'})
        code, reply = self.http('/api/gpu-power', {'gpu': test_power.MEASURED, 'action': 'revert'})
        self.assertEqual((code, reply['code']), (409, 'busy'))
        self.c.save('transaction.json', {'phase': 'succeeded'})
        self.assertEqual(self.http('/api/gpu-power', {'gpu': test_power.MEASURED,
            'action': 'revert'})[0], 200)
        # And a request that is not about a card this machine reports is a 400, not a guess.
        code, reply = self.http('/api/gpu-power', {'gpu': 'GPU-00000000-0000-0000-0000-000000000000',
            'action': 'revert'})
        self.assertEqual(code, 400)
        self.assertIn('no card with that UUID', reply['error'])

    def test_a_drivers_words_stay_off_the_page(self):
        """The card agent's failures name a device, a query field or a socket. The page is a browser
        two networks away, and it gets one sentence and a pointer to diagnostics."""
        self.c.power.merge({test_power.MEASURED: {'read_at': '2026-10-25T00:00:00+00:00',
            'error': 'nvidia-smi query failed for /dev/nvidia0: not supported'}})
        self.c.save('power/latest.json', dict(self.c.power.latest(),
            error='The rung table at /home/astigmatism/deployments/ai-runtime/config could not be read'))
        # The page's own path, because the sanitising belongs to the wire and not to the machine:
        # `controller.status()` keeps the full words, and a terminal that asks directly may read them.
        self.assertNotIn('could not be read', json.dumps(self.status()['power']
            ['cards'][test_power.MEASURED]['error']))
        with urllib.request.urlopen(self.base + '/api/status') as response:
            power = json.load(response)['power']
        self.assertNotIn('nvidia0', json.dumps(power))
        self.assertNotIn('deployments', json.dumps(power))
        self.assertIn('could not be read', power['cards'][test_power.MEASURED]['error'])
        self.assertNotIn('_why', json.dumps(power), 'the rung table reasons itself out in source, '
            'once; the page is not where that paragraph belongs four times over')
        # What is *not* private stays: the evidence string is the reason a rung is recommended.
        self.assertIn('energy per token', power['cards'][test_power.MEASURED]['measured'])

    def test_a_power_read_needs_no_controller_lock(self):
        code, reply = self.http('/api/gpu-power', {'gpu': test_power.TI, 'action': 'read'})
        self.assertEqual(code, 200)
        self.assertTrue(reply['ok'])
        self.assertEqual(self.agent.reads[-1], [test_power.TI])

    def test_a_stray_brace_does_not_eat_the_rule_after_it(self):
        """An unbalanced `}` in a stylesheet is not an error the page shows. A browser skips the one
        rule that follows it and carries on, which is how a GPU tile drew its power line at the
        default 16px inside a 141px box for as long as it took somebody to look. The cheapest thing
        that notices is arithmetic on the file the page actually gets.
        """
        css = urllib.request.urlopen(self.base + '/style.css').read().decode()
        depth = 0
        for line, text in enumerate(css.splitlines(), 1):
            for character in text:
                if character == '{':
                    depth += 1
                elif character == '}':
                    depth -= 1
                    self.assertGreaterEqual(depth, 0,
                        'an unmatched } on line ' + str(line) + ' costs the rule after it, silently')
        self.assertEqual(depth, 0, 'the stylesheet ends ' + str(depth) + ' rule(s) deep')

    def test_the_panel_is_cleared_before_anything_is_drawn_in_it(self):
        """The power panel is rebuilt while it is open, and for a while it appended instead of
        replacing: the document grew by a panel's height every poll, 2,194px to 57,553px in ten
        minutes, and it stayed perfectly readable the whole time, which is exactly why nobody noticed.
        The clearing has to sit directly above the drawing, and there has to be one on the path that
        closes the panel too — clearing it only when a card *is* open leaves the last panel of a closed
        card sitting under the map forever.
        """
        source = urllib.request.urlopen(self.base + '/app.js').read().decode()
        body = source.split('function renderPower() {')[1].split('\nfunction ')[0]
        self.assertIn('panel.replaceChildren(); panel.hidden = true;', body,
            'closing a card has to empty the panel, not leave it under the map')
        self.assertLess(body.index('powerSignature = signature;'), body.index('panel.hidden = false;'),
            'the panel has to be cleared immediately before it is drawn into')
        self.assertIn('panel.replaceChildren();\n  panel.hidden = false;', body,
            'the clear and the draw have to be the next two lines, or a future branch can slip between')

    def test_a_poll_that_changed_nothing_leaves_the_open_panel_alone(self):
        """The panel draws the driver's readings, which arrive on their own 30-second cadence, and it
        is drawn from a signature like the map and the list. Rebuilt on every five-second poll instead,
        it closed the drawer the operator had just opened and took the caret out of a watts field
        mid-number — both of which the operator feels as the page working against them.
        """
        source = urllib.request.urlopen(self.base + '/app.js').read().decode()
        body = source.split('function renderPower() {')[1].split('\nfunction ')[0]
        self.assertIn('if (signature === powerSignature) { powerAge(); return; }', body,
            'a poll that changed nothing this panel draws must not touch its DOM')
        self.assertIn('powerGate[powerOpen], powerCustom]);', body,
            'the signature has to cover the operator\'s own state, or a click that should redraw '
            'the panel is ignored')

    def test_an_open_drawer_is_the_operators_to_close(self):
        """"Values by hand" is open or closed because somebody wanted it that way. Keeping that on the
        `<details>` element alone means the next redraw answers with a shrug and a closed drawer, so it
        lives in module state and goes back on every draw."""
        source = urllib.request.urlopen(self.base + '/app.js').read().decode()
        self.assertIn('custom.open = powerCustom;', source)
        self.assertIn('powerCustom = custom.open', source)
        self.assertIn('let powerCustom = false', source)

    def test_typing_in_the_hand_fields_redraws_the_write_button(self):
        """Every control on this panel ends by drawing itself, because the button that does the writing
        is drawn from what is chosen, and picking something is what changes it. The hand-written fields
        were the one exception: the keystroke updated the draft that a *click* honours and stopped
        there, so the button kept the `disabled` it was given when the panel was drawn. On a card with
        measured rungs it stayed invisible — a rung radio clicked first does redraw, and by then the
        action is already enabled — but on an unmeasured card the drawer is the only way in, and the
        operator's click landed on a dead button until the next poll rescued it.
        """
        source = urllib.request.urlopen(self.base + '/app.js').read().decode()
        body = source.split('function renderPower() {')[1].split('\nfunction ')[0]
        handler = body.split("input.addEventListener('input', () => {")[1].split('});')[0]
        self.assertIn('powerDraft = {...powerDraft, [key]: input.value};', handler)
        self.assertTrue(handler.rstrip().endswith('render();'),
            'the hand fields have to redraw like the radio above them does; the button\'s label and '
            'its `disabled` are both worked out from the choice this keystroke just made')
        # And the button asks one function why it cannot be pressed, rather than working it out again
        # in a place the handler above cannot reach.
        self.assertIn('apply.disabled = !!powerBlocked(entry, target);', body)
        self.assertIn("if (apply.disabled) apply.title = powerBlocked(entry, target);", body,
            'a button that cannot be pressed should say why to whoever tries')

    def test_a_hand_written_box_shows_the_number_it_will_send(self):
        """The three boxes were prefilled for display from the last reading while the object actually
        submitted stayed empty until every one of them was edited, so an operator who changed the two
        numbers they cared about was refused for a third they could plainly see filled in — "Fill in
        the maximum clock in MHz", said about a box showing a number. Worse, that box was prefilled
        with `clock_sm`, the clock the card is *observed* at: a vision card idling at its floor offered
        to lock itself at 210 MHz. One resolver now answers both questions, and it has no honest
        default for a clock cap on a driver that reports none, so that box starts empty."""
        source = urllib.request.urlopen(self.base + '/app.js').read().decode()
        body = source.split('function renderPower() {')[1].split('\nfunction ')[0]
        self.assertIn("input.type = 'text'; input.value = powerHand(key, read) ?? '';", body,
            'what a box shows has to be what it sends')
        self.assertIn('for (const key of handKeys) body[key] = handValue(key, read);', source,
            'and what it sends has to be what it shows')
        resolver = source.split('function powerHand(key, read) {')[1].split('\n}\n')[0]
        self.assertNotIn('clock_sm', resolver,
            'the observed SM clock is what a card is doing, not what it is capped at: a clock lock '
            'cannot be verified on GeForce, which the panel says out loud two paragraphs above')
        self.assertIn('function powerSummary(read) {', source)
        self.assertIn('return `Values by hand — ${said.join(\' · \')}${gone}`;', source,
            'the drawer\'s own line names the numbers that go to the driver, and what is missing')

    def test_the_drawer_uses_the_bounds_of_the_card_it_is_drawn_under(self):
        """`status.power` carries no `envelope`: there is one per card, because two models of card do
        not share a clock floor. The drawer read it off the block four times and got `undefined` every
        time, which drew the clock hint as "locked from — up", left the offset limit to the fallback
        that happens to match this driver, and silently switched off the warning for a max clock at or
        below the floor — the one hand-written number that makes a card crash while nothing is asking
        it for anything, which is the failure the runtime's own refusal names.
        """
        source = urllib.request.urlopen(self.base + '/app.js').read().decode()
        body = source.split('function renderPower() {')[1].split('\nfunction ')[0]
        self.assertIn('const envelope = entry.envelope || {};', body)
        self.assertNotIn('powerBlock().envelope', body,
            'the bounds live on the card; the power block has never carried them')
        self.assertIn('powerBad(key, read, envelope)', body,
            'and the warning has to be computed from them, not from an undefined stand-in')

    def test_typing_in_the_rename_box_survives_a_poll(self):
        """The runtime gets asked about its own status every five seconds, and the box's draft text
        was in the list's signature. One keystroke changed the signature, so the next poll rebuilt the
        list: the text came back, and the focus did not. The draft is drawn once, into the box that is
        already holding it."""
        source = urllib.request.urlopen(self.base + '/app.js').read().decode()
        body = source.split('function renderRegistry() {')[1].split('\nfunction ')[0]
        signature = [line for line in body.split('if (signature')[0].splitlines() if 'const signature' in line]
        self.assertEqual(len(signature), 1, 'the list is drawn from one signature')
        self.assertNotIn('aliasDraft', signature[0],
            'a keystroke is not the runtime telling the page anything new')
        self.assertIn("field.dataset.focus = 'alias:' + model.model;", source,
            'the box has to name itself so a redraw that does happen can hand the caret back')

    def test_a_redraw_hands_the_keyboard_back(self):
        """Even a redraw that has every right to happen is a thing happening to a control the operator
        may be using. The page remembers which named control had the caret and where the caret was,
        and puts both back after drawing."""
        source = urllib.request.urlopen(self.base + '/app.js').read().decode()
        body = source.split('function render() {')[1].split('\n}\n')[0]
        self.assertIn('const focus = keepFocus();', body)
        self.assertLess(body.index('const focus = keepFocus();'), body.index('renderPower();'),
            'focus has to be remembered before anything is drawn')
        self.assertIn('restoreFocus(focus);', body)
        self.assertGreater(body.index('restoreFocus(focus);'), body.index('renderDetails();'),
            'and handed back after everything is drawn')
        self.assertIn("document.querySelector(`[data-focus=\"${at.name}\"]`)", source)
        # Every control an operator can be mid-keystroke on names itself, or a redraw that does have
        # every right to happen is allowed to drop them on the floor.
        for name in ("'alias:' + model.model", "'power:' + key", "'config:' + config.profile",
                     "'rung:' + name", "'gate:' + entry.uuid"):
            self.assertIn(name, source, 'a control without a name cannot be found again after a redraw')


# --------------------------------------------------------------------------------------------- the
# page's own arithmetic, run rather than read. The suite has no JS harness — there is no node in the
# runtime image — so the page is normally tested by reading its source, which is good for "the panel
# is cleared before it is drawn" and weak for "the button this operator is staring at can be pressed".
# The power drawer is a small function of three things: the card's state, what the driver last said,
# and what the operator has typed. Those four cards below are the shapes a real status carried on the
# day this broke, and node is on every machine that develops the page. Where it is not, this skips and
# the source assertions above still stand.

PAGE_LOGIC = ['const watts =', 'const mhz =', 'const handKeys =', 'const handName =', 'const handShort =',
              'function powerHand(', 'function handValue(', 'const handMissing =',
              'function powerHandRefusal(', 'const handSpelled =', 'function powerSummary(',
              'const powerTarget =', 'function powerBlocked(', 'function powerBody(',
              'function powerLabel(', 'function powerBad(']
PAGE_STATE = """
// The page's module state and the two facts it reads from `status`, which is all the functions above
// need. Everything else they are handed.
const CARD = 'GPU-b0c40094-1111-2222-3333-444444444444';
let powerDraft = {}, powerChoice = {}, powerBusy = false, powerGate = {};
let WRITES = {available: true}, UNWELL = [];
const powerWrites = () => WRITES;
const powerUnwell = () => UNWELL;
"""


def shipped(source, head):
    """The text of one definition from the page, from its header to its closing brace."""
    if head not in source:
        raise AssertionError('the page no longer defines `' + head + '`. This harness runs the page\'s '
            'own arithmetic, so it is tied to those names on purpose: move the decision and this is '
            'the test that says so, rather than one that quietly passes on a stub.')
    if head.startswith('function '):
        start, i = source.index(head), source.index('{', source.index(head))
        depth = 0
        while True:
            depth += source[i] == '{'
            depth -= source[i] == '}'
            i += 1
            if depth == 0:
                return source[start:i]
    start, end = source.index(head), source.index('\n', source.index(head))
    while not source[start:end].rstrip().endswith(';'):
        end = source.index('\n', end + 1)
    return source[start:end]


def run_page(source, script):
    """Run a script against the page's own definitions and hand back what it printed."""
    program = '\n'.join([PAGE_STATE] + [shipped(source, head) for head in PAGE_LOGIC] + [script])
    handle = tempfile.NamedTemporaryFile('w', suffix='.js', delete=False)
    handle.write(program)
    handle.close()
    try:
        running = subprocess.run([shutil.which('node'), handle.name], capture_output=True,
                                 text=True, timeout=120)
    finally:
        os.unlink(handle.name)
    assert running.returncode == 0, 'the page\'s own code would not run:\n' + running.stderr
    return json.loads(running.stdout)


@unittest.skipUnless(shutil.which('node'), 'no node on this machine: the page is judged by its source')
class HandWrittenRungTests(unittest.TestCase):
    """What the drawer decides, on the four cards the page was showing when it could not be used.
    The whole page is run once, in one node process, and the five tests below read that result."""

    # The 4080 of the report: never measured, so no plan and no rungs, and read back as `custom`
    # because a hand-set value matches no rung. The 210 MHz observed clock is a vision-card floor.
    SCRIPT = r"""
const card = over => Object.assign({uuid: CARD, name: 'NVIDIA GeForce RTX 4080', state: 'custom',
  rung: null, rungs: [], plan: null, read: {power_limit: 255, power_default: 320, power_min: 100,
  power_max: 340, offset: 105, clock_sm: 210, clock_max_sm: 2700}}, over);
const out = {};
let entry;

// Opened, and nothing chosen: a card at `custom` has no rung to pick.
powerDraft = {}; powerChoice = {}; entry = card({});
out.nothing_chosen_label = powerLabel(entry, powerTarget(entry));
out.nothing_chosen_blocked = powerBlocked(entry, powerTarget(entry));

// The drawer opened on that card — the operator said `by hand`, and that is the only way in.
powerChoice = {mode: 'custom'};
out.opened_blocked = powerBlocked(entry, powerTarget(entry));
out.opened_label = powerLabel(entry, powerTarget(entry));
out.untouched_refusal = powerHandRefusal(entry.read);
out.untouched_summary = powerSummary(entry.read);
out.observed_clock_offered = handValue('clock_max', entry.read);

// The numbers the operator came for: 255 W, 2640 MHz, +105. One box typed, two left as shown.
powerDraft = {clock_max: '2640'};
out.filled_refusal = powerHandRefusal(entry.read);
out.filled_body = powerBody(entry, powerTarget(entry), entry.read);
out.filled_summary = powerSummary(entry.read);

// A box cleared to type into is not a zero.
powerDraft = {power: ''};
out.cleared_refusal = powerHandRefusal(card({}).read);
out.cleared_shown = powerHand('power', card({}).read);

// A card that has never been read has no numbers to offer at all.
powerDraft = {}; powerChoice = {};
out.unread_refusal = powerHandRefusal({});
out.unread_summary = powerSummary({});

// The three measured cards, unchanged: where they are is what the button offers.
powerChoice = {}; entry = card({name: 'NVIDIA GeForce RTX 3090', state: 'rung', rung: 'sensible',
  rungs: [{name: 'safe'}, {name: 'sensible'}, {name: 'assertive'}]});
out.rung_body = powerBody(entry, powerTarget(entry), entry.read);
out.rung_label = powerLabel(entry, powerTarget(entry));
powerChoice = {}; entry = card({name: 'NVIDIA GeForce RTX 3080 Ti', state: 'rung', rung: 'assertive',
  rungs: [{name: 'assertive'}]});
out.ti_body = powerBody(entry, powerTarget(entry), entry.read);
powerChoice = {}; entry = card({name: 'NVIDIA GeForce RTX 3080', state: 'stock'});
out.stock_body = powerBody(entry, powerTarget(entry), entry.read);
out.stock_label = powerLabel(entry, powerTarget(entry));

// Why a button cannot be pressed, which is the other half of "the panel will not write this".
powerChoice = {}; entry = card({state: 'stock'});
UNWELL = [{name: 'qwen3-vl-8b'}];
out.unwell_blocked = powerBlocked(entry, powerTarget(entry));
powerGate[CARD] = true;
out.gated_blocked = powerBlocked(entry, powerTarget(entry));
powerGate = {}; WRITES = {available: false, reason: 'A source deployment has the machine.'};
out.refused_blocked = powerBlocked(entry, powerTarget(entry));
WRITES = {available: true}; powerBusy = true;
out.busy_blocked = powerBlocked(entry, powerTarget(entry));
powerBusy = false; UNWELL = [];

// A hint, not a gate: the driver refuses what it refuses.
powerDraft = {power: '1000'}; powerChoice = {mode: 'custom'}; entry = card({});
out.over_ceiling = powerBad('power', entry.read, {clock_min: 210, offset_max: 150});
out.odd_step = powerBad('offset', Object.assign({}, entry.read, {offset: 100}),
  {clock_min: 210, offset_max: 150});
// The floor one: a max clock at or below it is the setting that crashes a card at rest, and the page
// has the sentence for it — which only fires if it is handed a card's envelope at all.
powerDraft = {clock_max: '210'};
out.at_the_floor = powerBad('clock_max', entry.read, {clock_min: 210, offset_max: 150});
console.log(JSON.stringify(out));
"""

    def setUp(self):
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler(None, {'status': {}}))
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        base = 'http://127.0.0.1:' + str(self.server.server_address[1])
        with urllib.request.urlopen(base + '/app.js') as response:
            self.page = run_page(response.read().decode(), self.SCRIPT)

    def test_an_unmeasured_card_can_be_moved_by_hand(self):
        """`Power` on the 4080 → open the drawer → type 2640 → press the button. That is the whole
        request the owner came to make, and the page drew the button `disabled` when the panel opened,
        changed the choice under it as they typed, and never recomputed it: the click landed on a
        control marked disabled, and only the next poll (which draws the panel from a signature that
        covers the draft) quietly enabled it.
        """
        self.assertEqual(self.page['nothing_chosen_blocked'], 'nothing is chosen yet')
        self.assertEqual(self.page['opened_blocked'], '',
            'opening the drawer on a card with no rungs is the operator asking for a hand-written '
            'rung; the button cannot stay disabled, because there is nothing else on that panel to click')
        self.assertEqual(self.page['opened_label'], 'Write these values to NVIDIA GeForce RTX 4080')
        self.assertEqual(self.page['filled_body'], {'gpu': 'GPU-b0c40094-1111-2222-3333-444444444444',
            'action': 'custom', 'power': 255, 'clock_max': 2640, 'offset': 105},
            'the card\'s own power limit and offset, plus the clock that was typed')
        self.assertEqual(self.page['filled_refusal'], '')
        self.assertEqual(self.page['filled_summary'], 'Values by hand — 255 W · 2640 MHz · +105 MHz',
            'the line says what goes to the driver, because that is what goes to the driver')

    def test_a_drawer_nobody_edited_names_the_box_that_is_empty(self):
        """Pressing the button with a box still empty has to name that box. It used to refuse over a
        field the operator could see filled in — and, worse, an untouched drawer could read as a
        revert, which is a different action on a serving machine.
        """
        self.assertEqual(self.page['untouched_refusal'], 'Fill in the maximum clock in MHz.')
        self.assertEqual(self.page['untouched_summary'],
            'Values by hand — 255 W · +105 MHz · no maximum clock yet',
            'the summary names what is missing instead of showing a dash and implying it knows')
        self.assertIsNone(self.page['observed_clock_offered'],
            'a card seen at 210 MHz is a card idling at its floor, not a card capped at 210 MHz')
        self.assertEqual(self.page['cleared_refusal'], 'Fill in the power limit in watts.',
            'an emptied box is a missing number, not zero watts')
        self.assertEqual(self.page['cleared_shown'], '')
        self.assertEqual(self.page['unread_refusal'], 'Fill in the power limit in watts.')
        self.assertEqual(self.page['unread_summary'],
            'Values by hand — a trial rung this machine has not measured')

    def test_a_card_that_needs_no_decision_still_offers_the_one_it_needs(self):
        """Nothing here was meant to move the three measured cards: a card at a rung is re-applied, a
        card at stock is reverted, and neither is asked for numbers."""
        self.assertEqual(self.page['rung_body'],
            {'gpu': 'GPU-b0c40094-1111-2222-3333-444444444444', 'action': 'apply', 'rung': 'sensible'})
        self.assertEqual(self.page['rung_label'], 'Re-apply sensible to NVIDIA GeForce RTX 3090')
        self.assertEqual(self.page['ti_body']['rung'], 'assertive')
        self.assertEqual(self.page['stock_body'],
            {'gpu': 'GPU-b0c40094-1111-2222-3333-444444444444', 'action': 'revert'})
        self.assertEqual(self.page['stock_label'], 'Revert NVIDIA GeForce RTX 3080 to stock')
        # And a card that has nothing chosen asks for the choice in words rather than applying `null`.
        self.assertEqual(self.page['nothing_chosen_label'], 'Choose a rung, or write values by hand')

    def test_the_button_says_why_it_cannot_be_pressed(self):
        """Disabled is the page's least informative answer, and it is the one it used to give for four
        different reasons. Each of them now has a sentence, including the one the owner met.
        """
        self.assertIn('qwen3-vl-8b is not healthy', self.page['unwell_blocked'])
        self.assertEqual(self.page['gated_blocked'], '', '"apply anyway" is what unticks the gate')
        self.assertEqual(self.page['refused_blocked'], 'A source deployment has the machine.')
        self.assertEqual(self.page['busy_blocked'], 'a write is already running')

    def test_a_hint_stays_a_hint(self):
        """The boxes warn before the write and the runtime refuses after it. A prefilled number that
        happens to be one the driver dislikes is worth a sentence in the box.
        """
        self.assertEqual(self.page['over_ceiling'], 'this card\'s ceiling is 340 W')
        self.assertIn('15 MHz steps', self.page['odd_step'])
        self.assertIn('no P-state to idle in', self.page['at_the_floor'])

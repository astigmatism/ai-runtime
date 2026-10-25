"""The card power state: what the driver says, what a rung writes, and what refuses in between.

These tests hold the three promises the feature is judged on. A card is only ever reported at a rung
the *driver* agreed to, so every fake below answers like a driver and not like a wish. A write is
small and volatile, so nothing here asserts a boot path or a re-apply. And a refusal is a sentence an
operator can act on, which is what most of these assert.

The fake Docker answers `docker run … -m runtime.gpucard …` the way the real card agent does — one
JSON object — and mutates its idea of each card the way a driver would: the power limit and the offset
move, the clock lock does not appear anywhere, and a read-back can be made to disagree.
"""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import test_controller as base
from runtime.config import read_power_registry
from runtime.controller import Controller
from runtime.power import CADENCE_SECONDS, PowerError, classify, public_power
from runtime.system import atomic_json

ROOT = Path(__file__).resolve().parents[1]
# Full UUIDs, because a power request insists on one: it is naming a physical card, and the whole
# point of the form is that it cannot name two of them.
MEASURED = 'GPU-358353dd-5933-1dc6-ac0d-e1189b063e4c'      # RTX 3090, default rung `sensible`
OTHER = 'GPU-91239bb4-0f5b-b148-6d03-c098a5806e13'         # RTX 4080 SUPER, default rung `safe`
UNMEASURED = 'GPU-77777777-7777-7777-7777-777777777777'    # RTX 4080: nobody has run the ladder
TI = 'GPU-e4252fcb-6c59-6892-c5bb-bf8f75fbe523'            # RTX 3080 Ti, default rung `assertive`
STOCK = {'power_default': 350, 'power_min': 100, 'power_max': 350}


def power_host():
    """The controller fixture's host, with cards that have real UUIDs and one card left unmeasured.

    A power request names a card the way the driver does, so the fixture's short `GPU-day-1` ids will
    not do; and a machine where *every* card is measured would never exercise the state Daisy is in
    today, where two of them have no plan and must not be offered one.
    """
    host = copy.deepcopy(base.HOST)
    host['gpu_ids'] = {'daytime': [MEASURED, OTHER], 'nighttime': [UNMEASURED, TI]}
    host['gpu_names'] = {'daytime': ['RTX 3090', 'RTX 4080 SUPER'],
        'nighttime': ['RTX 4080', 'RTX 3080 Ti']}
    # Which card runs which measured plan is the operational half of the feature, and this is the file
    # that holds it: `UNMEASURED` is left out, which is the state of Daisy's cards today.
    host['power_plans'] = {MEASURED: 'rtx3090', OTHER: 'rtx4080-super', TI: 'rtx3080-ti'}
    return host


def stock_read(uuid, name, **changes):
    """What a card in its factory state reports, in the shape the card agent reports it."""
    card = {'uuid': uuid, 'name': name, 'driver': '560.35.03', 'pci': '0000:01:00.0',
        'power_limit': STOCK['power_default'], **STOCK, 'draw': 62.4, 'draw_average': 58.0,
        'clock_sm': 210, 'clock_max_sm': 2100 if '3090' in name else 2955,
        'clock_mem': 9750, 'pstate': 'P8', 'temperature': 31, 'fan': 0, 'utilization': 0,
        'memory_used': 0, 'memory_utilization': 0, 'power_capped_hours': 0.0,
        'offset': 0, 'offset_error': None}
    card.update(changes)
    return card


class CardAgent:
    """A driver that answers: reads report, writes move the two readable knobs and not the locked one."""

    def __init__(self):
        self.cards = {}
        self.reads, self.writes = [], []
        self.readback = None       # 'fail' makes a write land and then disagree
        self.error = None          # a driver that says no instead of answering
        self.answer = None         # or something that is not a card reading at all

    def read(self, uuid, **changes):
        self.cards[uuid] = dict(stock_read(uuid, self.names[uuid]), **changes)

    names = {'GPU-358353dd-5933-1dc6-ac0d-e1189b063e4c': 'NVIDIA GeForce RTX 3090',
        'GPU-91239bb4-0f5b-b148-6d03-c098a5806e13': 'NVIDIA GeForce RTX 4080 SUPER',
        'GPU-77777777-7777-7777-7777-777777777777': 'NVIDIA GeForce RTX 4080',
        'GPU-e4252fcb-6c59-6892-c5bb-bf8f75fbe523': 'NVIDIA GeForce RTX 3080 Ti'}

    def answer_for(self, argv):
        verb = argv[argv.index('runtime.gpucard') + 1]
        uuids = [argv[i + 1] for i, arg in enumerate(argv) if arg == '--uuid']
        if self.answer is not None:
            return self.answer
        if self.error:
            return {'ok': False, 'verb': verb, 'error': self.error}
        if verb == 'read':
            self.reads.append(uuids)
            return {'ok': True, 'verb': 'read', 'cards': {u: self.cards[u] for u in uuids}}
        self.writes.append(argv)
        uuid = uuids[0]
        asked = {'power': int(argv[argv.index('--power') + 1]),
            'offset': int(argv[argv.index('--offset') + 1]),
            'clock_max': None if '--clock-max' not in argv else int(argv[argv.index('--clock-max') + 1]),
            # A revert asks for no floor: `-rgc` drops the lock, and a runtime that invented one would
            # be claiming a state it never asked the driver for.
            'clock_min': int(argv[argv.index('--clock-min') + 1]) if '--clock-min' in argv else None}
        after = dict(self.cards[uuid], power_limit=asked['power'], offset=asked['offset'],
            clock_sm=asked['clock_max'] or self.cards[uuid]['clock_max_sm'])
        if self.readback == 'fail':
            # The write was accepted and did not land: the card is now between two states, which is
            # exactly the reading a page has to show rather than smooth over.
            after['offset'] = asked['offset'] - 15
        self.cards[uuid] = after
        verified = self.readback != 'fail'
        return {'ok': verified, 'verb': 'apply', 'apply': {
            'before': self.cards[uuid], 'after': after, 'attempts': 1, 'asked': asked,
            'commands': ['nvidia-smi -pl %d' % asked['power']],
            'readback': {'power': 'ok', 'offset': 'ok' if verified else 'fail',
                'clock_max': 'unverifiable'},
            'errors': [], 'ok': verified}}


class PowerSystem(base.FakeSystem):
    """The controller's Docker fake, plus the one subcommand the card agent needs."""

    def __init__(self, bundle):
        super().__init__(bundle)
        self.agent = CardAgent()

    def docker(self, *args, **kwargs):
        if args[0] != 'run' or 'runtime.gpucard' not in args:
            return super().docker(*args, **kwargs)
        self.events.append(('power', args))
        return json.dumps(self.agent.answer_for(list(args)))


class PowerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name)
        host = power_host()
        host['router_state_dir'] = str(self.state / 'router')
        atomic_json(self.state / 'host.json', host)
        (self.state / 'router-token').write_text('synthetic-token')
        self.bundle = base.render(ROOT / 'config', host, 'qwen27b-q8-pair')
        self.system = PowerSystem(self.bundle)
        self.c = Controller(ROOT / 'config', self.state, 'a' * 40, self.system)
        atomic_json(self.state / 'active.json',
            {'revision': 'a' * 40, 'image': 'old-image', 'bundle': self.bundle})
        self.sleep = patch('runtime.controller.time.sleep'); self.sleep.start()
        self.addCleanup(self.sleep.stop)
        self.power = self.c.power
        self.agent = self.system.agent
        for uuid in (MEASURED, OTHER, UNMEASURED, TI):
            self.agent.read(uuid)
        # gpu_inventory reports the host's own short ids, and UNMEASURED is the plain RTX 4080.
        self.host_uuids = [uuid for uuid, _ in self.power.cards()]

    def cached(self, uuid, read):
        """Put one reading in the cache, the way a background pass would have left it."""
        latest = self.power.latest()
        latest[uuid] = {'read_at': '2026-10-25T00:00:00+00:00', 'read': read}
        latest['swept_at'] = '2026-10-25T00:00:00+00:00'
        self.c.save('power/latest.json', latest)

    def block(self):
        return self.power.status()['cards']

    # ------------------------------------------------------------------------------------ what it says
    def test_the_page_is_shown_the_cache_and_never_a_driver_read_of_its_own(self):
        """A status poll costs a file read, not a container start. This is the whole reason the block
        carries an age: the alternative is a page that gets slower the more cards a machine has."""
        before = len(self.agent.reads)
        self.assertEqual(self.block()[MEASURED]['state'], 'unread')
        self.assertEqual(len(self.agent.reads), before, 'a status poll read a card from the driver')
        self.power.pass_once()
        block = self.block()
        self.assertEqual(block[MEASURED]['state'], 'stock')
        self.assertIsNotNone(block[MEASURED]['read_at'])
        self.assertEqual(len(self.agent.reads), before + 1, 'the cache was not used for the second poll')

    def test_a_card_sits_at_a_rung_only_when_the_driver_agrees_with_both_readable_knobs(self):
        rungs = read_power_registry(ROOT / 'config')['plans']['rtx3090']['rungs']
        plan = {'name': 'RTX 3090 (GA102)', 'rungs': rungs}
        at = dict(stock_read(MEASURED, 'x'), power_limit=330, offset=75)
        self.assertEqual(classify(at, plan), ('rung', 'sensible'))
        # One knob of the two: not the rung, and not stock either.
        self.assertEqual(classify(dict(at, power_limit=350), plan), ('custom', None))
        self.assertEqual(classify(dict(at, offset=90), plan), ('custom', None))
        self.assertEqual(classify(stock_read(MEASURED, 'x'), plan), ('stock', None))
        # A card whose offset could not be read has not been shown to be at zero.
        self.assertEqual(classify(dict(at, offset=None), plan), ('unknown', None))
        self.assertEqual(classify(None, plan), ('unknown', None))

    def test_the_word_on_the_tile_names_the_state_and_the_panel_carries_the_numbers(self):
        self.cached(MEASURED, dict(stock_read(MEASURED, CardAgent.names[MEASURED]),
            power_limit=330, offset=75))
        card = self.block()[MEASURED]
        self.assertEqual((card['state'], card['rung'], card['plan']), ('rung', 'sensible', 'rtx3090'))
        self.assertEqual(card['default'], 'sensible')
        self.assertIn('-12.5% energy per token', card['measured'])
        self.assertEqual([rung['name'] for rung in card['rungs']],
            ['safe', 'sensible', 'assertive'])
        self.assertEqual([rung['power'] for rung in card['rungs']], [350, 330, 315])
        self.assertEqual([rung['default'] for rung in card['rungs']], [False, True, False])
        self.assertEqual(card['read']['power_limit'], 330)
        # A clock lock has no query field, so the block reports the ceiling and the observation and
        # never claims a locked maximum it cannot see.
        self.assertNotIn('clock_lock', json.dumps(card))

    def test_a_card_nobody_measured_offers_no_rung_and_says_why(self):
        """`330 W / 1905 MHz / +75` is a GA102 result. Handing it to an unmeasured card because the
        page needs *something* in the radio list is how a card gets a rung nothing measured."""
        # `Controller.host` is the document this process read at startup, and the power plan a card
        # runs is read from it; a test changes the same document the runtime would restart to pick up.
        host = self.c.host
        plans = dict(host['power_plans']); plans.pop(TI)
        host['power_plans'] = plans
        self.c.save('host.json', host)
        card = self.block()[TI]
        self.assertEqual(card['rungs'], [])
        self.assertIsNone(card['plan'])
        self.assertEqual(card['state'], 'unread')
        self.cached(TI, dict(stock_read(TI, CardAgent.names[TI])))
        self.assertEqual(self.block()[TI]['state'], 'stock')
        with self.assertRaises(RuntimeError) as refused:
            self.power.write({'gpu': TI, 'action': 'apply', 'rung': 'sensible'})
        self.assertIn('has no undervolt plan', str(refused.exception))

    def test_a_plan_that_is_not_in_source_is_reported_instead_of_ignored(self):
        host = self.c.host
        plans = dict(host['power_plans'], **{UNMEASURED: 'rtx9090'})
        host['power_plans'] = plans
        self.c.save('host.json', host)
        self.cached(UNMEASURED, dict(stock_read(UNMEASURED, CardAgent.names[UNMEASURED])))
        card = self.block()[UNMEASURED]
        self.assertIn('rtx9090', card['plan_error'])
        self.assertIn('rtx3090', card['plan_error'], 'the refusal names the plans that do exist')
        self.assertEqual(card['rungs'], [])
        # The rest of the card still works: a reading is a reading, and Revert needs no plan.
        self.cached(UNMEASURED, dict(stock_read(UNMEASURED, CardAgent.names[UNMEASURED])))
        self.assertEqual(self.block()[UNMEASURED]['state'], 'stock')
        self.assertIsNone(self.block()[MEASURED]['plan_error'])

    # ----------------------------------------------------------------------------------- what it writes
    def test_a_rung_is_three_driver_calls_through_one_privileged_container(self):
        """The controller has the socket and no GPUs, so the write leaves the process. What the flags
        say is the contract with the host: privileged, root, no network, the project's own image."""
        self.cached(MEASURED, stock_read(MEASURED, CardAgent.names[MEASURED]))
        reply = self.power.write({'gpu': MEASURED, 'action': 'apply', 'rung': 'sensible'})
        self.assertTrue(reply['ok'])
        self.assertEqual(reply['to'], 'sensible')
        argv = self.agent.writes[-1]
        self.assertEqual(argv[:3], ['run', '-i', '--rm'])
        for flag in ('--privileged', '--user', '0:0', '--gpus', 'all', '--network', 'none'):
            self.assertIn(flag, argv)
        self.assertIn('runtime.gpucard', argv)
        self.assertEqual(argv[argv.index('--power') + 1], '330')
        self.assertEqual(argv[argv.index('--clock-max') + 1], '1905')
        self.assertEqual(argv[argv.index('--offset') + 1], '75')
        # The name the card reported on the last read travels with the write: a card swapped for
        # another model and re-enumerated into this slot must not be written blind.
        self.assertEqual(argv[argv.index('--expect-name') + 1], CardAgent.names[MEASURED])
        self.assertEqual(reply['asked'], {'power': 330, 'clock_max': 1905, 'offset': 75, 'clock_min': 210})

    def test_the_write_is_recorded_with_what_the_driver_said_afterwards(self):
        self.cached(MEASURED, stock_read(MEASURED, CardAgent.names[MEASURED]))
        self.power.write({'gpu': MEASURED, 'action': 'apply', 'rung': 'assertive'})
        entry = self.power.history()[0]
        self.assertEqual((entry['gpu'], entry['from'], entry['to'], entry['ok']),
            (MEASURED, 'stock', 'assertive', True))
        self.assertEqual(entry['readback']['clock_max'], 'unverifiable')
        self.assertEqual(entry['actor'], 'page')

    def test_a_write_that_did_not_read_back_is_an_answer_and_not_a_crash(self):
        """Half-landed is the state a page has to show: the card is between two rungs, the operator
        needs to know before the next one, and the way back is one click away."""
        self.cached(MEASURED, stock_read(MEASURED, CardAgent.names[MEASURED]))
        self.agent.readback = 'fail'
        reply = self.power.write({'gpu': MEASURED, 'action': 'apply', 'rung': 'safe'})
        self.assertFalse(reply['ok'])
        self.assertEqual(reply['readback']['offset'], 'fail')
        self.assertFalse(self.power.history()[0]['ok'])
        # And the cache now carries the reading the driver actually gave, not the rung that was asked.
        self.assertEqual(self.block()[MEASURED]['state'], 'custom')

    def test_revert_uses_this_cards_own_default_and_no_rung_table_at_all(self):
        """Reverting needs no measurement, which is why it is the one write an unmeasured card gets."""
        self.cached(UNMEASURED, stock_read(UNMEASURED, CardAgent.names[UNMEASURED], power_limit=255,
            power_default=320, offset=-30))
        reply = self.power.write({'gpu': UNMEASURED, 'action': 'revert'})
        argv = self.agent.writes[-1]
        self.assertEqual(reply['to'], 'stock')
        self.assertEqual(argv[argv.index('--power') + 1], '320')
        self.assertEqual(argv[argv.index('--offset') + 1], '0')
        self.assertNotIn('--clock-max', argv, 'stock means -rgc: the clock lock is dropped, not re-asked')

    def test_revert_without_a_reading_refuses_and_says_what_to_do_first(self):
        with self.assertRaises(RuntimeError) as refused:
            self.power.write({'gpu': MEASURED, 'action': 'revert'})
        self.assertIn('never been read', str(refused.exception))

    def test_values_by_hand_are_validated_against_the_card_and_the_envelope(self):
        self.cached(MEASURED, stock_read(MEASURED, CardAgent.names[MEASURED]))
        self.power.write({'gpu': MEASURED, 'action': 'custom', 'power': 300, 'clock_max': 1800,
            'offset': 60})
        argv = self.agent.writes[-1]
        self.assertEqual(argv[argv.index('--power') + 1], '300')
        self.assertEqual(argv[argv.index('--clock-max') + 1], '1800')
        with self.assertRaises(RuntimeError) as refused:
            self.power.write({'gpu': MEASURED, 'action': 'custom', 'power': 400, 'clock_max': 1800,
                'offset': 60})
        self.assertIn('maximum of 350 W', str(refused.exception))
        with self.assertRaises(RuntimeError) as refused:
            self.power.write({'gpu': MEASURED, 'action': 'custom', 'power': 300, 'clock_max': 1800,
                'offset': 400})
        self.assertIn('150', str(refused.exception))
        # Off the driver's step is rounded, and says so rather than silently moving the card.
        reply = self.power.write({'gpu': MEASURED, 'action': 'custom', 'power': 300, 'clock_max': 1800,
            'offset': 80})
        self.assertTrue(reply['rounded_offset'])
        self.assertEqual(reply['asked']['offset'], 75)

    def test_a_read_action_asks_the_driver_now_and_leaves_the_cache_fresher(self):
        reply = self.power.write({'gpu': MEASURED, 'action': 'read'})
        self.assertTrue(reply['ok'])
        self.assertEqual(self.agent.reads[-1], [MEASURED])
        self.assertEqual(self.block()[MEASURED]['state'], 'stock')
        with self.assertRaises(RuntimeError):
            self.power.write({'gpu': MEASURED, 'action': 'read', 'rung': 'safe'})

    def test_one_card_is_asked_for_and_the_whole_machine_is_read_at_once(self):
        self.power.pass_once()
        self.assertEqual(len(self.agent.reads[-1]), len(self.host_uuids),
            'a background pass is one container start for every card')
        self.power.write({'gpu': MEASURED, 'action': 'read'})
        self.assertEqual(self.agent.reads[-1], [MEASURED])

    # ------------------------------------------------------------------------------------ what refuses
    def test_an_unhealthy_engine_on_the_card_refuses_the_write(self):
        """An undervolt two notches too deep does not fail: it shows up later as strange turns or an
        Xid. So the write waits for someone to say they know better, and the words say that."""
        self.cached(MEASURED, stock_read(MEASURED, CardAgent.names[MEASURED]))
        self.system.containers[base.DAY]['State']['Running'] = False
        with self.assertRaises(PowerError) as refused:
            self.power.write({'gpu': MEASURED, 'action': 'apply', 'rung': 'sensible'})
        self.assertEqual((refused.exception.status, refused.exception.code), (409, 'engine'))
        self.assertIn('not healthy', str(refused.exception))
        self.assertIn('apply anyway', str(refused.exception))
        self.assertEqual(self.agent.writes, [], 'a refusal must not reach the driver')

    def test_the_override_is_the_explicit_way_past_the_gate(self):
        self.cached(MEASURED, stock_read(MEASURED, CardAgent.names[MEASURED]))
        self.system.containers[base.DAY]['State']['Running'] = False
        reply = self.power.write({'gpu': MEASURED, 'action': 'apply', 'rung': 'sensible',
            'override': True})
        self.assertTrue(reply['ok'])
        self.assertTrue(self.power.history()[0]['override'])

    def test_a_card_with_nothing_on_it_writes_without_being_asked_about_health(self):
        self.cached(UNMEASURED, stock_read(UNMEASURED, CardAgent.names[UNMEASURED]))
        self.assertTrue(self.power.write({'gpu': UNMEASURED, 'action': 'revert'})['ok'])

    def test_the_machine_being_busy_refuses_a_write_the_way_a_switch_does(self):
        self.cached(MEASURED, stock_read(MEASURED, CardAgent.names[MEASURED]))
        self.c.save('transaction.json', {'phase': 'drain'})
        with self.assertRaises(PowerError) as refused:
            self.power.write({'gpu': MEASURED, 'action': 'apply', 'rung': 'safe'})
        self.assertEqual((refused.exception.status, refused.exception.code), (409, 'busy'))
        self.assertIn('maintenance', str(refused.exception).lower())
        self.c.save('transaction.json', {'phase': 'complete'})
        self.c.save('update-job.json', {'phase': 'prepared', 'started_at': '2026-10-25T00:00:00+00:00'})
        with self.assertRaises(PowerError) as refused:
            self.power.write({'gpu': MEASURED, 'action': 'apply', 'rung': 'safe'})
        self.assertIn('deployment', str(refused.exception))

    def test_inspection_mode_neither_reads_nor_writes(self):
        from runtime.system import System
        state = Path(self.tmp.name) / 'inspection'
        state.mkdir()
        atomic_json(state / 'host.json', self.c.host)
        controller = Controller(ROOT / 'config', state, 'a' * 40, System(readonly=True))
        self.assertFalse(controller.power.availability()['available'])
        self.assertIn('inspecting', controller.power.availability()['reason'])
        with self.assertRaises(RuntimeError):
            controller.power.write({'gpu': MEASURED, 'action': 'read'})

    def test_a_driver_that_cannot_be_reached_is_a_reading_and_not_a_traceback(self):
        self.agent.error = 'nvidia-smi query failed: Statement could not be executed'
        self.power.pass_once()
        block = public_power(self.power.status())
        self.assertIn('could not be read', block['cards'][MEASURED]['error'])
        self.assertEqual(self.block()[MEASURED]['state'], 'error')
        self.assertIn('Statement could not be executed',
            (self.state / 'power' / 'last-error.txt').read_text())

    def test_something_that_is_not_a_card_reading_is_not_shown_to_the_browser(self):
        self.agent.answer = 'Error: No such image: local/ai-runtime:git-deadbeef'
        self.power.pass_once()
        self.assertIn('could not be read', public_power(self.power.status())['cards'][MEASURED]['error'])
        self.assertEqual(self.block()[MEASURED]['state'], 'error')
        self.assertIn('No such image', (self.state / 'power' / 'last-error.txt').read_text())

    # ------------------------------------------------------------------------------- what it promises
    def test_the_block_reports_whether_a_write_can_start_and_why_not(self):
        block = self.power.status()
        self.assertTrue(block['available'])
        self.assertTrue(block['writes']['available'])
        self.assertIsNone(block['writes']['reason'])
        self.assertEqual(block['cadence_seconds'], CADENCE_SECONDS)

    def test_no_private_word_of_a_driver_complaint_reaches_the_page(self):
        block = public_power({'error': 'nope /home/astigmatism/deployments/ai-runtime/.state',
            'cards': {MEASURED: {'error': 'docker: /run/docker.sock permission denied',
                'measured': 'kept: -12.5% energy per token'}}})
        self.assertNotIn('deployments', json.dumps(block))
        self.assertNotIn('docker.sock', json.dumps(block['cards'][MEASURED]['error']))
        self.assertIn('-12.5% energy per token', block['cards'][MEASURED]['measured'])

    def test_writes_are_kept_and_the_table_is_trimmed(self):
        self.cached(MEASURED, stock_read(MEASURED, CardAgent.names[MEASURED]))
        for _ in range(25):
            self.power.write({'gpu': MEASURED, 'action': 'revert'})
        self.assertEqual(len(self.power.history()), 20)

    def test_the_whole_status_carries_the_power_block_for_the_page(self):
        self.assertIn('power', self.c.status())


class CadenceTests(unittest.TestCase):
    """The background pass is a due-time and a thread, and the page must feel neither.

    A poll that started a container would make the page slower the more cards a machine has, and a
    poll that waited for one would make a hung daemon hold the whole status loop. So the decision is
    tested here without a thread running: what the loop is told to do, and how often.
    """

    def setUp(self):
        # The cadence wants a real power object with a real state directory, so this borrows the
        # fixture above by name rather than inventing a third shape of fake host.
        self.test = PowerTests('test_the_page_is_shown_the_cache_and_never_a_driver_read_of_its_own')
        self.test.setUp()
        self.addCleanup(self.test.tmp.cleanup)
        self.power = self.test.power

    def started(self, **kwargs):
        """Ask for a pass with the thread held back, and say what the loop was told to start."""
        started = []

        class Held:
            def __init__(self, target=None, **options):
                self.target = target

            def start(self):
                started.append(self.target)

        with patch('runtime.power.threading.Thread', Held):
            answer = self.power.refresh_if_due(**kwargs)
        return answer, started

    def test_a_pass_is_asked_for_when_the_last_one_ages_out(self):
        self.assertTrue(self.power.stale({}), 'never swept is due')
        self.assertTrue(self.power.stale({'swept_at': 'not a moment'}))
        self.assertTrue(self.power.stale({'swept_at': '2020-01-01T00:00:00+00:00'}))
        self.assertFalse(self.power.stale({'swept_at': '2099-01-01T00:00:00+00:00'}))
        answer, started = self.started()
        self.assertTrue(answer)
        self.assertEqual(started, [self.power.pass_once])

    def test_two_polls_start_one_pass_and_a_fresh_reading_starts_none(self):
        answer, started = self.started()
        self.assertTrue(answer)
        # The second poll of the same five seconds finds the first still running and says nothing.
        self.assertEqual(self.started(), (False, []), 'two containers for one card set')
        self.power.running = False  # the pass that poll started has since finished
        self.test.c.save('power/latest.json', {'swept_at': '2099-01-01T00:00:00+00:00'})
        self.assertEqual(self.started(), (False, []), 'a reading that has not aged out is not re-asked')
        self.assertTrue(self.started(force=True)[0],
            'a write asks the driver again right away, rather than leaving the page to show its own write')

    def test_a_failed_pass_leaves_the_door_open_for_the_next_one(self):
        """`swept_at` is when the machine last *heard from* the cards. A pass that could not run is
        not that, and stamping it anyway would leave the page confidently stale for a day."""
        self.test.agent.error = 'daemon not answering'
        self.power.pass_once()
        self.assertNotIn('swept_at', self.power.latest())
        answer, started = self.started()
        self.assertTrue(answer, 'a failed pass silence the machine')


if __name__ == '__main__':
    unittest.main()

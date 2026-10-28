"""Power sets: saving what the whole machine is sitting at, and putting it back a year later.

A single card's write is judged in `test_power.py`. What is judged here is the machine-wide act, and
specifically the three things that only appear once a machine has been running for months:

* a card that arrived after the save must get **nothing** from a set that does not name it;
* a card that left must be **skipped**, with the rest of the set still written;
* a slot that now holds a different model of card must be **refused**, because `sensible` is a number
  per model of board and a set that guessed would put 220 W on a board that has never seen it.

The fake driver is the one from `test_power.py`, so a card here moves the way a driver moves: the power
limit and the offset change, the clock lock appears nowhere, and a write is only `applied` once the
read-back agrees. A save is asserted to write **no card at all**, because a photograph that develops
itself onto the negative is the failure this feature cannot have.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_controller as base
import test_power as cards
from runtime.config import read_power_sets
from runtime.controller import Controller
from runtime.power import SETS, SETS_LIMIT, PowerError
from runtime.system import atomic_json

ROOT = Path(__file__).resolve().parents[1]
MEASURED, OTHER, UNMEASURED, TI = cards.MEASURED, cards.OTHER, cards.UNMEASURED, cards.TI
# A card the fixture host has never heard of: the one that arrives with a new purchase order.
ARRIVED = 'GPU-00000000-0000-4000-8000-000000000001'
# The three numbers each measured card's rungs are, restated here so a test that expects `sensible`
# fails loudly when the table moves under it rather than quietly agreeing with itself.
RUNGS = {MEASURED: ('sensible', 330, 75), OTHER: ('safe', 270, 45), TI: ('assertive', 300, 105)}


class SetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name)
        host = cards.power_host()
        host['router_state_dir'] = str(self.state / 'router')
        self.host = host
        atomic_json(self.state / 'host.json', host)
        (self.state / 'router-token').write_text('synthetic-token')
        self.bundle = base.render(ROOT / 'config', host, 'qwen27b-q8-pair')
        self.system = cards.PowerSystem(self.bundle)
        self.c = Controller(ROOT / 'config', self.state, 'a' * 40, self.system)
        atomic_json(self.state / 'active.json',
            {'revision': 'a' * 40, 'image': 'old-image', 'bundle': self.bundle})
        sleep = patch('runtime.controller.time.sleep'); sleep.start(); self.addCleanup(sleep.stop)
        self.power, self.agent = self.c.power, self.system.agent
        self.at_stock()

    # --------------------------------------------------------------------------------------------- helpers
    def at_stock(self):
        """Every card at its factory numbers, which is the state a set is saved *out of*."""
        for uuid in (MEASURED, OTHER, UNMEASURED, TI):
            self.agent.read(uuid)

    def move(self, uuid, power, offset, **changes):
        """Move one card on the fake driver, the way a rung write would have."""
        self.agent.cards[uuid].update(power_limit=power, offset=offset, **changes)

    def at_rungs(self):
        """The machine of the request: three cards on rungs and the fourth at stock.

        Exactly Rosalina's Tuesday — a 3090 on `sensible`, a 4080 SUPER on `safe`, a 3080 Ti on
        `assertive` and the vision card untouched — because the feature exists for that state and not
        for a machine where every card is the same.
        """
        for uuid, (rung, power, offset) in RUNGS.items():
            self.move(uuid, power, offset)

    def save(self, name='quiet-night', **body):
        return self.power.write_sets({'action': 'save', 'name': name, **body})

    def apply(self, name='quiet-night', **body):
        return self.power.write_sets({'action': 'apply', 'name': name, **body})

    def stored(self):
        return json.loads((self.state / SETS).read_text())['sets']

    def rows(self):
        """The block the page is drawn from, keyed by set name."""
        block = self.power.status()['sets']
        return {row['name']: row for row in block['sets']}, block

    def row(self, name):
        return self.rows()[0][name]

    def write_sets_file(self, sets):
        atomic_json(self.state / SETS, {'version': 1, 'sets': sets})

    def asked(self, uuid):
        """What the driver was told about one card, per container run: {uuid: (power, offset, clock)}."""
        wanted = {}
        for argv in self.agent.writes:
            gpu = argv[argv.index('--uuid') + 1]
            wanted[gpu] = (int(argv[argv.index('--power') + 1]), int(argv[argv.index('--offset') + 1]),
                None if '--clock-max' not in argv else int(argv[argv.index('--clock-max') + 1]))
        return wanted

    def take_out(self, uuid):
        """Take a card out of the machine: the settings stop naming it and the driver stops answering."""
        host = copy.deepcopy(self.c.host)
        for group in host['gpu_ids'].values():
            if uuid in group:
                group.remove(uuid)
        self.c.host = host
        self.c.save('host.json', host)
        self.agent.cards.pop(uuid, None)

    def move_in(self, uuid, name):
        """Put a card into the machine: the settings name it and the driver answers for it."""
        host = copy.deepcopy(self.c.host)
        host['gpu_ids']['nighttime'].append(uuid)
        host['gpu_names']['nighttime'].append(name)
        self.c.host = host
        self.c.save('host.json', host)
        self.agent.cards[uuid] = dict(cards.stock_read(uuid, 'NVIDIA GeForce ' + name))
        return uuid

    def replace(self, uuid, name):
        """Swap the card in one slot for another model, keeping the UUID the settings name.

        `CardAgent.names` is the class's table of what a fresh machine reports, so this copies it rather
        than writing through it: a shared table would quietly rename that card on every driver in every
        test that runs after this one, and they would agree with a lie.
        """
        self.agent.names = dict(self.agent.names, **{uuid: 'NVIDIA GeForce ' + name})
        self.agent.cards[uuid]['name'] = 'NVIDIA GeForce ' + name

    # --------------------------------------------------------------------------------------------- saving
    def test_a_save_records_the_whole_machine_and_writes_none_of_it(self):
        """A save is a photograph: one read pass, one file, and no card touched.

        The read pass is not the cache. A set built from a reading up to `CADENCE_SECONDS` old would be
        filed as a recording of *now*, and the one promise it has to keep is that it is what the cards
        were doing when the operator asked.
        """
        self.at_rungs()
        before = len(self.agent.reads)
        reply = self.save()
        self.assertTrue(reply['ok'])
        self.assertEqual(self.agent.writes, [], 'a save wrote a card')
        self.assertEqual(len(self.agent.reads), before + 1, 'a save did not ask the driver now')
        sets = self.stored()['quiet-night']['cards']
        self.assertEqual({uuid: entry['kind'] for uuid, entry in sets.items()},
            {MEASURED: 'rung', OTHER: 'rung', TI: 'rung', UNMEASURED: 'stock'})
        self.assertEqual(sets[MEASURED]['rung'], 'sensible')
        self.assertEqual(sets[MEASURED]['plan'], 'rtx3090')
        # The witness is what the driver read, kept only so the page can say a rung has moved.
        self.assertEqual(sets[MEASURED]['witness'], {'power': 330, 'offset': 75})
        self.assertIsNotNone(self.stored()['quiet-night']['saved_at'])

    def test_a_card_at_hand_written_values_refuses_the_save_and_says_what_to_do(self):
        """The clock cap of a hand-written state cannot be read back on GeForce.

        So a set that recorded `270 W / +45` would be promising to bring the card back to a state it has
        no way of naming, and the honest answer is to refuse, name the card, and offer to leave it out.
        """
        self.at_rungs()
        self.move(UNMEASURED, 280, 60)  # between rungs, and not by this card's ladder
        with self.assertRaises(RuntimeError) as refused:
            self.save()
        said = str(refused.exception)
        self.assertIn('RTX 4080', said, 'a refusal names the card that cannot be recorded')
        self.assertIn('clock cap', said)
        self.assertIn('--omit', said, 'a refusal says what would have made it pass')
        self.assertFalse((self.state / SETS).exists(), 'a refused save wrote the file anyway')

    def test_the_card_that_cannot_be_recorded_can_be_left_out_on_purpose(self):
        self.at_rungs()
        self.move(UNMEASURED, 280, 60)
        reply = self.save(omit=[UNMEASURED])
        self.assertTrue(reply['ok'])
        self.assertEqual([card['gpu'] for card in reply['omitted']], [UNMEASURED])
        self.assertNotIn(UNMEASURED, self.stored()['quiet-night']['cards'])
        # And by the name a person would type, when the name is one card's name.
        reply = self.save(name='by-name', omit=[UNMEASURED, '3090'])
        self.assertNotIn(MEASURED, self.stored()['by-name']['cards'])

    def test_an_ambiguous_card_name_refuses_rather_than_guessing(self):
        """`RTX 4080` is a substring of `RTX 4080 SUPER`, and leaving one out is not leaving both out."""
        self.at_rungs()
        with self.assertRaises(RuntimeError) as refused:
            self.save(omit=['4080'])
        self.assertIn('matches', str(refused.exception))
        self.assertFalse((self.state / SETS).exists())

    def test_a_card_the_driver_will_not_answer_refuses_the_save(self):
        """`custom` is a card whose state cannot be named; an unread card is a state that is not known."""
        self.at_rungs()
        self.agent.error = 'nvmlDeviceGetGpcClkVfOffset: not supported'
        with self.assertRaises(RuntimeError) as refused:
            self.save()
        self.assertIn('could not be read from the driver', str(refused.exception))
        self.assertEqual(self.agent.writes, [])

    def test_a_name_is_an_id_so_two_sets_cannot_share_one_without_saying_so(self):
        self.at_rungs()
        self.save()
        with self.assertRaises(RuntimeError) as refused:
            self.save()
        self.assertIn('already a power set named', str(refused.exception))
        self.assertIn('overwrite', str(refused.exception))
        self.save(overwrite=True)
        self.assertEqual(list(self.stored()), ['quiet-night'], 'overwriting added a second set')

    def test_stock_is_the_name_every_machine_has_already(self):
        self.at_rungs()
        for name in ('stock', 'STOCK', ' stock '):
            with self.assertRaises(RuntimeError) as refused:
                self.save(name=name)
            self.assertIn('every machine', str(refused.exception))
        with self.assertRaises(RuntimeError) as refused:
            self.power.write_sets({'action': 'delete', 'name': 'stock'})
        self.assertIn('every machine', str(refused.exception))
        self.assertFalse((self.state / SETS).exists(), 'a refused delete wrote the file')

    def test_a_name_that_is_not_a_name_is_refused(self):
        self.at_rungs()
        for name in ('', '   ', 'a' * 41, 'two\nlines'):
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                self.save(name=name)

    def test_sets_are_few_because_a_way_back_has_to_be_findable(self):
        self.at_rungs()
        for number in range(SETS_LIMIT):
            self.save(name='set-%02d' % number)
        with self.assertRaises(RuntimeError) as refused:
            self.save(name='one-too-many')
        self.assertIn(str(SETS_LIMIT), str(refused.exception))
        self.assertIn('superseded', str(refused.exception))

    def test_everything_the_save_recorded_survives_the_file_it_is_written_to(self):
        """What the runtime wrote is read back through the same validator it writes with.

        A set that cannot be read back is worse than one that was never saved: the operator pressed the
        button, the page said saved, and the recall is where it surfaces.
        """
        self.at_rungs()
        self.save()
        stored, note = read_power_sets(self.state / SETS)
        self.assertIsNone(note)
        self.assertEqual(sorted(stored['quiet-night']['cards']), sorted([MEASURED, OTHER, TI, UNMEASURED]))

    def test_a_malformed_file_is_a_note_on_the_page_and_a_refusal_to_rewrite(self):
        """Tolerant to read, strict to write: the page keeps working, and a save does not lose a line.

        The file is an operator's preference and the machine is serving, so a comma cannot take the page
        down. But a save rewrites the file, and a save that quietly dropped the line it could not parse
        would be losing a set while reporting a success.
        """
        (self.state / 'power').mkdir(parents=True, exist_ok=True)
        (self.state / SETS).write_text('{"sets": {"broken": {')
        self.assertIn('could not be read', self.rows()[1]['error'],
            'the page did not notice the file was unreadable')
        with self.assertRaises(RuntimeError) as refused:
            self.save()
        self.assertIn('cannot be rewritten', str(refused.exception))
        # One bad line in an otherwise readable file drops that line and names it, and the set beside
        # it is still something an operator can apply.
        self.write_sets_file({})
        self.at_rungs()
        self.save(name='good')
        sets = self.stored()
        sets['good']['cards'][MEASURED] = {'kind': 'sideways'}
        sets['also-broken'] = {'cards': {}}
        self.write_sets_file(sets)
        self.assertIn('good', self.rows()[0], 'a readable set went missing beside a broken line')
        self.assertIn('also-broken', self.rows()[1]['error'])
        self.assertIn(MEASURED, self.rows()[1]['error'])

    # ------------------------------------------------------------------------------------------- applying
    def test_applying_a_set_writes_each_card_the_rung_that_card_measured(self):
        """Three containers, three different pairs of numbers, one act.

        The numbers come from the plan, not from the file: `sensible` is what the ladder measures today,
        and a set that carried its own watts would keep applying a superseded campaign forever.
        """
        self.at_rungs()
        self.save()
        self.at_stock()
        self.agent.writes.clear()
        reply = self.apply()
        self.assertTrue(reply['ok'])
        self.assertEqual(reply['applied'], 4)
        asked = self.asked(None)
        self.assertEqual(set(asked), {MEASURED, OTHER, TI, UNMEASURED})
        for uuid, (rung, power, offset) in RUNGS.items():
            self.assertEqual(asked[uuid][:2], (power, offset),
                '%s was not written the rung it names' % uuid)
        # The card at stock is a revert: the driver's own default, no offset, and no clock lock at all.
        self.assertEqual(asked[UNMEASURED], (self.agent.cards[UNMEASURED]['power_default'], 0, None))
        self.assertEqual(sorted(entry['set'] for entry in self.power.history()),
            ['quiet-night'] * 4, 'a write from a set does not say which set')

    def test_the_rung_written_is_the_ladder_as_it_is_measured_not_as_it_was_saved(self):
        """A set points at a rung. If the ladder moved, the card follows the ladder, and the page says so."""
        self.at_rungs()
        self.save()
        self.write_sets_file({'quiet-night': dict(self.stored()['quiet-night'], cards={
            MEASURED: {'kind': 'rung', 'rung': 'sensible', 'plan': 'rtx3090', 'name':
                'NVIDIA GeForce RTX 3090', 'witness': {'power': 300, 'offset': 90}},
        })})
        self.at_stock()
        reply = self.apply()
        self.assertEqual(self.asked(None)[MEASURED][:2], (330, 75),
            'the saved witness was written instead of the rung')
        line = self.row('quiet-night')['cards'][0]
        # The page composes the sentence, so what has to survive to it is the two facts it is built from.
        self.assertTrue(line['moved'])
        self.assertEqual(line['witness'], {'power': 300, 'offset': 90})

    def test_a_card_that_left_the_machine_is_skipped_and_the_rest_are_written(self):
        """The gap rule the whole feature turns on: an absent card is a skip with a sentence, not a failure.

        A machine that lost a card to a capacitor is a machine that still wants the other three moved,
        and a set that refused there would leave the operator with no way back except doing it by hand.
        """
        self.at_rungs()
        self.save()
        self.take_out(TI)
        self.at_stock()
        reply = self.apply()
        self.assertTrue(reply['ok'], 'a card that left is not a failed apply')
        self.assertEqual(reply['applied'], 3)
        self.assertEqual(reply['skipped'], 1)
        self.assertNotIn(TI, self.asked(None), 'a card that is not in the machine was written')
        skipped = [card for card in reply['cards'] if card['state'] == 'absent']
        self.assertEqual(len(skipped), 1)
        self.assertIn('not in this machine any more', skipped[0]['reason'])

    def test_a_card_that_arrived_after_the_save_gets_nothing_from_the_set(self):
        """The other half of the rule, and the one that makes a set safe to press at all.

        A card added last week is running whatever it was set up to run. A set saved before it existed
        has no opinion about it, and "apply the saved set" must not become "reset the new card".
        """
        self.at_rungs()
        self.save()
        self.move_in(ARRIVED, 'RTX 4000 SFF Ada')
        self.at_stock()
        reply = self.apply()
        self.assertTrue(reply['ok'])
        self.assertNotIn(ARRIVED, self.asked(None), 'a set wrote a card it never named')
        self.assertEqual([card['gpu'] for card in reply['left_out']], [ARRIVED])
        self.assertEqual(self.agent.cards[ARRIVED]['offset'], 0, 'the new card moved anyway')
        # And the page says so before the click, not as a surprise afterwards.
        self.assertEqual([card['gpu'] for card in self.row('quiet-night')['left_out']], [ARRIVED])

    def test_a_slot_holding_another_model_of_card_is_refused_not_given_another_board_s_numbers(self):
        """220 W is a Turing number. It does not go on an Ampere card because a UUID is still in the file.

        The only word in a set that says which ladder a rung came from is the name the driver gave the
        card when it was saved, so a slot that changed hands is skipped rather than guessed at.
        """
        self.at_rungs()
        self.save()
        self.replace(MEASURED, 'RTX 2080 Ti')
        self.at_stock()
        reply = self.apply()
        self.assertEqual(reply['applied'], 3, 'the card that changed hands was written anyway')
        self.assertNotIn(MEASURED, self.asked(None))
        skipped = [card for card in reply['cards'] if card['state'] == 'card-changed']
        self.assertEqual(len(skipped), 1)
        self.assertIn('RTX 2080 Ti', skipped[0]['reason'], 'the refusal says what is in the slot now')

    def test_a_rung_that_left_source_skips_its_card_with_the_rungs_that_are_left(self):
        """A set outlives the ladder it was saved against, and the card must not be talked into a rung.

        This is what a revision that retires the rung a set was saved against does to it: the plan still
        resolves, the
        name still matches, and only the rung is gone. The refusal lists the rungs that are still there,
        which is the sentence that tells the operator what to re-save the set against.
        """
        # The 3090 is pointed at the 2080 Ti's short ladder, saved while it sits on a rung it has, and
        # the saved line is then moved to a rung only the long ladder has.
        host = copy.deepcopy(self.c.host)
        host['power_plans'][MEASURED] = 'rtx2080-ti'
        self.c.host = host
        self.c.save('host.json', host)
        safe = self.power.registry()['plans']['rtx2080-ti']['rungs']['safe']
        self.move(MEASURED, safe['power'], safe['offset'])
        self.save()
        sets = self.stored()
        sets['quiet-night']['cards'][MEASURED]['rung'] = 'assertive'
        self.write_sets_file(sets)
        # A revision retires the rung: the plan resolves, the name resolves, the rung is gone.
        retired = copy.deepcopy(self.power.registry())
        del retired['plans']['rtx2080-ti']['rungs']['assertive']
        self.power.registry = lambda: retired
        self.at_stock()
        reply = self.apply()
        self.assertNotIn(MEASURED, self.asked(None))
        skipped = [card for card in reply['cards'] if card['state'] == 'rung-gone']
        self.assertIn('assertive', skipped[0]['reason'])
        self.assertIn('safe, sensible', skipped[0]['reason'], 'a refusal lists the rungs that do exist')
        self.assertEqual(self.row('quiet-night')['skipped'], 1, 'the page did not see the gap either')

    def test_a_card_whose_plan_was_switched_off_says_which_setting_says_so(self):
        self.at_rungs()
        self.save()
        host = copy.deepcopy(self.c.host)
        host['power_plans'][MEASURED] = None
        self.c.host = host
        self.c.save('host.json', host)
        self.at_stock()
        reply = self.apply()
        self.assertNotIn(MEASURED, self.asked(None))
        skipped = [card for card in reply['cards'] if card['state'] == 'plan-off']
        self.assertIn('power_plans', skipped[0]['reason'])

    def test_a_card_moved_onto_another_plan_is_skipped_rather_than_reinterpreted(self):
        self.at_rungs()
        self.save()
        host = copy.deepcopy(self.c.host)
        host['power_plans'][MEASURED] = 'rtx3080-ti'
        self.c.host = host
        self.c.save('host.json', host)
        self.at_stock()
        reply = self.apply()
        self.assertNotIn(MEASURED, self.asked(None))
        skipped = [card for card in reply['cards'] if card['state'] == 'plan-changed']
        self.assertIn('rtx3080-ti', skipped[0]['reason'])

    def test_stock_is_every_card_the_machine_reports_and_needs_no_ladder_for_any_of_them(self):
        """The built-in set is not a memory, so a card nobody has measured reverts as happily as one
        somebody has, and a card added yesterday is in it before anybody has saved a thing."""
        self.at_rungs()
        self.move_in(ARRIVED, 'RTX 4000 SFF Ada')
        reply = self.apply(name='stock')
        asked = self.asked(None)
        self.assertEqual(reply['applied'], 5)
        self.assertTrue(reply['builtin'])
        for uuid in (MEASURED, OTHER, TI, UNMEASURED, ARRIVED):
            self.assertEqual(asked[uuid], (self.agent.cards[uuid]['power_default'], 0, None),
                '%s was not reverted' % uuid)
        self.assertEqual(reply['left_out'], [])
        self.assertFalse((self.state / SETS).exists(), 'applying stock saved a set')

    def test_a_set_that_names_nothing_the_machine_has_writes_nothing_and_says_which(self):
        self.at_rungs()
        self.save()
        for uuid in (MEASURED, OTHER, TI, UNMEASURED):
            self.take_out(uuid)
        reply = self.apply()
        self.assertTrue(reply['ok'], 'a set with nothing to write did not fail')
        self.assertEqual(reply['applied'], 0)
        self.assertEqual(self.agent.writes, [])
        self.assertIn('skipped', reply['note'])

    def test_an_unhealthy_engine_refuses_the_whole_set_before_a_single_card_moves(self):
        """Half a set is a machine in a state nobody saved, so the gate is taken over everything at once."""
        self.at_rungs()
        self.save()
        self.at_stock()
        self.system.containers['qwen38-daytime']['State']['Running'] = False
        with self.assertRaises(PowerError) as refused:
            self.apply()
        self.assertEqual((refused.exception.status, refused.exception.code), (409, 'engine'))
        self.assertIn('half a set', str(refused.exception))
        self.assertEqual(self.agent.writes, [], 'a refusal reached a card')
        reply = self.apply(override=True)
        self.assertEqual(reply['applied'], 4, 'the tick that means "apply anyway" did not apply')

    def test_one_card_s_driver_saying_no_is_that_card_s_and_the_rest_are_written(self):
        """A set that stopped at the first no would leave the operator guessing which cards moved."""
        self.at_rungs()
        self.save()
        self.at_stock()
        answer, agent = self.agent.answer_for, self.agent

        def failing(argv):
            if argv[argv.index('--uuid') + 1] == TI and 'apply' in argv:
                return {'ok': False, 'verb': 'apply', 'uuid': TI,
                    'error': 'Failed to set power limit: Insufficient Permissions'}
            return answer(argv)

        agent.answer_for = failing
        reply = self.apply()
        self.assertFalse(reply['ok'])
        self.assertEqual(reply['applied'], 3)
        self.assertEqual(reply['failed'], 1)
        failed = [card for card in reply['cards'] if card['state'] == 'failed']
        self.assertIn('Insufficient Permissions', failed[0]['reason'])
        self.assertEqual(set(self.asked(None)), {MEASURED, OTHER, UNMEASURED})

    def test_a_write_that_lands_and_does_not_read_back_is_reported_not_smoothed_over(self):
        self.at_rungs()
        self.save()
        self.at_stock()
        self.agent.readback = 'fail'
        reply = self.apply()
        self.assertFalse(reply['ok'])
        self.assertEqual(reply['applied'], 0)
        self.assertEqual(len([card for card in reply['cards'] if card['state'] == 'applied']), 4)
        self.assertIn('read back', json.dumps(reply['cards']))

    def test_a_forgotten_set_is_named_with_the_ones_this_machine_has(self):
        self.at_rungs()
        self.save(name='quiet-night')
        with self.assertRaises(RuntimeError) as refused:
            self.apply(name='last-summer')
        self.assertIn('quiet-night', str(refused.exception), 'a refusal lists what does exist')
        self.assertIn('stock', str(refused.exception))

    def test_a_delete_forgets_the_memory_and_touches_no_card(self):
        self.at_rungs()
        self.save()
        self.agent.writes.clear()
        reply = self.power.write_sets({'action': 'delete', 'name': 'quiet-night'})
        self.assertTrue(reply['deleted'])
        self.assertEqual(self.stored(), {})
        self.assertEqual(self.agent.writes, [])
        with self.assertRaises(RuntimeError):
            self.power.write_sets({'action': 'delete', 'name': 'quiet-night'})

    # --------------------------------------------------------------------------------- refusals, as ever
    def test_a_set_is_refused_while_the_machine_is_changing_its_own_shape(self):
        self.at_rungs()
        self.save()
        self.c.save('transaction.json', {'phase': 'draining'})
        with self.assertRaises(PowerError) as refused:
            self.apply()
        self.assertEqual((refused.exception.status, refused.exception.code), (409, 'busy'))
        # A save reads the driver and writes a file, so a switch in flight is not a reason to refuse it:
        # "what were the cards at, before this switch moves them" is exactly the question worth asking
        # then, and the answer costs a card nothing.
        self.assertTrue(self.save(name='during-a-switch')['ok'])
        for phase_file, value in (('update-job.json', {'phase': 'prepared'}),
                ('router-maintenance.json', {'reserved': True})):
            self.c.save(phase_file, value)
            with self.subTest(file=phase_file), self.assertRaises(PowerError) as refused:
                self.apply()
            self.assertEqual(refused.exception.code, 'busy')
            self.c.save(phase_file, {})
        self.c.save('transaction.json', {'phase': 'succeeded'})
        self.assertTrue(self.apply()['ok'], 'a finished switch still holds the cards')

    def test_a_save_does_not_move_the_lock_out_from_under_a_write(self):
        """The card lock is one act's worth of hands: a photograph taken mid-recall is a blur, said as one."""
        self.at_rungs()
        self.save()
        with patch('runtime.power.lock', side_effect=RuntimeError(
                'Another runtime operation holds the maintenance lock')):
            for call in (lambda: self.apply(), lambda: self.save(name='another')):
                with self.assertRaises(PowerError) as refused:
                    call()
                self.assertEqual((refused.exception.status, refused.exception.code), (409, 'busy'))
                self.assertIn('holds the cards', str(refused.exception))

    def test_inspection_mode_saves_nothing_applies_nothing_and_forgets_nothing(self):
        self.at_rungs()
        self.system.readonly = True
        for body in ({'action': 'save', 'name': 'quiet-night'}, {'action': 'apply', 'name': 'stock'},
                {'action': 'delete', 'name': 'quiet-night'}):
            with self.subTest(body=body), self.assertRaises(RuntimeError) as refused:
                self.power.write_sets(body)
            self.assertIn('inspection mode', str(refused.exception))
        self.assertEqual(self.agent.writes, [])

    def test_a_request_that_arrives_well_formed_but_wrong_is_the_browser_s_mistake(self):
        self.at_rungs()
        for body in ({'action': 'sideways', 'name': 'x'}, {'action': 'save'},
                {'action': 'apply', 'name': 'x', 'gpu': MEASURED},
                {'action': 'apply', 'name': 'x', 'omit': [MEASURED]},
                {'action': 'save', 'name': 'x', 'override': 'yes'}):
            with self.subTest(body=body), self.assertRaises(RuntimeError):
                self.power.write_sets(body)
        self.assertEqual(self.agent.writes, [])

    # --------------------------------------------------------------------------------------- what it shows
    def test_the_page_is_offered_stock_first_and_every_saved_set_after_it(self):
        self.at_rungs()
        self.save(name='quiet-night')
        self.save(name='assertive-everything')
        rows, block = self.rows()
        self.assertIsNone(block['error'])
        self.assertTrue(block['available'])
        listed = [row['name'] for row in self.power.status()['sets']['sets']]
        self.assertEqual(listed, ['stock', 'quiet-night', 'assertive-everything'])
        self.assertTrue(rows['stock']['builtin'])
        self.assertEqual([card['state'] for card in rows['stock']['cards']], ['ready'] * 4)
        self.assertEqual({card['kind'] for card in rows['stock']['cards']}, {'stock'})
        night = rows['quiet-night']
        self.assertEqual((night['ready'], night['skipped']), (4, 0))
        self.assertEqual({card['gpu']: card['rung'] for card in night['cards']},
            {MEASURED: 'sensible', OTHER: 'safe', TI: 'assertive', UNMEASURED: None})

    def test_the_page_shows_a_gap_the_same_way_the_write_will_find_it(self):
        """One resolution, two readers: the row an operator read is the row the driver carries out.

        A page that resolved a set from the cache and a write that resolved it from the driver would
        disagree the day a card left, which is the only day it matters.
        """
        self.at_rungs()
        self.save()
        self.take_out(TI)
        row = self.row('quiet-night')
        states = {card['gpu']: card['state'] for card in row['cards']}
        self.assertEqual(states[TI], 'absent')
        self.assertEqual((row['ready'], row['skipped']), (3, 1))
        self.assertEqual(self.apply()['applied'], row['ready'],
            'the page promised a different number of cards than the write wrote')

    def test_nothing_about_a_set_costs_the_page_a_container_start(self):
        """The five-second poll reads the sets, the rung table and the card cache. That is all it may read."""
        self.at_rungs()
        self.save()
        before = len(self.agent.reads)
        self.power.status()
        self.power.status()
        self.assertEqual(len(self.agent.reads), before, 'drawing the sets asked the driver')

    def test_a_sets_file_that_names_itself_stays_off_the_browser(self):
        """The complaint is a filesystem naming itself, and a browser is two networks away from it."""
        from runtime.power import public_power
        self.at_rungs()
        self.save()
        (self.state / SETS).write_text('not json at all')
        block = public_power(self.c.status()['power'])
        self.assertNotIn(str(self.state), json.dumps(block['sets']))
        self.assertNotIn('Expecting', json.dumps(block['sets']))
        self.assertIn('diagnostics', block['sets']['error'])


if __name__ == '__main__':
    unittest.main()

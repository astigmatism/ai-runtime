"""Card power state: read what the driver is doing, and move a card to a measured rung.

The functionality is the host script `ops/scripts/gpu-undervolt` — same three knobs, same rungs, same
refusals — reached from the page and the CLI. Two things about this machine make it a different piece
of code:

* **The controller has the Docker socket and no GPUs.** It cannot call `nvidia-smi` itself, so every
  read and every write is one privileged one-shot container of this project's own image, which is
  exactly how `gpu-undervolt --docker` works on Rosalina (where `sudo` prompts and an agent cannot
  answer). `runtime/gpucard.py` is the thing that runs in there.
* **A read is not free, and the page polls every five seconds.** Five cards times a container start
  every five seconds would be a feature that costs the machine more than it tells it, so the driver is
  asked on a slower cadence in a background pass, the answer is cached in `.state/power/latest.json`,
  and the page is shown the cache with its age. The age is the honest part: these are volatile driver
  settings, so a reading is a statement about a moment, and a reboot returns every card to stock
  without telling anyone.

Nothing here drains, publishes, or restores: a rung is three driver calls and a read-back. A write is
refused while a switch, a restoration or a source deployment owns the machine, and the write that
*did* land is recorded with what the driver said back afterwards.

**A set is the same act over the whole machine.** A plan belongs to a model of card and a rung to one
card, so neither answers "what were all the cards in this machine doing before I changed them". A
*power set* is that: the rung every physical card in the machine was sitting at, under a name, plus the
built-in `stock` that reverts whatever the machine reports right now. Saving is one read pass and a
JSON file; recalling is one read pass and then one card write per line of the set, under one lock. A
card the set does not name is not touched, a card the machine no longer has is skipped with a sentence,
and a slot now holding a different model of card is never handed the rung some other board earned.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import json
import threading
from datetime import datetime, timedelta, timezone

from .config import (STOCK_SET, SETS_LIMIT, card_model, power_plan, power_plan_for_name, power_plans,
    power_request, power_rung, power_set_cards, power_set_request, power_set_entry, read_power_registry,
    read_power_sets, require)
from .system import lock, now

# How often the background pass may ask the driver, and how long one container run is allowed to take
# before the pass gives up and leaves the cache to age. The cadence is a compromise: an apply reads
# back immediately, so the operator never waits on this; it is the *idle* freshness that costs.
CADENCE_SECONDS = 30
RUN_TIMEOUT = 120
# Writes per card kept so "what was this card doing yesterday at 21:00" has an answer. Volatile
# settings deserve a history precisely because the settings themselves do not survive a reboot.
HISTORY_LIMIT = 20
LATEST = 'power/latest.json'
HISTORY = 'power/history.json'
LOCK = 'power.lock'
# Saved power sets, keyed by the operator's own name for them. Operational state and not source: a set
# names this machine's physical cards, and an operator chose the name. It lives beside the cache
# because it survives an update the way the cache does — the state directory is mounted, an image is
# replaced — and a machine that is rebuilt from source still has the names its operator gave it.
SETS = 'power/sets.json'
SETS_VERSION = 1


class PowerError(Exception):
    """A refusal with the HTTP shape the handler needs and a sentence an operator can act on."""

    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code = status, code


def classify(card, plan):
    """What the driver says the card is sitting at, as one word the page can show.

    Deliberately read off the two knobs that can be verified — power limit and V/F offset — and never
    off the clock lock, which this driver does not report at all. `unknown` is not `stock`: a card
    whose offset could not be read has not been shown to be at zero.
    """
    if not card or card.get('error') or card.get('offset') is None:
        return 'unknown', None
    offset, limit, default = card['offset'], card.get('power_limit'), card.get('power_default')
    for name, rung in sorted((plan or {}).get('rungs', {}).items()):
        if offset == rung['offset'] and limit is not None and abs(limit - rung['power']) < 1:
            return 'rung', name
    if offset == 0 and limit is not None and default is not None and limit >= default:
        return 'stock', None
    return 'custom', None


class Power:
    def __init__(self, controller):
        self.controller = controller
        self.mutex = threading.Lock()
        self.running = False

    # ------------------------------------------------------------------ the two halves of the plan
    def registry(self):
        """The rung table, or None when it cannot be read; never an exception into a status poll."""
        try:
            return read_power_registry(self.controller.config_dir)
        except Exception:  # noqa: BLE001 - reported in status(), the page still has to draw
            return None

    def registry_error(self):
        try:
            read_power_registry(self.controller.config_dir)
        except Exception as error:  # noqa: BLE001
            return str(error)
        return None

    def host_plans(self):
        """{uuid: plan id} for this machine's overrides, or {} when the setting is malformed.

        This is the override table, not the plan of a card: a card absent from it is matched to the
        plan measured for its own model. A typo in `power_plans` is reported in `status()` and falls
        back to that inference rather than taking the page down: the machine is serving either way,
        and the page is where the typo gets read about.
        """
        try:
            return power_plans(self.controller.host)
        except Exception:  # noqa: BLE001
            return {}

    def plan_error(self):
        try:
            power_plans(self.controller.host)
        except Exception as error:  # noqa: BLE001
            return str(error)
        return None

    def names_for(self, uuid, reading=None, declared=None):
        """The names this card is known by, best evidence first.

        The name the *driver* reported for the card in the slot is the first one, because it is the
        only name that cannot be wrong about what is in the slot; the machine's own settings come
        second, and they are decorative — one host's list calls two cards `RTX 4080` and `RTX 4080
        SUPER` because an operator typed them, and a card nobody has read yet has nothing better.
        """
        if declared is None:
            declared = dict(self.cards()).get(uuid)
        return [(reading or {}).get('name'), declared]

    def plan_for(self, uuid, names, registry=None, plans=None):
        """Where one card's plan comes from: `(plan id, 'host')`, `(id, 'model')`, `(None, 'off')`.

        `power_plans` wins when it names this card, because that is an operator overruling the
        measurement for one board, and a `null` entry means this card is offered no rung even though
        its model has been measured. With nothing written down, the plan is the one measured for the
        model the driver reported — a ladder result belongs to a model of board, and the driver says
        which board is in the slot. A model the table has never seen gets no plan and no rung: an RTX
        4080 is not given the 4080 SUPER's numbers because they are nearby, and `330 W / +75` is not
        given to it either.
        """
        plans = self.host_plans() if plans is None else plans
        if uuid in plans:
            return (None, 'off') if plans[uuid] is None else (plans[uuid], 'host')
        registry = self.registry() if registry is None else registry
        for name in names if registry else []:
            plan_id = power_plan_for_name(registry, name)
            if plan_id:
                return plan_id, 'model'
        return None, None

    def plans_for_cards(self):
        """Every card this machine declares, with what it resolves to and why.

        `power --plans` prints this next to the table: a card whose rungs come from a name the driver
        reported is worth knowing about when the reason they are missing elsewhere is a model nobody
        has measured.
        """
        registry, plans, cached = self.registry(), self.host_plans(), self.latest()
        resolved = {}
        for uuid, name in self.cards():
            reading = (cached.get(uuid) or {}).get('read') or {}
            plan_id, source = self.plan_for(uuid, self.names_for(uuid, reading, name), registry, plans)
            resolved[uuid] = {'name': name, 'reported': reading.get('name'), 'plan': plan_id,
                'source': source}
        return resolved

    def cards(self):
        """Every physical GPU this machine declares, text cards first, in reservation order."""
        inventory = self.controller.gpu_inventory()
        return [(entry['uuid'], entry['name']) for entry in inventory['text']] + (
            [] if not inventory['vision'] else
            [(inventory['vision']['uuid'], inventory['vision']['name'])])

    def card(self, wanted):
        """One card from a name a person typed: its UUID, or enough of a name to be exactly one card.

        A full UUID is 40 characters and nobody types one from memory, on a terminal or in a list of
        cards to leave out of a save. What it must never take is an ambiguous name, because the whole
        point of naming a card here is that it is *that* card and not another of its model.
        """
        cards = self.cards()
        for uuid, name in cards:
            if uuid.lower() == str(wanted).lower():
                return uuid
        matches = [uuid for uuid, name in cards if str(wanted).lower() in uuid.lower()
            or str(wanted).lower() in (name or '').lower()]
        require(len(matches) == 1, '%r matches %s; name one card, or give its full UUID' % (wanted,
            ', '.join(matches) if matches else 'nothing this machine reports'))
        return matches[0]

    # ---------------------------------------------------------------------------- the driver, via docker
    def image(self):
        return self.controller.host.get('power_image') or self.controller.image

    def command(self, uuids, verb='read', wants=None, expect_names=None, retries=2):
        """One privileged one-shot container, and the card agent inside it.

        `--gpus all` with the card chosen by UUID inside the container is the arrangement the script
        has used for weeks: device-level `--gpus device=…` selection works, but it also makes a card
        invisible to the very read that has to say whether it is still the card we think it is.
        """
        argv = ['run', '-i', '--rm', '--pull=never', '--privileged', '--user', '0:0',
            '--gpus', 'all', '--network', 'none', '--label', 'ai-runtime.power=1',
            '--entrypoint', 'python3', self.image(), '-m', 'runtime.gpucard', verb]
        for uuid in uuids:
            argv += ['--uuid', uuid]
        if expect_names:
            for name in expect_names:
                argv += ['--expect-name', name]
        if verb == 'apply':
            argv += ['--power', '%.0f' % wants['power']]
            # A revert carries no clock floor because it asks for none: `-rgc` drops the lock, and
            # inventing a floor for it would be the runtime claiming a state it did not ask for.
            if wants.get('clock_max') is not None:
                argv += ['--clock-max', '%d' % wants['clock_max'],
                    '--clock-min', '%d' % (wants.get('clock_min') or 210)]
            argv += ['--offset', '%d' % wants['offset'], '--retries', str(retries)]
        return argv

    def invoke(self, argv):
        """Run one card-agent container and read its single JSON object back.

        Docker's own complaint is kept out of the returned structure beyond one line: it arrives on
        the browser with the rest of the status payload, and a path or a credential inside a daemon
        message belongs in the private journal, not on the page.
        """
        if self.controller.system.readonly:
            raise PowerError(409, 'inspection', 'The runtime is inspecting this machine and is not '
                'writing to or reading from the cards.')
        try:
            raw = self.controller.system.docker(*argv, check=False, timeout=RUN_TIMEOUT)
        except Exception as error:  # noqa: BLE001 - a daemon that will not answer is a reading
            self.journal(str(error))
            return {'ok': False, 'error': 'The one-shot GPU container could not be run.'}
        payload = (raw or '').strip()
        if not payload:
            self.journal(' '.join(argv))
            return {'ok': False, 'error': 'The one-shot GPU container answered nothing.'}
        try:
            report = json.loads(payload[payload.index('{'):])
        except Exception:  # noqa: BLE001
            self.journal(payload)
            return {'ok': False, 'error': 'The one-shot GPU container answered something that was '
                'not a card reading.'}
        if report.get('ok') is False and report.get('error'):
            # The driver's own words are kept here, and only here: they name a device, a query field
            # or a path, and what the page gets instead is one generic sentence per card.
            self.journal(str(report['error']))
        return report

    def journal(self, text):
        """Full failure text, kept privately — the same rule as an operation's journal."""
        try:
            path = self.controller.state / 'power' / 'last-error.txt'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(now() + '\n' + str(text)[-4000:] + '\n')
        except Exception:  # noqa: BLE001 - a diagnostic that cannot be written is not a failure
            pass

    # ------------------------------------------------------------------------------------ the cache
    def latest(self):
        return self.controller.load(LATEST, {}) or {}

    def history(self):
        return (self.controller.load(HISTORY, {}) or {}).get('writes', [])

    def remember(self, entry):
        writes = ([entry] + self.history())[:HISTORY_LIMIT]
        self.controller.save(HISTORY, {'writes': writes})

    def merge(self, cards):
        """Write a fresh pass to the cache, merging it over the last one card by card."""
        latest = self.latest()
        latest.update({uuid: card for uuid, card in cards.items()})
        self.controller.save(LATEST, latest)
        return latest

    def status(self):
        """The block `/api/status` carries: the cache, resolved against the plan table.

        Reads only files this process already has, because the five-second poll calls it. A card that
        has never been read is `unread`, which is not the same claim as `stock`. Each card also says
        where its plan came from — `host` for the override in the private settings, `model` for the
        plan measured for the card the driver reported, `off` for one an operator switched off — so a
        page that offers no rung can say which of three reasons it has.
        """
        registry = self.registry()
        plans, error = self.host_plans(), self.plan_error() or self.registry_error()
        cached = self.latest()
        writes = self.availability()
        history = self.history()
        block = {'available': bool(registry), 'cadence_seconds': CADENCE_SECONDS, 'error': error,
            'writes': writes, 'cards': {}}
        for uuid, name in self.cards():
            card = dict(cached.get(uuid) or {})
            reading = card.get('read') or {}
            # A host that names a plan this revision does not have is a typo in operational state, and
            # the page is where it gets read about: the card still reports its reading and its Revert,
            # and says why there is nothing to choose.
            plan_id, plan_source = self.plan_for(uuid, self.names_for(uuid, reading, name), registry,
                plans)
            plan, plan_error = None, None
            if registry and plan_id:
                try:
                    plan = power_plan(registry, plan_id)
                except Exception as error:  # noqa: BLE001 - one bad line, not a page that does not load
                    plan_error = str(error)
            # `error` is its own state: a card that could not be read is a different sentence from one
            # that was read and matches nothing, and the tile says `power unreadable` for it.
            if card.get('error'):
                state, rung = 'error', None
            else:
                state, rung = classify(reading, plan) if card else ('unread', None)
            block['cards'][uuid] = {
                'uuid': uuid, 'name': name, 'plan': plan_id, 'plan_source': plan_source,
                'plan_error': plan_error,
                'read_at': card.get('read_at'), 'error': card.get('error'),
                'state': state, 'rung': rung,
                # The envelope numbers, and only those: the table carries its reasoning as `_why`
                # comments, and a paragraph repeated once per card on a five-second poll is a page
                # that scrolls before it loads.
                'envelope': {key: value for key, value in ((registry or {}).get('envelope') or {}).items()
                    if not key.startswith('_')},
                'rungs': [] if not plan else [{'name': key, 'default': key == plan['default'],
                    **{field: value for field, value in rung.items() if field != 'clock_min'}}
                    for key, rung in sorted(plan['rungs'].items(), key=lambda item: item[1]['offset'])],
                'default': (plan or {}).get('default'), 'measured': (plan or {}).get('measured'),
                'note': (plan or {}).get('note'),
                'read': {key: reading.get(key) for key in ('power_limit', 'power_default', 'power_min',
                    'power_max', 'offset', 'offset_error', 'clock_sm', 'clock_max_sm', 'draw',
                    'draw_average', 'temperature', 'pstate', 'power_capped_hours', 'name')},
                'last_write': next((entry for entry in history if entry.get('gpu') == uuid), None),
            }
        # Every set this machine could put on itself, resolved against the cards it has this minute.
        # Files only: the poll that draws the page must never be the thing that starts a container.
        block['sets'] = self.sets_block(registry)
        return block

    def refresh_if_due(self, force=False):
        """Called by the status loop; starts a pass when one is due and never waits for it."""
        with self.mutex:
            if self.running:
                return False
            cached = self.latest()
            if not force and cached.get('swept_at') and not self.stale(cached):
                return False
            self.running = True
        threading.Thread(target=self.pass_once, daemon=True).start()
        return True

    def stale(self, cached):
        stamp = cached.get('swept_at')
        if not stamp:
            return True
        try:
            return datetime.fromisoformat(stamp) < datetime.now(timezone.utc) - timedelta(
                seconds=CADENCE_SECONDS)
        except ValueError:
            return True

    def read_pass(self):
        """One driver pass over every card this machine declares, straight into the cache.

        Returns `(cards, ok, moment)`, where `cards` is the *cache-shaped* entry for each card —
        `{'read': …}` or `{'error': …}` — so the caller that just asked the driver and the page that is
        reading the cache are looking at one shape and cannot disagree about what a card is doing. Two
        things call this: the cadence thread, and a save. A save that read the cache instead would be a
        photograph of a moment up to `CADENCE_SECONDS` old filed as a recording of *now*, and the one
        promise a saved set has to keep is that it is what the cards were doing when you saved.
        """
        declared = self.cards()
        if not declared:
            return {}, False, now()
        report = self.invoke(self.command([uuid for uuid, _ in declared], 'read'))
        cards = report.get('cards') or {}
        if not cards:
            # The container itself failed, so no card has a reading: say which cards that leaves
            # unread rather than letting the page go on showing readings from before the daemon
            # stopped answering.
            cards = {uuid: {'error': report.get('error') or 'the pass did not answer'}
                for uuid, _ in declared}
        moment = now()
        merged = {uuid: ({'read_at': moment, 'read': card} if 'error' not in card else
            {'read_at': moment, 'error': card['error']}) for uuid, card in cards.items()}
        self.merge(merged)
        return merged, bool(report.get('ok')), moment

    def pass_once(self):
        """One driver pass into the cache, on the cadence, keeping the stamp the cadence reads."""
        try:
            cards, ok, moment = self.read_pass()
            if ok:
                # `swept_at` is when the whole pass succeeded, and it is what the cadence compares
                # against; a pass that failed leaves it alone, so the next loop tries again.
                latest = self.latest()
                latest['swept_at'] = moment
                self.controller.save(LATEST, latest)
        finally:
            with self.mutex:
                self.running = False

    # ------------------------------------------------------------------------------------- refusals
    def blockers(self):
        """Why no power write can start right now, in the same categories a switch refuses for.

        A rung write does not need `ready`: the safest moment to move a card is while nothing is
        loaded on it. It does need the machine not to be in the middle of changing its own shape,
        because a drain moving models across cards and a hand changing their clocks is two hands.
        """
        # Both constants are the runtime's own, taken from the modules that write them, and read in the
        # same order `Operations.blockers` reads them: a power write and a switch are refused for the
        # same reasons or the two surfaces disagree about what the machine is doing. They are imported
        # here rather than at the top of the file because `controller` is what imports this module.
        from .controller import PENDING_PHASES
        from .operations import OPERATION_ID

        c = self.controller
        if c.system.readonly:
            return 'The runtime is inspecting this machine and is not writing to the cards.'
        if c.load('router-maintenance.json', {}).get('reserved'):
            return 'A router deployment reserves the runtime.'
        # A transaction does not delete itself when it works: it stays on disk with its terminal
        # `succeeded`, which is not pending and must not be read as if it were. Reading it that way is
        # not a refusal in a corner case — it refuses every card on every machine that has ever
        # finished one, which is what the first deploy of this feature did on Rosalina.
        if c.load('transaction.json', {}).get('phase') in PENDING_PHASES:
            return 'Runtime maintenance or recovery must finish first.'
        if c.load('update-job.json', {}).get('phase') in ('prepared', 'controller-replacement'):
            return 'A source deployment must finish or be recovered first.'
        pointer = c.load('latest-operation.json', {})
        wanted = pointer.get('id')
        # Guarded the way `Operations.read` guards it, because this runs on every status poll, and a
        # pointer with something unexpected in it means "no operation", not an exception in the page.
        operation = (c.load('operations/' + wanted + '.json')
                     if isinstance(wanted, str) and OPERATION_ID.fullmatch(wanted) else None)
        if (operation or {}).get('status') == 'running':
            return 'A configuration switch is in progress.'
        try:
            with lock(c.state / LOCK):
                pass
        except RuntimeError:
            return 'Another power write holds the cards.'
        return None

    def availability(self):
        reason = self.blockers()
        return {'available': reason is None, 'reason': reason}

    def gate(self, uuid):
        """Whether the model that owns this card is healthy enough to be written underneath.

        The script's health gate, kept for the reason the script keeps it: an undervolt that is two
        notches too aggressive does not announce itself as a failure, it shows up as strange turns and
        an Xid in the kernel log, and the 4080 SUPER of this house has that on record at 255 W. A card
        no model is on has nothing to be unhealthy, and writing it is the safe case.
        """
        active = self.controller.load('active.json')
        bundle = (active or {}).get('bundle')
        if not bundle:
            return True, 'Nothing is declared as the active configuration; nothing is serving here.'
        try:
            observed = self.controller.observe(bundle)
        except Exception:  # noqa: BLE001 - the observer failing is not a reason to refuse everything
            return True, 'The serving engines could not be asked; the write is allowed.'
        on_card = []
        for model, service in zip(bundle['catalog']['models'], bundle['manifest']['services']):
            held = set(model.get('text_gpu_uuids', model['gpu_uuids']) or [])
            if model.get('vision_gpu_uuid'):
                held.add(model['vision_gpu_uuid'])
            if uuid in held:
                on_card.append((service['service'], observed.get(service['service'], {})))
        if not on_card:
            return True, 'No model of the active configuration is on this card.'
        unhealthy = [(name, live) for name, live in on_card if not live.get('healthy')]
        if unhealthy:
            name, live = unhealthy[0]
            return False, ('%s is not healthy on this card%s' % (name, '' if live.get('running', True)
                else ' and is not running') + '. Nothing has been changed and nothing will be. '
                'Say you know better, or fix the model first.')
        return True, '%s is healthy on this card.' % on_card[0][0]

    # --------------------------------------------------------------------------------------- writes
    def wants_for(self, request, registry, reading, plan):
        """The three numbers to write, from a validated request plus the card's plan or the driver."""
        if request['action'] == 'apply':
            rung = power_rung(plan, request['rung'])
            envelope = registry['envelope']
            return {'power': rung['power'], 'clock_max': rung['clock_max'], 'offset': rung['offset'],
                'clock_min': envelope['clock_min'], 'label': request['rung']}
        if request['action'] == 'revert':
            default = (reading or {}).get('power_default')
            require(default, 'This card has never been read, so there is no stock power limit to '
                'revert to. Read it first — the Re-read control does that — and ask again.')
            return {'power': default, 'clock_max': None, 'offset': 0, 'clock_min': None,
                'label': 'stock'}
        return {'power': request['power'], 'clock_max': request['clock_max'], 'offset': request['offset'],
            'clock_min': registry['envelope']['clock_min'], 'label': 'custom'}

    def write(self, body):
        """Apply a validated request to one card and report what the driver said afterwards."""
        c = self.controller
        require(not c.system.readonly, 'State mutation is forbidden in inspection mode')
        registry = self.registry()
        require(registry, 'The rung table could not be read, so no write can be validated: ' +
            str(self.registry_error()))
        cached = (self.latest().get((body or {}).get('gpu')) or {}) if isinstance(body, dict) else {}
        reading = cached.get('read') or {}
        request = power_request(body, registry['envelope'],
            (reading.get('power_min'), reading.get('power_max')))
        if request['action'] == 'read':
            # The Re-read control: one card, one container, straight into the cache. It is a read, so
            # it is not refused by a switch being in flight -- and it is what an operator reaches for
            # when the page's age line has grown in a way they do not believe.
            report = self.invoke(self.command([request['gpu']], 'read'))
            card = (report.get('cards') or {}).get(request['gpu']) or {'error': report.get('error')}
            moment = now()
            self.merge({request['gpu']: {'error': card['error'], 'read_at': moment}
                if 'error' in card else {'read': card, 'read_at': moment}})
            return {'ok': 'error' not in card, 'action': 'read', 'gpu': request['gpu'],
                'read': card.get('read') or card, 'error': card.get('error'), 'read_at': moment}
        declared = {uuid for uuid, _ in self.cards()}
        require(request['gpu'] in declared, 'This machine reports no card with that UUID, so nothing '
            'has been written; the cards it reports are ' + ', '.join(sorted(declared)) +
            '. A power write names one of those.')
        blocker = self.blockers()
        if blocker:
            raise PowerError(409, 'busy', blocker)
        if cached.get('error'):
            raise PowerError(409, 'unread', 'This card cannot be read right now: ' + cached['error'])
        # The same resolution the page drew its radio list from, so a rung offered here is a rung the
        # write can carry out; a `null` in `power_plans` is the operator saying no to a card whose
        # model has been measured, and that deserves its own sentence rather than the generic one. It
        # says no to rungs, not to the writes that need no measurement.
        plan_id, plan_source = self.plan_for(request['gpu'], self.names_for(request['gpu'], reading),
            registry)
        require(request['action'] != 'apply' or plan_source != 'off',
            'This machine\'s settings turn the measured rungs off for this card, so it has no rung to '
            'apply. Its entry in `.state/host.json` is what turns them off; Revert and values by hand '
            'work regardless.')
        plan = None
        try:
            plan = power_plan(registry, plan_id) if plan_id else None
        except Exception:  # noqa: BLE001 - a typo'd mapping is reported in status(); Revert needs no plan
            plan = None
        if request['action'] == 'apply' and plan is None:
            power_plan(registry, plan_id)
        wants = self.wants_for(request, registry, reading, plan)
        allowed, why = self.gate(request['gpu'])
        if not allowed and not request['override']:
            raise PowerError(409, 'engine', why + ' Tick "apply anyway" to write regardless.')
        with lock(c.state / LOCK):
            return self.write_card(request['gpu'], wants, plan, reading, gate=why,
                override=request['override'], rounded=request.get('offset_rounded', False))

    def write_card(self, gpu, wants, plan, reading, gate, override, rounded=False, set_name=None):
        """Write three numbers to one card, record what the driver said back, refresh the cache.

        **The caller holds** `LOCK`. That is what makes a set possible: one card is one call and a set
        is the same call N times under the single lock it took, so a machine halfway through being
        moved to another set has one pair of hands and not two. `gate` is the health sentence the caller
        already weighed, kept because "what was serving on this card when it was written" is the first
        thing anybody asks an hour later.
        """
        name = reading.get('name')
        report = self.invoke(self.command([gpu], 'apply', wants=wants,
            expect_names=[name] if name else None))
        if not report.get('ok') and not report.get('apply'):
            raise PowerError(503, 'driver', report.get('error') or
                'The one-shot GPU container could not write this card.')
        applied = report['apply']
        after = applied['after']
        moment = now()
        before_state, before_rung = classify(reading, plan)
        self.remember({'at': moment, 'gpu': gpu, 'name': name,
            'from': before_rung or before_state, 'to': wants['label'], 'asked': applied['asked'],
            'readback': applied['readback'], 'ok': applied['ok'], 'attempts': applied['attempts'],
            'gate': gate, 'override': override, 'actor': 'page', 'set': set_name})
        latest = self.latest()
        latest[gpu] = {'read_at': moment, 'read': after}
        self.controller.save(LATEST, latest)
        state, rung = classify(after, plan)
        return {'ok': bool(applied['ok']), 'gpu': gpu, 'to': wants['label'],
            'state': state, 'rung': rung, 'asked': applied['asked'], 'readback': applied['readback'],
            'attempts': applied['attempts'], 'commands': applied['commands'], 'errors': applied['errors'],
            'read': after, 'gate': gate, 'rounded_offset': rounded, 'set': set_name, 'read_at': moment}


    # ------------------------------------------------------------------------------------ power sets
    # One card is a rung; a *set* is the machine. Everything below is the same three numbers reached a
    # different way: a save is one read pass and a JSON file, and a recall is one read pass and then one
    # ordinary card write per line of the set, under the one lock the recall took. Nothing about the
    # numbers themselves is new, which is the point — a set that wrote its own copy of the wattage would
    # be applying the ladder as it was remembered instead of the ladder as source says it is.
    def sets(self):
        """This machine's saved power sets, `{name: {saved_at, cards}}`, read tolerantly.

        The five-second poll reads this, so a file with something wrong in it is a note on the page and
        no sets, never an exception in a status poll. `stock` is not in here: it is derived from whatever
        the machine reports right now, by `stock_set`, so it cannot go stale the way a list of saved
        UUIDs can.
        """
        return read_power_sets(self.controller.state / SETS)[0]

    def sets_note(self):
        """Why the saved sets are not being used, or None. Same tolerance as `sets()`."""
        return read_power_sets(self.controller.state / SETS)[1]

    def stored_sets(self):
        """The saved sets for a *write*, which does not tolerate anything.

        Status tolerates a line it cannot read because the machine is serving either way. A save does
        not, because a save rewrites the file: silently dropping the line it could not parse and
        reporting a successful save is how a preference file quietly loses data.
        """
        stored, note = read_power_sets(self.controller.state / SETS)
        require(note is None, 'The saved power sets cannot be rewritten while part of them cannot be '
            'read: ' + str(note))
        return stored

    def save_sets(self, stored):
        self.controller.save(SETS, {'version': SETS_VERSION, 'sets': stored})

    def stock_set(self, cards=None):
        """The set every machine has: every card it reports, back at the driver's own numbers.

        Derived on every read and never stored, because it is not a memory — it is the answer to "what is
        in this machine today", and a saved list of UUIDs would be wrong the day a card is added. Its
        lines carry no card name and no plan either, which is what lets it fit a slot whose card was
        swapped and a board no ladder has ever measured: reverting needs neither.
        """
        cards = self.cards() if cards is None else cards
        return {'name': STOCK_SET, 'builtin': True, 'saved_at': None,
            'cards': {uuid: {'kind': 'stock', 'name': '', 'plan': None} for uuid, name in cards}}

    # ---------------------------------------------------------------------- one line of one set, now
    def resolve_entry(self, gpu, entry, card, declared=None, registry=None, plans=None):
        """Can this line of this set be put on this card as the machine stands? Why not?

        One resolution, used by the page's preview and by the write itself, so the row an operator read
        is the row the driver carries out. `card` is the cache-shaped entry — `{'read': …}`, `{'error':
         …}` or `None` for a card nobody reports — because the caller that asked the driver a second ago
        and the caller reading the cache must not need two code paths.

        The states are the gaps a machine grows over a year of service, and each has its own next move:

        `absent`       the card left the machine; its line is skipped and the rest are still written
        `unread`       the driver would not answer, or this card has never been read
        `card-changed` the slot holds another model of card, so the saved rung is not its rung
        `plan-off`     this machine's settings offer the card no plan, whatever its model measured
        `plan-changed` the card resolves to another plan than the one the rung belongs to
        `rung-gone`    that plan no longer has that rung
        `ready`        write it, with `want` as the numbers
        """
        registry = self.registry() if registry is None else registry
        plans = self.host_plans() if plans is None else plans
        label = declared or entry.get('name') or gpu
        row = {'gpu': gpu, 'name': label, 'kind': entry.get('kind'), 'rung': entry.get('rung'),
            'plan': entry.get('plan'), 'witness': entry.get('witness') or {}, 'state': None,
            'reason': None, 'want': None, 'plan_now': None, 'reading': {}, 'card_plan': None,
            'moved': False}
        # Two different silences, and the operator's next move differs. What decides between them is
        # the machine's own inventory and not the cache: a card this machine no longer declares has left
        # the chassis even while its last reading sits in the cache looking usable, and writing a rung to
        # it because the file remembered it is exactly the thing a set must not do.
        if not declared:
            row.update(state='absent', reason='%s is not in this machine any more. Its line was '
                'skipped; the cards that are here were written.' % label)
            return row
        if card is None:
            row.update(state='unread', reason='%s is declared by this machine and the driver did not '
                'answer for it, so nothing was written to it. It may be out of the chassis, or the '
                'driver may only be refusing: read it again to tell them apart.' % label)
            return row
        if card.get('error'):
            row.update(state='unread', reason='%s could not be read from the driver, so nothing was '
                'written to it. Read it again and try once more.' % label)
            return row
        reading = card.get('read') or {}
        row['reading'] = reading
        if not reading:
            row.update(state='unread', reason='%s has never been read from the driver, so this write '
                'has nothing to go on.' % label)
            return row
        # The card in the slot has to be the card the set was written for. A rung is a measurement of one
        # model of board, and the only word in the set that says which one is the name the driver gave.
        saved_as, seen_as = card_model(entry.get('name')), card_model(reading.get('name'))
        if entry.get('kind') == 'rung' and saved_as and seen_as and saved_as != seen_as:
            row.update(state='card-changed', reason='This slot holds %s now, and this line was saved '
                'against %s. A rung is what the ladder measured for one model of card, so nothing was '
                'written here: re-save the set, or put the card that was measured back.' % (
                    reading.get('name'), entry.get('name')))
            return row
        if entry.get('kind') == 'stock':
            default = reading.get('power_default')
            if not default:
                row.update(state='unread', reason='%s has never reported its stock power limit, so '
                    'there is nothing to revert it to. Read it again first.' % label)
                return row
            row.update(state='ready', want={'power': default, 'clock_max': None, 'offset': 0,
                'clock_min': None, 'label': STOCK_SET})
            return row
        plan_id, plan_source = self.plan_for(gpu, self.names_for(gpu, reading, declared), registry,
            plans)
        row['plan_now'] = plan_id
        if plan_source == 'off':
            row.update(state='plan-off', reason='this machine\'s settings offer this card no plan, so '
                'the rung in this line has no numbers to resolve to. The entry for %s in power_plans is '
                'what says so, and nothing was written here.' % gpu)
            return row
        if plan_id != entry.get('plan'):
            row.update(state='plan-changed', reason='this card resolves to plan %s now and this line '
                'was saved against %s. `%s` is that plan\'s number, not this card\'s, so nothing was '
                'written here.' % (plan_id or 'nothing', entry.get('plan'), entry.get('rung')))
            return row
        plan = None
        try:
            plan = power_plan(registry, plan_id)
            row['card_plan'] = plan
            rung = power_rung(plan, entry['rung'])
        except Exception as error:  # noqa: BLE001 - a rung that left source is a skip, not a crash
            row.update(state='rung-gone', reason=str(error) + ', so nothing was written to this card. '
                'Re-save the set once it has a rung worth putting it at.')
            return row
        envelope = (registry or {}).get('envelope') or {}
        row.update(state='ready', want={'power': rung['power'], 'clock_max': rung['clock_max'],
            'offset': rung['offset'], 'clock_min': envelope.get('clock_min'),
            'label': entry['rung']})
        witness = row['witness']
        row['moved'] = bool(witness.get('power') is not None and (
            witness.get('power') != rung['power'] or witness.get('offset') != rung['offset']))
        return row

    # --------------------------------------------------------------------------------- the three acts
    def write_sets(self, body):
        """The `/api/gpu-power-sets` half: save a photograph of the machine, put one back, forget one."""
        require(not self.controller.system.readonly,
            'State mutation is forbidden in inspection mode')
        request = power_set_request(body, taken=self.sets())
        if request['action'] == 'save':
            return self.save_set(request)
        if request['action'] == 'delete':
            return self.delete_set(request)
        return self.apply_set(request)

    @contextmanager
    def holding(self):
        """Hold the card lock across a whole set, or refuse in the words the page already shows.

        A set is one act, so it takes the lock once for all of it: a second pair of hands arriving
        mid-recall would find three cards moved and two not, which is a state no operator chose.
        """
        try:
            with lock(self.controller.state / LOCK):
                yield
        except RuntimeError as error:
            if 'maintenance lock' not in str(error):
                raise
            raise PowerError(409, 'busy', 'Another power write holds the cards.') from None

    def save_set(self, request):
        """Record what every card in the machine is sitting at right now, under a name.

        A save writes no card: it is a read pass and an atomic file, which is why it is not refused
        while a switch has the machine. It *is* refused rather than rounded off when a card cannot be
        recorded faithfully — a card at hand-written values has a clock cap nobody can read back on
        GeForce, and a set that filed `270 W / +45` for it would be promising a return to a state it
        never saw. Leaving a card out is an operator's decision, taken by name, not the runtime's.
        """
        registry = self.registry()
        require(registry, 'The rung table could not be read, so nothing can be saved: ' +
            str(self.registry_error()))
        stored = self.stored_sets()
        require(len(stored) < SETS_LIMIT or request['name'] in stored,
            'this machine already keeps ' + str(SETS_LIMIT) + ' power sets; delete one that has been '
            'superseded before saving another, because a set nobody can find is not a way back')
        omitted = {self.card(item) for item in request['omit']}
        cards = self.cards()
        with self.holding():
            readings, ok, moment = self.read_pass()
        plans = self.host_plans()
        entries, refused, left_out = {}, [], []
        for uuid, name in cards:
            if uuid in omitted:
                left_out.append({'gpu': uuid, 'name': name})
                continue
            entry, refusal = self.entry_for(uuid, name, readings.get(uuid), registry, plans)
            if refusal:
                refused.append(refusal)
            elif entry:
                entries[uuid] = entry
        require(not refused, 'Nothing was saved. ' + ' '.join(refused) + ' Move the card to a rung, '
            'or leave it out of this set on purpose: the page names the card, and from a terminal it is '
            '`--omit <card>`.')
        require(entries, 'nothing was left to save: every card this machine reports was left out, and a '
            'set of nothing is a name with nothing behind it')
        power_set_cards(entries)  # validate the runtime's own output before it can be read back
        stored[request['name']] = {'saved_at': moment, 'cards': entries}
        self.save_sets(stored)
        return {'ok': True, 'action': 'save', 'set': request['name'], 'saved_at': moment,
            'cards': len(entries), 'omitted': left_out,
            'note': None if ok else 'The driver did not answer for every card; this set is what it '
                'did answer for.'}

    def entry_for(self, gpu, name, card, registry, plans):
        """One card's line for a save: `(entry, None)` when it can be recorded, `(None, why not)`."""
        label = name or gpu
        if card is None:
            return None, '%s was not read, so there is nothing to record about it.' % label
        if card.get('error'):
            return None, ('%s could not be read from the driver.' % label)
        reading = card.get('read') or {}
        if not reading:
            return None, '%s has never been read from the driver.' % label
        plan_id, _ = self.plan_for(gpu, self.names_for(gpu, reading, name), registry, plans)
        plan = None
        try:
            plan = power_plan(registry, plan_id) if plan_id else None
        except Exception:  # noqa: BLE001 - an unresolvable plan is "not a rung", and says so below
            plan = None
        state, rung = classify(reading, plan)
        witness = {'power': reading.get('power_limit'), 'offset': reading.get('offset')}
        if state == 'rung':
            return {'kind': 'rung', 'rung': rung, 'plan': plan_id,
                'name': reading.get('name') or '', 'witness': witness}, None
        if state == 'stock':
            return {'kind': 'stock', 'name': reading.get('name') or '', 'plan': None,
                'witness': witness}, None
        if state == 'custom':
            return None, ('%s is at %.0f W and %+d MHz, which is none of this card\'s rungs: that is a '
                'hand-written state, and the clock cap it was given cannot be read back on GeForce, so '
                'a set saved now would bring the card back somewhere other than where it is.' % (
                    label, reading.get('power_limit') or 0, reading.get('offset') or 0))
        return None, ('%s has no readable V/F offset, so whether it is at stock or between rungs is not '
            'known, and a set does not record guesses.' % label)

    def apply_set(self, request):
        """Put a saved set — or `stock` — onto every card of this machine it still fits.

        One read pass first, because "is this card still here, and is it still this card" is a question
        for the driver and not for a file. Then the health gate over the whole set *before* any card is
        written: half a set applied is a machine in a state nobody ever saved, so an unhealthy engine
        refuses the act rather than carving a gap into it. Then one ordinary card write per line, under
        the lock this took, and a card whose driver says no is reported and does not stop its neighbours.
        """
        c = self.controller
        name = request['name']
        registry = self.registry()
        require(registry, 'The rung table could not be read, so no set can be applied: ' +
            str(self.registry_error()))
        cards = self.cards()
        if name.lower() == STOCK_SET:
            definition = self.stock_set(cards)
        else:
            stored = self.stored_sets()
            require(name in stored, 'There is no power set named ' + repr(name) + ' on this machine; it '
                'has ' + (', '.join(sorted(stored)) if stored else 'no saved sets yet') +
                ', and `stock` is always available.')
            definition = {'name': name, 'builtin': False, 'saved_at': None, **stored[name]}
        blocker = self.blockers()
        if blocker:
            raise PowerError(409, 'busy', blocker)
        with self.holding():
            readings, ok, moment = self.read_pass()
            rows = [self.resolve_entry(uuid, entry, readings.get(uuid),
                    dict(cards).get(uuid), registry) for uuid, entry in
                self.ordered(definition, cards)]
            # The gate runs over everything that would be written, before any of it is.
            gates, unwell = {}, []
            for row in rows:
                if row['state'] != 'ready':
                    continue
                allowed, why = self.gate(row['gpu'])
                gates[row['gpu']] = why
                if not allowed:
                    unwell.append(why)
            if unwell and not request['override']:
                raise PowerError(409, 'engine', ' '.join(unwell) + ' Nothing has been changed and '
                    'nothing will be: a set is one act over the whole machine, and half a set is a '
                    'machine in a state nobody saved. Say you know better, or fix the model first.')
            results, applied, failed = [], 0, 0
            for row in rows:
                if row['state'] != 'ready':
                    results.append({'gpu': row['gpu'], 'name': row['name'], 'state': row['state'],
                        'reason': row['reason'], 'rung': row['rung'], 'ok': True})
                    continue
                try:
                    wrote = self.write_card(row['gpu'], row['want'], row['card_plan'], row['reading'],
                        gate=gates[row['gpu']], override=request['override'],
                        set_name=None if definition['builtin'] else name)
                except PowerError as error:
                    # One card's driver is that card's bad day, not the machine's: the rest go ahead and
                    # this one is said, because a set that stopped at the first no would leave the
                    # operator guessing which cards moved.
                    failed += 1
                    results.append({'gpu': row['gpu'], 'name': row['name'], 'state': 'failed',
                        'reason': str(error), 'rung': row['rung'], 'ok': False})
                    continue
                applied += bool(wrote['ok'])
                failed += 0 if wrote['ok'] else 1
                results.append({'gpu': row['gpu'], 'name': row['name'], 'state': 'applied',
                    'to': wrote['to'], 'state_after': wrote['state'], 'rung': wrote['rung'],
                    'asked': wrote['asked'], 'readback': wrote['readback'], 'ok': wrote['ok'],
                    'reason': None if wrote['ok'] else 'The driver took the write and did not read back '
                        'as asked; Revert to stock is one act and needs no model to be healthy.'})
        declared = {uuid for uuid, _ in cards}
        left_out = [{'gpu': uuid, 'name': label} for uuid, label in cards
            if uuid not in definition['cards']]
        return {'ok': failed == 0, 'action': 'apply', 'set': name, 'builtin': definition['builtin'],
            'applied': applied, 'failed': failed, 'skipped': len(rows) - applied - failed,
            'cards': results, 'left_out': left_out, 'read_at': moment,
            'note': None if applied else 'Nothing was written: every line of the set was skipped, '
                'each for the reason it says.'}

    def delete_set(self, request):
        """Forget a saved set. The cards keep whatever they are sitting at, as always."""
        stored = self.stored_sets()
        name = request['name']
        require(name in stored, 'There is no power set named ' + repr(name) + ' to delete; this machine '
            'has ' + (', '.join(sorted(stored)) if stored else 'no saved sets yet'))
        stored.pop(name)
        self.save_sets(stored)
        return {'ok': True, 'action': 'delete', 'set': name, 'deleted': True,
            'note': 'Nothing was written to a card: a set is a memory, and forgetting one leaves the '
                'machine where it is.'}

    # ---------------------------------------------------------------------------------- sets, for the page
    def ordered(self, definition, cards):
        """A set's lines in the order the machine's tiles draw them, with the orphans last.

        A set whose first line is a card that left the machine draws a gap above four good ones, and an
        operator reading a five-line table should see the four first.
        """
        declared = [uuid for uuid, _ in cards]
        entries = definition['cards']
        return ([(uuid, entries[uuid]) for uuid in declared if uuid in entries] +
            [(uuid, entries[uuid]) for uuid in entries if uuid not in declared])

    def set_row(self, definition, cards, readings=None, registry=None, plans=None):
        """One set as this machine can offer it right now, one line per card, and what it leaves alone."""
        registry = self.registry() if registry is None else registry
        plans = self.host_plans() if plans is None else plans
        readings = self.latest() if readings is None else readings
        names = dict(cards)
        rows = [self.resolve_entry(uuid, entry, readings.get(uuid), names.get(uuid), registry, plans)
            for uuid, entry in self.ordered(definition, cards)]
        cards_rows = [{'gpu': row['gpu'], 'name': row['name'], 'kind': row['kind'], 'rung': row['rung'],
            'plan': row['plan'], 'plan_now': row['plan_now'], 'witness': row['witness'],
            'state': row['state'], 'reason': row['reason'], 'moved': bool(row.get('moved'))}
            for row in rows]
        return {'name': definition['name'], 'builtin': bool(definition.get('builtin')),
            'saved_at': definition.get('saved_at'), 'cards': cards_rows,
            'ready': sum(1 for row in rows if row['state'] == 'ready'),
            'skipped': sum(1 for row in rows if row['state'] != 'ready'),
            'left_out': ([] if definition.get('builtin') else
                [{'gpu': uuid, 'name': label} for uuid, label in cards if uuid not in
                    definition['cards']])}

    def sets_block(self, registry=None, error=None):
        """`status.power.sets`: the built-in set first, then every saved one, against this machine now.

        Reads only files this process already has — the saved sets, the rung table, the card cache —
        because the five-second poll calls it. `skipped` on a row is not a failure waiting to be fixed:
        it is a card this set has nothing to say about, or nothing true to say.
        """
        cards, plans = self.cards(), self.host_plans()
        rows = [self.set_row(self.stock_set(cards), cards, None, registry, plans)]
        saved, note = read_power_sets(self.controller.state / SETS)
        for name, definition in saved.items():
            rows.append(self.set_row({'name': name, 'builtin': False, **definition}, cards, None,
                registry, plans))
        return {'available': bool(registry), 'error': error or note, 'limit': SETS_LIMIT,
            'stock': STOCK_SET, 'sets': rows}


def public_power(block):
    """The power block with its private sentences replaced, the way `public_status` does.

    A card error is a Docker complaint or a driver message, and both can carry a path. The page's
    job is to say that the card could not be read and point at diagnostics, which is what this leaves.

    A set's per-card lines are not scrubbed and are not luck: every one of them is a sentence this file
    wrote — which card left the machine, which slot changed hands — and each is the thing the operator
    needs to read before applying. The one private sentence is the sets file's own complaint, because
    that is a filesystem naming itself.
    """
    if not block:
        return block
    block = copy.deepcopy(block)
    if block.get('error'):
        block['error'] = 'The rung table could not be read. View controller diagnostics for details.'
    if (block.get('sets') or {}).get('error'):
        block['sets']['error'] = ('The saved power sets could not be read and are not in use. View '
            'controller diagnostics for details.')
    for card in (block.get('cards') or {}).values():
        if card.get('error'):
            card['error'] = 'This card could not be read. View controller diagnostics for details.'
    return block

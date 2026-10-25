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
"""
from __future__ import annotations

import copy
import json
import threading
from datetime import datetime, timedelta, timezone

from .config import (power_plan, power_plans, power_request, power_rung, read_power_registry,
    require)
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
        """{uuid: plan id} for this machine, or {} when the setting is malformed.

        A typo in `power_plans` is reported in `status()` and degrades to no rungs rather than
        taking the page down: the machine is serving either way, and the page is where the typo gets
        read about.
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

    def cards(self):
        """Every physical GPU this machine declares, text cards first, in reservation order."""
        inventory = self.controller.gpu_inventory()
        return [(entry['uuid'], entry['name']) for entry in inventory['text']] + (
            [] if not inventory['vision'] else
            [(inventory['vision']['uuid'], inventory['vision']['name'])])

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
        has never been read is `unread`, which is not the same claim as `stock`.
        """
        registry = self.registry()
        plans, error = self.host_plans(), self.plan_error() or self.registry_error()
        cached = self.latest()
        writes = self.availability()
        history = self.history()
        block = {'available': bool(registry), 'cadence_seconds': CADENCE_SECONDS, 'error': error,
            'writes': writes, 'cards': {}}
        for uuid, name in self.cards():
            plan_id = plans.get(uuid)
            plan = None
            if registry and plan_id:
                try:
                    plan = power_plan(registry, plan_id)
                except Exception:  # noqa: BLE001 - one bad line, not a page that does not load
                    plan = None
            card = dict(cached.get(uuid) or {})
            # A host that names a plan this revision does not have is a typo in operational state, and
            # the page is where it gets read about: the card still reports its reading and its Revert,
            # and says why there is nothing to choose.
            plan_error = None
            if registry and plan_id and plan is None:
                try:
                    power_plan(registry, plan_id)
                except Exception as error:  # noqa: BLE001
                    plan_error = str(error)
            reading = card.get('read') or {}
            # `error` is its own state: a card that could not be read is a different sentence from one
            # that was read and matches nothing, and the tile says `power unreadable` for it.
            if card.get('error'):
                state, rung = 'error', None
            else:
                state, rung = classify(reading, plan) if card else ('unread', None)
            block['cards'][uuid] = {
                'uuid': uuid, 'name': name, 'plan': plan_id, 'plan_error': plan_error,
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

    def pass_once(self):
        """One driver pass over every card this machine declares, into the cache."""
        try:
            uuids = [uuid for uuid, _ in self.cards()]
            if not uuids:
                return
            report = self.invoke(self.command(uuids, 'read'))
            cards = report.get('cards') or {}
            if not cards:
                # The container itself failed, so no card has a reading: say which cards that leaves
                # unread rather than letting the page go on showing readings from before the daemon
                # stopped answering.
                cards = {uuid: {'error': report.get('error') or 'the pass did not answer'}
                    for uuid, _ in self.cards()}
            moment = now()
            self.merge({uuid: ({'read_at': moment, 'read': card} if 'error' not in card else
                {'read_at': moment, 'error': card['error']}) for uuid, card in cards.items()})
            if report.get('ok'):
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
    def wants_for(self, request, registry, reading):
        """The three numbers to write, from a validated request plus the rung table or the driver."""
        if request['action'] == 'apply':
            plan = power_plan(registry, self.host_plans().get(request['gpu']))
            rung = power_rung(plan, request['rung'])
            envelope = registry['envelope']
            return {'power': rung['power'], 'clock_max': rung['clock_max'], 'offset': rung['offset'],
                'clock_min': envelope['clock_min'], 'label': request['rung']}, plan
        if request['action'] == 'revert':
            default = (reading or {}).get('power_default')
            require(default, 'This card has never been read, so there is no stock power limit to '
                'revert to. Read it first — the Re-read control does that — and ask again.')
            return {'power': default, 'clock_max': None, 'offset': 0, 'clock_min': None,
                'label': 'stock'}, None
        return {'power': request['power'], 'clock_max': request['clock_max'], 'offset': request['offset'],
            'clock_min': registry['envelope']['clock_min'], 'label': 'custom'}, None

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
        wants, plan = self.wants_for(request, registry, reading)
        allowed, why = self.gate(request['gpu'])
        if not allowed and not request['override']:
            raise PowerError(409, 'engine', why + ' Tick "apply anyway" to write regardless.')
        name = reading.get('name')
        with lock(c.state / LOCK):
            report = self.invoke(self.command([request['gpu']], 'apply', wants=wants,
                expect_names=[name] if name else None))
        if not report.get('ok') and not report.get('apply'):
            raise PowerError(503, 'driver', report.get('error') or
                'The one-shot GPU container could not write this card.')
        applied = report['apply']
        after = applied['after']
        moment = now()
        before_state, before_rung = classify(reading, plan)
        self.remember({'at': moment, 'gpu': request['gpu'], 'name': reading.get('name'),
            'from': before_rung or before_state,
            'to': wants['label'], 'asked': applied['asked'], 'readback': applied['readback'],
            'ok': applied['ok'], 'attempts': applied['attempts'], 'gate': why,
            'override': request['override'], 'actor': 'page'})
        latest = self.latest()
        latest[request['gpu']] = {'read_at': moment, 'read': after}
        self.controller.save(LATEST, latest)
        state, rung = classify(after, plan)
        return {'ok': bool(applied['ok']), 'gpu': request['gpu'], 'to': wants['label'],
            'state': state, 'rung': rung, 'asked': applied['asked'], 'readback': applied['readback'],
            'attempts': applied['attempts'], 'commands': applied['commands'], 'errors': applied['errors'],
            'read': after, 'gate': why, 'rounded_offset': request.get('offset_rounded', False),
            'read_at': moment}


def public_power(block):
    """The power block with its private sentences replaced, the way `public_status` does.

    A card error is a Docker complaint or a driver message, and both can carry a path. The page's
    job is to say that the card could not be read and point at diagnostics, which is what this leaves.
    """
    if not block:
        return block
    block = copy.deepcopy(block)
    if block.get('error'):
        block['error'] = 'The rung table could not be read. View controller diagnostics for details.'
    for card in (block.get('cards') or {}).values():
        if card.get('error'):
            card['error'] = 'This card could not be read. View controller diagnostics for details.'
    return block

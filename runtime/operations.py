"""Durable browser operations around the controller's existing runtime transaction."""
from contextlib import ExitStack
import re
import threading
import uuid

from .config import (CONFIGURATIONS, configurations_for, everyday_for, is_exclusive, read,
    resolve_profile)
from .controller import PENDING_PHASES
from .system import lock, now

OPERATION_ID = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
# One vocabulary for "a runtime transaction has not finished", shared with the controller.
PENDING = PENDING_PHASES
MESSAGES = {
    'checking': 'Checking the configuration, model files, and runtime.',
    'draining': 'Waiting for existing requests to finish. New requests are paused.',
    'loading': 'Loading the selected model. The Everyday resident stays loaded; new requests remain paused.',
    'loading-stops-everyday': ('Stopping the Everyday resident, then loading the selected model across '
        "every text GPU this host has; new requests remain paused."),
    'loading-starts-everyday': ('Loading the selected model, then starting the Everyday resident again; new '
        'requests remain paused.'),
    'loading-everyday-off': 'Loading the selected model. The Everyday resident stays off; new requests remain paused.',
    'loading-changes-everyday': ('Restarting the Everyday resident with the selected variant, and the model '
        'beside it if that changed; new requests remain paused.'),
    'verifying': 'Checking generation and model discovery before reopening requests.',
    'restoring': 'The switch did not complete. Restoring the previous configuration.',
    'succeeded': 'The selected configuration is ready. Requests are open.',
    'failed': 'The switch failed before replacing a backend. The previous configuration was preserved.',
    'recovered': 'The switch failed. The previous configuration was restored and requests are open.',
    'needs-attention': 'Recovery is incomplete. Switching is disabled; an operator must check the runtime.',
    'drain-timeout': 'Existing work did not finish within five minutes. No active generation was stopped.',
    'interrupted': 'The operation was interrupted before a runtime transaction began. No switch was retried.',
    # The restoration is a different promise from a switch, so it says what it is: nothing is
    # serving, and this is the runtime bringing the previous configuration back by itself.
    'restore-restoring': 'Restoring the last-known-good configuration. Requests stay paused until it is serving.',
    'restore-succeeded': 'The last-known-good configuration is serving and requests are open.',
    'restore-recovered': 'The last-known-good configuration is serving and requests are open.',
    'restore-needs-attention': ('The restoration did not complete. It will try again on its own; the reason is in '
        'the runtime transaction, and the button runs the same restoration again.'),
    'restore-failed': ('The restoration could not begin. Nothing was changed; the reason is in the runtime '
        'transaction, and the button runs the same restoration again.'),
}
PUBLIC_FIELDS = ('id', 'kind', 'trigger', 'from_profile', 'target_profile', 'revision', 'status', 'phase',
                 'started_at', 'updated_at', 'finished_at', 'error_code', 'already_active')


def public_operation(operation):
    if not operation:
        return None
    result = {k: operation[k] for k in PUBLIC_FIELDS if k in operation}
    key = operation.get('error_code') or (operation['phase'] if operation['status'] == 'running' else operation['status'])
    if key == 'loading' and operation.get('everyday') in ('stops', 'starts', 'off', 'changes'):
        key = {'stops': 'loading-stops-everyday', 'starts': 'loading-starts-everyday',
            'off': 'loading-everyday-off', 'changes': 'loading-changes-everyday'}[operation['everyday']]
    if operation.get('kind') == 'restore':
        # A restoration is a different promise from a switch, so it is described in its own words
        # even when it ends in a phase a switch would also have used.
        named = 'restore-' + (operation['phase'] if operation['status'] == 'running' else operation['status'])
        key = named if named in MESSAGES else key
    result['message'] = MESSAGES.get(key, MESSAGES.get(operation['status'], 'Runtime status unavailable.'))
    return result


class OperationError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code = status, code


class Operations:
    def __init__(self, controller, state):
        self.controller, self.state = controller, state
        self.mutex = threading.Lock()
        self.worker = None

    def read(self, operation_id):
        if not isinstance(operation_id, str) or not OPERATION_ID.fullmatch(operation_id):
            return None
        return self.controller.load('operations/' + operation_id + '.json')

    def latest(self):
        pointer = self.controller.load('latest-operation.json', {})
        return self.read(pointer.get('id'))

    def save(self, operation):
        operation['updated_at'] = now()
        self.controller.save('operations/' + operation['id'] + '.json', operation)

    def blockers(self):
        c = self.controller
        if not self.state.get('started') or self.state.get('startup_error'):
            return 'Runtime startup checks have not completed.'
        if c.load('router-maintenance.json', {}).get('reserved'):
            return 'A router deployment reserves the runtime.'
        if c.load('transaction.json', {}).get('phase') in PENDING:
            return 'Runtime maintenance or recovery must finish first.'
        if c.load('update-job.json', {}).get('phase') in ('prepared', 'controller-replacement'):
            return 'A source deployment must finish or be recovered first.'
        if not (self.state.get('status') or {}).get('ready'):
            return 'Both models and router discovery must be ready before switching.'
        if (self.latest() or {}).get('status') == 'running':
            return 'A configuration switch is already in progress.'
        return None

    def availability(self):
        reason = self.blockers()
        if reason:
            return {'available': False, 'reason': reason}
        try:
            # Probe only; do no journal I/O while briefly holding the locks.
            with lock(self.controller.state / 'update.lock'), lock(self.controller.state / 'runtime.lock'):
                pass
            return {'available': True, 'reason': None}
        except RuntimeError:
            return {'available': False, 'reason': 'Another runtime operation or source deployment is in progress.'}

    def start(self, request):
        fields = {'profile', 'request_id', 'expected_revision', 'expected_profile'}
        # The target must be a configuration of *this* host, and the pair is checked asymmetrically
        # on purpose: `profile` is what this machine is about to load, so the host's own registry
        # gates it, while `expected_profile` is a report of what is already running -- refusing it
        # for belonging to another machine would strand a host that was given the wrong host_id with
        # no way to switch itself back.
        selectable = configurations_for(self.controller.config_dir, self.controller.host)
        if (not isinstance(request, dict) or set(request) != fields
                or any(not isinstance(request[k], str) for k in fields)
                or resolve_profile(request['profile']) not in selectable
                or resolve_profile(request['expected_profile']) not in CONFIGURATIONS
                or not OPERATION_ID.fullmatch(request['request_id'])
                or not re.fullmatch(r'[0-9a-f]{40}', request['expected_revision'])):
            raise OperationError(400, 'invalid-request',
                'Provide a configuration this host offers, a request UUID, and the observed '
                'configuration/revision.')
        # A renamed configuration is still the configuration the page showed a moment ago, so the old
        # id resolves rather than refusing. The request is stored as sent, which is what makes
        # replaying it by request id the same request.
        profile, expected = resolve_profile(request['profile']), resolve_profile(request['expected_profile'])
        with self.mutex:
            existing = self.read(request['request_id'])
            if existing:
                if existing['request'] != request:
                    raise OperationError(409, 'request-id-reused', 'This request ID belongs to a different switch.')
                return existing, False
            guards = ExitStack()
            try:
                try:
                    guards.enter_context(lock(self.controller.state / 'update.lock'))
                    guards.enter_context(lock(self.controller.state / 'runtime.lock'))
                except RuntimeError:
                    raise OperationError(409, 'busy', 'Another runtime operation or source deployment is in progress.') from None
                reason = self.blockers()
                if reason:
                    raise OperationError(409, 'unavailable', reason)
                active = self.controller.load('active.json', {})
                if (active.get('revision') != request['expected_revision']
                        or self.controller.revision != request['expected_revision']
                        or active.get('bundle', {}).get('profile') != expected):
                    raise OperationError(409, 'stale-selection', 'The active profile or revision changed. Refresh before switching.')
                # What the switch does to the second resident, so progress says it truthfully: a solo
                # configuration stops it, and leaving one starts it again.
                definitions = [read(self.controller.config_dir / 'profiles' / (name + '.json'))
                    for name in (expected, profile)]
                solo = [is_exclusive(definition) for definition in definitions]
                everyday = {(False, False): 'stays', (False, True): 'stops', (True, False): 'starts', (True, True): 'off'}[tuple(solo)]
                if everyday == 'stays' and everyday_for(definitions[0]) != everyday_for(definitions[1]):
                    everyday = 'changes'
                operation = {'id': request['request_id'], 'request': dict(request), 'everyday': everyday,
                    'from_profile': expected, 'target_profile': profile,
                    'revision': request['expected_revision'], 'status': 'running', 'phase': 'checking', 'started_at': now()}
                self.save(operation)
                self.controller.save('latest-operation.json', {'id': operation['id']})
                self.worker = threading.Thread(target=self.run, args=(operation, guards), daemon=True)
                try:
                    self.worker.start()
                except Exception:
                    self.complete(operation, 'failed')
                    raise
                return dict(operation), True
            except BaseException:
                guards.close()
                raise

    def restore_blockers(self):
        """Why a restoration may not start right now.

        Deliberately not the switch's list. A runtime that is not serving, is not ready, or has an
        interrupted transaction is precisely when a restoration is needed, so none of those can be
        reasons to refuse one. What is refused is a competing owner of the router or the checkout,
        and a host with no recorded release to bring back.
        """
        c = self.controller
        if c.load('router-maintenance.json', {}).get('reserved'):
            return 'A router deployment reserves the runtime.'
        if c.load('update-job.json', {}).get('phase') in ('prepared', 'controller-replacement'):
            return 'A source deployment must finish or be recovered first.'
        if (self.latest() or {}).get('status') == 'running' or (self.worker and self.worker.is_alive()):
            return 'Another runtime operation is already in progress.'
        if c.last_known_good() is None:
            return 'No last-known-good release is recorded; operator attention required.'
        return None

    def restore_availability(self):
        reason = self.restore_blockers()
        if reason:
            return {'available': False, 'reason': reason}
        try:
            # Probe only; do no journal I/O while briefly holding the locks.
            with lock(self.controller.state / 'update.lock'), lock(self.controller.state / 'runtime.lock'):
                pass
            return {'available': True, 'reason': None}
        except RuntimeError:
            return {'available': False, 'reason': 'Another runtime operation or source deployment is in progress.'}

    def start_restore(self, request):
        """Admit the one-action restore: no profile, and nothing observed beforehand."""
        if (not isinstance(request, dict) or set(request) != {'request_id'}
                or not isinstance(request.get('request_id'), str)
                or not OPERATION_ID.fullmatch(request['request_id'])):
            raise OperationError(400, 'invalid-request', 'Provide only a request UUID.')
        with self.mutex:
            existing = self.read(request['request_id'])
            if existing:
                if existing.get('kind') != 'restore' or existing.get('request') != dict(request):
                    raise OperationError(409, 'request-id-reused', 'This request ID belongs to a different operation.')
                return existing, False
            guards = ExitStack()
            try:
                try:
                    guards.enter_context(lock(self.controller.state / 'update.lock'))
                    guards.enter_context(lock(self.controller.state / 'runtime.lock'))
                except RuntimeError:
                    raise OperationError(409, 'busy', 'Another runtime operation or source deployment is in progress.') from None
                reason = self.restore_blockers()
                if reason:
                    raise OperationError(409, 'unavailable', reason)
                return self.begin_restore(guards, 'portal', operation_id=request['request_id']), True
            except BaseException:
                guards.close()
                raise

    def begin_restore(self, guards, trigger, operation_id=None):
        """Start the restoration in its own worker so the page keeps polling while it runs."""
        good = self.controller.last_known_good() or {}
        profile = good.get('bundle', {}).get('profile')
        operation = {'id': operation_id or str(uuid.uuid4()), 'kind': 'restore', 'trigger': trigger,
            'from_profile': self.controller.load('transaction.json', {}).get('profile') or profile,
            'target_profile': profile, 'revision': self.controller.revision,
            'status': 'running', 'phase': 'restoring', 'started_at': now()}
        if operation_id:
            operation['request'] = {'request_id': operation_id}
        self.save(operation)
        self.controller.save('latest-operation.json', {'id': operation['id']})
        self.worker = threading.Thread(target=self.run_restore, args=(operation, guards), daemon=True)
        try:
            self.worker.start()
        except Exception:
            self.complete(operation, 'needs-attention', 'The restoration could not be started.')
            raise
        return dict(operation)

    def restore_if_due(self):
        """Run a scheduled restoration when its timer has come due. This is the retry on a timer.

        A failed attempt schedules the next one, so as long as this loop is alive the runtime
        keeps trying until it is serving; nothing here waits for a person to notice.
        """
        with self.mutex:
            if self.worker and self.worker.is_alive():
                return None
        if not self.controller.restoration_due():
            return None
        guards = ExitStack()
        try:
            try:
                guards.enter_context(lock(self.controller.state / 'update.lock'))
                guards.enter_context(lock(self.controller.state / 'runtime.lock'))
            except RuntimeError:
                return None  # Something else owns the runtime this tick; the schedule still stands.
            if not self.controller.restoration_due():
                return None
            return self.begin_restore(guards, 'supervisor')
        except BaseException:
            guards.close()
            raise

    def run_restore(self, operation, guards):
        c = self.controller
        def progress(phase):
            operation['phase'] = phase
            self.save(operation)
        try:
            c._restore_locked(operation_id=operation['id'], progress=progress)
            self.publish_status(c.status())
            self.complete(operation, 'recovered')
        except BaseException as error:
            # This receipt is what a person reads when the box is not serving, so the reason stays
            # with it. The next attempt is already scheduled by the failed pass.
            tx = c.load('transaction.json', {})
            status = 'needs-attention'
            if tx.get('id') == operation['id'] and tx.get('phase') in ('succeeded', 'failed', 'recovered'):
                status = tx['phase']
            self.publish_status(c.status())
            self.complete(operation, status, str(error))
        finally:
            guards.close()

    def publish_status(self, snapshot):
        # A slow poll started before the switch must not overwrite newer health.
        with self.mutex:
            if snapshot.get('updated_at', '') >= (self.state.get('status') or {}).get('updated_at', ''):
                self.state['status'] = snapshot

    def complete(self, operation, status, error=None, error_code=None, finished_at=None):
        operation.update(status=status, phase='finished', finished_at=finished_at or now(), error=error, error_code=error_code)
        self.save(operation)

    def run(self, operation, guards):
        c = self.controller
        def progress(phase):
            operation['phase'] = phase
            self.save(operation)
        try:
            # Cheap admission used the cached health; recheck live health under both locks
            # before hashing or changing anything. The HTTP request never waits for this.
            if not c.status()['ready']:
                raise RuntimeError('Runtime is no longer ready')
            result = c._transition_locked(operation['target_profile'], operation_id=operation['id'],
                progress=progress, keep_everyday=True)
            operation['already_active'] = result.get('already_active', False)
            self.publish_status(c.status())
            self.complete(operation, 'succeeded')
        except BaseException as error:
            tx = c.load('transaction.json', {})
            status = 'failed'
            if tx.get('id') == operation['id']:
                status = tx['phase'] if tx['phase'] in ('succeeded', 'failed', 'recovered') else 'needs-attention'
            code = 'drain-timeout' if status == 'failed' and 'Drain timed out' in str(error) else None
            self.publish_status(c.status())
            self.complete(operation, status, str(error), code)
        finally:
            guards.close()

    def reconcile_interrupted(self):
        """Resolve receipts, never resume/retry a transition or alter a transaction."""
        try:
            # Receipt reconciliation never changes backends. Taking update.lock here
            # would prevent a replacement controller from reconciling receipts
            # while the Portal updater waits for its health.
            with lock(self.controller.state / 'runtime.lock'):
                # Scan receipts as well as the pointer: a process can stop between
                # persisting a request and updating latest-operation.json.
                records = [self.read(path.stem) for path in (self.controller.state / 'operations').glob('*.json')]
                records = [record for record in records if record]
                if not records:
                    return
                latest = max(records, key=lambda record: record['started_at'])
                if (self.latest() or {}).get('id') != latest['id']:
                    self.controller.save('latest-operation.json', {'id': latest['id']})
                tx = self.controller.load('transaction.json', {})
                for op in records:
                    if op['status'] not in ('running', 'needs-attention'):
                        continue
                    if tx.get('id') == op['id']:
                        phase = tx.get('phase')
                        status = phase if phase in ('succeeded', 'failed', 'recovered') else 'needs-attention'
                        if op['status'] != status:
                            self.complete(op, status, finished_at=tx.get('finished_at'))
                    elif op['status'] == 'running':
                        if op['phase'] == 'checking':
                            self.complete(op, 'failed', error_code='interrupted')
                        else:
                            self.complete(op, 'needs-attention')
        except RuntimeError:
            pass  # A live worker, CLI, or Portal updater owns these locks.

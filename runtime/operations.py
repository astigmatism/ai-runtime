"""Durable browser operations around the controller's existing runtime transaction."""
from contextlib import ExitStack
import re
import threading

from .config import DAYTIME_PROFILES
from .system import lock, now

OPERATION_ID = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
PENDING = ('prepared', 'draining', 'applying', 'verifying', 'needs-attention')
MESSAGES = {
    'checking': 'Checking the configuration, model files, and runtime.',
    'draining': 'Waiting for existing requests to finish. New requests are paused for both models.',
    'loading': 'Loading Daytime. Nighttime stays loaded; new requests remain paused.',
    'verifying': 'Checking generation and model discovery before reopening requests.',
    'restoring': 'The switch did not complete. Restoring the previous configuration.',
    'succeeded': 'The selected configuration is ready. Requests are open.',
    'failed': 'The switch failed before replacing a backend. The previous configuration was preserved.',
    'recovered': 'The switch failed. The previous configuration was restored and requests are open.',
    'needs-attention': 'Recovery is incomplete. Switching is disabled; an operator must check the runtime.',
    'drain-timeout': 'Existing work did not finish within five minutes. No active generation was stopped.',
    'interrupted': 'The operation was interrupted before a runtime transaction began. No switch was retried.',
}
PUBLIC_FIELDS = ('id', 'from_profile', 'target_profile', 'revision', 'status', 'phase', 'started_at',
                 'updated_at', 'finished_at', 'error_code', 'already_active')


def public_operation(operation):
    if not operation:
        return None
    result = {k: operation[k] for k in PUBLIC_FIELDS if k in operation}
    key = operation.get('error_code') or (operation['phase'] if operation['status'] == 'running' else operation['status'])
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
        if (not isinstance(request, dict) or set(request) != fields
                or any(not isinstance(request[k], str) for k in fields)
                or request['profile'] not in DAYTIME_PROFILES
                or request['expected_profile'] not in DAYTIME_PROFILES
                or not OPERATION_ID.fullmatch(request['request_id'])
                or not re.fullmatch(r'[0-9a-f]{40}', request['expected_revision'])):
            raise OperationError(400, 'invalid-request', 'Provide a registered profile, request UUID, and observed profile/revision.')
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
                        or active.get('bundle', {}).get('profile') != request['expected_profile']):
                    raise OperationError(409, 'stale-selection', 'The active profile or revision changed. Refresh before switching.')
                operation = {'id': request['request_id'], 'request': dict(request),
                    'from_profile': request['expected_profile'], 'target_profile': request['profile'],
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
                progress=progress, daytime_only=True)
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

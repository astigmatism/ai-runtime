"""Same-origin runtime UI with narrowly scoped, serialized profile switching."""
import copy
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlsplit

from .operations import Operations, OperationError, public_operation


def public_status(state):
    status = copy.deepcopy(state.get('status') or {'ready': False, 'error': 'Startup checks pending'})
    # Diagnostic exceptions can contain Docker output, private paths, or credentials.
    # Keep full errors in private journals/CLI, never return them to the browser.
    if status.get('error'):
        status['error'] = 'Runtime checks are unavailable. View controller diagnostics for details.'
    if state.get('startup_error'):
        status['startup_error'] = 'Startup or recovery checks need operator attention.'
        status['ready'] = False
    if status.get('configurations', {}).get('error'):
        status['configurations']['error'] = 'The configuration registry could not be read.'
    # A note about the alias file is a filesystem complaint, and it carries the state path with it.
    if (status.get('aliases') or {}).get('note'):
        status['aliases']['note'] = 'Saved aliases could not be read and are not in use.'
    for key in ('last_deployment', 'source_deployment'):
        if (status.get(key) or {}).get('error'):
            status[key]['error'] = 'Deployment did not complete. View the Service Portal deployment result.'
    # A restoration reason carries the same Docker detail as the private journals, including the
    # per-phase record that says what each attempt ran into.
    if (status.get('restoration') or {}).get('phases'):
        for phase in status['restoration']['phases']:
            if phase.get('error'):
                phase['error'] = 'The attempt did not complete.'
    for key in ('reason', 'last_error'):
        if (status.get('restoration') or {}).get(key):
            status['restoration'][key] = 'The restoration has not completed. View controller diagnostics for details.'
    return status


def handler(controller, state, operations=None):
    operations = operations or (Operations(controller, state) if controller else None)
    csrf_token = secrets.token_urlsafe(32)
    allowed_hosts = {value.strip().lower() for value in os.environ.get(
        'RUNTIME_ALLOWED_HOSTS', 'localhost,127.0.0.1,::1').split(',')}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send(self, status, data, kind='application/json'):
            body = data if isinstance(data, bytes) else json.dumps(data).encode()
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Leaving the page does not cancel an accepted operation.

        def valid_host(self):
            try:
                host = self.headers.get('Host', '')
                return (len(self.headers.get_all('Host', [])) == 1 and
                    urlsplit('//' + host).hostname in allowed_hosts and
                    not any(c in host for c in '/@?#\\'))
            except ValueError:
                return False

        def do_GET(self):
            if not self.valid_host():
                self.send(421, {'error': 'Unrecognized runtime host'})
                return
            path = self.path.split('?', 1)[0]
            if path in ('/api/status', '/healthz'):
                status = public_status(state)
                if operations:
                    operation = public_operation(operations.latest())
                    status['operation'] = operation
                    if path == '/api/status':
                        status['switching'] = operations.availability()
                        # The restore button needs its own availability: a runtime that refuses a
                        # switch because it is not ready is exactly when the restore is allowed.
                        status['restoring'] = operations.restore_availability()
                    if operation and operation['status'] == 'running':
                        status['ready'] = False
                if path == '/healthz':
                    self.send(200 if status['ready'] else 503, {'ready': status['ready']})
                else:
                    status['csrf_token'] = csrf_token
                    self.send(200, status)
            elif path.startswith('/api/operations/') and operations:
                operation = operations.read(path.removeprefix('/api/operations/'))
                if operation:
                    self.send(200, public_operation(operation))
                else:
                    self.send(404, {'error': 'Operation not found'})
            elif path in ('/', '/app.js', '/style.css'):
                name, kind = {'/': ('index.html', 'text/html; charset=utf-8'),
                    '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                    '/style.css': ('style.css', 'text/css; charset=utf-8')}[path]
                self.send(200, (Path(__file__).parent / 'static' / name).read_bytes(), kind)
            else:
                self.send(404, {'error': 'Not found'})

        def do_POST(self):
            route = self.path.split('?', 1)[0]
            if route not in ('/api/profile-switch', '/api/restore-last-known-good', '/api/alias'):
                self.send(405, {'error': 'Method not allowed'})
                return
            token = self.headers.get('X-Runtime-CSRF', '')
            if (not self.valid_host() or self.headers.get('Origin') != 'http://' + self.headers.get('Host', '')
                    or self.headers.get('Sec-Fetch-Site', 'same-origin') != 'same-origin'
                    or not token.isascii() or not secrets.compare_digest(token, csrf_token)):
                self.send(403, {'error': 'Refresh this page before switching. Same-origin requests are required.'})
                return
            if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
                self.send(415, {'error': 'JSON is required'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if self.headers.get('Transfer-Encoding') or not 0 < length <= 4096:
                    raise ValueError('Invalid body length')
                self.connection.settimeout(10)
                request = json.loads(self.rfile.read(length))
            except (ValueError, UnicodeError, TimeoutError):
                self.send(400, {'error': 'Invalid JSON request; maximum size is 4096 bytes'})
                return
            # Naming a model is not an operation: it drains nothing, restarts nothing and
            # cannot leave a machine between two releases, so it does not queue behind a switch and
            # has no receipt to survive a reload with. It shares the checks above, because a POST
            # that changes what the household is told about this machine deserves them.
            if route == '/api/alias':
                if not controller:
                    self.send(503, {'error': 'Runtime configuration is unavailable'})
                elif not isinstance(request, dict) or 'model' not in request:
                    self.send(400, {'error': 'A model and an alias are required'})
                else:
                    try:
                        reply = controller.set_alias(request.get('model'), request.get('alias'))
                    except RuntimeError as error:
                        detail = str(error)
                        # A refusal is the rule the alias broke, and the page shows it in the editor.
                        # Inspection mode is the one that is not the caller's fault, so it is a 409.
                        self.send(409 if 'inspection mode' in detail else 400, {'error': detail})
                        return
                    except Exception:
                        # Anything else is this machine's fault, and like every other unexpected fault
                        # it is reported as unavailable rather than in the words of a filesystem, which
                        # are how a private path reaches a browser.
                        self.send(503, {'error': 'The name could not be saved. Check it before trying again.'})
                        return
                    self.send(200, reply)
                return
            if not operations:
                self.send(503, {'error': 'Runtime switching is unavailable'})
                return
            # The restore route is the same one action the runtime's own supervisor takes: bring
            # back the release that last served. One button, and no reboot.
            start = operations.start_restore if route == '/api/restore-last-known-good' else operations.start
            try:
                operation, created = start(request)
                self.send(202 if created else 200, {'operation_id': operation['id'], 'operation': public_operation(operation)})
            except OperationError as error:
                self.send(error.status, {'error': str(error), 'code': error.code})
            except Exception:
                self.send(503, {'error': 'Switch status is unavailable. Reconnect to check the operation before trying again.'})

        def unsupported(self):
            self.send(405, {'error': 'Method not allowed'})

        do_PUT = do_PATCH = do_DELETE = unsupported
    return Handler


def refresh(controller, state, operations):
    try:
        operations.reconcile_interrupted()
        # A restoration that has come due outranks everything else in this loop. It is the
        # automatic action the runtime owes a host that is not serving, so the retry timer is
        # served here: a failed attempt schedules the next one and this is where it is taken up,
        # rather than waiting for a person to notice or for someone to reboot the machine.
        restoration = operations.restore_if_due()
        if not state['started'] and not restoration:
            controller.transition()
            state['started'] = True
        state.pop('startup_error', None)
    except Exception as error:
        state['startup_error'] = str(error)
        print('Runtime checks waiting: ' + str(error), flush=True)
    # Even an interrupted startup must keep assignments and diagnostic status visible.
    try:
        # Even an interrupted startup must keep assignments and diagnostic status visible.
        operations.publish_status(controller.status())
    except Exception:
        state['status'] = {'ready': False, 'error': 'Runtime checks unavailable'}
    # ...and keeps saying what this machine is serving, because a router that was never told is a
    # fact the page has to show and then let someone go and fix. It goes after the status and in its
    # own attempt: whether the catalog reached the router is news *about* the state, and a pass that
    # lost the page to a publication that could not start would remove the way to find out, which is
    # precisely the failure the record above is there to survive.
    try:
        controller.announce(state.get('status'))
    except Exception as error:
        print('Catalog publication check waiting: ' + str(error), flush=True)


def serve(controller):
    state = {'started': False}
    operations = Operations(controller, state)
    def supervise():
        while True:
            refresh(controller, state, operations)
            # A pending restoration is polled as often as a started runtime is, so a scheduled
            # retry is not held up by the slow startup loop while nothing is serving.
            pending = (state.get('status') or {}).get('service', {}).get('pending')
            time.sleep(5 if state['started'] or pending else 30)
    threading.Thread(target=supervise, daemon=True).start()
    ThreadingHTTPServer(('0.0.0.0', 8080), handler(controller, state, operations)).serve_forever()

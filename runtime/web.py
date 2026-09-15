"""Read-only, same-origin status UI. All administration stays in CLI and Service Portal."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time


def handler(controller, state):
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
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split('?', 1)[0]
            if path in ('/api/status', '/healthz'):
                status = dict(state.get('status') or {'ready': False, 'error': 'Startup checks pending'})
                if state.get('startup_error'):
                    status['startup_error'] = state['startup_error']
                    status['ready'] = False
                if path == '/healthz':
                    self.send(200 if status['ready'] else 503, {'ready': status['ready']})
                else:
                    self.send(200, status)
            elif path in ('/', '/app.js', '/style.css'):
                name, kind = {'/': ('index.html', 'text/html; charset=utf-8'),
                    '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                    '/style.css': ('style.css', 'text/css; charset=utf-8')}[path]
                self.send(200, (Path(__file__).parent / 'static' / name).read_bytes(), kind)
            else:
                self.send(404, {'error': 'Not found'})

        def do_POST(self):
            self.send(405, {'error': 'Read-only status service'})

        do_PUT = do_PATCH = do_DELETE = do_POST
    return Handler


def serve(controller):
    state = {}
    def supervise():
        started = False
        while True:
            if not started:
                try:
                    controller.transition()
                    state.pop('startup_error', None)
                    started = True
                except Exception as error:
                    state['startup_error'] = str(error)
                    print('Startup waiting: ' + str(error), flush=True)
            state['status'] = controller.status()
            time.sleep(5 if started else 30)
    threading.Thread(target=supervise, daemon=True).start()
    ThreadingHTTPServer(('0.0.0.0', 8080), handler(controller, state)).serve_forever()

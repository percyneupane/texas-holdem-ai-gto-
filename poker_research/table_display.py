"""Playable local table: the live game rendered in a browser, clickable like a real client."""

import argparse
import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .table_game import MAX_PLAYERS, MIN_PLAYERS, TableGame
from .table_layout import SIZE, hit_test
from .policy import POLICIES

TICK_S = 0.05
MAX_BODY_BYTES = 1024
MAX_SESSION_BYTES = 8192

HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Research Table</title>
<style>
html,body{margin:0;background:#0b0f13;color:#c9d1d9;font:13px -apple-system,Helvetica,Arial,sans-serif}
#table{display:block;width:%(w)dpx;height:%(h)dpx;cursor:pointer;user-select:none}
#log{box-sizing:border-box;width:%(w)dpx;height:150px;margin:0;padding:8px 16px;overflow:auto;
 font:12px Menlo,monospace;color:#9aa7b2;border-top:1px solid #1f2830;white-space:pre-wrap}
#session{display:block;box-sizing:border-box;width:%(w)dpx;margin:0;padding:6px 16px 0;color:#c9d1d9;
 font:12px Menlo,monospace;text-decoration:none}
#session:hover{text-decoration:underline}
</style></head><body>
<img id="table" src="/frame.png" width="%(w)d" height="%(h)d" alt="Poker table" draggable="false">
<a id="session" href="/session"></a>
<pre id="log"></pre>
<script>
const img = document.getElementById('table'), log = document.getElementById('log'),
      session = document.getElementById('session');
let shown = null, loading = false;
async function poll() {
  try {
    const state = await (await fetch('/state.json', {cache: 'no-store'})).json();
    if (state.key && state.key !== shown && !loading) {
      loading = true;
      const next = new Image();
      next.onload = () => { img.src = next.src; shown = state.key; loading = false; };
      next.onerror = () => { loading = false; };
      next.src = '/frame.png?k=' + encodeURIComponent(state.key);
    }
    log.textContent = state.log.join('\\n');
    log.scrollTop = log.scrollHeight;
    session.textContent = state.session || '';
  } catch (error) { /* server restarting; keep polling */ }
  setTimeout(poll, 120);
}
img.addEventListener('click', event => {
  const box = img.getBoundingClientRect();
  const x = Math.round((event.clientX - box.left) * %(w)d / box.width);
  const y = Math.round((event.clientY - box.top) * %(h)d / box.height);
  fetch('/click', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({x, y})});
});
poll();
</script></body></html>""" % {"w": SIZE[0], "h": SIZE[1]}

with open(os.path.join(os.path.dirname(__file__), 'session_page.html'), 'rb') as _session_page:
    SESSION_PAGE = _session_page.read()


_ACTIONS = {'fold', 'check', 'call', 'bet', 'raise'}
_LOOPBACK = {'localhost', '127.0.0.1', '::1'}


def whole_chips(amount):
    """Whole research-table chips. Fractional amounts are refused, not rounded into a different bet."""
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        raise ValueError('Action amount must be a number of chips.')
    if isinstance(amount, float):
        if not math.isfinite(amount):
            raise ValueError('Action amount must be a number of chips.')
        nearest = round(amount)
        if abs(amount - nearest) > 1e-6:
            raise ValueError('Action amount must be a whole number of chips.')
        amount = nearest
    if amount < 0:
        raise ValueError('Action amount cannot be negative.')
    return int(amount)


def _unit_interval(data, key: str):
    if key not in data or data[key] is None:
        return None
    value = data[key]
    label = key.replace('_', ' ')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError('%s must be between 0 and 1.' % label)
    if not 0 <= float(value) <= 1:
        raise ValueError('%s must be between 0 and 1.' % label)
    return float(value)


def parse_act(body: bytes):
    """Validate an autoplay action. Returns ``(action, chips to add, note)``."""
    data = json.loads(body.decode('utf-8'))
    if not isinstance(data, dict):
        raise ValueError('Action payload must be an object.')
    action = data.get('action')
    if not isinstance(action, str) or action.lower() not in _ACTIONS:
        raise ValueError('Action must be fold, check, call, bet or raise.')
    note = {}
    equity = _unit_interval(data, 'equity')
    bluff = _unit_interval(data, 'bluff_probability')
    if equity is not None:
        note['equity'] = equity
    if bluff is not None:
        note['bluff_probability'] = bluff
    if 'reason' in data and data['reason'] is not None:
        reason = data['reason']
        if not isinstance(reason, str) or len(reason) > 1200:
            raise ValueError('Reason must be a short string.')
        note['reason'] = reason
    return action.lower(), whole_chips(data.get('amount', 0)), note


def local_table_url(url: str) -> str:
    """Accept only this computer's research table. Auto-play never calls another host."""
    parsed = urlparse(url.strip())
    host = (parsed.hostname or '').lower()
    if parsed.scheme != 'http' or host not in _LOOPBACK or parsed.username or parsed.password:
        raise ValueError('Auto-play only controls the local research table on this computer.')
    if parsed.port is not None and not 1 <= parsed.port <= 65535:
        raise ValueError('Auto-play only controls the local research table on this computer.')
    return url.strip().rstrip('/')


def _post_json(base_url: str, path: str, payload: dict, timeout: float) -> dict:
    """POST JSON to the local research table and return the object it sends back."""
    base = local_table_url(base_url)
    body = json.dumps(payload).encode('utf-8')
    if len(body) > MAX_SESSION_BYTES:
        raise ValueError('Payload is too large.')
    request = Request(base + path, data=body, headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urlopen(request, timeout=timeout) as response:
            parsed = json.loads(response.read().decode('utf-8'))
    except HTTPError as error:
        try:
            message = ''
            try:
                message = json.loads(error.read().decode('utf-8')).get('error', '')
            except (ValueError, AttributeError, UnicodeDecodeError):
                message = ''
            raise ValueError(message or 'Research table rejected the request (%s).' % error.code) from error
        finally:
            error.close()
    except (URLError, TimeoutError, OSError) as error:
        raise ConnectionError('Research table is not running at %s.' % base) from error
    if not isinstance(parsed, dict) or not parsed.get('ok'):
        raise ValueError('Research table did not accept the request.')
    return parsed


def submit_table_action(base_url: str, action: str, amount, timeout: float = 1.0, note=None) -> str:
    """Send one decision to the local research table and return the hand-log line."""
    action = str(action).lower()
    chips = whole_chips(amount)
    if action not in _ACTIONS:
        raise ValueError('Action must be fold, check, call, bet or raise.')
    if action in ('bet', 'raise') and chips <= 0:
        raise ValueError('The decision has no chips to add, so it was not played.')
    payload = {'action': action, 'amount': chips}
    if isinstance(note, dict):
        for key in ('equity', 'bluff_probability', 'reason'):
            if note.get(key) is not None:
                payload[key] = note[key]
    body = _post_json(base_url, '/act', payload, timeout)
    return str(body.get('log') or 'Action played.')


def submit_observation(base_url: str, observation: dict, timeout: float = 0.4) -> dict:
    """Store one overlay read on the research-table session."""
    if not isinstance(observation, dict):
        raise ValueError('Observation must be an object.')
    return _post_json(base_url, '/session/observed', observation, timeout)


def parse_click(body: bytes):
    """Validate a click payload from the page; returns in-bounds (x, y) or raises ValueError."""
    data = json.loads(body.decode("utf-8"))
    x, y = data.get("x"), data.get("y")
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in (x, y)):
        raise ValueError("x and y must be integers")
    if not (0 <= x < SIZE[0] and 0 <= y < SIZE[1]):
        raise ValueError("click outside the table")
    return x, y


class Handler(BaseHTTPRequestHandler):
    game: TableGame = None

    def log_message(self, format, *args):  # keep the terminal quiet; the page shows the hand log
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/frame.png"):
            try:
                _, body = self.game.frame_png()
            except LookupError:
                self.send_error(503, "Dealing the first hand")
                return
            self._send(200, body, "image/png")
        elif self.path.startswith("/state.json"):
            payload = {"key": self.game.frame_key(), "log": list(self.game.log()),
                       "session": self.game.session_line()}
            self._send(200, json.dumps(payload).encode("utf-8"), "application/json")
        elif self.path.startswith("/session.json"):
            payload = json.dumps(self.game.session_snapshot()).encode("utf-8")
            self._send(200, payload, "application/json")
        elif self.path == "/session" or self.path.startswith("/session?"):
            self._send(200, SESSION_PAGE, "text/html; charset=utf-8")
        elif self.path == "/" or self.path.startswith("/?"):
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self.send_error(404)

    def _read_body(self, limit: int = MAX_BODY_BYTES):
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= limit:
            raise ValueError('Bad payload')
        return self.rfile.read(length)

    def _json_request(self, limit: int):
        # application/json is not a browser-simple request, so another
        # page cannot post here without a preflight this server ignores.
        kind = self.headers.get('Content-Type', '').split(';', 1)[0].strip().lower()
        if kind != 'application/json':
            raise ValueError('Payload must be JSON.')
        return self._read_body(limit)

    def do_POST(self):
        if self.path == '/act':
            try:
                action, amount, note = parse_act(self._json_request(MAX_SESSION_BYTES))
                log = self.game.act(action, amount, note=note)
            except (ValueError, UnicodeDecodeError, AttributeError, json.JSONDecodeError) as error:
                body = json.dumps({'ok': False, 'error': str(error)}).encode('utf-8')
                self._send(409, body, 'application/json')
                return
            self._send(200, json.dumps({'ok': True, 'log': log}).encode('utf-8'), 'application/json')
            return
        if self.path == '/session/observed':
            try:
                recorded = self.game.observe(json.loads(self._json_request(MAX_SESSION_BYTES).decode('utf-8')))
            except (ValueError, UnicodeDecodeError, AttributeError, json.JSONDecodeError) as error:
                body = json.dumps({'ok': False, 'error': str(error)}).encode('utf-8')
                self._send(409, body, 'application/json')
                return
            totals = self.game.session.totals()
            self._send(200, json.dumps({
                'ok': True, 'recorded': recorded,
                'observed_hands': totals['observed_hands'],
                'observed_decisions': totals['observed_decisions'],
            }).encode('utf-8'), 'application/json')
            return
        if self.path != "/click":
            self.send_error(404)
            return
        try:
            x, y = parse_click(self._read_body())
        except (ValueError, UnicodeDecodeError, AttributeError, json.JSONDecodeError) as error:
            self.send_error(400, "Bad click payload: %s" % error)
            return
        self.game.click(hit_test(x, y), x)
        self._send(204, b"", "text/plain")


def run_game_loop(game: TableGame, stop: threading.Event) -> None:
    while not stop.is_set():
        game.tick()
        time.sleep(TICK_S)


def main():
    parser = argparse.ArgumentParser(description="Serve a playable local research poker table")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--players", type=int, default=6, help="seats at the table (2-10)")
    parser.add_argument('--autoplay', action='store_true',
                        help='Let the model play your seat automatically from the first hand')
    parser.add_argument('--policy', choices=sorted(POLICIES), default='contextual',
                        help='Strategy for Auto-play my seat (default: contextual)')
    parser.add_argument('--opponents', choices=('bots', 'ai'), default='bots')
    parser.add_argument('--ai-max-calls', type=int, default=100, help='Paid opponent API request cap')
    args = parser.parse_args()
    if not MIN_PLAYERS <= args.players <= MAX_PLAYERS:
        parser.error("--players must be between %d and %d" % (MIN_PLAYERS, MAX_PLAYERS))
    try:
        Handler.game = TableGame(players=args.players, seed=args.seed, opponent_mode=args.opponents,
                                 ai_max_calls=args.ai_max_calls,
                                 session_dir=os.path.join(os.getcwd(), 'sessions'),
                                 autoplay=args.autoplay, policy=args.policy)
    except ValueError as error:
        parser.error(str(error))
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as error:
        parser.exit(1, "Cannot listen on port %d (%s). If a table display is already running, "
                       "open http://127.0.0.1:%d/ or pass --port.\n" % (args.port, error.strerror, args.port))
    stop = threading.Event()
    threading.Thread(target=run_game_loop, args=(Handler.game, stop), daemon=True).start()
    print("Open http://127.0.0.1:%d/ in a browser (Ctrl-C to stop)" % args.port)
    if args.autoplay:
        print('Auto-play is ON (%s). Uncheck Auto-play my seat to pause.' % args.policy)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nTable closed")
    finally:
        stop.set()
        Handler.game.close()
        server.server_close()


if __name__ == "__main__":
    main()

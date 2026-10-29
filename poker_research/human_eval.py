"""Local human-versus-policy benchmark for the controlled simulator."""

import argparse
import json
import random
import secrets
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List
from urllib.parse import parse_qs, urlparse

from .agent import VisionPolicyAgent
from .cards import Card, deck, equity
from .simulator import render_image_frame

TRIALS_PER_SESSION = 20

HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Poker Research Human Benchmark</title>
<style>
body{font:16px system-ui;background:#10251d;color:#eef7f1;max-width:900px;margin:24px auto;padding:0 18px}
.table{background:#176b49;border-radius:50%;padding:42px 30px;margin:20px 0;text-align:center;box-shadow:0 8px 30px #0006}
.cards{font-size:30px;letter-spacing:10px;background:#fff;color:#15251e;padding:15px;border-radius:8px;display:inline-block}
.meta{display:flex;gap:18px;justify-content:center;flex-wrap:wrap}.meta span{background:#0d3e2b;padding:9px 12px;border-radius:6px}
button{font-size:18px;margin:6px;padding:12px 20px;border:0;border-radius:7px;cursor:pointer;background:#f1c75b;color:#1c271e}
button:disabled{opacity:.4}.muted{color:#b7c9bf}.error{color:#ffb3a9}.hidden{display:none}
</style></head><body>
<h1>Controlled Poker Research Benchmark</h1>
<p class="muted">Local simulator only. Your response time starts when the state appears.</p>
<div id="status">Loading…</div><div id="table" class="table hidden">
<div id="street"></div><p>Board</p><div id="board" class="cards"></div>
<p>Your hand</p><div id="hero" class="cards"></div>
<div class="meta"><span>Pot: <b id="pot"></b></span><span>To call: <b id="call"></b></span><span>Stack: <b id="stack"></b></span><span>Players: <b id="players"></b></span></div>
</div><div id="actions" class="hidden"></div><pre id="result"></pre>
<script>
const params=new URLSearchParams(location.search); const sid=params.get('session'); let started=0, locked=false;
const $=id=>document.getElementById(id);
function showState(s){locked=false; started=performance.now(); $('table').classList.remove('hidden'); $('actions').classList.remove('hidden');
 $('street').textContent=s.street.toUpperCase(); $('board').textContent=s.board.length?s.board.join(' '):'—'; $('hero').textContent=s.hero.join(' ');
 $('pot').textContent=s.pot; $('call').textContent=s.to_call; $('stack').textContent=s.stack; $('players').textContent=s.active_players;
 $('actions').innerHTML=''; s.legal_actions.forEach(a=>{const b=document.createElement('button');b.textContent=a; b.onclick=()=>submit(a); $('actions').appendChild(b)});}
async function load(){const r=await fetch('/api/state?session='+sid); const d=await r.json(); if(d.done){finish(d);return} showState(d);}
async function submit(action){if(locked)return;locked=true; const response_ms=performance.now()-started; const r=await fetch('/api/action',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({session:sid,action,response_ms})}); const d=await r.json(); if(d.done)finish(d); else showState(d);}
function finish(d){$('table').classList.add('hidden');$('actions').classList.add('hidden');$('status').textContent='Session complete';$('result').textContent=JSON.stringify(d,null,2);}
load();
</script></body></html>"""


def _card_text(cards: List[Card]) -> List[str]:
    return [str(card) for card in cards]


def _action_ev(state: Dict, action: str, amount: int, sample_seed: int) -> Dict:
    """Compute a deterministic counterfactual EV proxy for one visible state."""
    hero = state["hero"]
    board = state["board"]
    opponents = max(1, state["active_players"] - 1)
    hand_equity = equity(hero, board, opponents, random.Random(sample_seed), samples=120)
    pot = state["pot"]
    to_call = state["to_call"]
    risk = to_call if action == "call" else (amount if action in {"bet", "raise"} else 0)
    if action == "fold":
        ev = 0.0
    elif action == "check":
        ev = hand_equity * pot
    else:
        ev = hand_equity * (pot + risk) - risk
    return {"equity": round(hand_equity, 6), "risk": risk, "ev_proxy": round(ev, 4)}


class HumanBenchmark:
    def __init__(self, seed: int, output: str = "human-results.jsonl"):
        self.rng = random.Random(seed)
        self.agent = VisionPolicyAgent(seed + 1)
        self.sessions: Dict[str, Dict] = {}
        self.output = Path(output)

    def new_session(self) -> str:
        session_id = secrets.token_urlsafe(12)
        self.sessions[session_id] = {"trial": 0, "records": [], "state": None, "started": 0.0}
        self._next_state(session_id)
        return session_id

    def _next_state(self, session_id: str) -> None:
        session = self.sessions[session_id]
        cards = deck()
        self.rng.shuffle(cards)
        board_count = self.rng.choice([0, 3, 4, 5])
        state = {
            "hero": cards[:2], "board": cards[2:2 + board_count],
            "active_players": self.rng.randint(2, 10), "pot": self.rng.randint(20, 300),
            "to_call": self.rng.randint(0, 80), "min_raise": 10,
            "stack": self.rng.randint(200, 1500),
            "street": ["preflop", "flop", "turn", "river"][[0, 3, 4, 5].index(board_count)],
        }
        image = render_image_frame(state["hero"], state["board"], state["active_players"],
                                   state["pot"], state["to_call"], state["min_raise"], state["stack"], state["street"])
        ai = self.agent.decide(image)
        state["ai_action"] = ai.action
        state["ai_amount"] = ai.amount
        state["ai_equity"] = round(ai.equity, 6)
        state["ai_bluff_probability"] = round(ai.bluff_probability, 6)
        state["ai_latency_ms"] = round(ai.latency_ms, 3)
        state["ev_seed"] = self.rng.randint(0, 2**31 - 1)
        state["legal_actions"] = ["fold", "check"] if state["to_call"] == 0 else ["fold", "call"]
        state["legal_actions"] += ["bet"] if state["to_call"] == 0 else ["raise"]
        session["state"] = state
        session["started"] = time.perf_counter()

    def public_state(self, session_id: str) -> Dict:
        session = self.sessions[session_id]
        if session["trial"] >= TRIALS_PER_SESSION:
            return self.results(session_id)
        state = session["state"]
        return {key: (value if key not in {"hero", "board"} else _card_text(value))
                for key, value in state.items() if key != "ai_action"}

    def action(self, session_id: str, action: str, response_ms: float) -> Dict:
        session = self.sessions[session_id]
        state = session["state"]
        if action not in state["legal_actions"]:
            raise ValueError("illegal action")
        human_amount = state["to_call"] if action == "call" else (
            max(state["min_raise"], int(state["pot"] * 0.5)) if action in {"bet", "raise"} else 0)
        human_ev = _action_ev(state, action, human_amount, state["ev_seed"])
        ai_ev = _action_ev(state, state["ai_action"], state["ai_amount"], state["ev_seed"])
        session["records"].append({
            "trial": session["trial"] + 1,
            "state": {"hero": _card_text(state["hero"]), "board": _card_text(state["board"]),
                      "active_players": state["active_players"], "pot": state["pot"],
                      "to_call": state["to_call"], "stack": state["stack"], "street": state["street"]},
            "human_action": action,
            "ai_action": state["ai_action"],
            "ai_amount": state["ai_amount"],
            "ai_equity": state["ai_equity"],
            "ai_bluff_probability": state["ai_bluff_probability"],
            "ai_latency_ms": state["ai_latency_ms"],
            "human_ev": human_ev,
            "ai_ev": ai_ev,
            "ev_delta_ai_minus_human": round(ai_ev["ev_proxy"] - human_ev["ev_proxy"], 4),
            "agreement": action == state["ai_action"],
            "response_ms": round(float(response_ms), 3),
        })
        session["trial"] += 1
        if session["trial"] >= TRIALS_PER_SESSION:
            return self.results(session_id)
        self._next_state(session_id)
        return self.public_state(session_id)

    def results(self, session_id: str) -> Dict:
        session = self.sessions[session_id]
        records = session["records"]
        latency = [record["response_ms"] for record in records]
        ev_deltas = [record["ev_delta_ai_minus_human"] for record in records]
        result = {"done": True, "trials": len(records),
                  "agreement_rate": round(sum(r["agreement"] for r in records) / max(1, len(records)), 3),
                  "median_response_ms": sorted(latency)[len(latency) // 2] if latency else 0,
                  "median_ai_latency_ms": sorted(r["ai_latency_ms"] for r in records)[len(records) // 2] if records else 0,
                  "mean_ev_delta_ai_minus_human": round(sum(ev_deltas) / max(1, len(ev_deltas)), 4),
                  "records": records}
        if not session.get("persisted"):
            self.output.parent.mkdir(parents=True, exist_ok=True)
            with self.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, sort_keys=True) + "\n")
            session["persisted"] = True
        return result


class Handler(BaseHTTPRequestHandler):
    benchmark = HumanBenchmark(7)

    def _json(self, payload: Dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            session = parse_qs(parsed.query).get("session", [""])[0]
            if not session:
                session = self.benchmark.new_session()
                self.send_response(302)
                self.send_header("Location", "/?session=" + session)
                self.end_headers()
                return
            body = HTML
            body = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/api/state":
            session = parse_qs(parsed.query).get("session", [""])[0]
            self._json(self.benchmark.public_state(session))
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if self.path != "/api/action":
            self._json({"error": "not found"}, 404)
            return
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            result = self.benchmark.action(payload["session"], payload["action"], payload["response_ms"])
            self._json(result)
        except (KeyError, ValueError):
            self._json({"error": "invalid action request"}, 400)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local human benchmark")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="human-results.jsonl")
    args = parser.parse_args()
    Handler.benchmark = HumanBenchmark(args.seed, args.output)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("Open http://127.0.0.1:%d/ in a browser" % args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()

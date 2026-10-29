"""Real-time table controller: deals hands, paces opponents and takes the hero's clicks."""

import io
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Callable, Optional, Tuple

from .agent import VisionPolicyAgent
from .ai_opponents import AIOpponentTable
from .engine import BIG_BLIND, STARTING_STACK, HandState, Seat, apply_action, legal_actions, start_hand
from .opponents import opponent_action
from .session_log import SessionLog, street_reached
from .table_layout import HERO_ONLY_BUTTONS, SLIDER_TRACK
from .table_render import OPPONENT_NAMES, render_table_image
from .table_view import build_view, decision_state, strategy_context

HERO = 0
MIN_PLAYERS, MAX_PLAYERS = 2, 10
ACTION_DELAY_S = 0.9
RESULT_PAUSE_S = 3.5
AUTOPLAY_DELAY_S = 1.0
TIMEBANK_S = 25.0
TIMER_STEPS = 50
POT_FRACTIONS = {"half": 0.5, "three_quarter": 0.75, "pot": 1.0}
LOG_LINES = 16


def _note_fields(note) -> dict:
    """Equity and reason from a policy decision or an auto-play payload."""
    if note is None:
        return {}
    if hasattr(note, 'equity'):
        return {
            'equity': note.equity,
            'bluff_probability': note.bluff_probability,
            'reason': note.reason or '',
            'recommended_action': note.action,
        }
    if not isinstance(note, dict):
        return {}
    fields = {}
    if note.get('equity') is not None:
        fields['equity'] = note['equity']
    if note.get('bluff_probability') is not None:
        fields['bluff_probability'] = note['bluff_probability']
    if isinstance(note.get('reason'), str):
        fields['reason'] = note['reason']
    return fields


def resize_seats(seats: Tuple[Seat, ...], count: int) -> Tuple[Seat, ...]:
    """Add or remove seats from the end (never the hero) and rebuy anyone who busted."""
    kept = seats[:count] + tuple(Seat(OPPONENT_NAMES[i]) for i in range(len(seats), count))
    return tuple(replace(seat, stack=STARTING_STACK) if seat.stack == 0 else seat for seat in kept)


class TableGame:
    def __init__(self, players: int = 6, seed: int = 7, clock: Callable[[], float] = time.monotonic,
                 opponent_mode: str = 'bots', ai_max_calls: int = 100, opponent_controller=None,
                 session_dir: Optional[str] = None, autoplay: bool = False,
                 policy: str = 'contextual'):
        if not MIN_PLAYERS <= players <= MAX_PLAYERS:
            raise ValueError("players must be between %d and %d" % (MIN_PLAYERS, MAX_PLAYERS))
        self.rng = random.Random(seed)
        self.agent = VisionPolicyAgent(seed + 1, policy=policy)
        if opponent_mode not in ('bots', 'ai'):
            raise ValueError('Opponent mode must be bots or ai')
        self.ai_opponents = opponent_controller or (AIOpponentTable.from_env(ai_max_calls) if opponent_mode == 'ai' else None)
        self.opponent_error = None
        self._opponent_executor = ThreadPoolExecutor(max_workers=1) if self.ai_opponents else None
        self._opponent_future = None
        self._opponent_state = None
        self._starting_stacks = ()
        self._closed = False
        self.clock = clock
        self.lock = threading.Lock()
        self.seats = tuple(Seat(OPPONENT_NAMES[i]) for i in range(players))
        self.player_count = players
        self.autoplay = autoplay
        self.state: Optional[HandState] = None
        self.dealer = players - 1
        self.hand_number = 0
        self.next_event = 0.0
        self.turn_started = 0.0
        self.raise_to = 0
        self.version = 0
        self.history: Tuple[str, ...] = ()
        self._png_cache: Tuple[Optional[str], bytes] = (None, b"")
        self.session = SessionLog(session_dir)
        self._hero_spots: list = []
        self._archived_hand = 0

    # -- game flow -------------------------------------------------------
    def tick(self) -> None:
        with self.lock:
            if self._closed or self.opponent_error:
                return
            now = self.clock()
            if self.state is None or self.state.complete:
                if now >= self.next_event:
                    self._start_hand(now)
                return
            if now < self.next_event:
                return
            if self.state.to_act != HERO:
                if not self.ai_opponents:
                    self._apply(*opponent_action(self.state, self.rng), now)
                elif self._opponent_future is None:
                    self._opponent_state = self.state
                    self._opponent_future = self._opponent_executor.submit(self.ai_opponents.choose, self.state)
                elif self._opponent_future.done():
                    future, self._opponent_future = self._opponent_future, None
                    try:
                        action = future.result()
                        if self.state is self._opponent_state:
                            self._apply(*action, now)
                    except Exception as error:
                        self.opponent_error = str(error)
                        self.version += 1
            elif self.autoplay:
                action, raise_to, decision = self._policy_action()
                self._apply(action, raise_to, now, note=decision)
            elif now - self.turn_started >= TIMEBANK_S:
                self._apply("check", 0, now)  # out of time: check if possible, otherwise fold

    def _start_hand(self, now: float) -> None:
        if self.state is not None:
            finished = ("--- Hand #%d ---" % self.hand_number,) + self.state.log
            self.history = (self.history + finished)[-LOG_LINES:]
            self.seats = self.state.seats
        self.seats = resize_seats(self.seats, self.player_count)
        self._starting_stacks = tuple(seat.stack for seat in self.seats)
        self.dealer = (self.dealer + 1) % len(self.seats)
        self.hand_number += 1
        self.state = start_hand(self.seats, self.dealer, self.rng)
        self.dealer = self.state.dealer
        self._hero_spots = []
        self._after_change(now)

    def _apply(self, action: str, raise_to: int, now: float, note=None) -> None:
        hero_turn = self.state is not None and self.state.to_act == HERO
        spot = self._hero_spot(action, raise_to, note) if hero_turn else None
        self.state = apply_action(self.state, action, raise_to)
        if spot is not None:
            spot['played'] = self.state.log[-1] if self.state.log else ''
            self._hero_spots.append(spot)
        if self.state.complete and self.ai_opponents:
            self.ai_opponents.finish_hand(self.state, self._starting_stacks)
        if self.state.complete:
            self._archive_hand()
        self._after_change(now)

    def _hero_spot(self, action: str, raise_to: int, note) -> dict:
        """How the hero acted, plus the model's equity and reason when we have them."""
        state = self.state
        legal = legal_actions(state)
        hero = state.seats[HERO]
        if action in ('bet', 'raise', 'allin'):
            added = max(0, raise_to - hero.bet)
        elif action == 'call':
            added = legal.to_call
        else:
            added = 0
        spot = {
            'street': state.street,
            'action': action,
            'chips_added': added,
            'position': strategy_context(state, HERO).get('hero_position', 'unknown'),
            'pot': state.pot,
            'to_call': legal.to_call,
            'stack': hero.stack,
            'equity': None,
            'bluff_probability': None,
            'reason': '',
        }
        spot.update(_note_fields(note))
        return spot

    def _archive_hand(self) -> None:
        if self._archived_hand == self.hand_number or self.state is None or not self._starting_stacks:
            return
        state = self.state
        start = self._starting_stacks[HERO]
        end = state.seats[HERO].stack
        context = strategy_context(state, HERO)
        self.session.add_table_hand({
            'hand_number': self.hand_number,
            'hero': [str(card) for card in state.seats[HERO].hand],
            'position': context.get('hero_position', 'unknown'),
            'board': [str(card) for card in state.board],
            'street': street_reached(state.board),
            'showdown': bool(state.showdown),
            'hero_folded': bool(state.seats[HERO].folded),
            'starting_stack': start,
            'ending_stack': end,
            'profit_chips': end - start,
            'profit_bb': (end - start) / BIG_BLIND,
            'pot': state.pot,
            'big_blind': BIG_BLIND,
            'players': len(state.seats),
            'hero_actions': list(self._hero_spots),
            'actions': list(context.get('betting_history') or []),
            'awards': [{
                'name': state.seats[award.seat].name,
                'amount': award.amount,
                'hand': award.hand_name,
                'hero': award.seat == HERO,
            } for award in state.awards],
            'log': list(state.log),
        })
        self._hero_spots = []
        self._archived_hand = self.hand_number

    def observe(self, payload: dict) -> bool:
        """Record one overlay read. It does not change the dealt hand or the chip total."""
        with self.lock:
            return self.session.observe(payload)

    def session_snapshot(self) -> dict:
        with self.lock:
            data = self.session.as_dict()
            if self.state is not None and not self.state.complete:
                data['current_hand'] = {
                    'hand_number': self.hand_number,
                    'hero': [str(card) for card in self.state.seats[HERO].hand],
                    'board': [str(card) for card in self.state.board],
                    'hero_actions': list(self._hero_spots),
                    'log': list(self.state.log),
                    'in_progress': True,
                }
            return data

    def session_line(self) -> str:
        with self.lock:
            return self.session.line()

    def close(self):
        with self.lock:
            self._closed = True
            if self._opponent_executor:
                self._opponent_executor.shutdown(wait=False, cancel_futures=True)

    def _after_change(self, now: float) -> None:
        self.version += 1
        if self.state.complete:
            self.next_event = now + RESULT_PAUSE_S
        elif self.state.to_act == HERO:
            self.turn_started = now
            self.raise_to = legal_actions(self.state).min_raise_to
            self.next_event = now + AUTOPLAY_DELAY_S
        else:
            self.next_event = now + ACTION_DELAY_S

    def _policy_action(self):
        decision = self.agent.decide(json.dumps(decision_state(build_view(self.state, HERO))))
        return decision.action, self.state.seats[HERO].bet + decision.amount, decision

    def act(self, action: str, amount: int = 0, note=None) -> str:
        """Play one hero decision. ``amount`` is whole chips to add, not a raise-to total."""
        with self.lock:
            if self._closed or self.state is None or self.state.complete or self.state.to_act != HERO:
                raise ValueError('Not your turn on the research table.')
            action = str(action).lower()
            if action not in {'fold', 'check', 'call', 'bet', 'raise'}:
                raise ValueError('Unsupported action.')
            legal = legal_actions(self.state)
            if action in ('bet', 'raise'):
                if not legal.can_raise:
                    raise ValueError('A raise is not available.')
                if amount <= 0:
                    raise ValueError('The decision has no chips to add, so it was not played.')
                raise_to = self.state.seats[HERO].bet + amount
            else:
                raise_to = 0
            self._apply(action, raise_to, self.clock(), note=note)
            return self.state.log[-1] if self.state.log else 'Action played.'

    # -- hero input ------------------------------------------------------
    def click(self, target: Optional[str], x: int = 0) -> None:
        with self.lock:
            if target in ("players_minus", "players_plus"):
                step = -1 if target == "players_minus" else 1
                self.player_count = max(MIN_PLAYERS, min(MAX_PLAYERS, self.player_count + step))
                self.version += 1
            elif target == "autoplay":
                self.autoplay = not self.autoplay
                self.version += 1
            elif target in HERO_ONLY_BUTTONS and self.state is not None and self.state.to_act == HERO:
                self._hero_click(target, x)

    def _hero_click(self, target: str, x: int) -> None:
        legal = legal_actions(self.state)
        now = self.clock()
        if target in ("fold", "call", "raise"):
            self._apply(target, self.raise_to, now)
            return
        if target in ("minus", "plus"):
            chosen = self.raise_to + (BIG_BLIND if target == "plus" else -BIG_BLIND)
        elif target == "slider":
            left, right = SLIDER_TRACK
            fraction = max(0.0, min(1.0, (x - left) / (right - left)))
            chosen = legal.min_raise_to + round(fraction * (legal.max_raise_to - legal.min_raise_to))
        elif target == "allin":
            chosen = legal.max_raise_to
        elif target in POT_FRACTIONS:
            chosen = self.state.current_bet + int((self.state.pot + legal.to_call) * POT_FRACTIONS[target])
        else:
            return
        self.raise_to = max(legal.min_raise_to, min(chosen, legal.max_raise_to))
        self.version += 1

    # -- presentation ----------------------------------------------------
    def _timer(self) -> Optional[float]:
        if self.state is None or self.state.to_act != HERO or self.autoplay:
            return None
        remaining = 1.0 - (self.clock() - self.turn_started) / TIMEBANK_S
        return round(max(0.0, remaining) * TIMER_STEPS) / TIMER_STEPS

    def frame_key(self) -> str:
        with self.lock:
            return "%d-%s" % (self.version, self._timer())

    def view(self) -> Optional[dict]:
        with self.lock:
            if self.state is None:
                return None
            return build_view(self.state, HERO, raise_to=self.raise_to, timer=self._timer(),
                              player_count=self.player_count, autoplay=self.autoplay,
                              hand_number=self.hand_number)

    def frame_png(self) -> Tuple[str, bytes]:
        key = self.frame_key()
        if self._png_cache[0] == key:
            return self._png_cache
        view = self.view()
        if view is None:
            raise LookupError("no hand has been dealt yet")
        buffer = io.BytesIO()
        render_table_image(view).save(buffer, format="PNG")
        self._png_cache = (key, buffer.getvalue())
        return self._png_cache

    def log(self) -> Tuple[str, ...]:
        with self.lock:
            current = self.state.log if self.state is not None else ()
            error = ('AI opponents paused: ' + self.opponent_error,) if self.opponent_error else ()
            return (self.history + ("--- Hand #%d ---" % self.hand_number,) + current + error)[-LOG_LINES:]

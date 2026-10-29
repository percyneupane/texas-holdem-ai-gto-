"""Session hand history for the research table and the overlay.

Table hands store the dealt cards, every action, and the hero's chip result.
Overlay reads store how a hand was played when the cards come from a screenshot.
Those reads are not added to the chip total. Nothing here is a real-money edge.
"""

import json
import math
import os
import secrets
import threading
import time
from typing import Optional

from .cards import RANKS
from .engine import BIG_BLIND

MAX_TABLE_HANDS = 2000
MAX_OBSERVED_HANDS = 2000
MAX_DECISIONS = 30
MAX_REASON = 1200
MAX_TEXT = 32
_SUITS = 'cdhs'
_ACTIONS = {'fold', 'check', 'call', 'bet', 'raise'}
NOTE = (
    'Research-table chip results for this session. They describe the recorded hands '
    'and are not a real-money profit edge. Overlay reads record the cards, the decision, '
    'and the reason. They are not added into the chip total.'
)


def street_reached(board) -> str:
    return {0: 'preflop', 3: 'flop', 4: 'turn', 5: 'river'}.get(len(board), 'river' if len(board) > 5 else 'preflop')


def card_label(card) -> str:
    """Accept ``As`` or ``[14, "s"]`` and return the two-character card."""
    if isinstance(card, str) and len(card) == 2 and card[0] in RANKS and card[1] in _SUITS:
        return card
    if isinstance(card, (list, tuple)) and len(card) == 2:
        rank, suit = card
        if isinstance(rank, int) and not isinstance(rank, bool) and 2 <= rank <= 14 and suit in _SUITS:
            return RANKS[rank - 2] + suit
    raise ValueError('Card must look like As or [14, "s"].')


def _number(value, name: str, low: float = 0.0, high: Optional[float] = None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('%s must be a number.' % name)
    if value < low or (high is not None and value > high):
        raise ValueError('%s is out of range.' % name)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _text(value, name: str, limit: int = MAX_TEXT) -> str:
    if not isinstance(value, str):
        raise ValueError('%s must be text.' % name)
    if len(value) > limit:
        raise ValueError('%s is too long.' % name)
    return value


def observation_from_decision(decision) -> dict:
    """The overlay fields worth keeping for one hero decision."""
    state = decision.state or {}
    return {
        'hero': state.get('hero') or [],
        'board': state.get('board') or [],
        'street': state.get('street') or '',
        'action': decision.action,
        'amount': decision.amount,
        'equity': decision.equity,
        'bluff_probability': decision.bluff_probability,
        'reason': decision.reason or '',
        'position': state.get('hero_position') or '',
        'pot': state.get('pot'),
        'to_call': state.get('to_call'),
        'stack': state.get('stack'),
        'big_blind': state.get('big_blind'),
        'active_players': state.get('active_players'),
        'preflop_action': state.get('preflop_action') or '',
        'postflop_position': state.get('postflop_position') or '',
    }


def _public_path(path: Optional[str]):
    if not path:
        return None
    try:
        relative = os.path.relpath(path, os.getcwd())
    except ValueError:
        return os.path.basename(path)
    if relative.startswith('..'):
        return os.path.basename(path)
    return relative


class SessionLog:
    """In-memory session, with one JSON line per event when a directory is set."""

    def __init__(self, directory: Optional[str] = None, prefix: str = 'table'):
        self.session_id = time.strftime('%Y%m%d-%H%M%S') + '-' + secrets.token_hex(2)
        self.started_at = time.strftime('%Y-%m-%dT%H:%M:%S')
        self.path = None
        self.write_error = ''
        if directory:
            os.makedirs(directory, exist_ok=True)
            self.path = os.path.join(directory, '%s-%s.jsonl' % (prefix, self.session_id))
        self._lock = threading.Lock()
        self._table_hands = []
        self._table_count = 0
        self._profit_chips = 0
        self._showdowns = 0
        self._won = 0
        self._lost = 0
        self._observed = []
        self._open = None
        self._observed_count = 0
        self._observed_decisions = 0

    def add_table_hand(self, hand: dict) -> None:
        profit = int(hand['profit_chips'])
        with self._lock:
            self._table_count += 1
            self._profit_chips += profit
            self._showdowns += int(bool(hand.get('showdown')))
            self._won += int(profit > 0)
            self._lost += int(profit < 0)
            self._table_hands.append(hand)
            if len(self._table_hands) > MAX_TABLE_HANDS:
                del self._table_hands[:-MAX_TABLE_HANDS]
            self._write({'kind': 'table_hand', **hand})

    def observe(self, raw: dict) -> bool:
        """Group one overlay decision into a hand. A repeated spot is ignored."""
        decision = _clean_observation(raw)
        with self._lock:
            opened = self._open
            if opened is None or opened['hero'] != decision['hero']:
                if opened is not None:
                    opened['in_progress'] = False
                    self._observed.append(opened)
                    if len(self._observed) > MAX_OBSERVED_HANDS:
                        del self._observed[:-MAX_OBSERVED_HANDS]
                self._observed_count += 1
                opened = {
                    'hand_number': self._observed_count,
                    'hero': decision['hero'],
                    'position': decision['position'],
                    'board': decision['board'],
                    'big_blind': decision['big_blind'],
                    'stack_first': decision['stack'],
                    'stack_last': decision['stack'],
                    'decisions': [],
                    'in_progress': True,
                }
                self._open = opened
            last = opened['decisions'][-1] if opened['decisions'] else None
            same = last and (last['street'], last['board'], last['action'], last['amount'], last['pot']) == (
                decision['street'], decision['board'], decision['action'], decision['amount'], decision['pot'])
            if same or len(opened['decisions']) >= MAX_DECISIONS:
                return False
            opened['decisions'].append(decision)
            opened['board'] = decision['board']
            if decision['position']:
                opened['position'] = decision['position']
            if decision['stack'] is not None:
                opened['stack_last'] = decision['stack']
            if opened['big_blind'] is None:
                opened['big_blind'] = decision['big_blind']
            self._observed_decisions += 1
            self._write({'kind': 'observed', 'hand_number': opened['hand_number'], **decision})
            return True

    def totals(self) -> dict:
        with self._lock:
            hands = self._table_count
            profit_bb = self._profit_chips / BIG_BLIND
            return {
                'hands': hands,
                'profit_chips': self._profit_chips,
                'profit_bb': profit_bb,
                'bb_per_100': (100.0 * profit_bb / hands) if hands else None,
                'showdowns': self._showdowns,
                'won_hands': self._won,
                'lost_hands': self._lost,
                'showdown_rate': (self._showdowns / hands) if hands else 0.0,
                'observed_hands': self._observed_count,
                'observed_decisions': self._observed_decisions,
                'saved_to': _public_path(self.path),
                'note': NOTE,
            }

    def line(self) -> str:
        totals = self.totals()
        if not totals['hands'] and not totals['observed_decisions']:
            return 'No completed hands yet. Finished hands show on the session page.'
        parts = []
        if totals['hands']:
            parts.append('Research table · %d %s · %+d chips · %+.1f bb · %+.1f bb/100' % (
                totals['hands'], 'hand' if totals['hands'] == 1 else 'hands',
                totals['profit_chips'], totals['profit_bb'], totals['bb_per_100']))
        if totals['observed_decisions']:
            parts.append('Overlay · %d %s in %d %s' % (
                totals['observed_decisions'], 'read' if totals['observed_decisions'] == 1 else 'reads',
                totals['observed_hands'], 'hand' if totals['observed_hands'] == 1 else 'hands'))
        return ' · '.join(parts) + '. Open the session page.'

    def as_dict(self) -> dict:
        with self._lock:
            observed = [dict(hand, decisions=list(hand['decisions'])) for hand in self._observed]
            if self._open is not None:
                observed.append(dict(self._open, decisions=list(self._open['decisions'])))
            table_hands = list(self._table_hands)
        totals = self.totals()
        return {
            'session_id': self.session_id,
            'started_at': self.started_at,
            'saved_to': totals['saved_to'],
            'note': NOTE,
            'table': {key: totals[key] for key in (
                'hands', 'profit_chips', 'profit_bb', 'bb_per_100', 'showdowns',
                'won_hands', 'lost_hands', 'showdown_rate')},
            'observed': {
                'hands': totals['observed_hands'],
                'decisions': totals['observed_decisions'],
            },
            'table_hands': table_hands,
            'observed_hands': observed,
            'write_error': self.write_error,
        }

    def _write(self, record: dict) -> None:
        if not self.path:
            return
        try:
            with open(self.path, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        except OSError as error:
            self.write_error = str(error)


def _cards(value, name: str, exact=None):
    if not isinstance(value, list) or (exact is not None and len(value) != exact) or len(value) > 5:
        raise ValueError('%s has the wrong number of cards.' % name)
    return [card_label(card) for card in value]


def _clean_observation(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError('Observation must be an object.')
    action = raw.get('action')
    if not isinstance(action, str) or action.lower() not in _ACTIONS:
        raise ValueError('Observation action must be fold, check, call, bet or raise.')
    street = raw.get('street') or ''
    if street:
        street = _text(street, 'Street', 16)
    board = _cards(raw.get('board') or [], 'Board')
    if not street:
        street = street_reached(board)
    equity = raw.get('equity')
    bluff = raw.get('bluff_probability')
    cleaned = {
        'hero': _cards(raw.get('hero'), 'Hero', exact=2),
        'board': board,
        'street': street,
        'action': action.lower(),
        'amount': _number(raw.get('amount', 0), 'Amount'),
        'equity': None if equity is None else _number(equity, 'Equity', 0, 1),
        'bluff_probability': None if bluff is None else _number(bluff, 'Bluff probability', 0, 1),
        'reason': _text(raw.get('reason') or '', 'Reason', MAX_REASON),
        'position': _text(raw.get('position') or '', 'Position', 16),
        'pot': _optional_number(raw.get('pot'), 'Pot'),
        'to_call': _optional_number(raw.get('to_call'), 'To call'),
        'stack': _optional_number(raw.get('stack'), 'Stack'),
        'big_blind': _optional_number(raw.get('big_blind') or None, 'Big blind', low=0.0001),
        'active_players': _optional_count(raw.get('active_players')),
        'preflop_action': _text(raw.get('preflop_action') or '', 'Preflop action', 16),
        'postflop_position': _text(raw.get('postflop_position') or '', 'Postflop position', 8),
    }
    return cleaned


def _optional_number(value, name: str, low: float = 0.0):
    if value is None:
        return None
    return _number(value, name, low)


def _optional_count(value):
    if value is None:
        return None
    count = _number(value, 'Players', 2, 10)
    if isinstance(count, float):
        raise ValueError('Players must be a whole number.')
    return count

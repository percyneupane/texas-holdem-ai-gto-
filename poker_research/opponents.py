"""Rule-based opponents for the simulated table."""

import random
from typing import Sequence, Tuple

from .cards import Card, evaluate
from .engine import HandState, legal_actions

# Rough made-hand strength by category: high card .. straight flush.
CATEGORY_STRENGTH = (0.15, 0.45, 0.66, 0.76, 0.82, 0.86, 0.92, 0.97, 0.99)
BLUFF_RATE = 0.08
LOOSE_CALL_RATE = 0.12
BET_FRACTIONS = (0.5, 0.66, 0.75)
OPPONENT_STYLES = {
    'balanced': (BLUFF_RATE, LOOSE_CALL_RATE, .15, .7),
    'tight': (.02, .02, .25, .8),
    'calling_station': (.01, .60, -.05, .9),
    'aggressive': (.25, .20, .05, .6),
}


def hand_strength(hand: Sequence[Card], board: Sequence[Card]) -> float:
    """Cheap 0..1 strength estimate; fast enough to act instantly at the table."""
    high, low = sorted((card.rank for card in hand), reverse=True)
    if len(board) < 3:
        if high == low:
            return 0.5 + (high - 2) * 0.035
        suited = hand[0].suit == hand[1].suit
        connected = high - low == 1
        return min(0.75, (high + low - 4) / 24 * 0.55 + 0.06 * suited + 0.04 * connected)
    category, ranks = evaluate(list(hand) + list(board))
    strength = CATEGORY_STRENGTH[category] + ranks[0] / 14 * 0.04
    if category == 1 and ranks[0] not in (high, low):
        strength = 0.25  # the pair is on the board, not in our hand
    return min(1.0, strength)


def opponent_action(state: HandState, rng: random.Random, style: str = 'balanced') -> Tuple[str, int]:
    """Choose (action, raise_to) for the seat due to act."""
    seat = state.seats[state.to_act]
    legal = legal_actions(state)
    strength = hand_strength(seat.hand, state.board)
    bluff_rate, loose_call_rate, price_margin, raise_threshold = OPPONENT_STYLES[style]

    def sized(fraction: float) -> int:
        return state.current_bet + max(state.min_raise, int((state.pot + legal.to_call) * fraction))

    if legal.to_call == 0:
        if legal.can_raise and (rng.random() < strength - 0.35 or rng.random() < bluff_rate):
            return "bet", sized(rng.choice(BET_FRACTIONS))
        return "check", 0
    price = legal.to_call / (state.pot + legal.to_call)
    if legal.can_raise and strength > raise_threshold and rng.random() < strength - 0.45:
        return "raise", sized(0.75)
    if strength > price + price_margin or rng.random() < loose_call_rate:
        return "call", 0
    return "fold", 0

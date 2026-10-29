"""No-limit hold'em hand engine with immutable states.

Every transition returns a new ``HandState``. Seats are listed clockwise, chip
amounts are integers, and the engine handles blinds (including heads-up),
turn order, min-raises, all-ins, side pots, showdowns and uncontested pots.
"""

import random
from dataclasses import dataclass, replace
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from .cards import Card, deck, evaluate

STARTING_STACK = 1000
SMALL_BLIND = 5
BIG_BLIND = 10
COMPLETE = "complete"
NEXT_STREET = {"preflop": ("flop", 3), "flop": ("turn", 1), "turn": ("river", 1)}
RAISE_ACTIONS = {"bet", "raise", "allin"}
RANK_NAMES = {2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight",
              9: "Nine", 10: "Ten", 11: "Jack", 12: "Queen", 13: "King", 14: "Ace"}


@dataclass(frozen=True)
class Seat:
    name: str
    stack: int = STARTING_STACK
    hand: Tuple[Card, ...] = ()
    bet: int = 0
    committed: int = 0
    folded: bool = False
    sitting_out: bool = False
    last_action: str = ""

    @property
    def in_hand(self) -> bool:
        return bool(self.hand) and not self.folded

    @property
    def can_act(self) -> bool:
        return self.in_hand and self.stack > 0


@dataclass(frozen=True)
class Award:
    seat: int
    amount: int
    hand_name: str = ""


@dataclass(frozen=True)
class Legal:
    to_call: int
    min_raise_to: int
    max_raise_to: int
    can_raise: bool


@dataclass(frozen=True)
class HandState:
    seats: Tuple[Seat, ...]
    dealer: int
    deck: Tuple[Card, ...]
    board: Tuple[Card, ...] = ()
    street: str = "preflop"
    current_bet: int = 0
    min_raise: int = BIG_BLIND
    to_act: Optional[int] = None
    pending: FrozenSet[int] = frozenset()
    awards: Tuple[Award, ...] = ()
    showdown: bool = False
    log: Tuple[str, ...] = ()

    @property
    def pot(self) -> int:
        return sum(seat.committed for seat in self.seats)

    @property
    def complete(self) -> bool:
        return self.street == COMPLETE


def _plural(rank: int) -> str:
    name = RANK_NAMES[rank]
    return name + ("es" if name.endswith("x") else "s")


def describe_hand(score: Tuple[int, Tuple[int, ...]]) -> str:
    category, ranks = score
    top = RANK_NAMES[ranks[0]]
    descriptions = {
        0: "High card, %s" % top,
        1: "Pair of %s" % _plural(ranks[0]),
        2: "Two pair, %s and %s" % (_plural(ranks[0]), _plural(ranks[1]) if len(ranks) > 1 else ""),
        3: "Three of a kind, %s" % _plural(ranks[0]),
        4: "Straight, %s high" % top,
        5: "Flush, %s high" % top,
        6: "Full house, %s over %s" % (_plural(ranks[0]), _plural(ranks[1]) if len(ranks) > 1 else ""),
        7: "Four of a kind, %s" % _plural(ranks[0]),
        8: "Royal flush" if ranks[0] == 14 else "Straight flush, %s high" % top,
    }
    return descriptions[category]


def current_hand_label(hand: Sequence[Card], board: Sequence[Card]) -> str:
    """Name of the best hand so far; preflop only pairs and high cards exist."""
    if len(hand) != 2:
        return ""
    if len(board) >= 3:
        return describe_hand(evaluate(list(hand) + list(board)))
    high = max(card.rank for card in hand)
    if hand[0].rank == hand[1].rank:
        return describe_hand((1, (high,)))
    return describe_hand((0, (high,)))


def _clockwise(count: int, start: int, wanted: Iterable[int]) -> List[int]:
    allowed = set(wanted)
    return [(start + step) % count for step in range(1, count + 1) if (start + step) % count in allowed]


def _update(seats: Tuple[Seat, ...], index: int, **changes) -> Tuple[Seat, ...]:
    return seats[:index] + (replace(seats[index], **changes),) + seats[index + 1:]


def _pay(seats: Tuple[Seat, ...], index: int, amount: int, label: str) -> Tuple[Seat, ...]:
    seat = seats[index]
    paid = max(0, min(seat.stack, amount))
    remaining = seat.stack - paid
    return _update(seats, index, stack=remaining, bet=seat.bet + paid, committed=seat.committed + paid,
                   last_action="ALL-IN" if remaining == 0 else label)


def start_hand(seats: Sequence[Seat], dealer: int, rng: random.Random) -> HandState:
    """Shuffle, deal two cards to every seated player with chips, and post blinds."""
    count = len(seats)
    playing = [i for i, seat in enumerate(seats) if not seat.sitting_out and seat.stack > 0]
    if len(playing) < 2:
        raise ValueError("a hand needs at least two seated players with chips")
    if dealer not in playing:
        dealer = _clockwise(count, dealer, playing)[0]
    cards = deck()
    rng.shuffle(cards)
    hands = {index: tuple(cards[n * 2:n * 2 + 2]) for n, index in enumerate(playing)}
    fresh = tuple(replace(seat, hand=hands.get(i, ()), bet=0, committed=0, folded=False, last_action="")
                  for i, seat in enumerate(seats))
    small = dealer if len(playing) == 2 else _clockwise(count, dealer, playing)[0]
    big = _clockwise(count, small, playing)[0]
    fresh = _pay(_pay(fresh, small, SMALL_BLIND, "SB"), big, BIG_BLIND, "BB")
    state = HandState(
        seats=fresh, dealer=dealer, deck=tuple(cards[len(playing) * 2:]), current_bet=BIG_BLIND,
        pending=frozenset(i for i in playing if fresh[i].can_act),
        log=("%s is the dealer" % fresh[dealer].name,
             "%s posts small blind %d" % (fresh[small].name, fresh[small].bet),
             "%s posts big blind %d" % (fresh[big].name, fresh[big].bet)))
    return _continue(state, after=big)


def legal_actions(state: HandState) -> Legal:
    if state.to_act is None:
        raise ValueError("no player is due to act")
    seat = state.seats[state.to_act]
    to_call = min(max(0, state.current_bet - seat.bet), seat.stack)
    max_to = seat.bet + seat.stack
    others_can_act = any(other.can_act for i, other in enumerate(state.seats) if i != state.to_act)
    return Legal(to_call, min(state.current_bet + state.min_raise, max_to), max_to,
                 max_to > state.current_bet and others_can_act)


def _normalise(action: str, legal: Legal) -> str:
    action = action.lower()
    if action not in RAISE_ACTIONS | {"fold", "check", "call"}:
        raise ValueError("unknown action: %s" % action)
    if action in RAISE_ACTIONS and not legal.can_raise:
        action = "call"
    if action in {"call", "fold"} and legal.to_call == 0:
        return "check"
    if action == "check" and legal.to_call > 0:
        return "fold"
    return action


def apply_action(state: HandState, action: str, raise_to: int = 0) -> HandState:
    """Apply the acting player's move; ``raise_to`` is the street total for bets and raises."""
    legal = legal_actions(state)
    index = state.to_act
    seat = state.seats[index]
    action = _normalise(action, legal)
    pending = state.pending - {index}
    if action == "fold":
        updated = replace(state, seats=_update(state.seats, index, folded=True, last_action="FOLD"), pending=pending)
        entry = "%s folds" % seat.name
    elif action == "check":
        updated = replace(state, seats=_update(state.seats, index, last_action="CHECK"), pending=pending)
        entry = "%s checks" % seat.name
    elif action == "call":
        updated = replace(state, seats=_pay(state.seats, index, legal.to_call, "CALL"), pending=pending)
        entry = "%s calls %d" % (seat.name, legal.to_call)
    else:
        target = legal.max_raise_to if action == "allin" else max(legal.min_raise_to, min(raise_to, legal.max_raise_to))
        verb = "BET" if state.current_bet == 0 else "RAISE"
        seats = _pay(state.seats, index, target - seat.bet, verb)
        updated = replace(state, seats=seats, current_bet=target,
                          min_raise=max(state.min_raise, target - state.current_bet),
                          pending=frozenset(i for i, other in enumerate(seats) if other.can_act and i != index))
        entry = "%s %s to %d" % (seat.name, "bets" if verb == "BET" else "raises", target)
    return _continue(replace(updated, log=state.log + (entry,)), after=index)


def _continue(state: HandState, after: int) -> HandState:
    in_hand = [i for i, seat in enumerate(state.seats) if seat.in_hand]
    able = [i for i in in_hand if state.seats[i].can_act]
    nobody_to_play = len(able) <= 1 and all(state.seats[i].bet >= state.current_bet for i in state.pending)
    if len(in_hand) > 1 and state.pending and not nobody_to_play:
        return replace(state, to_act=_clockwise(len(state.seats), after, state.pending)[0])
    return _end_round(replace(state, to_act=None, pending=frozenset()))


def _end_round(state: HandState) -> HandState:
    in_hand = [i for i, seat in enumerate(state.seats) if seat.in_hand]
    if len(in_hand) == 1:
        return _finish(state, (Award(in_hand[0], state.pot),), showdown=False)
    if state.street == "river":
        return _showdown(state)
    street, count = NEXT_STREET[state.street]
    # New street: clear action tags except the ones that still describe the seat.
    seats = tuple(replace(seat, bet=0, last_action=seat.last_action if seat.folded or (seat.in_hand and seat.stack == 0)
                          else "") for seat in state.seats)
    board = state.board + state.deck[:count]
    advanced = replace(state, seats=seats, board=board, deck=state.deck[count:], street=street, current_bet=0,
                       min_raise=BIG_BLIND, pending=frozenset(i for i, seat in enumerate(seats) if seat.can_act),
                       log=state.log + ("%s: %s" % (street.title(), " ".join(str(card) for card in board)),))
    return _continue(advanced, after=state.dealer)


def side_pots(seats: Sequence[Seat]) -> List[Tuple[int, Tuple[int, ...]]]:
    """Split contributions into (amount, eligible seats) layers, main pot first."""
    pots: List[Tuple[int, Tuple[int, ...]]] = []
    previous = 0
    for level in sorted({seat.committed for seat in seats if seat.committed > 0}):
        amount = sum(min(seat.committed, level) - min(seat.committed, previous) for seat in seats)
        eligible = tuple(i for i, seat in enumerate(seats) if seat.in_hand and seat.committed >= level)
        if eligible:
            pots.append((amount, eligible))
        elif pots:
            # Chips a folded player put in above every live player go to the last live pot.
            pots[-1] = (pots[-1][0] + amount, pots[-1][1])
        previous = level
    return pots


def _showdown(state: HandState) -> HandState:
    count = len(state.seats)
    scores = {i: evaluate(list(seat.hand) + list(state.board)) for i, seat in enumerate(state.seats) if seat.in_hand}
    totals: Dict[int, int] = {}
    for amount, eligible in side_pots(state.seats):
        best = max(scores[i] for i in eligible)
        winners = [i for i in _clockwise(count, state.dealer, eligible) if scores[i] == best]
        share, remainder = divmod(amount, len(winners))
        for position, winner in enumerate(winners):
            totals[winner] = totals.get(winner, 0) + share + (1 if position < remainder else 0)
    awards = tuple(Award(i, totals[i], describe_hand(scores[i])) for i in sorted(totals))
    shown = tuple("%s shows %s (%s)" % (state.seats[i].name, " ".join(map(str, state.seats[i].hand)),
                                        describe_hand(scores[i])) for i in sorted(scores))
    return _finish(replace(state, log=state.log + shown), awards, showdown=True)


def _finish(state: HandState, awards: Tuple[Award, ...], showdown: bool) -> HandState:
    seats = state.seats
    for award in awards:
        seats = _update(seats, award.seat, stack=seats[award.seat].stack + award.amount)
    lines = tuple("%s wins %d%s" % (state.seats[a.seat].name, a.amount, " with %s" % a.hand_name if a.hand_name else "")
                  for a in awards)
    return replace(state, seats=seats, street=COMPLETE, to_act=None, pending=frozenset(), awards=awards,
                   showdown=showdown, log=state.log + lines)

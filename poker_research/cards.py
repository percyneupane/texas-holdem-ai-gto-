from dataclasses import dataclass
import random
from typing import Iterable, List, Sequence, Tuple

RANKS = "23456789TJQKA"
SUITS = "cdhs"


@dataclass(frozen=True)
class Card:
    rank: int
    suit: str

    def __str__(self) -> str:
        return RANKS[self.rank - 2] + self.suit


def deck() -> List[Card]:
    return [Card(rank, suit) for rank in range(2, 15) for suit in SUITS]


def evaluate_five(cards: Sequence[Card]) -> Tuple[int, Tuple[int, ...]]:
    counts = {}
    for card in cards:
        counts[card.rank] = counts.get(card.rank, 0) + 1
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    unique = sorted(counts)
    straight = 0
    if len(unique) == 5:
        if unique == [2, 3, 4, 5, 14]:
            straight = 5
        elif unique[-1] - unique[0] == 4:
            straight = unique[-1]
    flush = len({card.suit for card in cards}) == 1
    if straight and flush:
        return (8, (straight,))
    if groups[0][0] == 4:
        return (7, (groups[0][1], groups[1][1]))
    if groups[0][0] == 3 and groups[1][0] == 2:
        return (6, (groups[0][1], groups[1][1]))
    if flush:
        return (5, tuple(sorted((c.rank for c in cards), reverse=True)))
    if straight:
        return (4, (straight,))
    if groups[0][0] == 3:
        return (3, (groups[0][1],) + tuple(sorted((r for r, c in counts.items() if c == 1), reverse=True)))
    if groups[0][0] == 2 and groups[1][0] == 2:
        pairs = sorted((groups[0][1], groups[1][1]), reverse=True)
        return (2, tuple(pairs + [groups[2][1]]))
    if groups[0][0] == 2:
        return (1, (groups[0][1],) + tuple(sorted((r for r, c in counts.items() if c == 1), reverse=True)))
    return (0, tuple(sorted((c.rank for c in cards), reverse=True)))


def evaluate(cards: Iterable[Card]) -> Tuple[int, Tuple[int, ...]]:
    values = list(cards)
    best = None
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            for k in range(j + 1, len(values)):
                for l in range(k + 1, len(values)):
                    for m in range(l + 1, len(values)):
                        score = evaluate_five([values[i], values[j], values[k], values[l], values[m]])
                        best = score if best is None or score > best else best
    return best


def equity(hero: Sequence[Card], board: Sequence[Card], opponents: int,
           rng: random.Random, samples: int = 80) -> float:
    known = set(hero) | set(board)
    wins = ties = 0.0
    for _ in range(samples):
        remaining = [card for card in deck() if card not in known]
        rng.shuffle(remaining)
        opp_hands = [remaining[i * 2:i * 2 + 2] for i in range(opponents)]
        runout = list(board) + remaining[opponents * 2:opponents * 2 + (5 - len(board))]
        hero_score = evaluate(list(hero) + runout)
        scores = [evaluate(hand + runout) for hand in opp_hands]
        best = max([hero_score] + scores)
        if hero_score == best:
            if scores.count(best):
                ties += 1.0 / (scores.count(best) + 1)
            else:
                wins += 1
    return (wins + ties) / max(1, samples)

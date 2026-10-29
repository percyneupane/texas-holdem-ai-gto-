"""Fixed geometry of the simulated table, shared by the renderer, reader and click handling."""

from typing import Dict, Optional, Tuple

SIZE = (1000, 720)
TABLE_CENTER = (500, 285)
RAIL_BOX = (126, 86, 874, 474)
FELT_BOX = (146, 106, 854, 454)
BETTING_LINE_BOX = (190, 146, 810, 414)
PANEL_TOP = 568

# Ten seat slots clockwise from the hero at the bottom centre.
SEAT_SLOTS = ((500, 520), (290, 490), (130, 400), (130, 220), (300, 95),
              (500, 70), (700, 95), (870, 220), (870, 400), (710, 490))
# Which slots are used for each table size, so short-handed tables stay balanced.
SEAT_LAYOUTS = {
    2: (0, 5), 3: (0, 3, 7), 4: (0, 2, 5, 8), 5: (0, 2, 4, 6, 8), 6: (0, 1, 3, 5, 7, 9),
    7: (0, 1, 3, 4, 6, 7, 9), 8: (0, 1, 2, 3, 5, 7, 8, 9), 9: (0, 1, 2, 3, 4, 6, 7, 8, 9),
    10: tuple(range(10)),
}
PILL_HALF = (75, 21)
# Bet chips sit between each seat and the pot, placed by hand so they never cover the board.
BET_POSITIONS = ((500, 362), (380, 410), (262, 352), (262, 247), (376, 175),
                 (500, 160), (624, 175), (748, 247), (748, 352), (628, 410))
SMALL_CARD_GAP = 2

CARD_W, CARD_H = 70, 100
BOARD_POS = tuple((309 + i * 78, 225) for i in range(5))
HERO_POS = ((428, 395), (502, 395))

ALIGN_LABEL = "NL HOLD'EM  5/10"
ALIGN_LABEL_POS = (16, 12)
ALIGN_LABEL_BOX = (10, 6, 170, 32)

POT_NUMBER = (500, 197, 18)
STACK_SIZE = 18
BET_SIZE = 15
BUTTON_NUMBER_SIZE = 17
BUTTON_NUMBER_Y = 604

BUTTONS: Dict[str, Tuple[int, int, int, int]] = {
    "fold": (250, 585, 410, 630),
    "call": (430, 585, 630, 630),
    "raise": (650, 585, 850, 630),
    "minus": (248, 645, 274, 671),
    "slider": (284, 645, 600, 671),
    "plus": (610, 645, 636, 671),
    "amount": (650, 645, 850, 671),
    "half": (250, 682, 385, 710),
    "three_quarter": (405, 682, 540, 710),
    "pot": (560, 682, 695, 710),
    "allin": (715, 682, 850, 710),
    "players_minus": (20, 607, 46, 633),
    "players_plus": (110, 607, 136, 633),
    "autoplay": (20, 652, 210, 676),
}
HERO_ONLY_BUTTONS = {"fold", "call", "raise", "minus", "slider", "plus", "amount",
                     "half", "three_quarter", "pot", "allin"}
SLIDER_TRACK = (296, 588)


def box_center(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    return (box[0] + box[2]) // 2, (box[1] + box[3]) // 2


def bet_position(slot: int) -> Tuple[int, int]:
    return BET_POSITIONS[slot]


def bet_number_center(slot: int) -> Tuple[int, int]:
    x, y = bet_position(slot)
    return x + 8, y - 9


def stack_number_center(slot: int) -> Tuple[int, int]:
    x, y = SEAT_SLOTS[slot]
    return x + 12, y - 10


def dealer_position(slot: int) -> Tuple[int, int]:
    x, y = SEAT_SLOTS[slot]
    return (x + 92, y) if x <= TABLE_CENTER[0] else (x - 92, y)


def small_cards_origin(slot: int) -> Tuple[int, int]:
    """Top-left of the first of an opponent's two small cards, peeking above the seat."""
    x, y = SEAT_SLOTS[slot]
    return x - 41, y - PILL_HALF[1] - 46


def hit_test(x: int, y: int) -> Optional[str]:
    for name, (left, top, right, bottom) in BUTTONS.items():
        if left <= x <= right and top <= y <= bottom:
            return name
    return None

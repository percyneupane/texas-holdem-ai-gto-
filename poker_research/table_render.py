"""Renders a complete table view (felt, seats, chips, cards, action controls) to an image."""

from functools import lru_cache
from typing import Dict, List

from PIL import Image, ImageDraw

from .card_art import (BLACK, SMALL_CARD_SIZE, WHITE, card_back, card_face, draw_chips,
                       draw_number, fit_text, font, paste)
from .cards import Card
from .table_layout import (ALIGN_LABEL, ALIGN_LABEL_POS, BET_SIZE, BETTING_LINE_BOX, BOARD_POS, CARD_H, CARD_W,
                           FELT_BOX, HERO_POS, PANEL_TOP, PILL_HALF, POT_NUMBER, RAIL_BOX, SEAT_SLOTS, SIZE,
                           SMALL_CARD_GAP, STACK_SIZE, bet_number_center, bet_position, dealer_position,
                           small_cards_origin, stack_number_center)
from .table_controls import draw_controls

STARTING_STACK_DISPLAY = 1000
OPPONENT_NAMES = ("Hero", "Ana", "Bo", "Cyd", "Dex", "Eli", "Fen", "Gus", "Hal", "Ivy")
BADGE_COLORS = ((0, 168, 150), (91, 110, 225), (230, 120, 40), (200, 70, 120), (60, 160, 80),
                (150, 90, 200), (220, 180, 40), (40, 140, 200), (200, 60, 60), (120, 130, 140))
STRIP_COLORS = {
    "name": ((44, 52, 60), (205, 212, 219)), "fold": ((58, 62, 66), (150, 156, 162)),
    "check": ((38, 92, 152), WHITE), "call": ((30, 122, 74), WHITE), "raise": ((196, 112, 28), WHITE),
    "allin": ((178, 40, 48), WHITE), "blind": ((70, 78, 90), WHITE), "win": ((222, 176, 46), BLACK),
    "out": ((38, 42, 46), (120, 126, 132)),
}
CHIP_COLORS = ((200, 50, 50), (40, 110, 200), (40, 150, 80), (30, 30, 30))
HIGHLIGHT = (242, 193, 78)


@lru_cache(maxsize=1)
def _background() -> Image.Image:
    """Static scenery drawn once at high resolution and downsampled for smooth edges."""
    k = 2
    image = Image.new("RGB", (SIZE[0] * k, SIZE[1] * k), (12, 17, 22))
    draw = ImageDraw.Draw(image)
    for row in range(0, PANEL_TOP * k, 4):
        shade = int(12 + 10 * (1 - abs(row / (PANEL_TOP * k) - 0.45) * 2))
        draw.rectangle((0, row, SIZE[0] * k, row + 4), fill=(shade, shade + 6, shade + 10))
    scaled = lambda box, grow=0: tuple(v * k + (-grow if i < 2 else grow) for i, v in enumerate(box))
    draw.ellipse(scaled(RAIL_BOX, 6), fill=(8, 10, 12))
    draw.ellipse(scaled(RAIL_BOX), fill=(58, 40, 28), outline=(104, 76, 50), width=3 * k)
    steps = 24
    for step in range(steps):
        t = step / steps
        inset = int(t * 150 * k)
        color = (int(24 + 16 * t), int(88 + 38 * t), int(60 + 26 * t))
        left, top, right, bottom = scaled(FELT_BOX)
        draw.ellipse((left + inset, top + inset * 0.5, right - inset, bottom - inset * 0.5), fill=color)
    draw.ellipse(scaled(BETTING_LINE_BOX), outline=(70, 150, 108), width=2 * k)
    draw.rectangle((0, PANEL_TOP * k, SIZE[0] * k, SIZE[1] * k), fill=(17, 22, 28))
    draw.line((0, PANEL_TOP * k, SIZE[0] * k, PANEL_TOP * k), fill=(40, 48, 56), width=2 * k)
    image = image.resize(SIZE, Image.LANCZOS)
    ImageDraw.Draw(image).text(ALIGN_LABEL_POS, ALIGN_LABEL, font=font(14), fill=(225, 230, 235))
    return image


def _synthetic_seats(state: Dict) -> List[Dict]:
    """Full-ring seats for a bare decision state: the hero plus ``active_players - 1`` live opponents."""
    live = max(0, int(state["active_players"]) - 1)
    seats = [{"slot": 0, "number": 1, "name": "Hero", "stack": state["stack"], "bet": 0, "cards": "hero",
              "status": "", "kind": "name", "to_act": True}]
    for slot in range(1, 10):
        folded = slot > live
        seats.append({"slot": slot, "number": slot + 1, "name": OPPONENT_NAMES[slot], "bet": 0,
                      "stack": STARTING_STACK_DISPLAY, "cards": "none" if folded else "backs",
                      "status": "FOLD" if folded else "", "kind": "fold" if folded else "name", "folded": folded})
    return seats


def _draw_pot(draw: ImageDraw.ImageDraw, pot: int) -> None:
    cx, y, size = POT_NUMBER
    draw.rounded_rectangle((cx - 62, y - 18, cx + 62, y + 24), radius=12, fill=(14, 52, 36))
    draw.text((cx, y - 9), "TOTAL POT", font=font(10), fill=(170, 210, 188), anchor="mm")
    draw_number(draw, pot, cx, y, size)


def _draw_board(image: Image.Image, draw: ImageDraw.ImageDraw, board: List) -> None:
    for index, (x, y) in enumerate(BOARD_POS):
        if index < len(board):
            paste(image, card_face(Card(*board[index])), (x, y))
            continue
        for offset in range(0, CARD_W + CARD_H, 8):
            # Dashed outline of an empty board slot.
            if offset < CARD_W:
                draw.line((x + offset, y, x + min(offset + 4, CARD_W), y), fill=(92, 160, 124))
                draw.line((x + offset, y + CARD_H, x + min(offset + 4, CARD_W), y + CARD_H), fill=(92, 160, 124))
            if offset < CARD_H:
                draw.line((x, y + offset, x, y + min(offset + 4, CARD_H)), fill=(92, 160, 124))
                draw.line((x + CARD_W, y + offset, x + CARD_W, y + min(offset + 4, CARD_H)), fill=(92, 160, 124))


def _draw_bet(draw: ImageDraw.ImageDraw, seat: Dict) -> None:
    if seat.get("bet", 0) <= 0:
        return
    x, y = bet_position(seat["slot"])
    cx, number_y = bet_number_center(seat["slot"])
    draw.rounded_rectangle((x - 52, y - 13, cx + 34, y + 11), radius=11, fill=(14, 52, 36))
    draw_chips(draw, x - 40, y + 2, CHIP_COLORS[seat["slot"] % len(CHIP_COLORS)])
    draw_number(draw, seat["bet"], cx, number_y, BET_SIZE)


def _draw_seat_cards(image: Image.Image, seat: Dict) -> None:
    if seat["cards"] not in ("backs", "faces"):
        return
    x, y = small_cards_origin(seat["slot"])
    step = SMALL_CARD_SIZE[0] + SMALL_CARD_GAP
    for index in range(2):
        sprite = card_back() if seat["cards"] == "backs" else card_face(Card(*seat["hand"][index]), SMALL_CARD_SIZE)
        paste(image, sprite, (x + index * step, y))


def _draw_strip(draw: ImageDraw.ImageDraw, seat: Dict, timer) -> None:
    x, y = SEAT_SLOTS[seat["slot"]]
    hero = seat["slot"] == 0
    half = 90 if hero else 68
    background, ink = STRIP_COLORS.get(seat.get("kind", "name"), STRIP_COLORS["name"])
    draw.rounded_rectangle((x - half, y + 12, x + half, y + 40), radius=6, fill=background)
    draw.text((x, y + 30), fit_text(seat.get("status") or seat["name"], 11, half * 2 - 10), font=font(11),
              fill=ink, anchor="mm")
    if hero and timer is not None:
        width = int((half * 2 - 12) * max(0.0, min(1.0, timer)))
        draw.rectangle((x - half + 6, y + 37, x - half + 6 + width, y + 39), fill=HIGHLIGHT)


def _draw_pill(draw: ImageDraw.ImageDraw, seat: Dict) -> None:
    x, y = SEAT_SLOTS[seat["slot"]]
    dim = seat.get("folded") or seat.get("out")
    outline = HIGHLIGHT if seat.get("to_act") or seat.get("winner") else (58, 68, 78)
    draw.rounded_rectangle((x - PILL_HALF[0], y - PILL_HALF[1], x + PILL_HALF[0], y + PILL_HALF[1]), radius=21,
                           fill=(22, 27, 33) if dim else (30, 37, 45), outline=outline,
                           width=3 if outline == HIGHLIGHT else 2)
    badge = BADGE_COLORS[seat["slot"]] if not dim else (70, 76, 82)
    draw.ellipse((x - 69, y - 15, x - 39, y + 15), fill=badge)
    draw.text((x - 54, y), str(seat["number"]), font=font(14), fill=WHITE, anchor="mm")
    cx, number_y = stack_number_center(seat["slot"])
    draw_number(draw, seat["stack"], cx, number_y, STACK_SIZE, fill=WHITE if not dim else (150, 156, 162))


def _draw_hand_label(draw: ImageDraw.ImageDraw, seat: Dict) -> None:
    if not seat.get("hand_label"):
        return
    x, y = SEAT_SLOTS[seat["slot"]]
    text = fit_text(seat["hand_label"], 11, 180)
    width = font(11).getlength(text) / 2 + 8
    draw.rounded_rectangle((x - width, y + 43, x + width, y + 61), radius=4, fill=(10, 12, 14))
    draw.text((x, y + 52), text, font=font(11), fill=(235, 238, 240), anchor="mm")


def _draw_dealer(draw: ImageDraw.ImageDraw, slot) -> None:
    if slot is None:
        return
    x, y = dealer_position(slot)
    draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=(245, 245, 240), outline=(120, 120, 120))
    draw.text((x, y), "D", font=font(13), fill=BLACK, anchor="mm")


def _draw_message(draw: ImageDraw.ImageDraw, message: str) -> None:
    if not message:
        return
    text = fit_text(message, 13, 460)
    width = font(13).getlength(text) / 2 + 14
    draw.rounded_rectangle((500 - width, 333, 500 + width, 355), radius=11, fill=(10, 14, 18))
    draw.text((500, 344), text, font=font(13), fill=HIGHLIGHT, anchor="mm")


def render_table_image(state: Dict) -> Image.Image:
    """Draw a table view; bare decision states get a synthetic full ring around the hero."""
    image = _background().copy()
    draw = ImageDraw.Draw(image)
    seats = state.get("seats") or _synthetic_seats(state)
    if state.get("hand_number"):
        draw.text((984, 20), "Hand #%d" % state["hand_number"], font=font(12), fill=(150, 160, 170), anchor="rm")
    _draw_pot(draw, state["pot"])
    _draw_board(image, draw, state["board"])
    for seat in seats:
        _draw_bet(draw, seat)
        _draw_seat_cards(image, seat)
    for seat in seats:
        _draw_strip(draw, seat, state.get("timer"))
        _draw_pill(draw, seat)
    for position, raw in zip(HERO_POS, state["hero"]):
        paste(image, card_face(Card(*raw)), position)
    _draw_dealer(draw, state.get("dealer_slot"))
    for seat in seats:
        _draw_hand_label(draw, seat)
    _draw_message(draw, state.get("message", ""))
    draw_controls(draw, state)
    return image

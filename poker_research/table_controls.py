"""Bottom control panel: table settings, action buttons and bet sizing."""

from typing import Dict

from PIL import ImageDraw

from .card_art import WHITE, draw_number, font
from .table_layout import BUTTON_NUMBER_SIZE, BUTTON_NUMBER_Y, BUTTONS, SLIDER_TRACK, box_center

FOLD_FILL = (122, 36, 40)
CALL_FILL = (28, 112, 70)
RAISE_FILL = (166, 92, 20)
DISABLED_FILL = (44, 50, 56)
SECONDARY_FILL = (36, 44, 52)
MUTED = (150, 160, 170)
PRESETS = (("half", "1/2 POT"), ("three_quarter", "3/4 POT"), ("pot", "POT"), ("allin", "ALL-IN"))


def _button(draw: ImageDraw.ImageDraw, name: str, fill, radius: int = 10) -> None:
    draw.rounded_rectangle(BUTTONS[name], radius=radius, fill=fill)


def _label(draw: ImageDraw.ImageDraw, name: str, text: str, size: int, dy: int = 0, fill=WHITE) -> None:
    x, y = box_center(BUTTONS[name])
    draw.text((x, y + dy), text, font=font(size), fill=fill, anchor="mm")


def _draw_table_settings(draw: ImageDraw.ImageDraw, state: Dict) -> None:
    draw.text((20, 594), "PLAYERS", font=font(11), fill=MUTED, anchor="lm")
    for name, sign in (("players_minus", "-"), ("players_plus", "+")):
        _button(draw, name, SECONDARY_FILL, radius=13)
        _label(draw, name, sign, 16)
    draw.text((78, 620), str(state.get("player_count", 10)), font=font(18), fill=WHITE, anchor="mm")
    left, top, _, bottom = BUTTONS["autoplay"]
    draw.rounded_rectangle((left, top + 3, left + 18, bottom - 3), radius=3, outline=MUTED, width=2)
    if state.get("autoplay"):
        draw.line((left + 4, top + 12, left + 8, bottom - 7, left + 14, top + 7), fill=(120, 220, 150), width=3)
    draw.text((left + 26, (top + bottom) // 2), "Auto-play my seat", font=font(12), fill=MUTED, anchor="lm")


def _draw_action_buttons(draw: ImageDraw.ImageDraw, state: Dict, raise_to: int, can_raise: bool) -> None:
    _button(draw, "fold", FOLD_FILL)
    _label(draw, "fold", "FOLD", 15)
    _button(draw, "call", CALL_FILL)
    if state["to_call"] > 0:
        _label(draw, "call", "CALL", 11, dy=-12)
        draw_number(draw, state["to_call"], box_center(BUTTONS["call"])[0], BUTTON_NUMBER_Y, BUTTON_NUMBER_SIZE)
    else:
        _label(draw, "call", "CHECK", 15, dy=-10)
    _button(draw, "raise", RAISE_FILL if can_raise else DISABLED_FILL)
    if can_raise:
        _label(draw, "raise", "BET" if state.get("current_bet", state["to_call"]) == 0 else "RAISE TO", 11, dy=-12)
        draw_number(draw, raise_to, box_center(BUTTONS["raise"])[0], BUTTON_NUMBER_Y, BUTTON_NUMBER_SIZE)
    else:
        _label(draw, "raise", "RAISE", 15, dy=-10, fill=MUTED)


def _draw_sizing(draw: ImageDraw.ImageDraw, state: Dict, raise_to: int, can_raise: bool) -> None:
    ink = WHITE if can_raise else MUTED
    for name, sign in (("minus", "-"), ("plus", "+")):
        _button(draw, name, SECONDARY_FILL, radius=13)
        _label(draw, name, sign, 16, fill=ink)
    _, top, _, bottom = BUTTONS["slider"]
    middle = (top + bottom) // 2
    left, right = SLIDER_TRACK
    draw.rounded_rectangle((left, middle - 3, right, middle + 3), radius=3, fill=SECONDARY_FILL)
    low, high = state.get("min_raise_to", raise_to), state.get("max_raise_to", raise_to)
    fraction = 0.0 if high <= low else (raise_to - low) / (high - low)
    thumb = int(left + fraction * (right - left))
    draw.rounded_rectangle((left, middle - 3, thumb, middle + 3), radius=3, fill=RAISE_FILL if can_raise else MUTED)
    draw.ellipse((thumb - 9, middle - 9, thumb + 9, middle + 9), fill=ink)
    _button(draw, "amount", SECONDARY_FILL, radius=8)
    if can_raise:
        draw_number(draw, raise_to, box_center(BUTTONS["amount"])[0], middle - 10, BUTTON_NUMBER_SIZE)
    for name, text in PRESETS:
        _button(draw, name, SECONDARY_FILL, radius=8)
        _label(draw, name, text, 12, fill=ink)


def draw_controls(draw: ImageDraw.ImageDraw, state: Dict) -> None:
    """Bare decision states (no ``hero_to_act`` key) are always drawn as the hero's turn."""
    _draw_table_settings(draw, state)
    if not state.get("hero_to_act", True):
        draw.text((550, 640), state.get("waiting_text") or "Waiting for your turn", font=font(14),
                  fill=MUTED, anchor="mm")
        return
    raise_to = state.get("raise_to", state["to_call"] + state["min_raise"])
    can_raise = state.get("can_raise", state["stack"] > state["to_call"])
    _draw_action_buttons(draw, state, raise_to, can_raise)
    _draw_sizing(draw, state, raise_to, can_raise)

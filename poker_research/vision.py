"""Reader for the simulator's own rendered table.

Recognition is template matching against this simulator's renderer at fixed
geometry, so the benchmark can measure vision latency independently of policy
latency. It is not connected to or designed for third-party poker clients.

What makes a frame "match", in order of importance:

1. Same art. Every template (cards, digits, header label) is produced by
   calling this simulator's own drawing code (card_art.py / table_render.py),
   so a match means "these pixels look like what our renderer draws".
2. Same geometry. Nothing is searched for except the table origin: cards,
   numbers, buttons and seats are read at fixed coordinates from
   table_layout.py. Once the table is located, everything else is a lookup.
3. Black/white comparison. Frames and templates are thresholded to pure
   black/white ink (INK_THRESHOLD), and similarity is the count of differing
   pixels (_distance). The closest template wins.
4. Colour probes. A few yes/no questions (is a card there? is it my turn?
   which surrounding seats hold card backs?) are answered by sampling small
   pixel regions for brightness or colour, not by templates.

The table zones are deliberately separate: BOARD_POS is the middle board,
HERO_POS is the bottom-center hero hand, and opponent card backs are searched
only in non-hero seat slots around the left, top, and right edges.
"""

from functools import lru_cache
from typing import Dict, List, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageStat

from .card_art import CARD_SIZE, card_face, digit_advance, font, number_start
from .cards import Card, SUITS
from .engine import SMALL_BLIND, BIG_BLIND
from .table_layout import (ALIGN_LABEL_BOX, BOARD_POS, BUTTON_NUMBER_SIZE, BUTTON_NUMBER_Y, BUTTONS, CARD_W,
                           HERO_POS, POT_NUMBER, SIZE, STACK_SIZE, BET_SIZE, bet_number_center, box_center,
                           small_cards_origin, stack_number_center)
from .table_render import render_table_image

__all__ = ["SIZE", "NotHeroTurn", "TableNotFoundError", "decode_table_image", "render_table_image"]

# Grey level (0-255) splitting "ink" from "background". Text and card faces are
# light (> 128 -> white); felt, pills, buttons and card ink are dark (-> black).
# Every colour the renderer uses was chosen to sit clearly on one side of this.
INK_THRESHOLD = 128
# A card slot counts as occupied if this pixel (top-right of the card, always
# plain white on a face) is at least this bright. Empty slots show dark felt.
CARD_PRESENT_MIN_BRIGHTNESS = 200
CARD_PROBE = (CARD_W - 12, 12)
# Numbers are read as up to this many digits on a fixed per-font digit grid.
MAX_DIGITS = 6
# Extra rows below the font size so descending glyph pixels fall inside the cell.
DIGIT_CELL_PAD = 6
# How far (pixels) the table may sit from where it is expected and still be found.
ALIGN_SEARCH_PX = 10
# Fraction of the header label that may differ before we decide the table is not in view.
ALIGN_MAX_MISMATCH = 0.2
# A remembered table position is reused while its label matches almost exactly (share of label ink).
CACHED_MAX_MISMATCH = 0.05
# Downscale factor for the first, rough pass when the capture box is much larger than the table.
COARSE_STEP = 4
# Region inside the FOLD button. Its dark-red fill only exists on the hero's turn.
FOLD_PROBE = (254, 588, 268, 596)
# Regions inside an opponent's first small card, relative to that card's corner,
# where the blue lattice of a card back is sampled. Using two separated probes
# avoids mistaking a nearby seat label or a single anti-aliased pixel for a card.
BACK_PROBES = ((8, 18, 16, 30), (24, 28, 32, 40))
# Slot 0 is the bottom-center hero. All other slots are opponent positions
# around the sides/top of the table; never use HERO_POS for this probe.
OPPONENT_SLOTS = tuple(range(1, 10))


_last_origin: Dict[Tuple[int, int], Tuple[int, int]] = {}


class TableNotFoundError(ValueError):
    """The captured frame does not contain the simulator table."""


class NotHeroTurn(ValueError):
    """The frame is valid but the hero has no decision to make right now."""


def _binarize(image: Image.Image) -> Image.Image:
    """Reduce to black/white so display colour profiles and mild blur don't matter."""
    return image.convert("L").point(lambda value: 255 if value > INK_THRESHOLD else 0)


def _distance(a: Image.Image, b: Image.Image) -> float:
    """The core similarity score: total pixel difference between two same-size images.

    On binarized images this is 255 x (number of pixels that disagree), so 0 is
    a perfect match and lower is better. Every template decision below picks
    the candidate with the smallest distance.
    """
    return ImageStat.Stat(ImageChops.difference(a, b)).sum[0]


def _ink(image: Image.Image, box: Tuple[int, int, int, int]) -> float:
    """Amount of white ink in a region (255 per white pixel); blank background scores 0."""
    return ImageStat.Stat(image.crop(box)).sum[0]


def _mean_rgb(image: Image.Image, box: Tuple[int, int, int, int]) -> Tuple[float, float, float]:
    return tuple(ImageStat.Stat(image.crop(box)).mean[:3])


@lru_cache(maxsize=1)
def _align_template() -> Image.Image:
    """The "NL HOLD'EM 5/10" header text, cut from an empty render.

    This label is the only thing used to find the table: it is drawn at the
    same spot in every frame and nothing else is ever drawn over it.
    """
    blank ={"pot": 0, "to_call": 0, "min_raise": 0, "stack": 0, "active_players": 0, "board": [], "hero": []}
    return _binarize(render_table_image(blank).crop(ALIGN_LABEL_BOX))


def _label_distance(ink: Image.Image, template: Image.Image, x: int, y: int) -> float:
    left, top, right, bottom = ALIGN_LABEL_BOX
    return _distance(template, ink.crop((x + left, y + top, x + right, y + bottom)))


def _search(ink: Image.Image, template: Image.Image, xs: range, ys: range) -> Tuple[float, int, int]:
    return min((_label_distance(ink, template, x, y), x, y) for x in xs for y in ys)


def _coarse_origin(ink: Image.Image, template: Image.Image, xs: range, ys: range) -> Tuple[int, int]:
    """Search a downscaled frame so a capture box much larger than the table stays fast."""
    def shrink(image: Image.Image) -> Image.Image:
        return image.resize((max(1, image.width // COARSE_STEP), max(1, image.height // COARSE_STEP)), Image.BOX)
    small_ink, small_template = shrink(ink), shrink(template)
    left, top = ALIGN_LABEL_BOX[0] // COARSE_STEP, ALIGN_LABEL_BOX[1] // COARSE_STEP
    width, height = small_template.size
    _, x, y = min((_distance(small_template, small_ink.crop((x + left, y + top, x + left + width, y + top + height))),
                   x, y)
                  for x in range(xs.start // COARSE_STEP, xs.stop // COARSE_STEP + 1)
                  for y in range(ys.start // COARSE_STEP, ys.stop // COARSE_STEP + 1))
    return x * COARSE_STEP, y * COARSE_STEP


def locate_table(image: Image.Image) -> Tuple[int, int]:
    """Top-left of the table inside a frame at table scale (the table itself or a larger box around it)."""
    ink = _binarize(image)
    template = _align_template()
    xs = range(-ALIGN_SEARCH_PX, max(0, image.width - SIZE[0]) + ALIGN_SEARCH_PX + 1)
    ys = range(-ALIGN_SEARCH_PX, max(0, image.height - SIZE[1]) + ALIGN_SEARCH_PX + 1)
    template_ink = ImageStat.Stat(template).sum[0]
    # Fast path: if the table was found at this spot last time and the label still
    # lines up almost perfectly, skip the search.
    cached = _last_origin.get(image.size)
    if cached and _label_distance(ink, template, *cached) <= template_ink * CACHED_MAX_MISMATCH:
        return cached
    if len(xs) * len(ys) <= (2 * ALIGN_SEARCH_PX + 1) ** 2:
        # Box is about table-sized: try every offset within +/- ALIGN_SEARCH_PX.
        mismatch, x, y = _search(ink, template, xs, ys)
    else:
        # Box is much larger: find the label roughly on a shrunken frame, then refine nearby.
        cx, cy = _coarse_origin(ink, template, xs, ys)
        around = lambda centre: range(centre - COARSE_STEP, centre + COARSE_STEP + 1)
        mismatch, x, y = _search(ink, template, around(cx), around(cy))
    # Even the best position must agree on at least 80% of the label's pixels,
    # otherwise whatever is on screen is not this table.
    if mismatch > ALIGN_MAX_MISMATCH * 255 * template.width * template.height:
        raise TableNotFoundError("simulator table not found in the captured frame")
    _last_origin[image.size] = (x, y)
    return x, y


def _align(image: Image.Image) -> Image.Image:
    """Crop the frame to exactly the renderer's table geometry."""
    x, y = locate_table(image)
    if (x, y) == (0, 0) and image.size == SIZE:
        return image
    return image.crop((x, y, x + SIZE[0], y + SIZE[1]))


@lru_cache(maxsize=1)
def _card_templates() -> Dict[Card, Image.Image]:
    """All 52 card faces drawn by card_art.card_face, binarized, on a dark background.

    Because the faces come from the renderer itself, the rank index, suit
    symbols and their exact placement are what distinguish one card from
    another. Red and black ink both fall below INK_THRESHOLD, so suits are told
    apart by shape (heart vs diamond, spade vs club), not by colour.
    """
    templates = {}
    for rank in range(2, 15):
        for suit in SUITS:
            patch = Image.new("RGB", CARD_SIZE, (0, 0, 0))
            face = card_face(Card(rank, suit))
            patch.paste(face, (0, 0), face)
            templates[Card(rank, suit)] = _binarize(patch)
    return templates


@lru_cache(maxsize=None)
def _digit_templates(size: int) -> Dict[str, Image.Image]:
    """Digits 0-9 in the renderer's font at one size, each in a cell one digit-advance wide.

    The renderer places each digit on the same fixed grid (card_art.draw_number),
    so a cell in the frame lines up exactly with one of these templates.
    """
    templates = {}
    for digit in "0123456789":
        patch = Image.new("L", (digit_advance(size), size + DIGIT_CELL_PAD), 0)
        ImageDraw.Draw(patch).text((0, 0), digit, fill=255, font=font(size))
        templates[digit] = _binarize(patch)
    return templates


def _read_cards(image: Image.Image, ink: Image.Image, positions) -> List[Card]:
    """Identify the cards at fixed slot positions (board or hero), skipping empty slots."""
    templates = _card_templates()
    found = []
    for x, y in positions:
        # Step 1: is a card face here at all? (one bright pixel check)
        if image.convert("L").getpixel((x + CARD_PROBE[0], y + CARD_PROBE[1])) < CARD_PRESENT_MIN_BRIGHTNESS:
            continue
        # Step 2: compare the whole 70x100 card to all 52 templates; fewest differing pixels wins.
        crop = ink.crop((x, y, x + CARD_SIZE[0], y + CARD_SIZE[1]))
        found.append(min(templates, key=lambda card: _distance(crop, templates[card])))
    return found


def _read_number(ink: Image.Image, cx: int, y: int, size: int) -> int:
    """Read a centred tabular number by testing each possible digit count.

    Numbers are centred on ``cx``, so their left edge depends on how many digits
    they have. For each guess of 1..MAX_DIGITS digits we know exactly where every
    digit cell would be, match each cell to its closest digit template, and add
    a penalty for any ink outside those cells (it should be blank background).
    The guess with the lowest total cost wins; "no number" (cost = all ink in
    the window) wins when the area is empty, which reads as 0.
    """
    templates = _digit_templates(size)
    advance = digit_advance(size)
    bottom = y + size + DIGIT_CELL_PAD
    half_window = (MAX_DIGITS * advance) // 2 + advance
    window_ink = _ink(ink, (cx - half_window, y, cx + half_window, bottom))
    best_cost, best_digits = window_ink, ""
    for count in range(1, MAX_DIGITS + 1):
        start = number_start(count, cx, size)
        digits, cost = "", 0.0
        for offset in range(count):
            cell = ink.crop((start + offset * advance, y, start + (offset + 1) * advance, bottom))
            digit, distance = min(((d, _distance(cell, t)) for d, t in templates.items()), key=lambda pair: pair[1])
            digits += digit
            cost += distance
        # Penalty: ink in the window that this guess leaves unexplained.
        cost += window_ink - _ink(ink, (start, y, start + count * advance, bottom))
        if cost < best_cost:
            best_cost, best_digits = cost, digits
    return int(best_digits) if best_digits else 0


def _is_hero_turn(image: Image.Image) -> bool:
    """Colour probe: the FOLD button (dark red) is only drawn when the hero must act.

    Otherwise that spot is the grey control panel, where red ~= green.
    """
    red, green, _ = _mean_rgb(image, FOLD_PROBE)
    return red > green + 40


def _count_card_backs(image: Image.Image) -> int:
    """Colour probe: count seats showing blue card backs (opponents still in the hand).

    Folded or empty seats have no cards there (dark background), and face-up
    showdown cards are white, so only the strongly blue backs count.
    """
    count = 0
    for slot in OPPONENT_SLOTS:
        x, y = small_cards_origin(slot)
        probes = []
        for left, top, right, bottom in BACK_PROBES:
            red, green, blue = _mean_rgb(image, (x + left, y + top, x + right, y + bottom))
            probes.append(blue > red + 50 and blue > green + 30)
        count += int(all(probes))
    return count


def decode_table_image(image: Image.Image) -> Dict:
    """Recover the hero's decision state from a simulator-rendered image.

    Every field below is read from a fixed spot defined in table_layout.py;
    the only search is ``_align`` locating the header label.
    """
    # 1. Find the table and crop to its exact 1000x720 geometry.
    image = _align(image.convert("RGB"))
    # 2. Only decode when the action buttons are showing.
    if not _is_hero_turn(image):
        raise NotHeroTurn("waiting for the hero's turn")
    # 3. One black/white copy shared by all card and number readers.
    ink = _binarize(image)
    # The renderer's central five slots are the board and the two bottom-center
    # slots are always the hero's cards. Opponent slots are intentionally not
    # used for card-face decoding: backs only indicate that a player remains.
    hero = _read_cards(image, ink, HERO_POS)
    board = _read_cards(image, ink, BOARD_POS)
    # Amount to call is the number on the CALL button (blank on CHECK -> 0).
    to_call = _read_number(ink, box_center(BUTTONS["call"])[0], BUTTON_NUMBER_Y, BUTTON_NUMBER_SIZE)
    # RAISE TO shows a street total; the hero's own bet chips are needed to turn it into a raise size.
    raise_to = _read_number(ink, box_center(BUTTONS["raise"])[0], BUTTON_NUMBER_Y, BUTTON_NUMBER_SIZE)
    hero_bet = _read_number(ink, *bet_number_center(0), BET_SIZE)
    return {
        "hero": [[card.rank, card.suit] for card in hero],
        "board": [[card.rank, card.suit] for card in board],
        "active_players": len(hero) // 2 + _count_card_backs(image),
        "pot": _read_number(ink, *POT_NUMBER),
        "to_call": to_call,
        "min_raise": max(0, raise_to - hero_bet - to_call) if raise_to else 0,
        "stack": _read_number(ink, *stack_number_center(0), STACK_SIZE),
        "hero_bet": hero_bet,
        "small_blind": SMALL_BLIND,
        "big_blind": BIG_BLIND,
        "street": "unknown",
        "action_history": [],
    }

"""Drawing primitives for the simulated table: fonts, cards, card backs, chips, digits."""

from functools import lru_cache
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont

from .cards import RANKS, Card

CARD_SIZE = (70, 100)
SMALL_CARD_SIZE = (40, 56)
SUPERSAMPLE = 4
WHITE = (255, 255, 255)
CARD_WHITE = (250, 250, 246)
CARD_EDGE = (170, 176, 184)
RED = (206, 40, 52)
BLACK = (24, 27, 31)
BACK_BLUE = (34, 70, 158)
BACK_LINE = (70, 108, 196)
FONT_PATHS = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "DejaVuSans-Bold.ttf",
)


@lru_cache(maxsize=None)
def font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def suit_color(suit: str) -> Tuple[int, int, int]:
    return RED if suit in "hd" else BLACK


def draw_suit(draw: ImageDraw.ImageDraw, suit: str, cx: float, cy: float, s: float, fill) -> None:
    """Draw a suit symbol of half-height ``s`` centred on (cx, cy) from basic shapes."""
    if suit == "d":
        draw.polygon([(cx, cy - s), (cx + 0.75 * s, cy), (cx, cy + s), (cx - 0.75 * s, cy)], fill=fill)
        return
    if suit == "h":
        r = 0.52 * s
        for dx in (-0.48 * s, 0.48 * s):
            draw.ellipse((cx + dx - r, cy - 0.45 * s - r, cx + dx + r, cy - 0.45 * s + r), fill=fill)
        draw.polygon([(cx - 0.98 * s, cy - 0.3 * s), (cx + 0.98 * s, cy - 0.3 * s), (cx, cy + s)], fill=fill)
        return
    if suit == "s":
        r = 0.5 * s
        for dx in (-0.47 * s, 0.47 * s):
            draw.ellipse((cx + dx - r, cy + 0.15 * s - r, cx + dx + r, cy + 0.15 * s + r), fill=fill)
        draw.polygon([(cx - 0.95 * s, cy + 0.05 * s), (cx + 0.95 * s, cy + 0.05 * s), (cx, cy - s)], fill=fill)
    else:
        r = 0.4 * s
        for dx, dy in ((0, -0.5 * s), (-0.48 * s, 0.1 * s), (0.48 * s, 0.1 * s)):
            draw.ellipse((cx + dx - r, cy + dy - r, cx + dx + r, cy + dy + r), fill=fill)
        draw.rectangle((cx - 0.12 * s, cy - 0.3 * s, cx + 0.12 * s, cy + 0.3 * s), fill=fill)
    draw.polygon([(cx, cy + 0.2 * s), (cx - 0.38 * s, cy + s), (cx + 0.38 * s, cy + s)], fill=fill)


def rank_label(rank: int) -> str:
    return "10" if rank == 10 else RANKS[rank - 2]


@lru_cache(maxsize=None)
def card_face(card: Card, size: Tuple[int, int] = CARD_SIZE) -> Image.Image:
    """Large-index card face (RGBA, transparent rounded corners)."""
    k = SUPERSAMPLE
    width, height = CARD_SIZE[0] * k, CARD_SIZE[1] * k
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=7 * k, fill=CARD_WHITE, outline=CARD_EDGE, width=2 * k)
    color = suit_color(card.suit)
    label = rank_label(card.rank)
    draw.text((6 * k, 3 * k), label, font=font((26 if len(label) > 1 else 30) * k), fill=color)
    draw_suit(draw, card.suit, 17 * k, 47 * k, 8 * k, color)
    draw_suit(draw, card.suit, 45 * k, 73 * k, 17 * k, color)
    return image.resize(size, Image.LANCZOS)


@lru_cache(maxsize=None)
def card_back(size: Tuple[int, int] = SMALL_CARD_SIZE) -> Image.Image:
    """Blue lattice card back (RGBA)."""
    k = SUPERSAMPLE
    width, height = size[0] * k, size[1] * k
    inset = 3 * k
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=5 * k, fill=CARD_WHITE, outline=CARD_EDGE, width=k)
    inner = Image.new("RGBA", (width - 2 * inset, height - 2 * inset), BACK_BLUE)
    lattice = ImageDraw.Draw(inner)
    step = 6 * k
    for offset in range(-inner.height, inner.width + inner.height, step):
        lattice.line((offset, 0, offset + inner.height, inner.height), fill=BACK_LINE, width=k)
        lattice.line((offset + inner.height, 0, offset, inner.height), fill=BACK_LINE, width=k)
    mask = Image.new("L", inner.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, inner.width - 1, inner.height - 1), radius=3 * k, fill=255)
    image.paste(inner, (inset, inset), mask)
    return image.resize(size, Image.LANCZOS)


def paste(image: Image.Image, sprite: Image.Image, position: Tuple[int, int]) -> None:
    image.paste(sprite, position, sprite)


def draw_chips(draw: ImageDraw.ImageDraw, cx: int, cy: int, color, count: int = 2) -> None:
    for level in range(count):
        y = cy - level * 3
        draw.ellipse((cx - 9, y - 5, cx + 9, y + 5), fill=color, outline=WHITE, width=1)
        draw.line((cx - 3, y - 4, cx + 3, y - 4), fill=WHITE, width=1)


def digit_advance(size: int) -> int:
    return round(max(font(size).getlength(digit) for digit in "0123456789"))


def number_start(digits: int, cx: int, size: int) -> int:
    return cx - (digits * digit_advance(size)) // 2


def draw_number(draw: ImageDraw.ImageDraw, value: int, cx: int, y: int, size: int, fill=WHITE) -> None:
    """Draw digits on a fixed grid centred on cx so the reader can locate each one."""
    text = str(max(0, int(value)))
    start = number_start(len(text), cx, size)
    advance = digit_advance(size)
    for offset, char in enumerate(text):
        draw.text((start + offset * advance, y), char, font=font(size), fill=fill)


def fit_text(text: str, size: int, width: int) -> str:
    """Trim text with an ellipsis so it fits within ``width`` pixels."""
    if font(size).getlength(text) <= width:
        return text
    while text and font(size).getlength(text + "...") > width:
        text = text[:-1]
    return text.rstrip() + "..."

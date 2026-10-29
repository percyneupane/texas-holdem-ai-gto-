"""Draggable, resizable capture box drawn as borderless always-on-top windows.

Edges and corners resize the box freely; the tab above it moves the whole box.
A box larger than the table is searched for the table; a smaller one is scaled
up to table size before decoding (see ``screen_capture.capture_size``).
"""

import tkinter as tk
from typing import Callable, Dict, Tuple

from .screen_capture import CaptureRegion
EDGE = 8
HANDLE = 18
# Just enough room that the corner handles never overlap.
MIN_WIDTH = MIN_HEIGHT = 3 * HANDLE
TAB_HEIGHT = 24
BOX_COLOR = "#ffd60a"
HANDLE_COLOR = "#fff8d6"
TAB_TEXT = "#1b1b1b"
EDGES = ("n", "s", "w", "e")
CORNERS = ("nw", "ne", "sw", "se")
CURSORS = {
    "n": "sb_v_double_arrow", "s": "sb_v_double_arrow",
    "w": "sb_h_double_arrow", "e": "sb_h_double_arrow",
    "nw": "top_left_corner", "ne": "top_right_corner",
    "sw": "bottom_left_corner", "se": "bottom_right_corner",
    "move": "fleur",
}


def drag_region(start: CaptureRegion, handle: str, dx: int, dy: int) -> CaptureRegion:
    """Return the region after dragging `handle` by (dx, dy); the opposite side stays put."""
    if handle == "move":
        return CaptureRegion(start.left + dx, start.top + dy, start.right + dx, start.bottom + dy)
    left = start.left + dx if "w" in handle else start.left
    right = start.right + dx if "e" in handle else start.right
    top = start.top + dy if "n" in handle else start.top
    bottom = start.bottom + dy if "s" in handle else start.bottom
    # Clamp the dragged side so the box never inverts or hides its handles.
    if "w" in handle:
        left = min(left, right - MIN_WIDTH)
    else:
        right = max(right, left + MIN_WIDTH)
    if "n" in handle:
        top = min(top, bottom - MIN_HEIGHT)
    else:
        bottom = max(bottom, top + MIN_HEIGHT)
    return CaptureRegion(left, top, right, bottom)


def _piece_geometry(region: CaptureRegion, name: str) -> Tuple[int, int, int, int]:
    """Width, height, x, y of one box piece in screen points."""
    left, top, right, bottom = region.bbox
    width, height = right - left, bottom - top
    tab_top = top - TAB_HEIGHT if top >= TAB_HEIGHT else top + EDGE
    return {
        "n": (width, EDGE, left, top),
        "s": (width, EDGE, left, bottom - EDGE),
        "w": (EDGE, height, left, top),
        "e": (EDGE, height, right - EDGE, top),
        "nw": (HANDLE, HANDLE, left, top),
        "ne": (HANDLE, HANDLE, right - HANDLE, top),
        "sw": (HANDLE, HANDLE, left, bottom - HANDLE),
        "se": (HANDLE, HANDLE, right - HANDLE, bottom - HANDLE),
        "move": (190, TAB_HEIGHT, left, tab_top),
    }[name]


class CaptureBox:
    def __init__(self, root: tk.Tk, region: CaptureRegion, on_change: Callable[[CaptureRegion], None]):
        self.root = root
        self.region = region
        self.on_change = on_change
        self._drag = None
        self.pieces: Dict[str, tk.Toplevel] = {}
        for name in EDGES + CORNERS + ("move",):
            self.pieces[name] = self._make_piece(name)
        self.set_region(region)

    def _make_piece(self, name: str) -> tk.Toplevel:
        window = tk.Toplevel(self.root)
        window.overrideredirect(True)
        window.attributes("-topmost", True)
        color = HANDLE_COLOR if name in CORNERS else BOX_COLOR
        window.configure(background=color, cursor=CURSORS[name])
        target = window
        if name == "move":
            target = tk.Label(window, text="✥  Drag to move  ·  edges resize", bg=BOX_COLOR, fg=TAB_TEXT,
                              font=("Helvetica Neue", 11, "bold"), cursor=CURSORS[name])
            target.pack(fill="both", expand=True)
        elif name in CORNERS:
            window.configure(highlightthickness=2, highlightbackground=BOX_COLOR)
        target.bind("<ButtonPress-1>", lambda event, handle=name: self._start_drag(event, handle))
        target.bind("<B1-Motion>", self._continue_drag)
        target.bind("<ButtonRelease-1>", self._end_drag)
        return window

    def _start_drag(self, event, handle: str):
        self._drag = (handle, event.x_root, event.y_root, self.region)

    def _continue_drag(self, event):
        if self._drag is None:
            return
        handle, x0, y0, start = self._drag
        self.set_region(drag_region(start, handle, event.x_root - x0, event.y_root - y0))
        self.on_change(self.region)

    def _end_drag(self, _event):
        self._drag = None

    def set_region(self, region: CaptureRegion):
        self.region = region
        # Edges first so corner handles and the tab stack above them.
        for name in EDGES + CORNERS + ("move",):
            window = self.pieces[name]
            window.geometry("%dx%d+%d+%d" % _piece_geometry(region, name))
            window.lift()

    def set_visible(self, visible: bool):
        for window in self.pieces.values():
            (window.deiconify if visible else window.withdraw)()

    def destroy(self):
        for window in self.pieces.values():
            window.destroy()

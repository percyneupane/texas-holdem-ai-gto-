"""Decision overlay and opt-in autoplay inside the selected capture area."""

import argparse
import os
import sys
import threading
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from PIL import Image

from .agent import Decision, VisionPolicyAgent
from .ai_screen_reader import AiScreenReaderError
from .stakes import parse_blinds, format_money, money_decimal, big_blind_count
from .cards import RANKS
from .overlay_box import CaptureBox
from .policy import POLICIES
from .metrics import Metrics
from .screen_capture import CaptureRegion, ScreenCapture, capture_size, display_scale, scale_region
from .session_log import SessionLog, observation_from_decision
from .table_display import submit_observation, submit_table_action
from .vision import NotHeroTurn, TableNotFoundError
from .turn_detection import LocalTurnDetector

CONTROLS_GAP = 12
MENU_BAR_HEIGHT = 30
DOCK_MARGIN = 90
PANEL_WIDTH = 290
BAR_HEIGHT = 6

BG = "#14171b"
CARD = "#1d2127"
BORDER = "#2b3038"
TEXT = "#eef1f4"
MUTED = "#8b949e"
FAINT = "#5c6570"
GREEN = "#3fb950"
AMBER = "#d29922"
RED = "#f85149"
BLUE = "#58a6ff"
FONT = "Helvetica Neue"

ACTION_COLORS = {"raise": GREEN, "bet": AMBER, "call": BLUE, "check": MUTED, "fold": RED}
SUIT_SYMBOLS = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
STATUS_STYLES = {"stopped": ("●  Stopped", MUTED), "live": ("●  Live", GREEN), "waiting": ("●  Waiting", BLUE),
                 "rejected": ("●  No table", AMBER), "error": ("●  Reader error", AMBER)}
# Action buttons sit in the lower part of the table. A mean per-channel change
# below this is a clock digit or anti-alias flicker, not a new decision.
ACTION_BAND_TOP = 0.72
ACTION_BAND_SIZE = (48, 12)
ACTION_BAND_WAKE = 3.0
# One cell of the whole capture. Above the thin shot clock, below buttons,
# a new board, or new hole cards, wherever they sit in the box.
CELL_WAKE = 3.0
WATCH_INTERVAL_MS = 700

METRICS = [
    ("equity", "Equity", "Estimated chance this hand wins at showdown against the players still in."),
    ("bluff", "Bluff chance", "How often a weak hand bets here when nobody has bet, so bluffs make up the "
                              "balanced share of bets. Lower with more opponents."),
    ("latency", "Latency", "Time from screenshot to decision: capture · read table · decide."),
]


def action_band(image: Image.Image) -> bytes:
    """Coarse colour sample of the action-button strip."""
    width, height = image.size
    if width < 8 or height < 8:
        return b''
    top = int(height * ACTION_BAND_TOP)
    band = image.convert('RGB').crop((0, top, width, height)).resize(ACTION_BAND_SIZE)
    return band.tobytes()


def action_band_changed(previous: Optional[bytes], current: bytes,
                        wake: float = ACTION_BAND_WAKE) -> bool:
    """True when the button strip changed enough to be a new turn or a new decision."""
    if not previous or not current or len(previous) != len(current):
        return True
    return sum(abs(a - b) for a, b in zip(previous, current)) / len(current) >= wake


# Share of the bottom button row that is strongly coloured. The row sits below
# the hero's cards, so cards and clocks do not keep a finished turn "active".
# Felt and "waiting" text stay below the absent line; fold/call/raise stay above it.
BUTTON_BAND_TOP = 0.78
BUTTON_BAND_BOTTOM = 0.97
BUTTONS_PRESENT = 0.02
BUTTONS_ABSENT = 0.008


def button_fraction(image: Image.Image) -> float:
    """Fraction of the action-button row that looks like coloured action buttons."""
    width, height = image.size
    if width < 8 or height < 8:
        return 0.0
    top = int(height * BUTTON_BAND_TOP)
    bottom = max(top + 1, int(height * BUTTON_BAND_BOTTOM))
    band = image.convert('RGB').crop((0, top, width, bottom)).resize(ACTION_BAND_SIZE)
    raw = band.tobytes()
    total = len(raw) // 3
    if total == 0:
        return 0.0
    saturated = 0
    for index in range(0, len(raw), 3):
        red, green, blue = raw[index], raw[index + 1], raw[index + 2]
        high = max(red, green, blue)
        low = min(red, green, blue)
        if high >= 48 and (high - low) / high >= 0.45:
            saturated += 1
    return saturated / total


def buttons_up(previous: Optional[bool], fraction: float) -> bool:
    """Hysteresis so a clock digit cannot flip the turn on and off."""
    if fraction >= BUTTONS_PRESENT:
        return True
    if fraction <= BUTTONS_ABSENT:
        return False
    return bool(previous)


def watch_signature(image: Image.Image):
    """6×4 colour samples covering the whole capture, not a fixed button row."""
    width, height = image.size
    if width < 8 or height < 8:
        return tuple()
    rgb = image.convert('RGB')
    cols, rows = 6, 4
    cells = []
    for row in range(rows):
        for col in range(cols):
            box = (col * width // cols, row * height // rows,
                   (col + 1) * width // cols, (row + 1) * height // rows)
            if box[2] <= box[0] or box[3] <= box[1]:
                cells.append(b'')
            else:
                cells.append(rgb.crop(box).resize((12, 8)).tobytes())
    return tuple(cells)


def watch_changed(previous, current, wake: float = CELL_WAKE) -> bool:
    """True when any part of the capture moved enough to be a new turn."""
    if not previous or not current or len(previous) != len(current):
        return True
    return any(action_band_changed(prev, cur, wake) for prev, cur in zip(previous, current))


def decision_signature(image: Image.Image):
    """Watch cards and controls, excluding opponent seats and their animations."""
    width, height = image.size
    rgb = image.convert('RGB')
    zones = ((0.30, 0.30, 0.71, 0.46),  # community cards
             (0.42, 0.54, 0.58, 0.69),  # hero cards
             (0.24, 0.81, 0.86, 0.88))  # fold/call/raise, excluding the clock
    return tuple(rgb.crop(tuple(int(value * (width if index % 2 == 0 else height))
                                                for index, value in enumerate(zone)))
                 .resize((48, 16)).tobytes() for zone in zones)


def turn_gate(state: str, active: bool, changed: bool):
    """One model read per turn. Returns (next_state, start_read).

    Only visible action buttons can start a read. Rejected frames stay locked
    until buttons disappear, so opponent motion cannot trigger paid retries.
    """
    if state == 'reading':
        return 'reading', False
    if not active:
        return 'idle', False
    if state == 'idle' or (state == 'answered' and changed):
        return 'reading', True
    return state, False


def format_cards(cards: List[List]) -> str:
    return "  ".join(RANKS[rank - 2] + SUIT_SYMBOLS.get(suit, suit) for rank, suit in cards) or "—"


def format_action(decision: Decision) -> str:
    state = decision.state or {}
    action = decision.action.upper() + (' ' + format_money(decision.amount, state.get('money_precision'))
                                        if decision.amount else '')
    street = {0: 'Preflop', 3: 'Flop', 4: 'Turn', 5: 'River'}.get(len(state.get('board', [])))
    return '%s\n%s' % (street, action) if 'board' in state and street else action


def format_stakes(state):
    sb, bb = state.get('small_blind', 0), state.get('big_blind', 0)
    precision = state.get('money_precision')
    text = 'SB %s · BB %s (%s)' % (format_money(sb, precision) if sb else '—',
                                   format_money(bb, precision) if bb else '—', state.get('blind_source', 'screen/state'))
    if bb:
        text += '\nStack %s bb · Pot %s bb · Call %s bb' % (
            big_blind_count(state['stack'], bb), big_blind_count(state['pot'], bb),
            big_blind_count(state['to_call'], bb))
    return text


def format_bet_sizing(decision):
    if decision.action not in ('raise', 'bet', 'call'):
        return ''
    state = decision.state or {}
    bb = state.get('big_blind', 0)
    precision = state.get('money_precision')
    text = 'Add %s' % format_money(decision.amount, precision)
    if bb:
        text += ' (%s bb)' % big_blind_count(decision.amount, bb, '0.01')
    if decision.action in ('raise', 'bet'):
        if 'hero_bet' in state:
            total = money_decimal(state['hero_bet']) + money_decimal(decision.amount)
            text += '\nRaise-to total: %s' % format_money(total, precision)
        else:
            text += '\nRaise-to total unknown: current street bet not read.'
    return text


def format_strategy_context(state):
    context = state.get('strategy_context', {})
    labels = {'raised': '2-bet/open', 'three_bet': '3-bet', 'four_bet': '4-bet',
              'five_bet': '5-bet+', 'unopened': 'unopened', 'limped': 'limped'}
    parts = [labels.get(state.get('preflop_action'), 'pot type unknown')]
    if context.get('squeeze'):
        parts.append('squeeze')
    if context.get('cold_four_bet'):
        parts.append('cold 4-bet')
    if context.get('effective_stack_bb') is not None:
        parts.append('Effective %.1f bb' % context['effective_stack_bb'])
    if context.get('spr') is not None:
        parts.append('SPR %.2f' % context['spr'])
    parts.append('history complete' if context.get('history_complete') else 'history incomplete')
    return ' · '.join(parts)


class ResearchOverlay:
    def __init__(self, root: tk.Tk, region: CaptureRegion, interval_ms: int = 250, policy: str = "contextual",
                 reader_mode: str = 'ai', table_url: str = 'http://127.0.0.1:8780', session_dir: Optional[str] = None,
                 autoplay_target: str = 'screen'):
        self.root = root
        self.region = region
        self.interval_ms = max(50, interval_ms)
        self.table_url = table_url
        if autoplay_target not in ('screen', 'local'):
            raise ValueError('Autoplay target must be screen or local.')
        self.autoplay_target = autoplay_target
        self.autoplay = False
        self._screen_player = None
        self._request_image = None
        self._played_signature = None
        self._play_not_before = 0.0
        self.agent = VisionPolicyAgent(policy=policy, reader_mode=reader_mode)
        if autoplay_target == 'screen' and self.agent.ai_reader is not None:
            for reader in getattr(self.agent.ai_reader, 'readers', (self.agent.ai_reader,)):
                reader.locate_controls = True
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self._probe = None
        self.turn_detector = LocalTurnDetector()
        self._read_once_requested = False
        self.request_manual = False
        self._timer = None
        self.generation = 0
        self.pixel_scale = display_scale(root.winfo_screenwidth())
        self.running = False
        self._turn_state = 'idle'
        self._turn_serial = 0
        self._turn_token = None
        self._locked_sig = None
        self._request_sig = None
        self._buttons_visible = False
        self.blind_overrides = {}
        self.last_decision: Optional[Decision] = None
        self.session_metrics = Metrics()
        self.observed = SessionLog(session_dir if session_dir is not None else os.path.join(os.getcwd(), 'sessions'),
                                   prefix='overlay')
        self.metric_values: Dict[str, tk.Label] = {}
        self.metric_bars: Dict[str, tk.Canvas] = {}
        self._build_panel()
        self.box = CaptureBox(root, region, self._on_box_dragged)
        self._place_controls()

    # ---------- layout ----------

    def _build_panel(self):
        self.root.title("Vision Policy")
        self.root.resizable(False, True)
        self.root.attributes("-topmost", True)
        self.root.configure(background=BG)
        viewport = tk.Frame(self.root, bg=BG)
        viewport.pack(fill="both", expand=True)
        self.scroll_canvas = tk.Canvas(viewport, bg=BG, highlightthickness=0,
                                       width=PANEL_WIDTH + 28)
        scrollbar = tk.Scrollbar(viewport, orient="vertical", command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=scrollbar.set)
        self.scroll_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        outer = tk.Frame(self.scroll_canvas, bg=BG, padx=14, pady=12)
        self._scroll_window = self.scroll_canvas.create_window((0, 0), window=outer, anchor="nw")
        outer.bind("<Configure>", self._update_scroll_region)
        self.scroll_canvas.bind("<Configure>", self._resize_scroll_content)
        self.scroll_canvas.bind("<Enter>", self._enable_mousewheel)
        self.scroll_canvas.bind("<Leave>", self._disable_mousewheel)
        # Clicking anywhere but a text field releases it, so its cursor stops blinking.
        self.root.bind_all("<ButtonPress-1>", self._release_entry_focus, add="+")

        self._build_header(outer)
        self._build_decision_card(outer)
        self._build_blind_controls(outer)
        self._build_metrics(outer)
        self._build_box_controls(outer)
        tk.Label(outer, text="Auto-play sends the decision to the local research table only. It never clicks other windows.",
                 bg=BG, fg=FAINT, font=(FONT, 10), wraplength=PANEL_WIDTH, justify="left").pack(anchor="w", pady=(8, 0))

        self.root.update_idletasks()
        content_height = outer.winfo_reqheight()
        max_height = max(240, self.root.winfo_screenheight() - MENU_BAR_HEIGHT - DOCK_MARGIN)
        self.root.geometry("%dx%d" % (PANEL_WIDTH + 44, min(content_height, max_height)))

    def _update_scroll_region(self, _event=None):
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def _resize_scroll_content(self, event):
        self.scroll_canvas.itemconfigure(self._scroll_window, width=event.width)

    def _enable_mousewheel(self, _event=None):
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        self.root.bind_all("<Button-4>", self._on_mousewheel)
        self.root.bind_all("<Button-5>", self._on_mousewheel)

    def _disable_mousewheel(self, _event=None):
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")

    def _release_entry_focus(self, event=None):
        if event is None or not isinstance(event.widget, tk.Entry):
            self.root.focus_set()

    def _on_mousewheel(self, event):
        if getattr(event, "num", None) == 4:
            amount = -1
        elif getattr(event, "num", None) == 5:
            amount = 1
        else:
            amount = -int(event.delta / 120) if event.delta else 0
            if amount == 0 and event.delta:
                amount = -1 if event.delta > 0 else 1
        if amount:
            self.scroll_canvas.yview_scroll(amount, "units")

    def _build_header(self, parent):
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x")
        tk.Label(row, text="Vision Policy", bg=BG, fg=TEXT, font=(FONT, 16, "bold")).pack(side="left")
        self.toggle = self._button(row, "Start", self.toggle_running, GREEN, "#0d1117")
        self.toggle.configure(width=6)
        self.toggle.pack(side="right")
        self.status_label = tk.Label(parent, bg=BG, font=(FONT, 11, "bold"), anchor="w")
        self.status_label.pack(fill="x", pady=(6, 0))
        self.detail_label = tk.Label(parent, text="Reads the table inside the yellow box.", bg=BG,
                                     fg=MUTED, font=(FONT, 11), anchor="w", justify="left",
                                     wraplength=PANEL_WIDTH)
        self.detail_label.pack(fill="x", pady=(0, 10))
        self._button(parent, 'Read now', self.read_now, BORDER, TEXT).pack(fill='x', pady=(0, 6))
        self._set_status("stopped")
        self.autoplay_button = self._button(parent, "Auto-play · OFF", self.toggle_autoplay, BORDER, TEXT)
        self.autoplay_button.pack(fill="x", pady=(0, 3))
        self.autoplay_hint = tk.Label(
            parent, text=self._autoplay_hint(),
            bg=BG, fg=MUTED, font=(FONT, 10), anchor="w",
            justify="left", wraplength=PANEL_WIDTH,
        )
        self.autoplay_hint.pack(fill="x", pady=(0, 10))

    def _card(self, parent, title: str) -> tk.Frame:
        frame = tk.Frame(parent, bg=CARD, padx=12, pady=10, highlightthickness=1, highlightbackground=BORDER)
        frame.pack(fill="x", pady=(0, 8))
        tk.Label(frame, text=title.upper(), bg=CARD, fg=FAINT, font=(FONT, 9, "bold")).pack(anchor="w")
        return frame

    def _build_decision_card(self, parent):
        card = self._card(parent, "Recommended action")
        self.action_label = tk.Label(card, text="—", bg=CARD, fg=MUTED, font=(FONT, 30, "bold"), justify='left')
        self.action_label.pack(anchor="w")
        self.hand_label = tk.Label(card, text="Hand  —", bg=CARD, fg=TEXT, font=(FONT, 13), anchor="w")
        self.hand_label.pack(fill="x", pady=(4, 0))
        self.board_label = tk.Label(card, text="Board  —", bg=CARD, fg=TEXT, font=(FONT, 13), anchor="w")
        self.board_label.pack(fill="x")
        self.table_label = tk.Label(card, text="Pot —  ·  To call —  ·  Players —", bg=CARD, fg=MUTED,
                                    font=(FONT, 11), anchor="w")
        self.table_label.pack(fill="x", pady=(4, 0))
        self.stakes_label = tk.Label(card, text='SB — · BB —', bg=CARD, fg=TEXT, font=(FONT, 11),
                                     justify='left', anchor='w', wraplength=PANEL_WIDTH - 26)
        self.stakes_label.pack(fill='x', pady=(4, 0))
        self.sizing_label = tk.Label(card, text='', bg=CARD, fg=TEXT, font=(FONT, 11),
                                     justify='left', anchor='w', wraplength=PANEL_WIDTH - 26)
        self.sizing_label.pack(fill='x', pady=(4, 0))
        self.strategy_label = tk.Label(card, text='Strategy: contextual · position unknown', bg=CARD, fg=MUTED,
                                       font=(FONT, 10), justify='left', anchor='w', wraplength=PANEL_WIDTH - 26)
        self.strategy_label.pack(fill='x', pady=(4, 0))

    def _build_blind_controls(self, parent):
        card = self._card(parent, 'Table blinds')
        row = tk.Frame(card, bg=CARD)
        row.pack(fill='x', pady=(4, 0))
        self.blind_entries = {}
        for key, name in (('small_blind', 'Small blind'), ('big_blind', 'Big blind')):
            cell = tk.Frame(row, bg=CARD)
            cell.pack(side='left', expand=True, fill='x', padx=(0, 6))
            tk.Label(cell, text=name, bg=CARD, fg=MUTED, font=(FONT, 10)).pack(anchor='w')
            entry = tk.Entry(cell, width=10, bg=BG, fg=TEXT, insertbackground=TEXT, font=('Menlo', 12))
            entry.pack(fill='x')
            entry.bind('<Return>', lambda _event: self.apply_blinds())
            self.blind_entries[key] = entry
        self._button(card, 'Apply blinds', self.apply_blinds, BORDER, TEXT).pack(fill='x', pady=(8, 0))
        tk.Label(card, text='Example: 0.05 / 0.10. Both blank = read from screen. Use the table’s displayed units.',
                 bg=CARD, fg=MUTED, font=(FONT, 10), justify='left', wraplength=PANEL_WIDTH - 26).pack(fill='x', pady=(4, 0))

    def apply_blinds(self):
        try:
            blinds = parse_blinds(self.blind_entries['small_blind'].get(), self.blind_entries['big_blind'].get())
        except ValueError as error:
            self._set_status('rejected', str(error))
            return
        self.blind_overrides = blinds
        self.generation += 1  # Discard any pending result using the previous stakes.
        self._reset_turn()
        self.last_decision = None
        self.action_label.configure(text='—', fg=MUTED)
        self.sizing_label.configure(text='')
        self.stakes_label.configure(text=('SB %s · BB %s (manual)' % (
            format_money(blinds['small_blind']), format_money(blinds['big_blind']))) if blinds else 'SB — · BB — (read from screen)')
        self._set_status('live' if self.running else 'stopped',
                         'Blinds applied. Next read uses these settings.' if blinds else 'Manual blinds cleared. Next read uses screen values.')

    def _build_metrics(self, parent):
        card = self._card(parent, "Metrics")
        for key, name, explanation in METRICS:
            row = tk.Frame(card, bg=CARD)
            row.pack(fill="x", pady=(6, 0))
            tk.Label(row, text=name, bg=CARD, fg=TEXT, font=(FONT, 12, "bold")).pack(side="left")
            self.metric_values[key] = tk.Label(row, text="—", bg=CARD, fg=TEXT, font=("Menlo", 12))
            self.metric_values[key].pack(side="right")
            if key != "latency":
                bar = tk.Canvas(card, height=BAR_HEIGHT, bg=BORDER, highlightthickness=0, width=PANEL_WIDTH - 26)
                bar.pack(fill="x", pady=(3, 0))
                self.metric_bars[key] = bar
            tk.Label(card, text=explanation, bg=CARD, fg=MUTED, font=(FONT, 10), justify="left", anchor="w",
                     wraplength=PANEL_WIDTH - 26).pack(fill="x", pady=(2, 0))

    def _build_box_controls(self, parent):
        card = self._card(parent, "Capture box (screen points)")
        row = tk.Frame(card, bg=CARD)
        row.pack(fill="x", pady=(6, 0))
        self.entries = {}
        for key, value in zip(("left", "top", "right", "bottom"), self.region.bbox):
            cell = tk.Frame(row, bg=CARD)
            cell.pack(side="left", padx=(0, 6))
            tk.Label(cell, text=key.capitalize(), bg=CARD, fg=MUTED, font=(FONT, 10)).pack(anchor="w")
            entry = tk.Entry(cell, width=5, bg="#0d1117", fg=TEXT, insertbackground=TEXT, relief="flat",
                             highlightthickness=1, highlightbackground=BORDER, highlightcolor=BLUE,
                             font=("Menlo", 12))
            entry.insert(0, str(value))
            entry.bind("<Return>", lambda _event: (self.apply_region(), self._release_entry_focus()))
            entry.bind("<Escape>", lambda _event: self._release_entry_focus())
            entry.pack()
            self.entries[key] = entry
        self._button(card, "Apply box", self.apply_region, BORDER, TEXT).pack(fill="x", pady=(8, 0))
        tk.Label(card, text="Drag the yellow edges or corners to resize, or the tab to move.",
                 bg=CARD, fg=MUTED, font=(FONT, 10), justify="left", anchor="w",
                 wraplength=PANEL_WIDTH - 26).pack(fill="x", pady=(6, 0))

    def _button(self, parent, text: str, command, bg: str, fg: str) -> tk.Label:
        # Labels instead of tk.Button: macOS ignores Button background colours.
        button = tk.Label(parent, text=text, bg=bg, fg=fg, font=(FONT, 12, "bold"), padx=10, pady=6,
                          cursor="hand2")
        button.bind("<Button-1>", lambda _event: command())
        return button

    def _place_controls(self):
        """Keep the control panel outside the capture box so it never covers the table."""
        self.root.update_idletasks()
        panel_width = self.root.winfo_reqwidth()
        screen_width = self.root.winfo_screenwidth()
        if self.region.right + CONTROLS_GAP + panel_width <= screen_width:
            x = self.region.right + CONTROLS_GAP
        else:
            x = max(0, self.region.left - CONTROLS_GAP - panel_width)
        lowest_top = self.root.winfo_screenheight() - DOCK_MARGIN - self.root.winfo_reqheight()
        y = max(MENU_BAR_HEIGHT, min(self.region.top, lowest_top))
        self.root.geometry("+%d+%d" % (x, y))

    # ---------- state updates ----------

    def _set_status(self, kind: str, detail: Optional[str] = None):
        text, color = STATUS_STYLES[kind]
        self.status_label.configure(text=text, fg=color)
        if detail is not None:
            self.detail_label.configure(text=detail, fg=AMBER if kind in ('rejected', 'error') else MUTED)

    def _set_bar(self, key: str, fraction: float, color: str):
        bar = self.metric_bars[key]
        bar.delete("all")
        width = int(bar.winfo_width() or bar["width"])
        bar.create_rectangle(0, 0, int(width * max(0.0, min(1.0, fraction))), BAR_HEIGHT, fill=color, width=0)

    def _show_decision(self, decision: Decision, capture_ms: float, total_ms: float):
        self.session_metrics.record(decision)
        color = ACTION_COLORS.get(decision.action, TEXT)
        self.action_label.configure(text=format_action(decision), fg=color)
        state = decision.state or {}
        self.stakes_label.configure(text=format_stakes(state))
        self.sizing_label.configure(text=format_bet_sizing(decision))
        self.strategy_label.configure(text='%s · %s\n%s\n%s' % (
            self.agent.policy_name, state.get('hero_position', 'unknown'),
            format_strategy_context(state), decision.reason))
        self.hand_label.configure(text="Hand   " + format_cards(state.get("hero", [])))
        self.board_label.configure(text="Board  " + format_cards(state.get("board", [])))
        self.table_label.configure(text="Pot %s  ·  To call %s  ·  Players %s" % (
            format_money(state['pot'], state.get('money_precision')),
            format_money(state['to_call'], state.get('money_precision')), state.get("active_players", "—")))
        self.metric_values["equity"].configure(text="%.0f%%" % (decision.equity * 100))
        self._set_bar("equity", decision.equity, GREEN if decision.equity >= 0.5 else BLUE)
        self.metric_values["bluff"].configure(text="%.0f%%" % (decision.bluff_probability * 100))
        self._set_bar("bluff", decision.bluff_probability, AMBER)
        stages = decision.stages_ms
        self.metric_values["latency"].configure(text="%.0f ms" % total_ms)
        aggregate = self.session_metrics.as_dict()
        self._set_status("live", "Session %d decisions · mean %.0f ms · equity %.0f%% · "
                                 "capture %.0f · read %.0f · decide %.0f ms" % (
                                     aggregate["decisions"],
                                     aggregate["median_latency_ms"],
                                     aggregate["mean_equity"] * 100,
                                     capture_ms, stages.get("perception_ms", 0), stages.get("policy_ms", 0)))
        self._remember_decision(decision)

    def _remember_decision(self, decision: Decision):
        """Keep this read locally and on the research-table session."""
        try:
            payload = observation_from_decision(decision)
            self.observed.observe(payload)
        except (ValueError, OSError, TypeError):
            return
        threading.Thread(target=self._post_observation, args=(payload,), daemon=True).start()

    def _post_observation(self, payload: dict):
        try:
            submit_observation(self.table_url, payload)
        except (ConnectionError, ValueError, OSError):
            return

    def _on_box_dragged(self, region: CaptureRegion):
        if region != self.region:
            self._selection_changed()
        self.region = region
        for key, value in zip(("left", "top", "right", "bottom"), region.bbox):
            self.entries[key].delete(0, "end")
            self.entries[key].insert(0, str(value))

    def apply_region(self):
        try:
            values = {key: int(entry.get()) for key, entry in self.entries.items()}
        except ValueError:
            self._set_status("rejected", "Box values must be whole numbers.")
            return
        if values["right"] <= values["left"] or values["bottom"] <= values["top"]:
            self._set_status("rejected", "Right must exceed Left and Bottom must exceed Top.")
            return
        self._selection_changed()
        self.region = CaptureRegion(values["left"], values["top"], values["right"], values["bottom"])
        self.box.set_region(self.region)
        self._place_controls()
        self._reset_turn()
        self._set_status("live" if self.running else "stopped", "Box updated.")

    def _autoplay_hint(self):
        if self.autoplay_target == 'local':
            return ('On. Sending decisions to %s.' % self.table_url if self.autoplay else
                    'Off. Auto-play controls the local research table. Leave its own Auto-play unchecked.')
        return ('On. Playing only inside your yellow box. Stop or moving the box cancels pending input.'
                if self.autoplay else
                'Off. Point the yellow box at one table, then turn Auto-play on to play there.')

    def _refresh_autoplay(self):
        self.autoplay_button.configure(
            text='Auto-play · ON' if self.autoplay else 'Auto-play · OFF',
            bg=GREEN if self.autoplay else BORDER,
            fg='#0d1117' if self.autoplay else TEXT,
        )
        self.autoplay_hint.configure(text=self._autoplay_hint())

    def _make_screen_player(self):
        if self.agent.reader_mode != 'ai' or self.agent.ai_reader is None:
            raise ValueError('Selected-window Auto-play requires the AI reader and a configured Gemini key.')
        try:
            import pyautogui
        except ImportError as error:
            raise ValueError('Install screen controls with .venv/bin/python -m pip install -r requirements-screen.txt') from error
        from .selected_area_play import SelectedAreaPlayer
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = .15
        return SelectedAreaPlayer(self.agent, self.executor, self._capture,
                                  lambda: (self.region, self.generation), pyautogui)

    def _disable_autoplay(self, message=None):
        self.autoplay = False
        if self._screen_player is not None:
            self._screen_player.cancel()
        self._refresh_autoplay()
        if message:
            self._set_status('rejected', message + ' Auto-play is off.')

    def _selection_changed(self):
        self.generation += 1
        self._request_image = None
        if self.autoplay:
            self._disable_autoplay('Selection changed.')
        elif self._screen_player is not None:
            self._screen_player.cancel()
        self._reset_turn()

    def toggle_autoplay(self):
        if self.autoplay:
            self._disable_autoplay()
            return
        if self.autoplay_target == 'screen':
            try:
                if self._screen_player is None:
                    self._screen_player = self._make_screen_player()
                self._screen_player.resume()
            except Exception as error:
                self._disable_autoplay(str(error))
                return
        self.autoplay = True
        self.generation += 1
        self._request_image = None
        self._reset_turn()
        self._refresh_autoplay()
        if not self.running:
            self.toggle_running()

    def _play_decision(self, decision: Decision, signature: Optional[bytes]):
        """Submit one fresh decision. The same button strip is never played twice."""
        if not self.autoplay or signature is None or signature == self._played_signature:
            return
        if time.monotonic() < self._play_not_before:
            return
        if self.autoplay_target == 'screen':
            if not self.running or self._request_image is None:
                return
            from .screen_play import ScreenChanged
            try:
                played = self._screen_player.start(decision, self._request_image, self.blind_overrides)
                self._played_signature = signature
                if played:
                    self._set_status('live', 'Verifying the entered bet…' if self._screen_player.busy else
                                     'Action sent inside your selected box.')
            except ScreenChanged:
                self._request_image = None
                self._reset_turn()
                self._set_status('waiting', 'Table changed while reading. Reading a fresh decision…')
            except Exception as error:
                self._disable_autoplay(str(error))
            return
        try:
            log = submit_table_action(self.table_url, decision.action, decision.amount, note={
                'equity': decision.equity,
                'bluff_probability': decision.bluff_probability,
                'reason': decision.reason,
            })
        except ConnectionError as error:
            self._play_not_before = time.monotonic() + 2
            self._set_status('rejected', str(error))
            return
        except ValueError as error:
            self._played_signature = signature
            self._set_status('rejected', str(error))
            return
        self._played_signature = signature
        self._set_status('live', 'Played on the research table: %s' % log)

    def _reset_turn(self):
        """Allow one new read the next time the action buttons are up."""
        self._turn_state = 'idle'
        self._turn_token = None
        self._locked_sig = None
        self._request_sig = None
        self._buttons_visible = False
        if self._probe is not None:
            self._probe[0].cancel()
            self._probe = None

    def read_now(self):
        """One explicitly requested read, even if local turn detection is unavailable."""
        if self.pending is not None:
            self._set_status('waiting', 'A read is already in progress.')
            return
        self._read_once_requested = True
        if self._probe is not None:
            self._probe[0].cancel()
            self._probe = None
        if not self.running:
            self.toggle_running()

    def _inspect_turn(self, image):
        if self.autoplay_target == 'screen' and self.autoplay:
            return True, watch_signature(image)
        if self.agent.reader_mode == 'ai':
            return self.turn_detector.inspect(image)
        return buttons_up(self._buttons_visible, button_fraction(image)), decision_signature(image)

    def _snapshot_changed(self, previous, current):
        if self.autoplay_target == 'screen' and self.autoplay:
            return watch_changed(previous, current)
        return previous != current if self.agent.reader_mode == 'ai' else watch_changed(previous, current)

    def _capture(self):
        """Capture without the overlay's border, restoring it even on failure."""
        self.box.set_visible(False)
        try:
            self.root.update_idletasks()
            box_size = (self.region.right - self.region.left, self.region.bottom - self.region.top)
            output_size = box_size if self.agent.reader_mode == 'ai' else capture_size(box_size)
            return ScreenCapture(scale_region(self.region, self.pixel_scale), output_size).grab()
        finally:
            self.box.set_visible(True)

    def _begin_read(self, image, started, capture_ms, signature=None, manual=False):
        self._request_image = image
        self._turn_serial += 1
        self._request_sig = signature if signature is not None else decision_signature(image)
        self.request_manual = manual
        self.request_token = b'%d' % self._turn_serial
        self.request_started = started
        self.request_region = self.region
        self.request_generation = self.generation
        self.capture_ms = capture_ms
        if self.last_decision is None:
            self.action_label.configure(text='Reading…', fg=MUTED)
            self.sizing_label.configure(text='')
        self._set_status('live', 'Reading your turn…' if self.last_decision
                         else 'Reading selected table and calculating…')
        self.pending = self.executor.submit(self.agent.decide, image, dict(self.blind_overrides))
        self._turn_state = 'reading'

    def _keep_decision(self):
        """Leave the current recommendation on screen and retry auto-play once."""
        if self.last_decision is not None and self._turn_state == 'answered':
            self._set_status('live', 'Decision stays up for this turn.')
            self._play_decision(self.last_decision, self._turn_token)
            return
        self._set_status('waiting', 'Waiting for your action buttons. Use Read now if a turn is missed.')

    def toggle_running(self):
        if self._timer is not None:
            self.root.after_cancel(self._timer)
            self._timer = None
        self.generation += 1
        self.running = not self.running
        self.toggle.configure(text="Stop" if self.running else "Start", bg=RED if self.running else GREEN)
        if self.running:
            self._reset_turn()
            self._played_signature = None
            self._set_status("live", "Starting…")
            self._tick()
        else:
            self._disable_autoplay()
            self.action_label.configure(text='—', fg=MUTED)
            self.sizing_label.configure(text='')
            self._set_status("stopped", "Paused. Press Start to resume.")

    def _tick(self):
        self._timer = None
        if not self.running:
            return
        if self.autoplay_target == 'screen' and self.autoplay and self._screen_player is not None:
            try:
                was_busy = self._screen_player.busy
                if self._screen_player.poll():
                    self._timer = self.root.after(50, self._tick)
                    return
                if was_busy:
                    self._set_status('live', 'Verified bet sent inside your selected box.')
            except Exception as error:
                self._disable_autoplay(str(error))
        if self.pending is not None:
            if not self.pending.done():
                self._timer = self.root.after(50, self._tick)
                return
            future, self.pending = self.pending, None
            retry_ms = max(1000, self.interval_ms)
            same_request = self.request_region == self.region and self.request_generation == self.generation
            try:
                decision = future.result()
                age = time.perf_counter() - self.request_started
                if same_request and not self.request_manual:
                    current = self._capture()
                    self._buttons_visible, signature = self._inspect_turn(current)
                    same_request = self._buttons_visible and not self._snapshot_changed(
                        self._request_sig, signature)
                if not same_request:
                    self._turn_state = 'idle'
                    if self.last_decision is None:
                        self.action_label.configure(text='—', fg=MUTED)
                    self._set_status('waiting', 'Turn or box changed. Waiting for your next decision…')
                else:
                    self._turn_state = 'rejected' if self.request_manual else 'answered'
                    self._locked_sig = self._request_sig
                    self._turn_token = None if self.request_manual else self.request_token
                    self.last_decision = decision
                    self._show_decision(decision, self.capture_ms, age * 1000)
                    model = getattr(self.agent.ai_reader, 'last_model', None)
                    self.detail_label.configure(text='%sSnapshot read %.1f seconds ago. Verify cards and amounts.' % (
                        (model + ' · ') if model else '', age))
                    self._play_decision(decision, self._turn_token)
            except NotHeroTurn:
                if same_request and self._screen_player is not None:
                    self._screen_player.waiting()
                self._turn_state = 'rejected' if same_request else 'idle'
                if same_request:
                    self._locked_sig = self._request_sig
                if self.last_decision is None:
                    self.action_label.configure(text='—', fg=MUTED)
                self._set_status('waiting', 'Waiting for your turn. The decision stays up until the next one.'
                                 if self.last_decision else
                                 'Waiting for your turn. Reading stays off until your action buttons are up.')
            except AiScreenReaderError as error:
                self._turn_state = 'idle'
                if self.last_decision is None:
                    self.action_label.configure(text='—', fg=MUTED)
                observed = error.observation or {}
                self.hand_label.configure(text='Hand   ' + format_cards(observed.get('hero', [])))
                self.board_label.configure(text='Board  ' + format_cards(observed.get('board', [])))
                if self.last_decision is None:
                    self.table_label.configure(text='No current decision')
                if error.fatal:
                    self.running = False
                    self.toggle.configure(text='Start', bg=GREEN)
                    self._set_status('error', str(error) + ' Reader paused.')
                    return
                retry_ms = max(retry_ms, int(error.retry_after * 1000) + 100)
                suffix = (' Retrying in %.0f seconds.' % error.retry_after) if error.retry_after else ''
                self._set_status('waiting' if error.retry_after else 'rejected', str(error) + suffix)
            except Exception as error:
                self._turn_state = 'idle'
                if self.last_decision is None:
                    self.action_label.configure(text='—', fg=MUTED)
                self._set_status('rejected', str(error))
            self._timer = self.root.after(retry_ms, self._tick)
            return
        reader = self.agent.ai_reader
        if reader is not None and reader.retry_in > 0:
            delay = reader.retry_in
            self._timer = self.root.after(int(delay * 1000) + 100, self._tick)
            return
        started = time.perf_counter()
        hold = False
        try:
            if self._read_once_requested:
                self._read_once_requested = False
                image = self._capture()
                self._begin_read(image, started, (time.perf_counter() - started) * 1000, manual=True)
                self._timer = self.root.after(50, self._tick)
                return
            if self._probe is None:
                image = self._capture()
                capture_ms = (time.perf_counter() - started) * 1000
                self._probe = (self.executor.submit(self._inspect_turn, image), image, started, capture_ms)
            if not self._probe[0].done():
                self._timer = self.root.after(50, self._tick)
                return
            probe, image, started, capture_ms = self._probe
            self._probe = None
            self._buttons_visible, signature = probe.result()
            changed = self._snapshot_changed(self._locked_sig, signature)
            if self._turn_state == 'reading':
                self._turn_state = 'idle'
            if self.autoplay_target == 'screen' and self.autoplay:
                start = self._turn_state != 'reading' and (self._turn_state == 'idle' or changed)
                state = 'reading' if start else self._turn_state
            else:
                state, start = turn_gate(self._turn_state, self._buttons_visible, changed)
            if start:
                self._begin_read(image, started, capture_ms, signature=signature)
            else:
                self._turn_state = state
                hold = True
                self._keep_decision()
        except TableNotFoundError:
            self.session_metrics.record_frame("error")
            self._set_status("rejected", "No table in the box. Drag the box so it surrounds the whole "
                                         "table, and keep the browser at 100% zoom.")
        except NotHeroTurn:
            self.session_metrics.record_frame("waiting")
            # Keep the last decision on screen; there is simply nothing to decide yet.
            self._set_status("waiting", "Table found. Waiting for your turn…")
        except (ValueError, TypeError, KeyError):
            self.session_metrics.record_frame("error")
            self._set_status("rejected", "Found the table but couldn't read this frame. Retrying…")
        except RuntimeError as error:
            self._set_status('rejected', str(error))
            hold = True
        delay = 50 if self.pending else (WATCH_INTERVAL_MS if hold else self.interval_ms)
        self._timer = self.root.after(delay, self._tick)

    def close(self):
        self.running = False
        if self._screen_player is not None:
            self._screen_player.cancel()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.box.destroy()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Read and optionally autoplay the table inside the selected box")
    # The local table page renders its 1000x720 image at the browser content
    # origin by default. Keeping this default exact prevents the capture from
    # clipping the top/left of the table before CV gets a chance to align it.
    parser.add_argument("--bbox", default="0,0,1000,720", help="left,top,right,bottom")
    parser.add_argument("--interval-ms", type=int, default=250)
    parser.add_argument("--policy", choices=sorted(POLICIES), default="contextual")
    parser.add_argument('--reader', choices=('ai', 'hybrid', 'local'), default='ai',
                        help='AI for unfamiliar layouts; hybrid/local for the simulator')
    parser.add_argument('--table-url', default='http://127.0.0.1:8780',
                        help='Local research table that Auto-play controls')
    parser.add_argument('--autoplay-target', choices=('screen', 'local'), default='screen',
                        help='Auto-play inside the selected box (default), or use the local table API')
    args = parser.parse_args()
    from .table_display import local_table_url
    try:
        table_url = local_table_url(args.table_url)
    except ValueError as error:
        parser.error(str(error))
    values = [int(value) for value in args.bbox.split(",")]
    if len(values) != 4:
        parser.error("--bbox must be left,top,right,bottom")
    if tk.TkVersion < 8.6:
        print("The overlay requires Tk 8.6 or newer; use the project's .venv Python.", file=sys.stderr)
        return 2
    root = tk.Tk()
    overlay = ResearchOverlay(root, CaptureRegion(*values), args.interval_ms, args.policy, args.reader,
                              table_url=table_url, autoplay_target=args.autoplay_target)
    root.protocol("WM_DELETE_WINDOW", overlay.close)
    root.mainloop()


if __name__ == "__main__":
    raise SystemExit(main())

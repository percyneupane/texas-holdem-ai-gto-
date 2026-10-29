"""Resolve controls from each observed frame and verify typed amounts before submitting."""

from decimal import Decimal

from .screen_play import ScreenController, money, spot_key


def control_profile(table_ui, region, action):
    """Reject ambiguous metadata before converting normalized bounds to desktop points."""
    if (not isinstance(table_ui, dict) or table_ui.get('game') != 'nlhe' or
            type(table_ui.get('table_count')) is not int or table_ui['table_count'] != 1):
        raise ValueError('Automatic controls require one unambiguous No-Limit Hold’em table.')
    entries = table_ui.get('controls')
    if not isinstance(entries, list) or not 1 <= len(entries) <= 8:
        raise ValueError('No unambiguous visible table controls.')
    controls = {}
    width, height = region.right - region.left, region.bottom - region.top
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Invalid control observation.')
        role, bounds = entry.get('role'), entry.get('bounds')
        if role not in ('fold', 'check', 'call', 'bet', 'raise', 'amount') or role in controls:
            raise ValueError('Unknown or duplicate action control.')
        if (not isinstance(bounds, list) or len(bounds) != 4 or
                any(type(value) is not int or not 0 <= value <= 1000 for value in bounds) or
                bounds[0] >= bounds[2] or bounds[1] >= bounds[3]):
            raise ValueError('Control bounds are outside the capture or inverted.')
        if type(entry.get('enabled')) is not bool:
            raise ValueError('Unknown control availability.')
        label = entry.get('label')
        kind = entry.get('kind')
        if not isinstance(label, str) or not label.strip() or len(label) > 120:
            raise ValueError('Unlabeled table control.')
        if kind != ('input' if role == 'amount' else 'button'):
            raise ValueError('Unsupported table control type: %s.' % role)
        # Keep disabled entries in the duplicate check, but never expose them
        # as clickable controls.
        controls[role] = ([int((bounds[0] + bounds[2]) * width / 2000),
                           int((bounds[1] + bounds[3]) * height / 2000)]
                          if entry['enabled'] else None)
    if controls.get(action) is None:
        raise ValueError('The requested action is missing, disabled or ambiguous.')
    mode = table_ui.get('amount_mode')
    if action in ('raise', 'bet'):
        if (mode not in ('raise_to', 'chips_to_add') or
                table_ui.get('amount_units') not in ('chips', 'bb') or
                table_ui.get('decimal_separator') not in ('dot', 'comma') or
                controls.get('amount') is None):
            raise ValueError('Bet input semantics are unknown or unsupported.')
    return dict(bbox=list(region.bbox), amount_mode=mode if mode in ('raise_to', 'chips_to_add') else 'raise_to',
                controls={key: value for key, value in controls.items() if value is not None})


class AdaptiveScreenController(ScreenController):
    automatic = True

    def __init__(self, region, mouse, clock=None, ack_timeout=15):
        kwargs = dict(ack_timeout=ack_timeout)
        if clock is not None:
            kwargs['clock'] = clock
        super().__init__(dict(bbox=list(region.bbox), amount_mode='raise_to', controls={'fold': [0, 0]}),
                         mouse, **kwargs)
        self.table_ui = None

    def _resolve(self, decision):
        state = decision.state or {}
        observed_ui = state.get('table_ui')
        profile = control_profile(observed_ui, self.region, decision.action)
        resolved = ScreenController(profile, self.mouse, self.clock, self.ack_timeout)
        self.controls = resolved.controls
        self.amount_mode = resolved.amount_mode
        self.table_ui = observed_ui

    def format_amount(self, total, state):
        if self.table_ui['amount_units'] == 'bb':
            blind = money(state.get('big_blind'), 'Big blind')
            if blind <= 0:
                raise ValueError('BB input requires a known positive big blind.')
            # Require an exact representable amount rather than silently rounding.
            converted = total / blind
            if converted != converted.quantize(Decimal('.01')) or converted * blind != total:
                raise ValueError('The BB input cannot represent this amount exactly to two decimals.')
            total = converted
        text = super().format_amount(total, state)
        return text.replace('.', ',') if self.table_ui['decimal_separator'] == 'comma' else text

    def play(self, decision, observed, current, verify=None):
        self.check_timeout()
        if spot_key(decision.state or {}) == self.pending:
            return False
        if decision.action in ('bet', 'raise') and verify is None:
            raise ValueError('Automatic sizing requires input readback before submission.')
        return super().play(decision, observed, current, verify=verify)

    def prepare(self, decision, observed, current):
        self.check_timeout()
        if spot_key(decision.state or {}) == self.pending:
            return None
        self._resolve(decision)
        return super().prepare(decision, observed, current)

    def verify_entry(self, decision, text, readback):
        """Fresh reader result must match the spot, semantics, and typed input value."""
        state, after = decision.state or {}, readback.state or {}
        if (spot_key(state) != spot_key(after) or readback.action != decision.action or
                money(readback.amount, 'Read-back recommendation') != money(decision.amount, 'Recommendation')):
            return False
        current_ui = after.get('table_ui')
        if not isinstance(current_ui, dict):
            return False
        for key in ('amount_mode', 'amount_units', 'decimal_separator'):
            if current_ui.get(key) != self.table_ui.get(key):
                return False
        value = current_ui.get('amount_value')
        if money(value, 'Read-back input') != money(Decimal(text.replace(',', '.')), 'Entered amount'):
            return False
        # Control positions may change as the field is edited. Resolve the
        # fresh enabled submit button instead of reusing the earlier center.
        self._resolve(readback)
        return True

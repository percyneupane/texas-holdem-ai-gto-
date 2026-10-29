"""Display-unit money with exact cent conversion at the strategy boundary."""

import math
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

MONEY_FIELDS = ('pot', 'to_call', 'min_raise', 'stack', 'small_blind', 'big_blind',
                'hero_bet', 'current_bet', 'min_raise_to', 'max_raise_to')
CHIP_FIELDS = tuple(key for key in MONEY_FIELDS if key not in ('small_blind', 'big_blind'))
NESTED_CHIPS = (('opponents', ('stack', 'bet')), ('betting_history', ('amount', 'raise_to', 'pot_before')))


def money_decimal(value):
    if isinstance(value, bool):
        raise ValueError('Amounts must be numbers, not booleans.')
    try:
        amount = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as error:
        raise ValueError('Enter an amount such as 0.05, 0.10 or 10.') from error
    if not amount.is_finite() or amount < 0 or amount * 100 != (amount * 100).to_integral_value():
        raise ValueError('Amounts must be nonnegative, finite numbers with at most two decimal places.')
    return amount


def numeric_amount(value):
    amount = money_decimal(value)
    return int(amount) if amount == amount.to_integral_value() else float(amount)


def big_blind_count(amount, big_blind, places='0.1'):
    """Format amount/big_blind without binary float residue."""
    count = money_decimal(amount) / money_decimal(big_blind)
    return format(count.quantize(Decimal(places), rounding=ROUND_HALF_UP), 'f')


def scale_chip_amounts(state, factor):
    """Scale chip amounts into the big blind's unit. Blinds themselves stay put."""
    factor = Decimal(factor)
    scaled = dict(state)
    for key in CHIP_FIELDS:
        if key in scaled:
            scaled[key] = numeric_amount(money_decimal(scaled[key]) * factor)
    for field, keys in NESTED_CHIPS:
        if field in scaled and isinstance(scaled[field], list):
            scaled[field] = [dict(item, **{key: numeric_amount(money_decimal(item[key]) * factor)
                                          for key in keys if key in item})
                             for item in scaled[field] if isinstance(item, dict)]
    return scaled


def _plausible_depth(stack_bb, pot_bb):
    if not (Decimal(8) <= stack_bb <= Decimal(400)):
        return False
    if pot_bb is None:
        return True
    return Decimal(1) <= pot_bb <= Decimal(60)


def _is_decade(factor):
    """True for 10, 100, 1000 and their reciprocals, within a rounding tolerance."""
    if factor <= 0 or factor == 1:
        return False
    exponent = round(math.log10(float(factor)))
    if exponent == 0 or abs(exponent) > 3:
        return False
    return abs(factor - (Decimal(10) ** exponent)) <= Decimal('0.000001') * abs(factor)


def align_chip_units(state, reported_big_blind=None):
    """Repair chip amounts that are not in the same unit as the big blind.

    A reader often copies a "100 BB" stack label, or multiplies a 10.00 stack
    by 100, while the blind stays 0.10. Either makes stack/blind absurd.
    A depth that is already between 8 and 400 big blinds is left alone, so a
    real 100bb stack of 10.00 is not rewritten. Dividing by 100 or 1000 needs
    a positive pot that lands in range too, so an unread pot cannot turn a
    "100 BB" label into a 10bb stack. A conversion that would leave a fraction
    of a cent is skipped.
    """
    big_blind = state.get('big_blind') or 0
    stack = state.get('stack') or 0
    if not big_blind or not stack:
        return state
    blind = money_decimal(big_blind)
    stack_amount = money_decimal(stack)
    pot_amount = money_decimal(state.get('pot') or 0)
    if reported_big_blind:
        reported = money_decimal(reported_big_blind)
        if reported > 0 and reported != blind:
            factor = blind / reported
            reported_pot = pot_amount / reported if pot_amount else None
            if _is_decade(factor) and _plausible_depth(stack_amount / reported, reported_pot):
                scaled = _try_scale(state, factor)
                if scaled is not None:
                    return scaled
    stack_bb = stack_amount / blind
    pot_bb = pot_amount / blind if pot_amount else None
    if _plausible_depth(stack_bb, pot_bb):
        return state
    if pot_amount:
        for divisor in (Decimal(100), Decimal(1000)):
            factor = Decimal(1) / divisor
            if _plausible_depth(stack_bb * factor, pot_bb * factor):
                scaled = _try_scale(state, factor)
                if scaled is not None:
                    return scaled
    # The stack and pot numbers are already big-blind counts ("100" and "1.5")
    # while the blind is a cash stake such as 0.10.
    if (blind < 1 and Decimal(8) <= stack_amount <= Decimal(400)
            and (not pot_amount or Decimal('0.5') <= pot_amount <= Decimal(40))
            and _plausible_depth(stack_amount, pot_amount or None)):
        scaled = _try_scale(state, blind)
        if scaled is not None:
            return scaled
    return state


def _try_scale(state, factor):
    try:
        return scale_chip_amounts(state, factor)
    except (ValueError, InvalidOperation):
        return None


def policy_units(state):
    """Convert every monetary field together, never just the blinds.

    Decimal tables are calculated in integer cents; whole-chip simulator
    tables keep their original one-chip granularity and rounding behavior.
    """
    amounts = {key: money_decimal(state[key]) for key in MONEY_FIELDS if key in state}
    nested = [money_decimal(item[key]) for field, keys in (
        ('opponents', ('stack', 'bet')), ('betting_history', ('amount', 'raise_to', 'pot_before')))
        for item in state.get(field, []) for key in keys if key in item]
    scale = 100 if any(v != v.to_integral_value() for v in list(amounts.values()) + nested) else 1
    normalized = dict(state)
    normalized.update({key: int(value * scale) for key, value in amounts.items()})
    for field, keys in [('opponents', ('stack', 'bet')), ('betting_history', ('amount', 'raise_to', 'pot_before'))]:
        if field in state:
            normalized[field] = [dict(item, **{key: int(money_decimal(item[key]) * scale)
                                              for key in keys if key in item}) for item in state[field]]
    return normalized, scale


def format_money(value, precision=None):
    amount = money_decimal(value)
    if precision is None:
        precision = 0 if amount == amount.to_integral_value() else 2
    return format(amount, '.%df' % precision)


def display_units(amount, scale):
    return numeric_amount(Decimal(amount) / Decimal(scale))


def validate_blinds(values):
    if not values:
        return {}
    if set(values) != {'small_blind', 'big_blind'}:
        raise ValueError('Enter both small blind and big blind, or leave both blank.')
    sb, bb = money_decimal(values['small_blind']), money_decimal(values['big_blind'])
    if not 0 < sb <= bb:
        raise ValueError('Blinds must be positive amounts, with small blind ≤ big blind.')
    return {'small_blind': numeric_amount(sb), 'big_blind': numeric_amount(bb)}


def parse_blinds(small, big):
    if not small.strip() and not big.strip():
        return {}
    if not small.strip() or not big.strip():
        raise ValueError('Enter both blinds, or leave both blank for screen reading.')
    return validate_blinds({'small_blind': small, 'big_blind': big})

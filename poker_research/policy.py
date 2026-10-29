"""Betting policies that turn a decoded table state and hand equity into an action.

Amounts are chips to add on top of the hero's current street bet; the engine
clamps raises to the legal range.

``gto`` is a GTO-principled approximation, not a solved equilibrium: it uses
the closed-form results from simplified game-theory models (pot-odds calling,
polarized betting, and the bluff-to-value ratio that makes a bet size
unexploitable) instead of solving the full game.
"""

import random
from dataclasses import dataclass, replace
from typing import Callable, Dict
from .cards import Card
from .ranges import hand_class, preflop_chart_action, preflop_street_target
from .betting_context import enrich_context

# Polarized bet size as a fraction of the pot, used for both value bets and bluffs.
GTO_BET_FRACTION = 0.75
# Raise size, as a fraction of the pot after calling.
GTO_RAISE_FRACTION = 0.75
# Equity needed to bet / raise for value against the players still in.
GTO_VALUE_BET_EQUITY = 0.60
GTO_VALUE_RAISE_EQUITY = 0.70
# Hands below this equity have little showdown value, so they are the bluff candidates.
GTO_BLUFF_EQUITY_MAX = 0.30
# Bluff with at most this share of candidates on streets with cards to come.
GTO_MAX_BLUFF_PROBABILITY = 0.5
# Share of random heads-up hands that are value hands / bluff candidates by street,
# measured by sampling 300 random spots per street with this module's thresholds.
GTO_VALUE_SHARE = {"flop": 0.30, "turn": 0.34, "river": 0.40}
GTO_BLUFF_SHARE = {"flop": 0.17, "turn": 0.30, "river": 0.30}
STREETS = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}


@dataclass(frozen=True)
class PolicyChoice:
    action: str
    amount: int
    bluff_probability: float
    reason: str = ''


def street_of(state: Dict) -> str:
    return STREETS.get(len(state["board"]), "unknown")


def pot_odds(to_call: int, pot: int) -> float:
    """Equity a call needs to break even: the call's share of the final pot."""
    return to_call / (pot + to_call) if to_call > 0 else 0.0


def bluff_share_for_bet(bet_fraction: float) -> float:
    """Share of a betting range that is bluffs so the caller is indifferent: s / (1 + 2s)."""
    return bet_fraction / (1 + 2 * bet_fraction)


def gto_bluff_probability(street: str, opponents: int) -> float:
    """How often a bluff candidate bets so bluffs make up the balanced share of all bets."""
    if street not in GTO_VALUE_SHARE:
        return 0.0
    share = bluff_share_for_bet(GTO_BET_FRACTION)
    bluffs_per_value_hand = share / (1 - share)
    probability = bluffs_per_value_hand * GTO_VALUE_SHARE[street] / GTO_BLUFF_SHARE[street]
    if street != "river":
        probability = min(GTO_MAX_BLUFF_PROBABILITY, probability)
    # Each extra opponent is another player who can call, so bluffs work less often.
    return min(1.0, probability) / max(1, opponents)


def gto_policy(state: Dict, hand_equity: float, rng: random.Random) -> PolicyChoice:
    pot, to_call, stack = state["pot"], state["to_call"], state["stack"]
    min_raise = state["min_raise"]
    opponents = max(1, int(state["active_players"]) - 1)
    bluff_probability = gto_bluff_probability(street_of(state), opponents)
    if to_call > 0:
        if hand_equity >= GTO_VALUE_RAISE_EQUITY and stack > to_call:
            raise_by = max(min_raise, round((pot + to_call) * GTO_RAISE_FRACTION))
            return PolicyChoice("raise", min(stack, to_call + raise_by), bluff_probability)
        if hand_equity >= pot_odds(to_call, pot):
            return PolicyChoice("call", min(stack, to_call), bluff_probability)
        return PolicyChoice("fold", 0, bluff_probability)
    bet = min(stack, max(min_raise, round(pot * GTO_BET_FRACTION)))
    if hand_equity >= GTO_VALUE_BET_EQUITY:
        return PolicyChoice("bet", bet, bluff_probability)
    if hand_equity < GTO_BLUFF_EQUITY_MAX and rng.random() < bluff_probability:
        return PolicyChoice("bet", bet, bluff_probability)
    return PolicyChoice("check", 0, bluff_probability)


def baseline_policy(state: Dict, hand_equity: float, rng: random.Random) -> PolicyChoice:
    """The original fixed-threshold heuristic, kept for comparison."""
    pot, to_call, stack, min_raise = state["pot"], state["to_call"], state["stack"], state["min_raise"]
    bluff_probability = max(0.0, min(1.0, 0.18 + (0.5 - hand_equity) * 0.22))
    if hand_equity >= 0.66:
        return PolicyChoice("raise", min(stack, max(min_raise, int(pot * 0.65))), bluff_probability)
    if hand_equity >= 0.42:
        if to_call <= pot * 0.35:
            return PolicyChoice("call", min(stack, to_call), bluff_probability)
        return PolicyChoice("fold", 0, bluff_probability)
    if to_call == 0 and rng.random() < bluff_probability:
        return PolicyChoice("bet", min(stack, max(min_raise, int(pot * 0.5))), bluff_probability)
    if to_call == 0:
        return PolicyChoice("check", 0, bluff_probability)
    return PolicyChoice("fold", 0, bluff_probability)


def contextual_policy(state: Dict, hand_equity: float, rng: random.Random) -> PolicyChoice:
    """Evidence-aware cash heuristics, intentionally not a full GTO solver."""
    state = enrich_context(state)
    context = state['strategy_context']
    position = state.get('hero_position', 'unknown')
    bb = state.get('big_blind', 0)
    stack = state['stack']
    call = min(stack, state['to_call'])
    hero_bet = state.get('hero_bet', 0)
    can_raise = state.get('can_raise', True) and stack > call
    depth = context['effective_stack_bb']
    depth = depth if depth is not None else (stack + hero_bet) / bb if bb else 0

    def aggressive(target, reason, probability=0):
        if not can_raise:
            return PolicyChoice('call' if call else 'check', call, 0, reason + '; raising unavailable')
        minimum = state.get('min_raise_to', hero_bet + call + state['min_raise'])
        maximum = min(hero_bet + stack, state.get('max_raise_to', hero_bet + stack))
        target = max(minimum, round(target))
        # Avoid recommending an uncallable overbet against a known short stack.
        if context['effective_stack_bb'] is not None and bb:
            target = min(target, max(minimum, round(context['effective_stack_bb'] * bb)))
        amount = max(0, min(maximum, target) - hero_bet)
        return PolicyChoice('raise' if call else 'bet', amount, probability, reason)

    def passive(reason):
        return PolicyChoice('fold' if call else 'check', 0, 0, reason)

    if not state['board'] and bb and 30 <= depth <= 200:
        hand = hand_class([Card(*raw) for raw in state['hero']])
        chart = preflop_chart_action(hand, state, context)
        if chart is not None:
            action, kind, reason = chart
            depth_note = '%.1fbb effective depth%s' % (
                depth, ' (hero-only estimate)' if context['effective_stack_bb'] is None else '')
            reason = '%s; %s' % (reason, depth_note)
            if action == 'raise':
                if kind == 'open':
                    target = round(bb * (3 if position == 'SB' else 2.5))
                elif kind == 'iso':
                    base = 3.5 if position == 'BB' else 3 if position == 'SB' else 2.5
                    target = round(bb * base) + bb * max(1, state.get('limper_count', 1))
                else:
                    current = preflop_street_target(state)
                    ip = state.get('postflop_position') == 'IP'
                    multiplier = (3 if ip else 4) if kind == 'three_bet' else (2.2 if ip else 2.5)
                    target = current * multiplier + context['callers_since_raise'] * current
                    if kind == 'five_bet' or target - hero_bet > stack * .6:
                        target = hero_bet + stack
                return aggressive(target, reason)
            if action == 'call':
                return PolicyChoice('call' if call else 'check', call, 0, reason)
            if action == 'check':
                return PolicyChoice('check', 0, 0, reason)
            return passive(reason)

    if state['board']:
        opponents = max(1, state['active_players'] - 1)
        margin = {'IP': .02, 'OOP': .06}.get(state.get('postflop_position'), 0)
        # River/all-in removes future-street realization costs, not uncertainty
        # about a stronger range after observed aggression. We do not yet have
        # calibrated postflop range updating, so retain the pressure margins.
        if len(state['board']) == 5 or call == stack:
            margin = 0
        margin += .025 * (opponents - 1)
        margin += .06 if context['facing_postflop_raise'] else 0
        margin += .03 if (context['barrel_streets'] or 0) >= 2 else 0
        margin += .03 if (context['last_bet_pot_ratio'] or 0) > 1 else 0
        price = pot_odds(call, state['pot'])
        required = min(1, price + margin)
        context['adjusted_call_threshold'] = required
        label = 'postflop raise' if context['facing_postflop_raise'] else 'postflop bet'
        reason = '%s: price %.1f%%, adjusted call %.1f%%; %s, SPR %s' % (
            label, price * 100, required * 100, 'multiway' if context['multiway'] else 'heads-up',
            'unknown' if context['spr'] is None else '%.2f' % context['spr'])
        value_raise = min(.95, .70 + .05 * (opponents - 1) + .08 * context['facing_postflop_raise'])
        if call:
            if hand_equity >= value_raise and can_raise:
                return aggressive(hero_bet + call + (state['pot'] + call) * .75, reason + '; value raise')
            if hand_equity >= required:
                return PolicyChoice('call', call, 0, reason)
            return passive(reason)
        wet = context['board_texture'] in ('two-tone', 'monotone') or context['connected_board']
        fraction = .75 if wet else .33
        value_bet = min(.90, .60 + .05 * (opponents - 1))
        target = max(state['min_raise'], round(state['pot'] * fraction))
        if hand_equity >= value_bet:
            if context['spr'] is not None and context['spr'] <= 1 and hand_equity >= .70:
                target = hero_bet + stack
            return aggressive(target, reason + '; %s board value sizing' % context['board_texture'])
        # No automatic multiway air bluffs. Draws can semi-bluff heads-up;
        # low-equity air only uses the existing polarized river approximation.
        draw = context['flush_draw'] or context['open_ended']
        probability = .25 if draw and opponents == 1 and hand_equity >= .25 else 0
        if len(state['board']) == 5 and opponents == 1 and hand_equity < GTO_BLUFF_EQUITY_MAX:
            probability = gto_bluff_probability('river', 1)
        if probability and rng.random() < probability:
            return aggressive(target, reason + ('; semi-bluff draw' if draw else '; river bluff'), probability)
        return PolicyChoice('check', 0, probability, reason + '; check')

    choice = gto_policy(state, hand_equity, rng)
    reason = 'Equity fallback; position/chart context incomplete or outside 30–200bb assumptions'
    if choice.action in ('raise', 'bet'):
        return aggressive(hero_bet + choice.amount, reason, choice.bluff_probability)
    if not can_raise and choice.action in ('raise', 'bet'):
        choice = PolicyChoice('call' if call else 'check', min(stack, call), choice.bluff_probability)
    return replace(choice, reason=reason)


POLICIES: Dict[str, Callable[[Dict, float, random.Random], PolicyChoice]] = {
    "contextual": contextual_policy,
    "gto": gto_policy,
    "baseline": baseline_policy,
}

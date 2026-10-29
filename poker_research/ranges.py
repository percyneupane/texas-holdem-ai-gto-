"""100bb cash preflop charts. One action per hand.

Raise-first-in ranges are the 6-max lists published by PokerCoaching
(Jonathan Little's implementable 100bb charts, no ante). Facing-an-open
3-bet/call/fold ranges are the 6-max 100bb lists published by HoldemPro.
A mixed hand follows the action listed at 50% or more; a smaller mix is a
call when that page also lists the call, and a fold when it does not.
4-bet and 5-bet lines are the same books' pure-action simplification:
4-bet QQ+/AK plus A5s–A4s, continue QQ/AK against a 4-bet, and continue
only KK+ against a 5-bet or a cold 4-bet. Full-ring early seats use a
tighter subset of the 6-max opens. These charts are not a solver file
and they do not copy a paid chart image.
"""

from functools import lru_cache
from itertools import combinations

from .cards import RANKS, deck, evaluate

POSITIONS = ('UTG', 'UTG1', 'UTG2', 'UTG3', 'LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB')
PRESSURES = ('unopened', 'limped', 'raised', 'three_bet', 'four_bet', 'five_bet', 'unknown')
ROLES = ('open', 'limp', 'call', 'three_bet', 'four_bet', 'five_bet', 'call_three_bet', 'call_four_bet', 'unknown')
# 6-max raise-first-in. LJ is the first seat at a 6-handed table.
OPEN_6 = {
    'LJ': '66+ A3s+ K8s+ Q9s+ J9s+ T9s ATo+ KJo+ QJo',
    'HJ': '55+ A2s+ K6s+ Q9s+ J9s+ T9s 98s 87s 76s ATo+ KTo+ QTo+',
    'CO': '33+ A2s+ K3s+ Q6s+ J8s+ T7s+ 97s+ 87s 76s A8o+ KTo+ QTo+ JTo',
    'BTN': '33+ A2s+ K2s+ Q3s+ J4s+ T6s+ 96s+ 85s+ 75s+ 64s+ 53s+ A4o+ K8o+ Q9o+ J9o+ T8o+ 98o',
    'SB': '22+ A2s+ K2s+ Q2s+ J2s+ T3s+ 94s+ 84s+ 74s+ 63s+ 53s+ 43s A2o+ K4o+ Q5o+ J7o+ T7o+ 96o+ 86o+ 76o',
}
# Full ring only. Late seats keep the 6-max lists; early seats are tighter.
OPEN_EARLY = {
    'UTG': '77+ ATs+ A5s A4s KQs KJs QJs JTs T9s AJo+ KQo',
    'UTG1': '66+ A9s+ A5s A4s KTs+ QJs JTs T9s 98s ATo+ KQo',
    'UTG2': '66+ A5s+ KTs+ QTs+ JTs T9s 98s 87s ATo+ KJo+ QJo',
    'UTG3': '55+ A4s+ K9s+ QTs+ J9s+ T9s 98s 87s ATo+ KJo+ QJo',
}
HU_OPEN = ('22+ A2s+ A2o+ K2s+ K2o+ Q2s+ Q4o+ J2s+ J6o+ T2s+ T6o+ 92s+ 96o+ '
           '82s+ 85o+ 72s+ 74o+ 62s+ 64o+ 52s+ 53o+ 42s+ 43o 32s')
# (hero group, opener group) -> (3-bet, call). Groups: UTG MP CO BTN SB BB.
FACING_OPEN = {
    ('BB', 'UTG'): ('JJ+ AKs AKo',
                    'TT-33 AQs-A9s A5s-A2s KTs+ QTs+ JTs T9s 98s 87s 76s 65s 54s AQo'),
    ('BB', 'MP'): ('TT+ AQs+ AKo',
                   '99-22 AJs-A6s A5s-A2s KTs+ K9s K8s-K7s Q8s+ J9s+ T9s 98s 87s 76s 65s 54s AQo-ATo KTo+ QJo'),
    ('BB', 'CO'): ('TT+ AQs+ AQo+',
                   '99-22 AJs-A6s A5s-A2s KTs+ K9s K8s-K4s QTs+ Q9s Q8s-Q7s JTs J9s-J8s T8s+ 98s 87s 76s 65s 53s+ 43s AJo-ATo KTo+ QTo+ JTo'),
    ('BB', 'BTN'): ('TT+ AJs+ KQs AQo+ A5s A4s',
                    '99-22 ATs-A6s A3s-A2s KJs-K2s Q2s+ J5s+ T9s T8s-T5s 95s+ 85s+ 76s 75s-74s 65s 64s 53s+ 43s AJo-A5o K9o+ Q9o+ J9o+ T9o 98o'),
    ('BB', 'SB'): ('99+ AJs+ KQs AJo+ KQo A5s',
                   '88-22 ATs-A6s A4s-A2s KJs-KTs K9s K8s-K2s QTs+ Q9s Q8s-Q2s J6s+ T6s+ 95s+ 85s+ 76s 75s-74s 64s+ 53s+ 43s 32s ATo-A5o KJo-K4o Q6o+ J7o+ T7o+ 97o+ 87o 76o'),
    ('SB', 'UTG'): ('TT+ AJs+ KQs AKo',
                    '99-66 ATs A5s-A4s KJs-KTs QJs JTs T9s 98s 87s AQo'),
    ('SB', 'MP'): ('99+ AJs+ KJs+ QJs AQo+ KQo',
                   '88-55 ATs-A9s A5s-A4s KTs QTs JTs T9s 98s 87s 76s'),
    ('SB', 'CO'): ('88+ ATs+ KJs+ QJs JTs AQo+ A3s',
                   '77-44 A9s-A8s A5s-A4s KTs QTs T9s 98s 87s 76s AJo KQo'),
    ('SB', 'BTN'): ('77+ ATs+ KTs+ QJs JTs AJo+ KQo A5s-A3s',
                    '66-22 A9s-A6s K9s QTs-Q9s J9s T9s 98s 87s 76s 65s ATo KJo'),
    ('BTN', 'UTG'): ('JJ+ AQs+ AKo',
                     'TT-22 AJs-A9s A5s-A2s KTs+ QTs+ JTs T9s 98s 87s 76s 65s 54s AQo-AJo KQo'),
    ('BTN', 'MP'): ('JJ+ AJs+ KQs AQo+',
                    'TT-22 ATs-A2s KJs-K9s Q9s+ J9s+ T8s+ 97s+ 87s 76s 65s 54s AJo-ATo KJo+ QJo'),
    ('BTN', 'CO'): ('TT+ AJs+ KQs AQo+ A5s A4s',
                    '99-22 ATs-A6s A3s-A2s KJs-KTs K9s K8s-K7s QTs+ Q9s Q8s J8s+ T8s+ 97s+ 87s 76s 65s 54s AJo-ATo KTo+ QTo+ JTo'),
    ('CO', 'UTG'): ('JJ+ AQs+ AKo',
                    'TT-33 AJs-A9s A5s-A2s KTs+ QTs+ JTs T9s 98s 87s 76s 65s AQo KQo'),
    ('CO', 'MP'): ('JJ+ AQs+ KQs AKo',
                   'TT-22 AJs-A9s A5s-A2s KJs-KTs QTs+ JTs T9s 98s 87s 76s 65s 54s AQo-AJo KQo'),
    ('MP', 'UTG'): ('JJ+ AQs+ AKo',
                    'TT-55 AJs-ATs A5s-A4s KTs+ QJs JTs T9s 98s 87s 76s AQo'),
}
HUGE_3BET = 'JJ+ AKs AKo'
HUGE_CALL = 'TT AQs AQo'


def position_names(count):
    if count == 2:
        return ('BTN', 'BB')
    if count < 3 or count > 10:
        raise ValueError('Expected 2–10 dealt players')
    if count <= 6:
        return ('LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB')[-count:]
    return ('UTG', 'UTG1', 'UTG2', 'UTG3')[:count - 6] + ('LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB')


def hand_class(hand):
    a, b = sorted(hand, key=lambda c: c.rank, reverse=True)
    ranks = RANKS[a.rank - 2] + RANKS[b.rank - 2]
    return ranks if a.rank == b.rank else ranks + ('s' if a.suit == b.suit else 'o')


def _span(left, right):
    if len(left) == 2 and left[0] == left[1] and len(right) == 2 and right[0] == right[1]:
        low, high = sorted((RANKS.index(left[0]), RANKS.index(right[0])))
        return {RANKS[i] * 2 for i in range(low, high + 1)}
    if (len(left) == 3 and len(right) == 3 and left[0] == right[0] and left[2] == right[2]
            and left[0] != left[1] and right[0] != right[1]):
        low, high = sorted((RANKS.index(left[1]), RANKS.index(right[1])))
        return {left[0] + RANKS[i] + left[2] for i in range(low, high + 1)}
    raise ValueError('Invalid chart span %s-%s' % (left, right))


@lru_cache(maxsize=64)
def expand_chart(chart):
    result = set()
    for token in chart.split():
        if '-' in token:
            result.update(_span(*token.split('-')))
        elif not token.endswith('+'):
            result.add(token)
        elif len(token) == 3 and token[0] == token[1]:
            result.update(rank * 2 for rank in RANKS[RANKS.index(token[0]):])
        else:
            high, low, suited = token[:3]
            result.update(high + rank + suited for rank in RANKS[RANKS.index(low):RANKS.index(high)])
    return frozenset(result)


def _charts(raises, calls):
    raised = expand_chart(raises)
    return raised, expand_chart(calls) - raised


def opening_range(position, table_size=6):
    if table_size == 2 and position == 'BTN':
        return expand_chart(HU_OPEN)
    if table_size <= 6:
        return expand_chart(OPEN_6.get(position, ''))
    return expand_chart(OPEN_EARLY.get(position, OPEN_6.get(position, '')))


def _opener_group(position):
    if position in ('UTG', 'UTG1', 'UTG2', 'UTG3', 'LJ'):
        return 'UTG'
    if position == 'HJ':
        return 'MP'
    if position in ('CO', 'BTN', 'SB'):
        return position
    return None


def _hero_group(position):
    if position in ('LJ', 'HJ', 'UTG1', 'UTG2', 'UTG3'):
        return 'MP'
    if position in ('CO', 'BTN', 'SB', 'BB'):
        return position
    return None


def facing_open_charts(hero, opener):
    """3-bet and call sets for a standard open. Unknown openers use the tightest chart."""
    groups = (_hero_group(hero), _opener_group(opener) or 'UTG')
    charts = FACING_OPEN.get(groups) or FACING_OPEN.get((groups[0], 'UTG'))
    if charts is None:
        return frozenset(), frozenset()
    return _charts(*charts)


def preflop_street_target(state):
    """Latest preflop raise-to, in the same units as the state."""
    hero_bet = state.get('hero_bet', 0)
    current = state.get('current_bet', hero_bet + min(state['stack'], state['to_call']))
    targets = [event['raise_to'] for event in state.get('betting_history') or []
               if event.get('street') == 'preflop' and event.get('action') in ('bet', 'raise')
               and event.get('raise_to')]
    return max([current] + targets)


def preflop_chart_action(hand, state, context):
    """Chart action for one preflop hand. None when no chart applies.

    The sampled showdown equity is not an input. Oversized opens (above 5bb)
    keep only the value continue range.
    """
    position = state.get('hero_position', 'unknown')
    if position not in POSITIONS:
        return None
    pressure = state.get('preflop_action', 'unknown')
    table_size = state.get('table_size') or 6
    bb = state.get('big_blind') or 0
    call = min(state['stack'], state['to_call'])
    opening = opening_range(position, table_size)

    if pressure in ('unopened', 'limped') and call <= bb:
        if pressure == 'unopened' and call == 0:
            return 'check', 'open', '%s: no raise; checking the option' % position
        if pressure == 'limped' and position == 'BB':
            opening = opening_range('SB', table_size)
        if hand in opening:
            kind = 'iso' if pressure == 'limped' else 'open'
            return 'raise', kind, '%s opening chart: %s' % (position, hand)
        if call == 0:
            return 'check', 'open', '%s: %s outside opening chart' % (position, hand)
        return 'fold', 'open', '%s: %s outside opening chart' % (position, hand)

    aggressor = state.get('aggressor_position', 'unknown')
    if pressure == 'raised':
        if table_size == 2 and position == 'BB':
            raises, calls = _charts(*FACING_OPEN[('BB', 'SB')])
            calls = calls | (expand_chart(HU_OPEN) - raises)
        else:
            raises, calls = facing_open_charts(position, aggressor)
        target = preflop_street_target(state)
        if bb and target > 5 * bb:
            raises, calls = _charts(HUGE_3BET, HUGE_CALL)
            label = 'oversized open'
        elif context.get('squeeze') or context.get('callers_since_raise'):
            if position != 'BB':
                calls = frozenset()
            label = 'squeeze' if context.get('squeeze') else 'open with callers'
        else:
            label = '%s open' % (aggressor if aggressor in POSITIONS else 'unknown')
        action = 'raise' if hand in raises else 'call' if hand in calls else 'fold'
        verb = {'raise': '3-bets', 'call': 'calls', 'fold': 'folds'}[action]
        return action, 'three_bet', '%s vs %s chart: %s %s' % (position, label, hand, verb)

    if pressure == 'three_bet':
        ip = state.get('postflop_position') == 'IP' or position in ('CO', 'BTN')
        raises = expand_chart('JJ+ AQs+ AKo A5s A4s' if ip else 'QQ+ AKs AKo A5s A4s')
        calls = frozenset() if position == 'SB' else expand_chart(
            'JJ TT-88 AJs ATs KQs QJs JTs AQo' if ip else 'JJ TT-99 AQs KQs')
        action = 'raise' if hand in raises else 'call' if hand in calls - raises else 'fold'
        verb = {'raise': '4-bets', 'call': 'calls', 'fold': 'folds'}[action]
        return action, 'four_bet', 'vs 3-bet chart: %s %s' % (hand, verb)

    if pressure == 'four_bet':
        if context.get('cold_four_bet'):
            action = 'raise' if hand in expand_chart('KK+') else 'fold'
            verb = '5-bets' if action == 'raise' else 'folds'
            return action, 'five_bet', 'cold 4-bet chart: %s %s' % (hand, verb)
        action = 'raise' if hand in expand_chart('KK+') else 'call' if hand in expand_chart('QQ AKs AKo') else 'fold'
        verb = {'raise': '5-bets', 'call': 'calls', 'fold': 'folds'}[action]
        return action, 'five_bet', 'vs 4-bet chart: %s %s' % (hand, verb)

    if pressure == 'five_bet':
        action = 'raise' if hand in expand_chart('KK+') else 'fold'
        verb = 'continues' if action == 'raise' else 'folds'
        return action, 'five_bet', 'vs 5-bet chart: %s %s' % (hand, verb)
    return None


def range_classes(position, role, table_size=6):
    if role == 'five_bet':
        return expand_chart('KK+')
    if role == 'call_four_bet':
        return expand_chart('QQ+ AKs AKo')
    if role == 'call_three_bet':
        return expand_chart('JJ TT-88 AQs-ATs KQs AQo')
    if role == 'four_bet':
        return expand_chart('QQ+ AKs AKo A5s A4s')
    if role == 'three_bet':
        if position in ('SB', 'BB'):
            return expand_chart('99+ ATs+ AJo+ KQs A5s-A2s')
        if position in ('CO', 'BTN'):
            return expand_chart('TT+ AJs+ AQo+ A5s A4s')
        return expand_chart('JJ+ AQs+ AKo')
    if role in ('open', 'call', 'limp') and position in POSITIONS:
        return opening_range(position, table_size) or None
    return None


@lru_cache(maxsize=128)
def _pool(position, role, table_size):
    classes = range_classes(position, role, table_size)
    return tuple(pair for pair in combinations(deck(), 2) if classes is None or hand_class(pair) in classes)


def opponent_profiles(state):
    count = max(1, int(state['active_players']) - 1)
    profiles = list(state.get('opponents', []))[:count]
    if not profiles and not state['board'] and state.get('aggressor_position') in POSITIONS:
        role = {'raised': 'open', 'three_bet': 'three_bet', 'four_bet': 'four_bet',
                'five_bet': 'five_bet'}.get(state.get('preflop_action'))
        if role and state['to_call'] > 0:
            profiles.append({'position': state['aggressor_position'], 'preflop_role': role})
    return profiles + [{} for _ in range(count - len(profiles))]


def range_equity(hero, board, state, rng, samples=200):
    """Uniform joint sampling of legal combos within each observed range.

    Reject the whole joint draw on a blocker collision, so later seats are not
    silently assigned random hands when their ranges conflict with earlier ones.
    """
    known = set(hero) | set(board)
    size = state.get('table_size') or 6
    pools = [tuple(pair for pair in _pool(p.get('position'), p.get('preflop_role'), size)
                   if not known.intersection(pair)) for p in opponent_profiles(state)]
    if any(not pool for pool in pools):
        raise ValueError('Opponent range has no combinations after known-card blockers')
    wins, accepted = 0.0, 0
    for _ in range(max(1, samples) * 100):
        hands = [rng.choice(pool) for pool in pools]
        dealt = [card for hand in hands for card in hand]
        if len(set(dealt)) != len(dealt):
            continue
        dead = known | set(dealt)
        runout = list(board) + rng.sample([c for c in deck() if c not in dead], 5 - len(board))
        scores = [evaluate(list(hand) + runout) for hand in [hero] + hands]
        if scores[0] == max(scores):
            wins += 1 / scores.count(scores[0])
        accepted += 1
        if accepted >= samples:
            break
    if accepted < max(1, samples):
        raise ValueError('Opponent ranges too restrictive to sample reliably')
    return wins / accepted

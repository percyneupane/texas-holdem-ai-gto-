"""Public betting evidence and explainable features; no inferred hidden actions.

History amounts use the same units as the parent state. Complete history is
required to classify a squeeze/cold four-bet or override an observed pot type.
These features support heuristics, not an equilibrium solver.
"""

from collections import Counter

from .cards import Card, evaluate
from .ranges import POSITIONS
from .stakes import numeric_amount

STREETS = ('preflop', 'flop', 'turn', 'river')
ACTIONS = ('fold', 'check', 'call', 'bet', 'raise')
HISTORY_MONEY = ('amount', 'raise_to', 'pot_before')


def validate_betting_context(state):
    """Validate both structured simulator input and optional vision fields."""
    result = dict(state)
    complete = state.get('history_complete', False)
    if type(complete) is not bool:
        raise ValueError('history_complete must be boolean')
    history = state.get('betting_history', [])
    if not isinstance(history, list) or len(history) > 160:
        raise ValueError('Invalid betting history length')
    clean, last_street = [], 0
    for item in history:
        if (not isinstance(item, dict) or item.get('street') not in STREETS
                or item.get('position') not in POSITIONS or item.get('action') not in ACTIONS):
            raise ValueError('Invalid public betting event')
        street = STREETS.index(item['street'])
        if street < last_street:
            raise ValueError('Betting history must be chronological')
        board_street = {0: 0, 3: 1, 4: 2, 5: 3}.get(len(state.get('board', [])), 0)
        if street > board_street:
            raise ValueError('Betting history contains a future street')
        last_street = street
        event = {key: item[key] for key in ('street', 'position', 'action')}
        for key in HISTORY_MONEY:
            if key in item:
                if type(item[key]) not in (int, float):
                    raise ValueError('Invalid betting amount')
                event[key] = numeric_amount(item[key])
        if item['action'] in ('bet', 'raise') and not event.get('raise_to', 0):
            raise ValueError('An aggressive event requires a positive raise_to')
        clean.append(event)
    opponents = state.get('opponents', [])
    if not isinstance(opponents, list) or len(opponents) > 9:
        raise ValueError('Invalid public opponent data')
    profiles = []
    for profile in opponents:
        if not isinstance(profile, dict):
            raise ValueError('Invalid public opponent data')
        profile = dict(profile)
        for key in ('stack', 'bet'):
            if key in profile:
                if type(profile[key]) not in (int, float):
                    raise ValueError('Invalid opponent amount')
                profile[key] = numeric_amount(profile[key])
        profiles.append(profile)
    for key in ('current_bet', 'min_raise_to', 'max_raise_to'):
        if key in state:
            if type(state[key]) not in (int, float):
                raise ValueError('Invalid legal betting amount')
            result[key] = numeric_amount(state[key])
    if 'can_raise' in state and type(state['can_raise']) is not bool:
        raise ValueError('can_raise must be boolean')
    result.update(betting_history=clean, history_complete=complete, opponents=profiles)
    return result


def board_features(hero, board):
    features = dict(board_texture='preflop', paired_board=False, connected_board=False,
                    flush_draw=False, open_ended=False, gutshot=False, made_hand_category=None)
    if len(board) < 3:
        return features
    board_suits = Counter(c[1] for c in board)
    features['board_texture'] = ('monotone' if max(board_suits.values()) == len(board)
                                 else 'two-tone' if max(board_suits.values()) >= 2 else 'rainbow')
    ranks = {c[0] for c in board}
    features['paired_board'] = len(ranks) < len(board)
    if 14 in ranks:
        ranks.add(1)
    features['connected_board'] = any(len(ranks.intersection(range(low, low + 5))) >= 3 for low in range(1, 11))
    cards = hero + board
    features['made_hand_category'] = evaluate([Card(*c) for c in cards])[0]
    if len(board) < 5:
        suits = Counter(c[1] for c in cards)
        features['flush_draw'] = any(n == 4 and any(c[1] == suit for c in hero) for suit, n in suits.items())
        # Only count straight draws using at least one private rank, not a
        # four-straight entirely on the public board, and never a made straight.
        ranks = {c[0] for c in cards}
        if 14 in ranks:
            ranks.add(1)
        if not any(set(range(low, low + 5)) <= ranks for low in range(1, 11)):
            outs = set()
            for low in range(1, 11):
                window = set(range(low, low + 5))
                missing = window - ranks
                private = {c[0] for c in hero} - {c[0] for c in board}
                if len(missing) == 1 and window.intersection(private):
                    outs.update(14 if r == 1 else r for r in missing)
            features['open_ended'] = len(outs) >= 2
            features['gutshot'] = len(outs) == 1
    return features


def enrich_context(state):
    """Derive dimensionless policy features from known public information."""
    result = validate_betting_context(state)
    history, complete = result['betting_history'], result['history_complete']
    preflop = [e for e in history if e['street'] == 'preflop']
    street = STREETS[{0: 0, 3: 1, 4: 2, 5: 3}.get(len(state['board']), 0)]
    current = [e for e in history if e['street'] == street]
    hero = state.get('hero_position', 'unknown')
    raises, limpers, callers, roles, aggressor = [], 0, 0, {}, None
    squeeze, cold_four = False, False
    for e in preflop:
        pos, action = e['position'], e['action']
        if action in ('bet', 'raise'):
            if len(raises) == 1:
                squeeze = callers > 0 and pos != raises[0]['position']
            if len(raises) == 2:
                cold_four = pos not in roles
            raises.append(e)
            roles[pos] = ('open' if len(raises) == 1 else 'three_bet' if len(raises) == 2
                          else 'four_bet' if len(raises) == 3 else 'five_bet')
            aggressor, callers = pos, 0
        elif action == 'call':
            callers += 1
            if not raises:
                limpers += 1
            roles[pos] = ('limp' if not raises else 'call' if len(raises) == 1
                          else 'call_three_bet' if len(raises) == 2 else 'call_four_bet')
    if complete:
        result['preflop_action'] = ('unopened' if not limpers else 'limped') if not raises else (
            'raised' if len(raises) == 1 else 'three_bet' if len(raises) == 2
            else 'four_bet' if len(raises) == 3 else 'five_bet')
        result['aggressor_position'] = aggressor if aggressor != hero and aggressor else 'unknown'
        result['limper_count'] = limpers
        result['opponents'] = [dict(p, preflop_role=roles.get(p.get('position'), p.get('preflop_role', 'unknown')))
                               for p in result['opponents']]
    bb, pot, stack = state.get('big_blind', 0), state['pot'], state['stack']
    hero_bet = state.get('hero_bet', 0)
    profiles = result['opponents']
    # Do not understate depth if any live opponent stack is missing. Heads-up
    # depth is exact; multiway depth is the largest live effective matchup.
    known = bool(profiles) and len(profiles) == max(1, state['active_players'] - 1)
    known = known and all('stack' in p and 'bet' in p for p in profiles)
    effective_total = min(stack + hero_bet, max(p['stack'] + p['bet'] for p in profiles)) if known else None
    remaining = min(stack, max(p['stack'] for p in profiles)) if known else None
    aggression = [e for e in current if e['action'] in ('bet', 'raise')]
    last = aggression[-1] if aggression else None
    last_ratio = last.get('amount', 0) / last['pot_before'] if last and last.get('pot_before', 0) else None
    barrel_streets = {e['street'] for e in history if last and e['position'] == last['position']
                      and e['street'] != 'preflop' and e['action'] in ('bet', 'raise')}
    call = min(stack, state['to_call'])
    metrics = dict(history_complete=complete, preflop_bet_level=(len(raises) + 1 if complete else None),
                   squeeze=squeeze if complete else False, cold_four_bet=cold_four if complete else False,
                   hero_preflop_role=roles.get(hero, 'unknown') if complete else 'unknown',
                   callers_since_raise=callers if complete else 0,
                   street_bet_level=len(aggression) if complete else None,
                   facing_postflop_raise=bool(complete and state['board'] and call and len(aggression) >= 2),
                   barrel_streets=len(barrel_streets) if complete else None,
                   last_bet_pot_ratio=last_ratio,
                   last_raise_to_bb=last['raise_to'] / bb if last and bb else None,
                   effective_stack_bb=effective_total / bb if effective_total is not None and bb else None,
                   spr=remaining / pot if remaining is not None and pot else None,
                   to_call_bb=call / bb if bb else None,
                   call_equity_required=call / (pot + call) if call else 0,
                   multiway=state['active_players'] > 2)
    metrics.update(board_features(state['hero'], state['board']))
    result['strategy_context'] = metrics
    return result

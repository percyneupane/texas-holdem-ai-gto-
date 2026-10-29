"""Turns an engine ``HandState`` into the plain dict the table renderer draws."""

from typing import Dict, List, Optional

from .cards import evaluate
from .engine import BIG_BLIND, SMALL_BLIND, HandState, Seat, current_hand_label, describe_hand, legal_actions
from .ranges import position_names
from .table_layout import SEAT_LAYOUTS

ACTION_KINDS = {"FOLD": "fold", "CHECK": "check", "CALL": "call", "BET": "raise", "RAISE": "raise",
                "ALL-IN": "allin", "SB": "blind", "BB": "blind"}


def cards_json(cards) -> List[List]:
    return [[card.rank, card.suit] for card in cards]


def _seat_view(index: int, seat: Seat, slot: int, state: HandState, hero: int) -> Dict:
    award = next((a for a in state.awards if a.seat == index), None)
    shown = state.showdown and seat.in_hand
    if not seat.hand or (seat.folded and index != hero):
        cards = "none"
    elif index == hero:
        cards = "hero"
    else:
        cards = "faces" if shown else "backs"
    if award:
        status, kind = "WINS %d" % award.amount, "win"
    elif not seat.hand:
        status, kind = "SITTING OUT", "out"
    elif index == hero and seat.in_hand:
        status, kind = current_hand_label(seat.hand, state.board), "name"
    elif seat.last_action:
        status, kind = seat.last_action, ACTION_KINDS.get(seat.last_action, "name")
    else:
        status, kind = seat.name, "name"
    return {
        "slot": slot, "number": index + 1, "name": seat.name, "stack": seat.stack, "bet": seat.bet,
        "cards": cards, "hand": cards_json(seat.hand) if cards == "faces" else [],
        "status": status, "kind": kind, "to_act": state.to_act == index, "winner": award is not None,
        "folded": seat.folded, "out": not seat.hand,
        "hand_label": describe_hand(evaluate(list(seat.hand) + list(state.board))) if shown and index != hero else "",
    }


def result_message(state: HandState) -> str:
    if not state.complete:
        return ""
    return "; ".join("%s wins %d%s" % (state.seats[a.seat].name, a.amount,
                                       " with %s" % a.hand_name if a.hand_name else "") for a in state.awards)


def _waiting_text(state: HandState, hero_seat: Seat) -> str:
    if state.complete:
        return "Next hand starting..."
    if not hero_seat.hand:
        return "Sitting out - you'll be dealt in next hand"
    if hero_seat.folded:
        return "You folded - waiting for the next hand"
    return "Waiting for your turn"


def strategy_context(state: HandState, hero: int) -> Dict:
    dealt = [i for i, seat in enumerate(state.seats) if seat.hand]
    if len(dealt) < 2:
        return {}
    clockwise = [(state.dealer + step) % len(state.seats) for step in range(1, len(state.seats) + 1)]
    clockwise = [i for i in clockwise if i in dealt]
    order = ([state.dealer] + [i for i in dealt if i != state.dealer]) if len(dealt) == 2 else clockwise[2:] + clockwise[:2]
    positions = dict(zip(order, position_names(len(dealt))))
    history = public_betting_history(state, positions)
    roles, raises, limpers, aggressor = {}, 0, 0, None
    for line in state.log:
        if line.startswith('Flop:'):
            break
        for index, seat in enumerate(state.seats):
            if line.startswith(seat.name + ' raises to ') or line.startswith(seat.name + ' bets to '):
                raises += 1
                roles[index] = ('open' if raises == 1 else 'three_bet' if raises == 2
                                else 'four_bet' if raises == 3 else 'five_bet')
                aggressor = index
                break
            if line.startswith(seat.name + ' calls '):
                roles.setdefault(index, 'call')
                if not raises:
                    limpers += 1
                break
    live_order = [i for i in clockwise if state.seats[i].in_hand]
    return {
        'hero_position': positions.get(hero, 'unknown'), 'table_size': len(dealt),
        'small_blind': SMALL_BLIND, 'big_blind': BIG_BLIND,
        'hero_bet': state.seats[hero].bet, 'can_raise': legal_actions(state).can_raise if state.to_act == hero else False,
        'preflop_action': ('raised' if raises == 1 else 'three_bet' if raises == 2
                          else 'four_bet' if raises == 3 else 'five_bet') if raises
                          else 'limped' if limpers else 'unopened',
        'aggressor_position': positions.get(aggressor, 'unknown') if aggressor != hero else 'unknown',
        'postflop_position': 'IP' if live_order and live_order[-1] == hero else 'OOP',
        'limper_count': limpers,
        'betting_history': history, 'history_complete': True,
        'opponents': [{'position': positions[i], 'preflop_role': roles.get(i, 'unknown'),
                       'stack': state.seats[i].stack, 'bet': state.seats[i].bet}
                      for i in dealt if i != hero and state.seats[i].in_hand],
    }


def public_betting_history(state, positions):
    """Replay only public log actions, retaining amounts before each action."""
    street, pot, bets, events = 'preflop', 0, {}, []
    # Longest name first prevents a prefix name from matching another seat.
    names = sorted(((s.name, i) for i, s in enumerate(state.seats)), key=lambda pair: -len(pair[0]))
    for line in state.log:
        changed = next((s for s in ('flop', 'turn', 'river') if line.startswith(s.title() + ':')), None)
        if changed:
            street, bets = changed, {}
            continue
        for name, index in names:
            if not line.startswith(name + ' '):
                continue
            text = line[len(name) + 1:]
            if text.startswith(('posts small blind ', 'posts big blind ')):
                amount = int(text.rsplit(' ', 1)[1])
                bets[index] = amount
                pot += amount
            elif index in positions and (text in ('folds', 'checks') or text.startswith(('calls ', 'raises to ', 'bets to '))):
                action = 'fold' if text == 'folds' else 'check' if text == 'checks' else (
                    'call' if text.startswith('calls ') else 'raise' if text.startswith('raises ') else 'bet')
                old = bets.get(index, 0)
                target = int(text.rsplit(' ', 1)[1]) if action in ('raise', 'bet') else 0
                amount = target - old if target else int(text.split()[-1]) if action == 'call' else 0
                events.append(dict(street=street, position=positions[index], action=action,
                                   amount=amount, raise_to=target, pot_before=pot))
                bets[index] = old + amount
                pot += amount
            break
    return events


def build_view(state: HandState, hero: int = 0, raise_to: Optional[int] = None, timer: Optional[float] = None,
               player_count: Optional[int] = None, autoplay: bool = False, hand_number: int = 0) -> Dict:
    slots = SEAT_LAYOUTS[len(state.seats)]
    hero_seat = state.seats[hero]
    hero_turn = state.to_act == hero
    view = {
        "hero": cards_json(hero_seat.hand) if hero_seat.in_hand else [],
        "board": cards_json(state.board),
        "pot": state.pot, "to_call": 0, "min_raise": state.min_raise, "stack": hero_seat.stack,
        "active_players": sum(1 for seat in state.seats if seat.in_hand), "street": state.street,
        "current_bet": state.current_bet, "hero_bet": hero_seat.bet,
        "seats": [_seat_view(i, seat, slots[i], state, hero) for i, seat in enumerate(state.seats)],
        "dealer_slot": slots[state.dealer], "hero_to_act": hero_turn, "timer": timer if hero_turn else None,
        "player_count": player_count or len(state.seats), "autoplay": autoplay, "hand_number": hand_number,
        "message": result_message(state), "waiting_text": _waiting_text(state, hero_seat),
    }
    if hero_turn:
        legal = legal_actions(state)
        chosen = legal.min_raise_to if raise_to is None else raise_to
        view.update(to_call=legal.to_call, can_raise=legal.can_raise, min_raise_to=legal.min_raise_to,
                    max_raise_to=legal.max_raise_to,
                    raise_to=max(legal.min_raise_to, min(chosen, legal.max_raise_to)))
    view.update(strategy_context(state, hero))
    return view


def decision_state(view: Dict) -> Dict:
    """The subset of a view the policy agent consumes (same keys as ``simulator.render_frame``)."""
    keys = ("hero", "board", "active_players", "pot", "to_call", "min_raise", "stack", "street",
            'hero_position', 'table_size', 'small_blind', 'big_blind', 'hero_bet', 'can_raise', 'preflop_action',
            'aggressor_position', 'postflop_position', 'limper_count', 'opponents',
            'betting_history', 'history_complete', 'current_bet')
    return {**{key: view[key] for key in keys}, "action_history": []}

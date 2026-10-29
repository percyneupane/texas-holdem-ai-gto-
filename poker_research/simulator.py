import json
import random
from typing import List, Optional, Tuple

from .agent import Decision, VisionPolicyAgent
from .cards import Card
from .engine import (BIG_BLIND, RAISE_ACTIONS, STARTING_STACK, HandState, Legal, Seat, apply_action,
                     legal_actions, start_hand)
from .metrics import Metrics
from .opponents import opponent_action
from .table_view import build_view
from .vision import render_table_image


def render_frame(hero: List[Card], board: List[Card], players: int, pot: int,
                 to_call: int, min_raise: int = BIG_BLIND, stack: int = STARTING_STACK,
                 street: str = "preflop", action_history: Optional[List[str]] = None) -> str:
    """Render the controlled table state consumed by the vision adapter."""
    return json.dumps({
        "hero": [[card.rank, card.suit] for card in hero],
        "board": [[card.rank, card.suit] for card in board],
        "active_players": players,
        "pot": pot,
        "to_call": to_call,
        "min_raise": min_raise,
        "stack": stack,
        "street": street,
        "action_history": action_history or [],
    })


def render_image_frame(hero: List[Card], board: List[Card], players: int, pot: int,
                       to_call: int, min_raise: int = BIG_BLIND,
                       stack: int = STARTING_STACK, street: str = "preflop",
                       action_history: Optional[List[str]] = None):
    state = json.loads(render_frame(hero, board, players, pot, to_call,
                                    min_raise, stack, street))
    return render_table_image(state)


HERO = 0


def decision_to_action(state: HandState, decision: Decision) -> Tuple[str, int]:
    """Map the agent's (action, chips to add) onto the engine's (action, raise-to total)."""
    return decision.action, state.seats[state.to_act].bet + decision.amount


def _is_legal(action: str, legal: Legal) -> bool:
    if action == "check":
        return legal.to_call == 0
    if action in RAISE_ACTIONS:
        return legal.can_raise
    return True


def play_hand(players_count: int, dealer: int, rng: random.Random,
              agent: VisionPolicyAgent, metrics: Metrics) -> HandState:
    """Play one full hand; the hero decides from a rendered table image each time it acts."""
    seats = tuple(Seat("Hero" if i == HERO else "P%d" % (i + 1)) for i in range(players_count))
    state = start_hand(seats, dealer, rng)
    while state.to_act is not None:
        if state.to_act == HERO:
            decision: Decision = agent.decide(render_table_image(build_view(state, HERO)))
            metrics.record(decision, valid=_is_legal(decision.action, legal_actions(state)))
            action = decision_to_action(state, decision)
        else:
            action = opponent_action(state, rng)
        state = apply_action(state, *action)
    metrics.record_hand(state.seats[HERO].stack - STARTING_STACK, state.showdown)
    return state


def run_experiment(hands: int, players: int, seed: int, policy: str = "gto") -> str:
    rng = random.Random(seed)
    agent = VisionPolicyAgent(seed + 1, policy=policy)
    metrics = Metrics()
    for hand_number in range(hands):
        play_hand(players, hand_number % players, rng, agent, metrics)
    return f"policy={policy} players={players} {metrics.summary()}"

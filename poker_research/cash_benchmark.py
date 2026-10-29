"""Paired cash-game strategy experiments, independent of screen recognition.

Each strategy receives identical shuffled deals, resets stacks each hand and
rotates the hero through the positions. Opponents are transparent heuristics,
not representative samples of real players. Rake is a configurable simulation
assumption, charged proportionally to awards after removing uncalled returns.
No inferred real-money profitability or global 'best strategy' is reported.
"""

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from statistics import mean, stdev

from .agent import VisionPolicyAgent
from .engine import BIG_BLIND, Seat, apply_action, legal_actions, start_hand
from .opponents import OPPONENT_STYLES, opponent_action
from .policy import POLICIES
from .table_view import build_view, decision_state


@dataclass(frozen=True)
class CashConfig:
    players: int = 6
    hands: int = 1000
    seed: int = 41001
    stack_bb: int = 100
    rake_rate: float = .05
    rake_cap_bb: float = 3.0
    no_flop_no_drop: bool = True
    equity_samples: int = 200

    def validate(self):
        if not 2 <= self.players <= 10 or self.hands < 1 or self.stack_bb < 1 or self.equity_samples < 1:
            raise ValueError('Invalid players, hands, stack depth or equity sample count')
        if not math.isfinite(self.rake_rate) or not 0 <= self.rake_rate <= 1:
            raise ValueError('Rake rate must be between zero and one')
        if not math.isfinite(self.rake_cap_bb) or self.rake_cap_bb < 0:
            raise ValueError('Rake cap must be nonnegative')


def profit_after_rake(state, starting_stack, config):
    gross = state.seats[0].stack - starting_stack
    if config.no_flop_no_drop and not state.board:
        return float(gross), 0.0
    # The unique largest commitment's excess above the next largest is an
    # uncalled return, not winnings subject to rake.
    contributions = sorted((seat.committed, i) for i, seat in enumerate(state.seats))
    refund = max(0, contributions[-1][0] - contributions[-2][0])
    refund_owner = contributions[-1][1]
    contested = state.pot - refund
    rake = min(contested * config.rake_rate, config.rake_cap_bb * BIG_BLIND)
    award = sum(a.amount for a in state.awards if a.seat == 0)
    hero_won = max(0, award - (refund if refund_owner == 0 else 0))
    hero_rake = rake * hero_won / contested if contested else 0.0
    return gross - hero_rake, hero_rake


def play_trial_hand(config, hand_number, policy, style, agent):
    starting_stack = config.stack_bb * BIG_BLIND
    seats = tuple(Seat('Hero' if i == 0 else 'P%d' % i, stack=starting_stack) for i in range(config.players))
    # Reset the shuffle independently of policy and action-count RNG use.
    deal_seed = config.seed + hand_number * 1000003
    state = start_hand(seats, hand_number % config.players, random.Random(deal_seed))
    decisions, invalid = 0, 0
    for step in range(2000):
        if state.to_act is None:
            net, rake = profit_after_rake(state, starting_stack, config)
            return {'net_bb': net / BIG_BLIND, 'gross_bb': (state.seats[0].stack - starting_stack) / BIG_BLIND,
                    'rake_bb': rake / BIG_BLIND, 'decisions': decisions, 'invalid_actions': invalid}
        if state.to_act == 0:
            observation = decision_state(build_view(state, 0))
            decision = agent.decide(json.dumps(observation))
            legal = legal_actions(state)
            if decision.action in ('raise', 'bet'):
                target = state.seats[0].bet + decision.amount
                invalid += int(not legal.can_raise or not legal.min_raise_to <= target <= legal.max_raise_to)
            elif decision.action == 'check':
                invalid += int(legal.to_call != 0)
            elif decision.action == 'call':
                invalid += int(decision.amount != legal.to_call)
            action = decision.action, state.seats[0].bet + decision.amount
            decisions += 1
        else:
            action = opponent_action(state, random.Random(deal_seed ^ ((step + 1) * 7919)), style)
        state = apply_action(state, *action)
    raise RuntimeError('Hand exceeded action limit')


def rate_summary(profits):
    n = len(profits)
    value = mean(profits) * 100
    # Normal approximation; no interval for tiny smoke-test samples.
    half_width = 1.96 * stdev(profits) / math.sqrt(n) * 100 if n >= 30 else None
    return {'hands': n, 'bb_per_100': value,
            'approx_95_ci': [value - half_width, value + half_width] if half_width is not None else None}


def compare(config, policies=('gto', 'contextual'), styles=tuple(OPPONENT_STYLES)):
    config.validate()
    if not policies or len(set(policies)) != len(policies) or any(p not in POLICIES for p in policies):
        raise ValueError('Choose distinct known policies')
    if not styles or any(style not in OPPONENT_STYLES for style in styles):
        raise ValueError('Choose known opponent styles')
    records, comparisons = [], []
    for style in styles:
        outcomes = {}
        for policy in policies:
            agent = VisionPolicyAgent(seed=config.seed ^ 17011, policy=policy, equity_samples=config.equity_samples)
            rows = [play_trial_hand(config, hand, policy, style, agent) for hand in range(config.hands)]
            outcomes[policy] = [row['net_bb'] for row in rows]
            records.append(dict(rate_summary(outcomes[policy]), policy=policy, opponent_style=style,
                                gross_bb_per_100=100 * mean(row['gross_bb'] for row in rows),
                                rake_bb_per_100=100 * mean(row['rake_bb'] for row in rows),
                                profitable_hand_fraction=mean(row['net_bb'] > 0 for row in rows),
                                invalid_actions=sum(row['invalid_actions'] for row in rows)))
        baseline = policies[0]
        for policy in policies[1:]:
            differences = [a - b for a, b in zip(outcomes[policy], outcomes[baseline])]
            comparisons.append(dict(rate_summary(differences), candidate=policy, baseline=baseline,
                                    opponent_style=style, metric='paired_net_bb_per_100_difference'))
    return {'config': asdict(config), 'observation_mode': 'structured_state_no_screen_errors',
            'scope': 'Synthetic opponents only; no demonstrated real-game profitability.',
            'results': records, 'paired_comparisons': comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--players', type=int, default=6)
    parser.add_argument('--hands', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=41001)
    parser.add_argument('--stack-bb', type=int, default=100)
    parser.add_argument('--rake-rate', type=float, default=.05)
    parser.add_argument('--rake-cap-bb', type=float, default=3.0)
    parser.add_argument('--equity-samples', type=int, default=200)
    args = parser.parse_args()
    config = CashConfig(players=args.players, hands=args.hands, seed=args.seed, stack_bb=args.stack_bb,
                        rake_rate=args.rake_rate, rake_cap_bb=args.rake_cap_bb, equity_samples=args.equity_samples)
    print(json.dumps(compare(config), indent=2))


if __name__ == '__main__':
    main()

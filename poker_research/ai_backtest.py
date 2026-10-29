"""Accelerated local-table backtest against paid AI role-play opponents."""

import argparse
import json
import time
from collections import Counter
from statistics import median

from .cash_benchmark import CashConfig, profit_after_rake, rate_summary
from .engine import BIG_BLIND, legal_actions
from .table_game import TableGame


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def run(hands=100, players=6, seed=7, max_calls=3000, progress=None, controller=None):
    if type(hands) is not int or hands < 1:
        raise ValueError('Hands must be positive')
    clock = Clock()
    game = TableGame(players=players, seed=seed, clock=clock, opponent_mode='ai',
                     ai_max_calls=max_calls, opponent_controller=controller)
    game.autoplay = True
    config = CashConfig(players=players, hands=hands, seed=seed)
    actions, latencies, rows = Counter(), [], []
    invalid, rebuys = 0, 0
    decide = game.agent.decide

    def record(*args, **kwargs):
        nonlocal invalid
        result = decide(*args, **kwargs)
        legal = legal_actions(game.state)
        if result.action in ('raise', 'bet'):
            target = game.state.seats[0].bet + result.amount
            invalid += int(not legal.can_raise or not legal.min_raise_to <= target <= legal.max_raise_to)
        elif result.action == 'check':
            invalid += int(legal.to_call != 0)
        elif result.action == 'call':
            invalid += int(result.amount != legal.to_call)
        actions[result.action] += 1
        latencies.append(result.latency_ms)
        return result

    game.agent.decide = record
    started, start_stack, last_hand, error = time.perf_counter(), None, 0, None
    try:
        while len(rows) < hands:
            previous = game.hand_number
            busted = game.state is not None and game.state.complete and game.state.seats[0].stack == 0
            clock.now += 10
            game.tick()
            if game.opponent_error:
                error = game.opponent_error
                break
            if game.hand_number != previous:
                start_stack = game.state.seats[0].stack + game.state.seats[0].committed
                rebuys += int(busted)
            if game.state.complete and game.hand_number > last_hand:
                gross = game.state.seats[0].stack - start_stack
                net, rake = profit_after_rake(game.state, start_stack, config)
                rows.append({'hand': game.hand_number, 'gross_bb': gross / BIG_BLIND,
                             'rake_adjusted_bb': net / BIG_BLIND, 'hypothetical_rake_bb': rake / BIG_BLIND})
                last_hand = game.hand_number
                if progress:
                    progress({'completed_hands': last_hand, 'requested_hands': hands,
                              'api_calls': game.ai_opponents.calls,
                              'elapsed_seconds': round(time.perf_counter() - started, 1)})
            if game._opponent_future is not None and not game._opponent_future.done():
                time.sleep(.02)
    finally:
        game.close()
    gross = sum(r['gross_bb'] for r in rows)
    net = sum(r['rake_adjusted_bb'] for r in rows)
    return {
        'requested_hands': hands, 'completed_hands': len(rows), 'complete': len(rows) == hands, 'error': error,
        'players': players, 'seed': seed, 'policy': game.agent.policy_name,
        'opponents': game.ai_opponents.metrics(),
        'scope': 'AI role-play remains synthetic; no validated human realism or real-game profitability.',
        'observation': 'Structured acting-seat/public state; no screenshots or hidden opponent cards sent.',
        'reproducibility': 'Deal seed is fixed; provider LLM actions are not guaranteed reproducible.',
        'hero_rebuys': rebuys, 'gross_profit_bb': gross, 'rake_adjusted_profit_bb': net,
        'gross_profit_at_0_10_bb': round(gross * .10, 2), 'rake_adjusted_profit_at_0_10_bb': round(net * .10, 2),
        'illustrative_rake': {'rate': .05, 'cap_bb': 3, 'no_flop_no_drop': True, 'applied_to_table': False},
        'gross_bb_per_100': rate_summary([r['gross_bb'] for r in rows]) if rows else None,
        'rake_adjusted_bb_per_100': rate_summary([r['rake_adjusted_bb'] for r in rows]) if rows else None,
        'profitable_hands': sum(r['gross_bb'] > 0 for r in rows),
        'losing_hands': sum(r['gross_bb'] < 0 for r in rows),
        'break_even_hands': sum(r['gross_bb'] == 0 for r in rows),
        'decisions': sum(actions.values()), 'actions': dict(actions), 'invalid_actions': invalid,
        'median_decision_ms': round(median(latencies), 2) if latencies else None,
        'runtime_seconds': round(time.perf_counter() - started, 2), 'hand_results': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hands', type=int, default=100)
    parser.add_argument('--players', type=int, default=6)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--max-ai-calls', type=int, default=3000)
    args = parser.parse_args()
    def progress(data):
        if data['completed_hands'] % 5 == 0:
            print('PROGRESS_JSON ' + json.dumps(data), flush=True)
    report = run(args.hands, args.players, args.seed, args.max_ai_calls, progress)
    print('RESULT_JSON ' + json.dumps(report), flush=True)
    if not report['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

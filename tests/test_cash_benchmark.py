import random
import unittest
from dataclasses import replace

from poker_research.cash_benchmark import CashConfig, compare, profit_after_rake, rate_summary
from poker_research.cards import Card
from poker_research.engine import Award, Seat, start_hand


class CashBenchmarkTests(unittest.TestCase):
    def state(self, commitments, awards, board=True):
        base = start_hand((Seat('Hero'), Seat('Other')), 0, random.Random(3))
        seats = tuple(replace(seat, committed=commitments[i],
                              stack=1000 - commitments[i] + sum(a.amount for a in awards if a.seat == i))
                      for i, seat in enumerate(base.seats))
        return replace(base, seats=seats, awards=awards,
                       board=(Card(2, 'c'), Card(3, 'h'), Card(4, 's')) if board else ())

    def test_rake_excludes_uncalled_return(self):
        state = self.state((300, 100), (Award(0, 400),))
        net, rake = profit_after_rake(state, 1000, CashConfig())
        self.assertEqual(rake, 10)
        self.assertEqual(net, 90)

    def test_split_pot_allocates_only_hero_share_of_rake(self):
        state = self.state((100, 100), (Award(0, 100), Award(1, 100)))
        self.assertEqual(profit_after_rake(state, 1000, CashConfig()), (-5, 5))

    def test_no_flop_no_drop_and_cap(self):
        state = self.state((100, 100), (Award(0, 200),), board=False)
        self.assertEqual(profit_after_rake(state, 1000, CashConfig()), (100, 0))
        state = self.state((500, 500), (Award(0, 1000),))
        self.assertEqual(profit_after_rake(state, 1000, CashConfig())[1], 30)

    def test_small_samples_have_no_confidence_claim(self):
        self.assertIsNone(rate_summary([1, -1])['approx_95_ci'])
        summary = rate_summary([1, -1] * 30)
        self.assertEqual(summary['bb_per_100'], 0)
        self.assertLess(summary['approx_95_ci'][0], 0)
        self.assertGreater(summary['approx_95_ci'][1], 0)

    def test_paired_experiment_is_reproducible_and_reports_rake(self):
        config = CashConfig(players=2, hands=4, equity_samples=5)
        first = compare(config, styles=('balanced',))
        self.assertEqual(first, compare(config, styles=('balanced',)))
        self.assertEqual(len(first['results']), 2)
        self.assertEqual(len(first['paired_comparisons']), 1)
        for result in first['results']:
            self.assertAlmostEqual(result['bb_per_100'], result['gross_bb_per_100'] - result['rake_bb_per_100'])

    def test_invalid_configuration_is_rejected(self):
        for config in (CashConfig(hands=0), CashConfig(players=11), CashConfig(rake_rate=-1),
                       CashConfig(rake_cap_bb=float('nan'))):
            with self.assertRaises(ValueError):
                config.validate()

import json
import random
import unittest

from poker_research.agent import VisionPolicyAgent
from poker_research.policy import (GTO_BLUFF_EQUITY_MAX, bluff_share_for_bet, gto_bluff_probability, gto_policy,
                                   pot_odds)


def spot(**overrides):
    state = {"hero": [[14, "s"], [13, "h"]], "board": [[2, "c"], [7, "d"], [11, "s"]], "active_players": 2,
             "pot": 100, "to_call": 0, "min_raise": 10, "stack": 900, "street": "flop", "action_history": []}
    return {**state, **overrides}


class GtoMathTests(unittest.TestCase):
    def test_pot_odds_is_calls_share_of_final_pot(self):
        self.assertAlmostEqual(pot_odds(50, 150), 0.25)
        self.assertEqual(pot_odds(0, 150), 0.0)

    def test_balanced_bluff_share_for_three_quarter_pot_bet(self):
        self.assertAlmostEqual(bluff_share_for_bet(0.75), 0.3)
        self.assertAlmostEqual(bluff_share_for_bet(1.0), 1 / 3)

    def test_bluffing_drops_with_more_opponents_and_never_preflop(self):
        heads_up = gto_bluff_probability("river", 1)
        self.assertAlmostEqual(gto_bluff_probability("river", 3), heads_up / 3)
        self.assertEqual(gto_bluff_probability("preflop", 1), 0.0)


class GtoDecisionTests(unittest.TestCase):
    def test_calls_exactly_when_equity_beats_pot_odds(self):
        state = spot(pot=150, to_call=50)  # needs 25%
        rng = random.Random(1)
        self.assertEqual(gto_policy(state, 0.26, rng).action, "call")
        self.assertEqual(gto_policy(state, 0.24, rng).action, "fold")

    def test_raises_strong_hands_facing_a_bet(self):
        choice = gto_policy(spot(pot=150, to_call=50), 0.8, random.Random(1))
        self.assertEqual(choice.action, "raise")
        self.assertGreater(choice.amount, 50)

    def test_bets_value_and_checks_medium_hands(self):
        rng = random.Random(1)
        self.assertEqual(gto_policy(spot(), 0.65, rng), gto_policy(spot(), 0.65, rng))
        self.assertEqual(gto_policy(spot(), 0.65, rng).action, "bet")
        self.assertEqual(gto_policy(spot(), 0.45, rng).action, "check")

    def test_only_weak_hands_bluff(self):
        actions = {gto_policy(spot(board=[[2, "c"], [7, "d"], [11, "s"], [4, "h"], [9, "c"]]),
                              GTO_BLUFF_EQUITY_MAX - 0.05, random.Random(seed)).action for seed in range(200)}
        self.assertEqual(actions, {"bet", "check"})


class ConsistencyTests(unittest.TestCase):
    def test_same_spot_gives_identical_decision_on_every_read(self):
        agent = VisionPolicyAgent(seed=5)
        frame = json.dumps(spot(to_call=20, pot=120))
        first = agent.decide(frame)
        for _ in range(5):
            again = agent.decide(frame)
            self.assertEqual((again.action, again.amount, again.equity), (first.action, first.amount, first.equity))

    def test_fresh_agent_with_same_seed_agrees(self):
        frame = json.dumps(spot(board=[], to_call=10, pot=15))
        self.assertEqual(VisionPolicyAgent(seed=5).decide(frame).action, VisionPolicyAgent(seed=5).decide(frame).action)

    def test_unknown_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            VisionPolicyAgent(policy="magic")


if __name__ == "__main__":
    unittest.main()

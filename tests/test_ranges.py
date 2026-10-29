import json
import random
import unittest
from unittest.mock import patch

from poker_research.agent import VisionPolicyAgent
from poker_research.ai_screen_reader import validate_table_state
from poker_research.cards import Card, deck
from poker_research.engine import Seat, start_hand, apply_action
from poker_research.policy import contextual_policy
from poker_research.ranges import hand_class, opening_range, position_names, range_equity
from poker_research.table_view import build_view, decision_state


def spot(**updates):
    value = dict(hero=[[14, 's'], [11, 'h']], board=[], active_players=6, pot=15,
                 to_call=10, min_raise=10, stack=1000, hero_position='LJ', table_size=6,
                 big_blind=10, hero_bet=0, preflop_action='unopened',
                 aggressor_position='unknown', postflop_position='OOP', action_history=[])
    return dict(value, **updates)


class ChartTests(unittest.TestCase):
    def test_all_positions_and_hand_classes_are_valid(self):
        valid = {hand_class([a, b]) for i, a in enumerate(deck()) for b in deck()[i + 1:]}
        for count in range(2, 11):
            self.assertEqual(len(position_names(count)), count)
            for position in position_names(count):
                self.assertTrue(opening_range(position, count) <= valid)
        self.assertIn('AA', opening_range('UTG', 10))
        self.assertNotIn('AKo', opening_range('unknown'))

    def test_aq_and_aj_open_from_six_max_early_position(self):
        for rank in (11, 12):
            choice = contextual_policy(spot(hero=[[14, 's'], [rank, 'h']]), .25, random.Random(1))
            self.assertEqual(choice.action, 'raise')
            self.assertEqual(choice.amount, 25)

    def test_position_changes_same_hands_opening_decision(self):
        state = spot(hero=[[10, 's'], [9, 'h']])
        self.assertEqual(contextual_policy(state, .3, random.Random(1)).action, 'fold')
        self.assertEqual(contextual_policy(dict(state, hero_position='BTN'), .3, random.Random(1)).action, 'raise')

    def test_strong_hand_can_fold_to_four_bet(self):
        state = spot(hero=[[14, 's'], [11, 'h']], preflop_action='four_bet', to_call=300)
        self.assertEqual(contextual_policy(state, .3, random.Random(1)).action, 'fold')

    def test_blind_contribution_subtracted_from_raise_total(self):
        state = spot(hero_position='SB', hero_bet=5, to_call=5)
        self.assertEqual(contextual_policy(state, .4, random.Random(1)).amount, 25)

    def test_no_chart_open_when_raise_context_contradicts_call_amount(self):
        choice = contextual_policy(spot(to_call=500), .2, random.Random(1))
        self.assertEqual(choice.action, 'fold')

    def test_postflop_position_changes_marginal_call(self):
        state = spot(board=[[2, 's'], [7, 'h'], [9, 'c']], pot=150, to_call=50, active_players=2)
        self.assertEqual(contextual_policy(dict(state, postflop_position='IP'), .29, random.Random(1)).action, 'call')
        self.assertEqual(contextual_policy(dict(state, postflop_position='OOP'), .29, random.Random(1)).action, 'fold')


class RangeEquityTests(unittest.TestCase):
    def test_tight_range_changes_equity_vs_random_hands(self):
        state = spot(active_players=2, opponents=[{'position': 'UTG', 'preflop_role': 'four_bet'}])
        hero = [Card(14, 's'), Card(11, 'h')]
        tight = range_equity(hero, [], state, random.Random(8), samples=200)
        random_hands = range_equity(hero, [], dict(state, opponents=[]), random.Random(8), samples=200)
        self.assertLess(tight, random_hands - .1)

    def test_position_and_pressure_are_part_of_decision_cache(self):
        agent = VisionPolicyAgent(equity_samples=5)
        for position in ('LJ', 'BTN'):
            agent.decide(json.dumps(spot(hero_position=position)))
        self.assertEqual(len(agent._spots), 2)

    def test_published_open_frequencies_and_chart_ignores_equity(self):
        def weight(hands):
            return sum(6 if hand[0] == hand[1] else 4 if hand.endswith('s') else 12 for hand in hands)
        expected = {'LJ': 0.176, 'HJ': 0.214, 'CO': 0.278, 'BTN': 0.435}
        for seat, share in expected.items():
            self.assertAlmostEqual(weight(opening_range(seat, 6)) / 1326, share, delta=0.02)
        state = spot(hero_position='BTN', preflop_action='raised', aggressor_position='CO',
                     to_call=25, pot=40, active_players=2)
        self.assertEqual(contextual_policy(state, .10, random.Random(1)).action, 'call')
        self.assertEqual(contextual_policy(dict(state, hero=[[14, 's'], [14, 'h']]), .10, random.Random(1)).action, 'raise')
        self.assertEqual(contextual_policy(dict(state, hero=[[7, 's'], [2, 'h']]), .80, random.Random(1)).action, 'fold')

    def test_unknown_context_falls_back_without_inventing_position(self):
        state = spot(hero_position='unknown', preflop_action='unknown')
        self.assertNotIn('opening chart:', contextual_policy(state, .3, random.Random(1)).reason)

    def test_screen_reader_preserves_context_and_rejects_invalid_position(self):
        raw = dict(spot(), readable=True, hero_to_act=True)
        self.assertEqual(validate_table_state(raw)['hero_position'], 'LJ')
        # A made-up seat must not receive a chart, and it must not block the decision.
        self.assertEqual(validate_table_state(dict(raw, hero_position='invented'))['hero_position'], 'unknown')
        self.assertEqual(validate_table_state(dict(raw, hero_position='button'))['hero_position'], 'BTN')

    def test_conflicting_ai_position_labels_do_not_create_range_priors(self):
        raw = dict(spot(), readable=True, hero_to_act=True, hero_position='BTN',
                   opponents=[{'position': 'UTG', 'preflop_role': 'open'},
                              {'position': 'LJ', 'preflop_role': 'call'}])
        result = validate_table_state(raw)
        self.assertEqual(result['hero_position'], 'BTN')
        self.assertTrue(all(p['position'] == 'unknown' and p['preflop_role'] == 'unknown'
                            for p in result['opponents']))


class EngineContextTests(unittest.TestCase):
    def test_heads_up_and_full_ring_positions(self):
        for count in range(2, 11):
            state = start_hand(tuple(Seat('P%d' % i) for i in range(count)), 0, random.Random(3))
            view = build_view(state, 0)
            self.assertEqual(view['hero_position'], 'BTN')
            self.assertEqual(view['table_size'], count)

    def test_raise_history_reaches_backend_and_positions_survive_folds(self):
        state = start_hand(tuple(Seat('P%d' % i) for i in range(6)), 0, random.Random(3))
        state = apply_action(state, 'raise', 30)
        state = apply_action(state, 'fold')
        view = build_view(state, 0)
        self.assertEqual(view['hero_position'], 'BTN')
        self.assertEqual(view['table_size'], 6)
        self.assertEqual(view['preflop_action'], 'raised')
        self.assertEqual(view['aggressor_position'], 'LJ')
        self.assertEqual(decision_state(view)['opponents'][2]['preflop_role'], 'open')

import json
import random
import unittest
from unittest.mock import patch

from poker_research.agent import VisionPolicyAgent
from poker_research.ai_screen_reader import validate_table_state
from poker_research.engine import Seat, start_hand, apply_action
from poker_research.policy import contextual_policy
from poker_research.stakes import policy_units
from poker_research.table_view import build_view, decision_state


def event(position, action, target=0, street='preflop', amount=None, pot_before=15):
    return dict(position=position, action=action, raise_to=target,
                amount=target if amount is None else amount, street=street, pot_before=pot_before)


def spot(**updates):
    return dict(dict(hero=[[12, 's'], [12, 'h']], board=[], active_players=2,
                     pot=145, to_call=65, min_raise=65, stack=975, hero_bet=25,
                     big_blind=10, small_blind=5, hero_position='BTN', table_size=6,
                     postflop_position='IP', preflop_action='three_bet', can_raise=True,
                     opponents=[dict(position='BB', preflop_role='three_bet', stack=910, bet=90)],
                     betting_history=[], history_complete=False), **updates)


class BettingContextTests(unittest.TestCase):
    def test_local_history_reaches_backend_and_detects_squeeze(self):
        state = start_hand(tuple(Seat('P%d' % i) for i in range(6)), 0, random.Random(3))
        for action, target in [('raise', 25), ('call', 0), ('raise', 110)]:
            state = apply_action(state, action, target)
        data = decision_state(build_view(state, 0))
        self.assertTrue(data['history_complete'])
        self.assertEqual([e['action'] for e in data['betting_history']], ['raise', 'call', 'raise'])
        result = VisionPolicyAgent(equity_samples=5).decide(json.dumps(data))
        self.assertTrue(result.state['strategy_context']['squeeze'])
        self.assertEqual(result.state['strategy_context']['preflop_bet_level'], 3)
        self.assertTrue(all('stack' in p and 'bet' in p for p in data['opponents']))

    def test_complete_history_distinguishes_cold_four_bet_and_five_bet(self):
        from poker_research.betting_context import enrich_context
        history = [event('LJ', 'raise', 25), event('HJ', 'raise', 90), event('CO', 'raise', 220)]
        state = enrich_context(spot(betting_history=history, history_complete=True))
        self.assertTrue(state['strategy_context']['cold_four_bet'])
        self.assertEqual(state['preflop_action'], 'four_bet')
        state = enrich_context(spot(betting_history=history + [event('LJ', 'raise', 600)], history_complete=True))
        self.assertEqual(state['preflop_action'], 'five_bet')

    def test_no_unseen_history_or_opponent_stack_is_invented(self):
        from poker_research.betting_context import enrich_context
        state = enrich_context(spot(opponents=[], preflop_action='unknown'))
        self.assertIsNone(state['strategy_context']['effective_stack_bb'])
        self.assertIsNone(state['strategy_context']['preflop_bet_level'])
        self.assertFalse(state['strategy_context']['cold_four_bet'])

    def test_nested_currency_fields_scale_together(self):
        raw = spot(pot=.95, to_call=.65, stack=9.75, hero_bet=.25, big_blind=.10,
                   small_blind=.05, min_raise=.65,
                   opponents=[dict(position='BB', preflop_role='three_bet', stack=9.10, bet=.90)],
                   betting_history=[event('BB', 'raise', .90, amount=.80, pot_before=.45)])
        normalized, scale = policy_units(raw)
        self.assertEqual(scale, 100)
        self.assertEqual(normalized['opponents'][0]['stack'], 910)
        self.assertEqual(normalized['betting_history'][0]['raise_to'], 90)
        self.assertEqual(normalized['betting_history'][0]['pot_before'], 45)
        self.assertEqual(raw['opponents'][0]['stack'], 9.10)

    def test_screen_validates_optional_history_and_public_stacks(self):
        raw = dict(spot(), readable=True, hero_to_act=True,
                   betting_history=[event('BB', 'raise', 90)], history_complete=True)
        self.assertEqual(validate_table_state(raw)['betting_history'], raw['betting_history'])
        kept = validate_table_state(dict(raw, history_complete='yes'))
        self.assertFalse(kept['history_complete'])
        self.assertEqual(kept['betting_history'], raw['betting_history'])
        self.assertEqual(kept['pot'], raw['pot'])
        for updates in [dict(betting_history=[event('BB', 'magic', 90)]),
                        dict(betting_history=[event('BB', 'raise', -.1)])]:
            with self.subTest(updates=updates):
                parsed = validate_table_state(dict(raw, **updates))
                self.assertEqual(parsed['betting_history'], [])
                self.assertFalse(parsed['history_complete'])
                self.assertEqual(parsed['pot'], raw['pot'])
        unclear_stack = validate_table_state(dict(
            raw, opponents=[dict(position='BB', preflop_role='call', stack=float('nan'))]))
        self.assertNotIn('stack', unclear_stack['opponents'][0])
        self.assertEqual(unclear_stack['opponents'][0]['position'], 'BB')
        self.assertEqual(unclear_stack['pot'], raw['pot'])

    def test_sizing_depth_and_history_invalidate_decision_cache(self):
        agent = VisionPolicyAgent(equity_samples=5)
        for data in [spot(), spot(betting_history=[event('BB', 'raise', 90)], history_complete=True),
                     spot(opponents=[dict(position='BB', preflop_role='three_bet', stack=200, bet=90)])]:
            agent.decide(json.dumps(data))
        self.assertEqual(len(agent._spots), 3)

    def test_history_validation_rejects_bad_types_and_reversed_streets(self):
        from poker_research.betting_context import validate_betting_context
        for changes in [dict(betting_history='raise'), dict(betting_history=[event('BB', 'raise', 0)]),
                        dict(betting_history=[event('BB', 'raise', '90')]),
                        dict(betting_history=[event('BB', 'check', street='turn'), event('BB', 'check', street='flop')]),
                        dict(opponents='BB'), dict(opponents=[False]), dict(can_raise=1),
                        dict(current_bet=True), dict(opponents=[dict(stack='10')])]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_betting_context(spot(**changes))

    def test_preflop_calls_after_reraises_narrow_the_caller_role(self):
        from poker_research.betting_context import enrich_context
        history = [event('LJ', 'raise', 25), event('BB', 'raise', 90), event('LJ', 'call', amount=65)]
        result = enrich_context(spot(opponents=[dict(position='LJ')], history_complete=True,
                                     betting_history=history))
        self.assertEqual(result['opponents'][0]['preflop_role'], 'call_three_bet')
        self.assertFalse(result['strategy_context']['squeeze'])

    def test_limp_and_barrel_features_use_complete_public_history(self):
        from poker_research.betting_context import enrich_context
        history = [event('LJ', 'call', amount=10), event('BTN', 'raise', 40),
                   event('BB', 'call', amount=30), event('BB', 'bet', 50, 'flop', pot_before=120),
                   event('BB', 'bet', 200, 'turn', pot_before=220)]
        result = enrich_context(spot(hero=[[14, 's'], [13, 'h']], history_complete=True,
                                     betting_history=history, board=[[2, 's'], [7, 'h'], [9, 'c'], [10, 'd']]))
        self.assertEqual(result['limper_count'], 1)
        self.assertEqual(result['strategy_context']['hero_preflop_role'], 'open')
        self.assertEqual(result['strategy_context']['barrel_streets'], 2)
        self.assertAlmostEqual(result['strategy_context']['last_bet_pot_ratio'], 200 / 220)

    def test_history_seat_aliases_are_normalized_like_opponent_positions(self):
        raw = dict(spot(), readable=True, hero_to_act=True,
                   betting_history=[event('UTG', 'raise', 25)], history_complete=True)
        self.assertEqual(validate_table_state(raw)['betting_history'][0]['position'], 'LJ')

    def test_future_street_history_is_rejected(self):
        from poker_research.betting_context import enrich_context
        with self.assertRaises(ValueError):
            enrich_context(spot(betting_history=[event('BB', 'bet', 50, 'river')]))


class ContextPolicyTests(unittest.TestCase):
    def test_overlay_shows_pot_type_depth_and_unknown_history(self):
        from poker_research.overlay import format_strategy_context
        from poker_research.betting_context import enrich_context
        data = enrich_context(spot())
        text = format_strategy_context(data)
        self.assertIn('3-bet', text)
        self.assertIn('100.0 bb', text)
        self.assertIn('history incomplete', text)
        self.assertIn('SPR', text)
    def test_three_bet_uses_observed_target_and_position_not_just_pot(self):
        base = spot(preflop_action='raised', hero_bet=0, to_call=25, min_raise=15,
                    pot=40, stack=1000, aggressor_position='CO',
                    opponents=[dict(position='CO', preflop_role='open', stack=975, bet=25)],
                    hero=[[14, 's'], [14, 'h']])
        ip = contextual_policy(base, .85, random.Random(1))
        oop = contextual_policy(dict(base, postflop_position='OOP'), .85, random.Random(1))
        self.assertEqual(ip.amount, 75)
        self.assertEqual(oop.amount, 100)

    def test_huge_open_is_not_defended_like_normal_open(self):
        base = spot(hero=[[14, 's'], [11, 'h']], hero_bet=0, preflop_action='raised',
                    aggressor_position='CO', pot=40, to_call=25, min_raise=15)
        self.assertEqual(contextual_policy(base, .45, random.Random(1)).action, 'call')
        self.assertEqual(contextual_policy(dict(base, pot=215, to_call=200, min_raise=190),
                                          .45, random.Random(1)).action, 'fold')

    def test_cold_four_bet_chart_continues_kings_and_folds_ace_jack(self):
        history = [event('LJ', 'raise', 25), event('HJ', 'raise', 90), event('CO', 'raise', 240)]
        kings = spot(hero=[[13, 's'], [13, 'h']], pot=370, to_call=215, min_raise=150,
                     betting_history=history, history_complete=True)
        self.assertEqual(contextual_policy(kings, .20, random.Random(1)).action, 'raise')
        jack = spot(hero=[[14, 's'], [11, 'h']], pot=370, to_call=215, min_raise=150,
                    betting_history=history, history_complete=True)
        self.assertEqual(contextual_policy(jack, .80, random.Random(1)).action, 'fold')

    def test_postflop_raise_and_multiway_increase_marginal_call_threshold(self):
        base = spot(board=[[2, 's'], [7, 'h'], [9, 'c']], pot=150, to_call=50,
                    stack=800, hero_bet=0)
        self.assertEqual(contextual_policy(base, .29, random.Random(1)).action, 'call')
        history = [event('BB', 'bet', 20, 'flop'), event('CO', 'raise', 70, 'flop')]
        raised = dict(base, active_players=3, history_complete=True, betting_history=history)
        self.assertEqual(contextual_policy(raised, .29, random.Random(1)).action, 'fold')

    def test_river_raise_still_accounts_for_unmodeled_range_pressure(self):
        data = spot(board=[[2, 's'], [7, 'h'], [9, 'c'], [4, 'd'], [5, 's']],
                    pot=150, to_call=50, stack=800, hero_bet=0,
                    betting_history=[event('BB', 'bet', 20, 'river'), event('CO', 'raise', 70, 'river')],
                    history_complete=True)
        self.assertEqual(contextual_policy(data, .29, random.Random(1)).action, 'fold')

    def test_board_texture_and_draws_are_exposed(self):
        from poker_research.betting_context import enrich_context
        data = enrich_context(spot(hero=[[14, 's'], [11, 's']],
                                   board=[[12, 's'], [10, 's'], [2, 'h']]))
        features = data['strategy_context']
        self.assertTrue(features['flush_draw'])
        self.assertTrue(features['gutshot'])
        self.assertEqual(features['board_texture'], 'two-tone')

    def test_public_stack_gives_spr_and_five_bet_prior_is_tight(self):
        from poker_research.betting_context import enrich_context
        from poker_research.ranges import range_classes
        data = enrich_context(spot())['strategy_context']
        self.assertAlmostEqual(data['spr'], 910 / 145)
        self.assertEqual(data['effective_stack_bb'], 100)
        self.assertNotIn('AQo', range_classes('BTN', 'five_bet'))

    def test_disabled_raise_cannot_be_recommended(self):
        choice = contextual_policy(spot(hero=[[14, 's'], [14, 'h']], can_raise=False), .85, random.Random(1))
        self.assertEqual(choice.action, 'call')

    def test_agent_end_to_end_preserves_decimal_raise_size_and_metrics(self):
        data = spot(hero=[[14, 's'], [14, 'h']], hero_bet=0, preflop_action='raised',
                    to_call=.25, pot=.40, stack=10, min_raise=.15, big_blind=.10,
                    small_blind=.05, opponents=[dict(position='CO', preflop_role='open', stack=9.75, bet=.25)])
        with patch('poker_research.agent.range_equity', return_value=.85):
            decision = VisionPolicyAgent(equity_samples=5).decide(json.dumps(data))
        self.assertEqual(decision.amount, .75)
        self.assertAlmostEqual(decision.state['strategy_context']['effective_stack_bb'], 100)

    def test_legal_raise_bounds_and_known_short_opponent_cap(self):
        base = spot(hero=[[14, 's'], [14, 'h']], to_call=25, hero_bet=0, preflop_action='raised',
                    pot=40, min_raise=15, min_raise_to=110, max_raise_to=150,
                    opponents=[dict(position='CO', preflop_role='open', stack=475, bet=25)])
        self.assertEqual(contextual_policy(base, .85, random.Random(1)).amount, 110)
        self.assertEqual(contextual_policy(dict(base, min_raise_to=40, max_raise_to=60), .85, random.Random(1)).amount, 60)

    def test_draws_do_not_include_made_straights_or_board_only_draws(self):
        from poker_research.betting_context import board_features
        made = board_features([[14, 's'], [5, 'h']], [[2, 'h'], [3, 'c'], [4, 'd']])
        self.assertFalse(made['gutshot'] or made['open_ended'])
        board_only = board_features([[14, 's'], [13, 'h']], [[5, 'h'], [6, 'c'], [7, 'd'], [8, 's']])
        self.assertFalse(board_only['gutshot'] or board_only['open_ended'])
        draw = board_features([[10, 's'], [9, 'h']], [[8, 'h'], [7, 'c'], [2, 'd']])
        self.assertTrue(draw['open_ended'])

    def test_multiway_air_is_not_bluffed_and_low_spr_value_can_commit(self):
        base = spot(hero=[[14, 's'], [13, 'h']], board=[[2, 'h'], [7, 'c'], [9, 'd']],
                    pot=150, to_call=0, hero_bet=0, active_players=3)
        self.assertEqual(contextual_policy(base, .15, random.Random(1)).action, 'check')
        shallow = dict(base, active_players=2, stack=50,
                       opponents=[dict(position='BB', stack=50, bet=0)])
        self.assertEqual(contextual_policy(shallow, .80, random.Random(1)).amount, 50)


if __name__ == '__main__':
    unittest.main()

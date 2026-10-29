import json
import unittest
from unittest.mock import Mock

from PIL import Image

from poker_research.agent import VisionPolicyAgent, Decision
from poker_research.overlay import format_action, format_bet_sizing, format_stakes
from poker_research.stakes import align_chip_units, parse_blinds


def spot(scale=1):
    return dict(hero=[[14, 's'], [12, 'h']], board=[], active_players=2, pot=15 * scale,
                to_call=5 * scale, min_raise=10 * scale, stack=1000 * scale,
                hero_position='SB', preflop_action='unopened', table_size=6,
                small_blind=5 * scale, big_blind=10 * scale, hero_bet=5 * scale)


class BlindSettingsTests(unittest.TestCase):
    def test_blank_uses_screen_and_nonstandard_ratio_is_allowed(self):
        self.assertEqual(parse_blinds('', ''), {})
        self.assertEqual(parse_blinds('2', '5'), {'small_blind': 2, 'big_blind': 5})

    def test_invalid_blinds_are_rejected(self):
        for sb, bb in [('5', ''), ('', '10'), ('0', '10'), ('20', '10'), ('-5', '10'), ('0.005', '0.01'), ('NaN', '1')]:
            with self.subTest(sb=sb, bb=bb), self.assertRaises(ValueError):
                parse_blinds(sb, bb)

    def test_decimal_blinds_and_currency_amounts_are_not_truncated(self):
        settings = parse_blinds('0.05', '0.10')
        self.assertEqual(settings, {'small_blind': .05, 'big_blind': .10})
        state = dict(spot(), pot=.15, to_call=.05, min_raise=.10, stack=10.00,
                     small_blind=.05, big_blind=.10, hero_bet=.05)
        decision = VisionPolicyAgent(equity_samples=5).decide(json.dumps(state), settings)
        self.assertEqual(decision.amount, .25)
        self.assertEqual(format_action(decision), 'RAISE 0.25')
        self.assertEqual(decision.state['stack'], 10)
        self.assertIn('SB 0.05 · BB 0.10', format_stakes(decision.state))
        self.assertIn('Stack 100.0 bb', format_stakes(decision.state))
        self.assertIn('Add 0.25 (2.50 bb)', format_bet_sizing(decision))
        self.assertIn('Raise-to total: 0.30', format_bet_sizing(decision))

    def test_ai_validation_preserves_cents_and_rejects_nonfinite_or_subcent_values(self):
        from poker_research.ai_screen_reader import validate_table_state, AiScreenReaderError
        raw = dict(spot(), readable=True, hero_to_act=True, pot=.15, to_call=.05,
                   min_raise=.10, stack=10.25, small_blind=.05, big_blind=.10, hero_bet=.05)
        decoded = validate_table_state(raw)
        self.assertEqual(decoded['stack'], 10.25)
        self.assertEqual(decoded['to_call'], .05)
        for bad in (float('inf'), float('nan'), .005, True, -.01):
            with self.subTest(bad=bad), self.assertRaises(AiScreenReaderError):
                validate_table_state(dict(raw, pot=bad))

    def test_manual_decimal_blinds_are_passed_to_ai_without_rescaling(self):
        reader = Mock()
        reader.read.return_value = dict(spot(), pot=.15, to_call=.05, min_raise=.10,
                                      stack=10, hero_bet=.05)
        agent = VisionPolicyAgent(ai_reader=reader, reader_mode='ai', equity_samples=5)
        image = Image.new('RGB', (10, 10))
        blinds = parse_blinds('0.05', '0.10')
        result = agent.decide(image, blinds)
        reader.read.assert_called_once_with(image, blinds=blinds)
        self.assertEqual(result.amount, .25)
        self.assertEqual(result.state['money_precision'], 2)

    def test_decimal_postflop_pot_bet_is_rounded_in_cents_not_whole_units(self):
        from poker_research.stakes import policy_units, display_units
        from poker_research.policy import gto_policy
        import random
        state = dict(spot(), board=[[2, 's'], [3, 'h'], [4, 'c']], pot=.40, to_call=0,
                     min_raise=.10, stack=10, big_blind=.10, small_blind=.05, hero_bet=0)
        normalized, scale = policy_units(state)
        choice = gto_policy(normalized, .8, random.Random(1))
        self.assertEqual(display_units(choice.amount, scale), .30)

    def test_stakes_scale_opening_size_and_subtract_posted_small_blind(self):
        agent = VisionPolicyAgent(equity_samples=5)
        small = agent.decide(json.dumps(spot()))
        large = agent.decide(json.dumps(spot(10)))
        self.assertEqual((small.action, small.amount), ('raise', 25))
        self.assertEqual((large.action, large.amount), ('raise', 250))
        self.assertIn('Raise-to total: 30', format_bet_sizing(small))
        self.assertIn('Raise-to total: 300', format_bet_sizing(large))
        self.assertIn('Stack 100.0 bb', format_stakes(small.state))

    def test_manual_blinds_reach_reader_and_override_bad_detection(self):
        image = Image.new('RGB', (1000, 720))
        reader = Mock()
        reader.read.return_value = dict(spot(4), small_blind=5, big_blind=10, min_raise=10)
        agent = VisionPolicyAgent(ai_reader=reader, reader_mode='ai', equity_samples=5)
        settings = {'small_blind': 20, 'big_blind': 40}
        decision = agent.decide(image, settings)
        reader.read.assert_called_once_with(image, blinds=settings)
        self.assertEqual(decision.state['blind_source'], 'manual')
        self.assertEqual(decision.state['min_raise'], 40)
        self.assertEqual(decision.amount, 100)
        self.assertIn('Raise-to total: 120', format_bet_sizing(decision))

    def test_larger_raise_increment_is_not_replaced_by_big_blind(self):
        agent = VisionPolicyAgent(equity_samples=5)
        decision = agent.decide(json.dumps(dict(spot(), min_raise=80)), parse_blinds('5', '10'))
        self.assertEqual(decision.state['min_raise'], 80)
        self.assertEqual(decision.amount, 85)

    def test_invalid_settings_fail_before_api_request(self):
        reader = Mock()
        agent = VisionPolicyAgent(ai_reader=reader, reader_mode='ai')
        with self.assertRaises(ValueError):
            agent.decide(Image.new('RGB', (10, 10)), {'small_blind': 10, 'big_blind': 0})
        reader.read.assert_not_called()

    def test_unknown_posted_amount_does_not_invent_raise_to_total(self):
        decision = Decision('raise', 25, .5, 0, 1, {}, {'big_blind': 10})
        self.assertIn('total unknown', format_bet_sizing(decision))

    def test_bb_label_and_cent_multiplier_become_blind_units(self):
        labelled = align_chip_units(dict(
            spot(), stack=100, pot=1.5, to_call=.5, min_raise=1, hero_bet=.5,
            small_blind=.05, big_blind=.10,
            opponents=[dict(position='BB', stack=80, bet=1)],
            betting_history=[dict(street='preflop', position='SB', action='raise',
                                  amount=3, raise_to=3, pot_before=1.5)]))
        self.assertEqual(labelled['stack'], 10)
        self.assertEqual(labelled['pot'], .15)
        self.assertEqual(labelled['big_blind'], .10)
        self.assertEqual(labelled['opponents'][0]['stack'], 8)
        self.assertEqual(labelled['betting_history'][0]['raise_to'], .3)
        cents = align_chip_units(dict(
            spot(), stack=1000, pot=15, to_call=5, min_raise=10, hero_bet=5,
            small_blind=.05, big_blind=.10))
        self.assertEqual((cents['stack'], cents['pot'], cents['to_call']), (10, .15, .05))
        self.assertEqual(cents['small_blind'], .05)
        # An unread pot must not divide a "100 BB" label by 100.
        unread = align_chip_units(dict(
            spot(), stack=100, pot=0, to_call=0, min_raise=.1, hero_bet=0,
            small_blind=.05, big_blind=.10))
        self.assertEqual(unread['stack'], 10)
        deep = align_chip_units(dict(
            spot(), stack=100, pot=.15, to_call=.10, min_raise=.10, hero_bet=0,
            small_blind=.05, big_blind=.10))
        self.assertEqual(deep['stack'], 100)
        self.assertEqual(align_chip_units(spot())['stack'], 1000)
        mixed = dict(spot(), stack=100, pot=1.5, to_call=.05, min_raise=.10, hero_bet=.05,
                     small_blind=.05, big_blind=.10)
        self.assertEqual(align_chip_units(mixed)['stack'], 100)

    def test_manual_blinds_repair_a_big_blind_count_before_the_stats_line(self):
        state = dict(spot(), stack=100, pot=1.5, to_call=.5, min_raise=1, hero_bet=.5,
                     small_blind=1, big_blind=1,
                     opponents=[dict(position='BB', preflop_role='unknown', stack=100, bet=1)])
        decision = VisionPolicyAgent(equity_samples=5).decide(json.dumps(state), parse_blinds('0.05', '0.10'))
        self.assertEqual(decision.state['stack'], 10)
        self.assertEqual(decision.state['pot'], .15)
        self.assertEqual(decision.state['to_call'], .05)
        stakes = format_stakes(decision.state)
        self.assertIn('Stack 100.0 bb', stakes)
        self.assertIn('Pot 1.5 bb', stakes)
        self.assertIn('Call 0.5 bb', stakes)
        self.assertAlmostEqual(decision.state['strategy_context']['effective_stack_bb'], 100.5)

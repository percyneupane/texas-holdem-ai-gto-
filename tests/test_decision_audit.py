import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from poker_research.agent import Decision
from poker_research.cash_benchmark import CashConfig, play_trial_hand
from poker_research.agent import VisionPolicyAgent


def decision(action='call', amount=.10, equity=.20):
    return Decision(action, amount, equity, 0, 1.5, {'policy_ms': 1}, {
        'hero': [[14, 's'], [11, 'h']], 'board': [[2, 'c'], [7, 'd'], [9, 'h']],
        'pot': .30, 'to_call': .10, 'min_raise': .10, 'stack': 10,
        'hero_bet': 0, 'small_blind': .05, 'big_blind': .10,
        'active_players': 2, 'table_size': 6, 'hero_position': 'BTN',
        'preflop_action': 'raised', 'postflop_position': 'IP',
        'opponents': [{'position': 'BB', 'preflop_role': 'open', 'stack': 9,
                       'bet': .10, 'hand': [[13, 's'], [13, 'h']]}],
        'history_complete': False, 'betting_history': [],
        'deck': ['secret'], 'api_key': 'secret', 'strategy_context': {'spr': 30},
    }, 'test reason')


class DecisionAuditTests(unittest.TestCase):
    def test_record_retains_public_evidence_without_private_opponent_data(self):
        from poker_research.decision_audit import decision_record
        original = decision()
        row = decision_record(original, hand=3, index=1, policy='contextual', equity_samples=200)
        self.assertEqual(row['observation']['hero'], original.state['hero'])
        self.assertEqual(row['reason'], 'test reason')
        self.assertEqual(row['range_priors'][0]['position'], 'BB')
        self.assertIn('AA', row['range_priors'][0]['hand_classes'])
        self.assertNotIn('secret', json.dumps(row))
        self.assertNotIn('hand', row['observation']['opponents'][0])
        original.state['hero'][0][0] = 2
        self.assertEqual(row['observation']['hero'][0][0], 14)

    def test_decimal_call_proxy_has_correct_units_and_explicit_assumptions(self):
        from poker_research.decision_audit import decision_record
        row = decision_record(decision(), hand=1, index=1)
        self.assertAlmostEqual(row['call_price'], .25)
        self.assertAlmostEqual(row['call_showdown_proxy_bb'], -.2)
        self.assertIn('call_below_raw_pot_odds', row['review_flags'])
        self.assertIn('not_action_ev', row['proxy_limitations'])

    def test_raise_target_includes_already_posted_amount(self):
        from poker_research.decision_audit import decision_record
        value = decision('raise', .75, .80)
        value.state['hero_bet'] = .05
        row = decision_record(value, hand=1, index=1)
        self.assertEqual(row['recommendation']['raise_to'], .80)
        self.assertIsNone(row['call_showdown_proxy_bb'])

    def test_missing_blinds_do_not_invent_bb_or_ev_units(self):
        from poker_research.decision_audit import decision_record
        value = decision()
        value.state['big_blind'] = 0
        row = decision_record(value, hand=1, index=1)
        self.assertIsNone(row['amount_bb'])
        self.assertIsNone(row['call_showdown_proxy_bb'])
        self.assertIn('blinds_unknown', row['review_flags'])

    def test_leak_summary_does_not_count_hand_profit_once_per_decision(self):
        from poker_research.decision_audit import decision_record, analyze
        rows = [decision_record(decision(), hand=1, index=i) for i in (1, 2)]
        hands = [{'hand': 1, 'gross_bb': -2, 'rake_adjusted_bb': -2,
                  'hero_position': 'BTN', 'decision_ids': [r['decision_id'] for r in rows]}]
        report = analyze(rows, hands)
        self.assertEqual(report['gross_profit_bb'], -2)
        self.assertEqual(report['by_position']['BTN']['gross_bb'], -2)
        self.assertEqual(report['flagged_hand_outcomes']['call_below_raw_pot_odds']['gross_bb'], -2)
        self.assertEqual(report['biggest_losses'][0]['hand'], 1)
        self.assertEqual(report['decision_count'], 2)
        self.assertIn('not_proven_leaks', report['limitations'])

    def test_big_loss_is_review_candidate_not_a_confirmed_mistake(self):
        from poker_research.decision_audit import analyze
        report = analyze([], [{'hand': 8, 'gross_bb': -100, 'rake_adjusted_bb': -100,
                               'hero_position': 'BB', 'decision_ids': []}])
        self.assertEqual(report['biggest_losses'][0]['hand'], 8)
        self.assertEqual(report['flag_counts'], {})
        self.assertNotIn('mistakes', report)

    def test_empty_analysis_is_valid(self):
        from poker_research.decision_audit import analyze
        report = analyze([], [])
        self.assertEqual(report['completed_hands'], 0)
        self.assertEqual(report['gross_profit_bb'], 0)


class AuditBacktestTests(unittest.TestCase):
    def test_run_saves_decisions_hands_and_report_without_ai_calls(self):
        from poker_research.audit_backtest import run
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'audit'
            result = run(hands=4, players=3, seed=7, equity_samples=5, output_dir=output)
            self.assertTrue(result['complete'])
            self.assertEqual(result['completed_hands'], 4)
            self.assertEqual(result['api_calls'], 0)
            decisions = [json.loads(line) for line in (output / 'decisions.jsonl').read_text().splitlines()]
            hands = [json.loads(line) for line in (output / 'hands.jsonl').read_text().splitlines()]
            self.assertEqual(len(hands), 4)
            self.assertEqual(len(decisions), result['analysis']['decision_count'])
            self.assertEqual(result['analysis']['invalid_actions'], 0)
            self.assertTrue((output / 'report.md').is_file())
            self.assertEqual(json.loads((output / 'summary.json').read_text())['completed_hands'], 4)
            self.assertEqual(sum(h['gross_bb'] for h in hands), result['analysis']['gross_profit_bb'])

    def test_logging_does_not_change_decisions_or_seeded_results(self):
        from poker_research.audit_backtest import run
        a = run(hands=5, players=3, seed=9, equity_samples=5)
        with tempfile.TemporaryDirectory() as root:
            b = run(hands=5, players=3, seed=9, equity_samples=5, output_dir=Path(root) / 'audit')
        self.assertEqual(a['hand_results'], b['hand_results'])
        self.assertEqual(a['analysis'], b['analysis'])

    def test_existing_output_is_not_overwritten(self):
        from poker_research.audit_backtest import run
        with tempfile.TemporaryDirectory() as root:
            marker = Path(root) / 'keep.txt'
            marker.write_text('keep')
            with self.assertRaises(FileExistsError):
                run(hands=1, equity_samples=5, output_dir=root)
            self.assertEqual(marker.read_text(), 'keep')

    def test_interrupt_keeps_completed_hands_and_marks_incomplete(self):
        from poker_research.audit_backtest import run
        with tempfile.TemporaryDirectory() as root:
            def interrupt(progress):
                if progress['completed_hands'] == 2:
                    raise KeyboardInterrupt
            output = Path(root) / 'audit'
            result = run(hands=5, players=3, equity_samples=5, output_dir=output, progress=interrupt)
            self.assertFalse(result['complete'])
            self.assertEqual(result['completed_hands'], 2)
            self.assertEqual(result['stop_reason'], 'interrupted')
            self.assertEqual(len((output / 'hands.jsonl').read_text().splitlines()), 2)
            self.assertFalse(json.loads((output / 'summary.json').read_text())['complete'])

    def test_invalid_configuration_fails_before_creating_output(self):
        from poker_research.audit_backtest import run
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'invalid'
            for updates in [dict(hands=0), dict(players=11), dict(equity_samples=0), dict(policy='unknown')]:
                with self.subTest(updates=updates), self.assertRaises(ValueError):
                    run(output_dir=output, **updates)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()

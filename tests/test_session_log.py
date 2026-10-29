import json
import tempfile
import unittest

from poker_research.agent import Decision
from poker_research.session_log import SessionLog, observation_from_decision


def read_hand(hero, board, action, amount=0, stack=1000, **extra):
    payload = {
        'hero': hero, 'board': board, 'action': action, 'amount': amount, 'stack': stack,
        'equity': 0.4, 'reason': 'chart', 'position': 'BTN', 'pot': 15, 'to_call': 10,
        'big_blind': 10,
    }
    payload.update(extra)
    return payload


class SessionLogTests(unittest.TestCase):
    def test_table_hands_keep_chip_results_without_calling_them_a_real_edge(self):
        with tempfile.TemporaryDirectory() as directory:
            log = SessionLog(directory)
            log.add_table_hand({'hand_number': 1, 'profit_chips': -10, 'showdown': False, 'hero': ['As', 'Kd']})
            log.add_table_hand({'hand_number': 2, 'profit_chips': 30, 'showdown': True, 'hero': ['Qh', 'Qc']})
            totals = log.totals()
            self.assertEqual(totals['hands'], 2)
            self.assertEqual(totals['profit_chips'], 20)
            self.assertEqual(totals['profit_bb'], 2)
            self.assertEqual(totals['bb_per_100'], 100)
            self.assertEqual(totals['won_hands'], 1)
            self.assertEqual(totals['lost_hands'], 1)
            self.assertEqual(totals['showdowns'], 1)
            self.assertIn('not a real-money profit edge', totals['note'])
            self.assertIn('bb/100', log.line())
            self.assertTrue(totals['saved_to'].endswith('.jsonl'))
            with open(log.path, encoding='utf-8') as handle:
                lines = handle.read().strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])['kind'], 'table_hand')
            self.assertEqual(json.loads(lines[1])['hero'], ['Qh', 'Qc'])

    def test_overlay_reads_group_by_hole_cards_and_ignore_a_repeated_spot(self):
        log = SessionLog()
        self.assertTrue(log.observe(read_hand([[14, 's'], [13, 'h']], [], 'raise', 25, stack=990)))
        self.assertFalse(log.observe(read_hand(['As', 'Kh'], [], 'raise', 25, stack=990)))
        self.assertTrue(log.observe(read_hand(
            ['As', 'Kh'], [[2, 'c'], [7, 'd'], [9, 's']], 'bet', 15, stack=975, pot=40)))
        self.assertTrue(log.observe(read_hand(['Qh', 'Qc'], [], 'fold', stack=1100)))
        data = log.as_dict()
        self.assertEqual(data['table']['hands'], 0)
        self.assertEqual(data['table']['profit_chips'], 0)
        self.assertEqual(data['observed']['hands'], 2)
        self.assertEqual(data['observed']['decisions'], 3)
        first, second = data['observed_hands']
        self.assertEqual(first['hero'], ['As', 'Kh'])
        self.assertEqual(first['board'], ['2c', '7d', '9s'])
        self.assertEqual([item['action'] for item in first['decisions']], ['raise', 'bet'])
        self.assertEqual(first['stack_first'], 990)
        self.assertEqual(first['stack_last'], 975)
        self.assertFalse(first['in_progress'])
        self.assertEqual(second['hero'], ['Qh', 'Qc'])
        self.assertTrue(second['in_progress'])
        self.assertNotIn('profit_chips', first)
        self.assertIn('Overlay', log.line())

    def test_a_decision_becomes_one_observation(self):
        decision = Decision(
            'raise', 25, 0.62, 0.0, 12.0, {},
            {'hero': [[14, 'd'], [14, 'c']], 'board': [], 'street': 'preflop', 'pot': 15,
             'to_call': 10, 'stack': 990, 'big_blind': 10, 'hero_position': 'CO',
             'active_players': 6, 'preflop_action': 'unopened'},
            'opening chart')
        log = SessionLog()
        self.assertTrue(log.observe(observation_from_decision(decision)))
        hand = log.as_dict()['observed_hands'][0]
        self.assertEqual(hand['hero'], ['Ad', 'Ac'])
        self.assertEqual(hand['decisions'][0]['reason'], 'opening chart')
        self.assertEqual(hand['decisions'][0]['equity'], 0.62)
        self.assertEqual(hand['position'], 'CO')

    def test_bad_observations_are_refused(self):
        log = SessionLog()
        for bad in (
            {'hero': ['As'], 'action': 'fold'},
            {'hero': ['As', 'Kd'], 'action': 'allin'},
            {'hero': ['Zz', 'Kd'], 'action': 'fold'},
            {'hero': ['As', 'Kd'], 'action': 'bet', 'amount': -1},
            {'hero': ['As', 'Kd'], 'action': 'call', 'equity': 1.2},
            [],
        ):
            with self.assertRaises(ValueError):
                log.observe(bad)
        self.assertEqual(log.totals()['observed_decisions'], 0)


if __name__ == '__main__':
    unittest.main()

import json
import random
import unittest
from dataclasses import replace
from io import BytesIO
from unittest.mock import patch

from poker_research.ai_opponents import AIOpponentTable, AIOpponentError, PersonalityState
from poker_research.engine import Seat, start_hand
from poker_research.table_game import TableGame


class AIPlayerTests(unittest.TestCase):
    def state(self):
        return start_hand(tuple(Seat('P%d' % i) for i in range(6)), 0, random.Random(10))

    def response(self, action, target=0):
        text = json.dumps({'action': action, 'raise_to': target})
        return BytesIO(json.dumps({'candidates': [{'content': {'parts': [{'text': text}]}}],
                                  'usageMetadata': {'promptTokenCount': 10, 'candidatesTokenCount': 5}}).encode())

    def test_only_acting_players_cards_and_public_state_are_sent(self):
        state = self.state()
        controller = AIOpponentTable('test')
        observed = controller.observe(state)
        changed_seats = list(state.seats)
        other = [i for i in range(len(changed_seats)) if i != state.to_act]
        a, b = other[:2]
        changed_seats[a], changed_seats[b] = (replace(state.seats[a], hand=state.seats[b].hand),
                                            replace(state.seats[b], hand=state.seats[a].hand))
        changed = replace(state, seats=tuple(changed_seats), deck=tuple(reversed(state.deck)))
        self.assertEqual(controller.observe(changed), observed)
        # Explicitly inspect the network payload rather than relying on prompt wording.
        with patch('poker_research.ai_opponents.urlopen', return_value=self.response('call')) as request:
            controller.choose(state)
        body = json.loads(request.call_args.args[0].data)
        payload = json.loads(body['contents'][0]['parts'][0]['text'])
        self.assertEqual(payload['own_cards'], [[c.rank, c.suit] for c in state.seats[state.to_act].hand])
        self.assertNotIn('deck', payload)
        self.assertTrue(all('hand' not in p and 'cards' not in p for p in payload['players']))
        self.assertEqual(payload['board'], [])

    def test_budget_blocks_network_and_invalid_raise_is_not_clamped(self):
        state = self.state()
        controller = AIOpponentTable('test', max_calls=1)
        with patch('poker_research.ai_opponents.urlopen', return_value=self.response('raise', 999999)) as request:
            with self.assertRaises(AIOpponentError):
                controller.choose(state)
            with self.assertRaises(AIOpponentError):
                controller.choose(state)
        request.assert_called_once()

    def test_unused_raise_field_cannot_change_a_call(self):
        controller = AIOpponentTable('test')
        with patch('poker_research.ai_opponents.urlopen', return_value=self.response('call', 15)):
            self.assertEqual(controller.choose(self.state()), ('call', 0))
        self.assertEqual(controller.normalized_unused_amounts, 1)

    def test_emotional_state_changes_after_losses_then_recovers(self):
        p = PersonalityState('cautious')
        p.settle(-500, 1000)
        self.assertGreater(p.tilt, 0)
        self.assertLess(p.confidence, .5)
        self.assertEqual(p.loss_streak, 1)
        previous = p.tilt
        p.settle(500, 1000)
        self.assertLess(p.tilt, previous)
        self.assertEqual(p.loss_streak, 0)

    def test_table_pauses_without_silent_bot_fallback(self):
        from unittest.mock import Mock
        controller = Mock()
        controller.choose.side_effect = AIOpponentError('quota exhausted')
        game = TableGame(players=6, clock=lambda: 1000, opponent_controller=controller)
        try:
            game.tick()
            game.state = self.state()
            game.next_event = 0
            game.tick()
            game._opponent_future.exception(timeout=2)
            with patch('poker_research.table_game.opponent_action') as bots:
                game.tick()
                game.tick()
            self.assertEqual(game.opponent_error, 'quota exhausted')
            bots.assert_not_called()
        finally:
            game.close()

    def test_backtest_finishes_with_mocked_legal_ai_actions(self):
        from poker_research.ai_backtest import run
        controller = AIOpponentTable('test', max_calls=100)
        def answer(request, **kwargs):
            payload = json.loads(request.data)
            observation = json.loads(payload['contents'][0]['parts'][0]['text'])
            return self.response('call' if 'call' in observation['allowed_actions'] else 'check')
        with patch('poker_research.ai_opponents.urlopen', side_effect=answer):
            report = run(hands=2, players=2, controller=controller)
        self.assertTrue(report['complete'])
        self.assertEqual(report['completed_hands'], 2)
        self.assertGreater(report['opponents']['api_calls'], 0)
        self.assertEqual(report['opponents']['input_tokens'], report['opponents']['api_calls'] * 10)

    def test_backtest_marks_budget_stop_incomplete(self):
        from poker_research.ai_backtest import run
        controller = AIOpponentTable('test', max_calls=1)
        def answer(request, **kwargs):
            payload = json.loads(request.data)
            observation = json.loads(payload['contents'][0]['parts'][0]['text'])
            return self.response('call' if 'call' in observation['allowed_actions'] else 'check')
        with patch('poker_research.ai_opponents.urlopen', side_effect=answer):
            report = run(hands=10, players=2, controller=controller)
        self.assertFalse(report['complete'])
        self.assertLess(report['completed_hands'], 10)
        self.assertIn('budget reached', report['error'])

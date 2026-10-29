import unittest
import json
from io import BytesIO
from urllib.error import HTTPError, URLError
from unittest.mock import Mock, patch

from PIL import Image

from poker_research.ai_screen_reader import (
    GEMINI_HISTORY_LIMIT, PROMPT, TABLE_STATE_SCHEMA, AiScreenReaderError,
    GeminiScreenReader, FallbackScreenReader, validate_table_state)
from poker_research.agent import VisionPolicyAgent
from poker_research.vision import NotHeroTurn


def state():
    return dict(readable=True, hero_to_act=True, hero=[['14', 's'], ['13', 'h']],
                board=[], active_players=3, pot=100, to_call=10, min_raise=10,
                stack=1000, street='preflop', action_history=[])


class AiReaderTests(unittest.TestCase):
    def test_control_schema_uses_smaller_history_without_mutating_table_schema(self):
        from poker_research.table_ui_schema import TABLE_UI_SCHEMA
        for enabled in (True, False, True):
            with self.subTest(locate_controls=enabled):
                reader = GeminiScreenReader('test-key', locate_controls=enabled)
                response = BytesIO(json.dumps({'candidates': [{'content': {'parts': [
                    {'text': json.dumps(state())}]}}]}).encode())
                with patch('poker_research.ai_screen_reader.urlopen', return_value=response) as request:
                    reader.read(Image.new('RGB', (100, 100)))
                payload = json.loads(request.call_args.args[0].data)
                schema = payload['generationConfig']['responseSchema']
                limit = schema['properties']['betting_history']['maxItems']
                self.assertEqual(limit, 8 if enabled else GEMINI_HISTORY_LIMIT)
                prompt = payload['contents'][0]['parts'][1]['text']
                self.assertIn('at most %d events' % limit, prompt)
                self.assertIn('%d and set history_complete=false' % limit, prompt)
                self.assertEqual('table_ui' in schema['required'], enabled)
                if enabled:
                    self.assertEqual(schema['properties']['table_ui'], TABLE_UI_SCHEMA)
                self.assertEqual(TABLE_STATE_SCHEMA['properties']['betting_history']['maxItems'],
                                 GEMINI_HISTORY_LIMIT)

    def test_backup_takes_over_and_is_preferred_on_next_frame(self):
        primary = GeminiScreenReader('key', 'primary')
        backup = GeminiScreenReader('key', 'backup')
        primary.read = Mock(side_effect=AiScreenReaderError('limited', status=429))
        backup.read = Mock(return_value=state())
        reader = FallbackScreenReader([primary, backup])
        self.assertEqual(reader.read(Image.new('RGB', (10, 10))), state())
        self.assertEqual(reader.last_model, 'backup')
        reader.read(Image.new('RGB', (10, 10)))
        primary.read.assert_called_once()
        self.assertEqual(backup.read.call_count, 2)

    def test_authentication_errors_do_not_try_backup(self):
        primary = GeminiScreenReader('key', 'primary')
        backup = GeminiScreenReader('key', 'backup')
        primary.read = Mock(side_effect=AiScreenReaderError('denied', status=403, fatal=True))
        backup.read = Mock()
        with self.assertRaises(AiScreenReaderError):
            FallbackScreenReader([primary, backup]).read(Image.new('RGB', (10, 10)))
        backup.read.assert_not_called()

    def test_missing_model_is_disabled_but_backup_still_works(self):
        primary = GeminiScreenReader('key', 'primary')
        backup = GeminiScreenReader('key', 'backup')
        primary.read = Mock(side_effect=AiScreenReaderError('missing', status=404, fatal=True))
        backup.read = Mock(return_value=state())
        reader = FallbackScreenReader([primary, backup])
        reader.read(Image.new('RGB', (10, 10)))
        self.assertEqual(primary._next_request, float('inf'))
        self.assertEqual(reader.last_model, 'backup')
        self.assertEqual(reader.retry_in, 0)

    def test_incomplete_table_does_not_trigger_backup(self):
        primary = GeminiScreenReader('key', 'primary')
        backup = GeminiScreenReader('key', 'backup')
        primary.read = Mock(side_effect=AiScreenReaderError('missing call amount'))
        backup.read = Mock()
        with self.assertRaises(AiScreenReaderError):
            FallbackScreenReader([primary, backup]).read(Image.new('RGB', (10, 10)))
        backup.read.assert_not_called()

    def test_deadline_prevents_additional_calls(self):
        primary = GeminiScreenReader('key', 'primary')
        backup = GeminiScreenReader('key', 'backup')
        clock = [0.0]
        def timeout(*args, **kwargs):
            clock[0] = 8.0
            raise AiScreenReaderError('timeout', status='network/timeout')
        primary.read = Mock(side_effect=timeout)
        backup.read = Mock()
        with patch('poker_research.ai_screen_reader.time.monotonic', side_effect=lambda: clock[0]):
            with self.assertRaises(AiScreenReaderError):
                FallbackScreenReader([primary, backup]).read(Image.new('RGB', (10, 10)))
        backup.read.assert_not_called()

    def test_rate_limit_waits_without_repeated_network_calls(self):
        reader = GeminiScreenReader('test-key')
        error = HTTPError('https://example.test', 429, 'rate limited', {'Retry-After': '30'},
                          BytesIO(b'{"error":{"message":"quota reached"}}'))
        with patch('poker_research.ai_screen_reader.urlopen', side_effect=error) as request:
            with self.assertRaises(AiScreenReaderError) as first:
                reader.read(Image.new('RGB', (10, 10)))
            self.assertGreaterEqual(first.exception.retry_after, 30)
            self.assertFalse(first.exception.fatal)
            with self.assertRaises(AiScreenReaderError) as second:
                reader.read(Image.new('RGB', (10, 10)))
            self.assertGreater(second.exception.retry_after, 0)
            request.assert_called_once()

    def test_model_error_is_fatal_and_key_is_redacted(self):
        reader = GeminiScreenReader('test-key')
        error = HTTPError('https://example.test', 404, 'missing', {},
                          BytesIO(b'{"error":{"message":"test-key unknown model"}}'))
        with patch('poker_research.ai_screen_reader.urlopen', side_effect=error):
            with self.assertRaises(AiScreenReaderError) as caught:
                reader.read(Image.new('RGB', (10, 10)))
        self.assertTrue(caught.exception.fatal)
        self.assertNotIn('test-key', str(caught.exception))
        self.assertIn('model is unavailable', str(caught.exception))

    def test_gemini_history_cap_stays_inside_decoder_limit(self):
        history = TABLE_STATE_SCHEMA['properties']['betting_history']
        self.assertEqual(history['maxItems'], GEMINI_HISTORY_LIMIT)
        self.assertLessEqual(GEMINI_HISTORY_LIMIT, 20)
        self.assertIn('at most %d events' % GEMINI_HISTORY_LIMIT, PROMPT)
        self.assertIn('history_complete=false', PROMPT)

    def test_incomplete_read_explains_missing_field_and_retains_cards(self):
        with self.assertRaises(AiScreenReaderError) as caught:
            validate_table_state(dict(state(), readable=False, unreadable_reason='Call amount missing'))
        self.assertEqual(str(caught.exception), 'Call amount missing')
        self.assertEqual(caught.exception.observation['hero'], [[14, 's'], [13, 'h']])

    def test_normalizes_cards_and_rejects_invalid_states(self):
        self.assertEqual(validate_table_state(state())['hero'][0], [14, 's'])
        for changes in ({'readable': False}, {'hero': [[14, 's'], [14, 's']]},
                        {'to_call': True}, {'board': [[2, 's'], [3, 'h']]},
                        {'stack': float('nan')}, {'active_players': 11}):
            with self.subTest(changes=changes), self.assertRaises(AiScreenReaderError):
                validate_table_state(dict(state(), **changes))

    def test_face_ranks_suit_names_and_compact_tokens(self):
        parsed = validate_table_state(dict(
            state(), hero=[['A', 'h'], ['K', 'spades']],
            board=[['T', 'd'], ['10', 'c'], 'Q♥']))
        self.assertEqual(parsed['hero'], [[14, 'h'], [13, 's']])
        self.assertEqual(parsed['board'], [[10, 'd'], [10, 'c'], [12, 'h']])
        compact = validate_table_state(dict(state(), hero=[['As'], ['10h']]))
        self.assertEqual(compact['hero'], [[14, 's'], [10, 'h']])
        ace = validate_table_state(dict(state(), hero=[['1', 'c'], [13, 'd']]))
        self.assertEqual(ace['hero'][0], [14, 'c'])
        with self.assertRaises(AiScreenReaderError) as duplicate:
            validate_table_state(dict(state(), hero=[['A', 's'], ['14', 's']]))
        self.assertIn('valid poker state', str(duplicate.exception))
        with self.assertRaises(AiScreenReaderError) as bad_rank:
            validate_table_state(dict(state(), hero=[['Z', 's'], ['K', 'h']]))
        self.assertEqual(str(bad_rank.exception), 'AI returned an invalid card rank')
        with self.assertRaises(AiScreenReaderError) as bad_card:
            validate_table_state(dict(state(), hero=[['15', 's'], ['K', 'h']]))
        self.assertEqual(str(bad_card.exception), 'AI returned an invalid card')

    def test_waiting_frame_never_generates_decision(self):
        with self.assertRaises(NotHeroTurn):
            validate_table_state(dict(state(), hero_to_act=False))

    def test_direct_ai_reader_bypasses_fixed_layout(self):
        reader = Mock()
        reader.read.return_value = validate_table_state(state())
        agent = VisionPolicyAgent(ai_reader=reader, reader_mode='ai', equity_samples=5)
        with patch('poker_research.agent.decode_table_image') as local:
            decision = agent.decide(Image.new('RGB', (810, 600)))
        local.assert_not_called()
        reader.read.assert_called_once()
        self.assertEqual(decision.state['pot'], 100)

    def test_messy_opponents_do_not_block_the_decision(self):
        aliased = validate_table_state(dict(state(), active_players=2, opponents=[{
            'position': 'bb', 'preflop_role': '3-bet', 'stack': '9.10', 'bet': None}]))
        self.assertEqual(aliased['opponents'][0]['position'], 'BB')
        self.assertEqual(aliased['opponents'][0]['preflop_role'], 'three_bet')
        self.assertAlmostEqual(aliased['opponents'][0]['stack'], 9.1)
        self.assertNotIn('bet', aliased['opponents'][0])
        self.assertEqual(validate_table_state(dict(state(), opponents=None))['opponents'], [])
        extra = validate_table_state(dict(state(), active_players=2, opponents=[
            {'position': 'BB', 'preflop_role': 'open'},
            {'position': 'SB', 'preflop_role': 'call'}]))
        self.assertEqual(extra['opponents'], [])
        self.assertEqual(extra['pot'], 100)
        folded = validate_table_state(dict(state(), hero_position='BTN', active_players=2, opponents=[
            {'position': 'BTN', 'preflop_role': 'open'},
            {'position': 'bb', 'preflop_role': 'fold'},
            {'position': 'SB', 'preflop_role': 'call'}]))
        self.assertEqual(folded['opponents'], [{'position': 'SB', 'preflop_role': 'call'}])

        def read(image, blinds=None):
            return validate_table_state(dict(state(), opponents=[{'position': 'middle', 'preflop_role': None}, False]))

        agent = VisionPolicyAgent(ai_reader=Mock(read=read), reader_mode='ai', equity_samples=5)
        decision = agent.decide(Image.new('RGB', (32, 32)))
        self.assertIn(decision.action, {'fold', 'call', 'check', 'bet', 'raise'})
        self.assertEqual(decision.state['opponents'][0]['position'], 'unknown')

    def test_read_timeout_retries_on_the_next_shot_not_after_a_long_lockout(self):
        reader = GeminiScreenReader('test-key')
        with patch('poker_research.ai_screen_reader.urlopen', side_effect=URLError(TimeoutError('timed out'))):
            with self.assertRaises(AiScreenReaderError) as caught:
                reader.read(Image.new('RGB', (10, 10)))
        self.assertLessEqual(caught.exception.retry_after, 1)
        self.assertFalse(caught.exception.fatal)
        self.assertEqual(caught.exception.status, 'network/timeout')

    def test_ai_failure_is_not_retried_within_same_decision(self):
        reader = Mock()
        reader.read.side_effect = AiScreenReaderError('unclear')
        agent = VisionPolicyAgent(ai_reader=reader, reader_mode='ai')
        with self.assertRaises(AiScreenReaderError):
            agent.decide(Image.new('RGB', (800, 600)))
        reader.read.assert_called_once()

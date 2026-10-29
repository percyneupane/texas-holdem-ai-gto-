import copy
import json
import unittest
from io import BytesIO
from unittest.mock import Mock, patch

from PIL import Image

from poker_research.agent import Decision
from poker_research.screen_capture import CaptureRegion


def ui(mode='raise_to', units='chips', separator='dot'):
    return dict(game='nlhe', table_count=1, amount_mode=mode, amount_units=units,
                decimal_separator=separator, amount_value=.3,
                controls=[dict(role=role, bounds=box, enabled=True, label=role,
                               kind='input' if role == 'amount' else 'button')
                          for role, box in [
                              ('fold', [50, 800, 200, 900]),
                              ('check', [300, 800, 450, 900]),
                              ('call', [300, 800, 450, 900]),
                              ('bet', [600, 800, 750, 900]),
                              ('raise', [600, 800, 750, 900]),
                              ('amount', [600, 700, 750, 780])]])


def decision(action='raise', amount=.25, **changes):
    state = dict(hero=[[14, 's'], [13, 's']], board=[], active_players=6,
                 pot=.15, to_call=.05, min_raise=.1, stack=9.95, hero_bet=.05,
                 big_blind=.1, can_raise=True, table_ui=ui())
    state.update(changes)
    return Decision(action, amount, .65, 0, 1, {}, state)


class AdaptiveControlTests(unittest.TestCase):
    def controller(self, width=1000, height=720):
        from poker_research.table_ui import AdaptiveScreenController
        self.mouse = Mock()
        self.mouse.size.return_value = (3000, 2000)
        self.image = Image.new('RGB', (width, height), 'green')
        return AdaptiveScreenController(CaptureRegion(100, 100, 100 + width, 100 + height),
                                        self.mouse, clock=lambda: 1)

    def test_layout_scale_position_and_table_size_matrix(self):
        for width, height in ((600, 432), (1000, 720), (1600, 900)):
            for seats in range(2, 11):
                with self.subTest(size=(width, height), seats=seats):
                    controller = self.controller(width, height)
                    read = decision('call', active_players=seats)
                    self.assertTrue(controller.play(read, self.image, self.image))
                    self.mouse.click.assert_called_once_with(100 + int(.375 * width),
                                                             100 + int(.85 * height))

    def test_rediscover_controls_for_next_decision(self):
        controller = self.controller()
        controller.play(decision('call'), self.image, self.image)
        controller.waiting()
        read = decision('call')
        read.state['table_ui']['controls'][2]['bounds'] = [100, 100, 200, 200]
        controller.play(read, self.image, self.image)
        self.assertEqual(self.mouse.click.call_args.args, (250, 208))

    def test_sizing_conventions_units_and_decimal_separators(self):
        cases = [('raise_to', 'chips', 'dot', '0.3'),
                 ('chips_to_add', 'chips', 'comma', '0,25'),
                 ('raise_to', 'bb', 'dot', '3'),
                 ('chips_to_add', 'bb', 'comma', '2,5')]
        for mode, units, separator, expected in cases:
            with self.subTest(mode=mode, units=units, separator=separator):
                controller = self.controller()
                read = decision(table_ui=ui(mode, units, separator))
                verify = Mock(return_value=True)
                controller.play(read, self.image, self.image, verify=verify)
                self.mouse.write.assert_called_once_with(expected, interval=.03)
                verify.assert_called_once_with(expected)

    def test_no_raise_without_readback_and_failed_readback_never_submits(self):
        controller = self.controller()
        with self.assertRaisesRegex(ValueError, 'readback'):
            controller.play(decision(), self.image, self.image)
        self.mouse.click.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'verified'):
            controller.play(decision(), self.image, self.image, verify=lambda text: False)
        # Only the amount input was clicked; the raise was not submitted.
        self.assertEqual(self.mouse.click.call_count, 1)
        self.assertFalse(controller.play(decision(), self.image, self.image, verify=lambda text: True))

    def test_unknown_ambiguous_disabled_and_out_of_bounds_controls_rejected(self):
        bad = []
        for change in (dict(game='plo'), dict(table_count=2), dict(amount_mode='unknown'),
                       dict(amount_units='unknown'), dict(decimal_separator='unknown')):
            value = ui()
            value.update(change)
            bad.append(value)
        for mutate in (
            lambda value: value['controls'].append(copy.deepcopy(value['controls'][-2])),
            lambda value: value['controls'][-2].update(enabled=False),
            lambda value: value['controls'][-2].update(bounds=[-1, 0, 1001, 1000]),
            lambda value: value['controls'][-2].update(bounds=[700, 900, 600, 800]),
            lambda value: value['controls'][-2].update(label=''),
        ):
            value = ui()
            mutate(value)
            bad.append(value)
        for value in bad:
            controller = self.controller()
            with self.assertRaises(ValueError):
                controller.play(decision(table_ui=value), self.image, self.image, verify=lambda text: True)
            self.mouse.click.assert_not_called()

    def test_nonterminating_bb_amount_is_rejected_instead_of_rounded(self):
        controller = self.controller()
        with self.assertRaises(ValueError):
            controller.play(decision(big_blind=.07, table_ui=ui(units='bb')),
                            self.image, self.image, verify=lambda text: True)
        self.mouse.click.assert_not_called()

    def test_readback_validates_value_spot_semantics_and_submit_control(self):
        cases = [
            ('valid', lambda read: None, True),
            ('wrong value', lambda read: read.state['table_ui'].update(amount_value=.4), False),
            ('unreadable value', lambda read: read.state['table_ui'].pop('amount_value'), False),
            ('changed pot', lambda read: read.state.update(pot=.25), False),
            ('changed policy', lambda read: setattr(read, 'action', 'fold'), False),
            ('changed semantics', lambda read: read.state['table_ui'].update(amount_units='bb'), False),
            ('disabled raise', lambda read: read.state['table_ui']['controls'][-2].update(enabled=False), False),
            ('missing metadata', lambda read: read.state.pop('table_ui'), False),
        ]
        for name, mutate, expected in cases:
            with self.subTest(name=name):
                controller = self.controller()
                original = decision()
                controller._resolve(original)
                read = decision()
                mutate(read)
                try:
                    result = controller.verify_entry(original, '0.3', read)
                except ValueError:
                    result = False
                self.assertEqual(result, expected)
                self.mouse.click.assert_not_called()

    def test_fresh_readback_can_relocate_submit_button(self):
        controller = self.controller()
        original = decision()
        controller._resolve(original)
        read = decision()
        read.state['table_ui']['controls'][-2]['bounds'] = [100, 100, 200, 200]
        self.assertTrue(controller.verify_entry(original, '0.3', read))
        self.assertEqual(controller.controls['raise'], (250, 208))

    def test_loop_readback_happens_after_typing_and_before_submit(self):
        from poker_research.screen_play import run
        controller = self.controller()
        agent = Mock()
        agent.ai_reader.retry_in = .75
        def read(image):
            if agent.decide.call_count == 2:
                self.mouse.write.assert_called_once_with('0.3', interval=.03)
                self.assertEqual(self.mouse.click.call_count, 1)
            return decision()
        agent.decide.side_effect = read
        capture = Mock(side_effect=[self.image] * 4 + [KeyboardInterrupt])
        with patch('poker_research.screen_play.time.sleep') as sleep, patch('builtins.print'):
            with self.assertRaises(KeyboardInterrupt):
                run(controller, agent, capture, play=True)
        sleep.assert_any_call(.8)
        self.assertEqual(agent.decide.call_count, 2)
        self.assertEqual(self.mouse.click.call_count, 2)
        self.assertEqual(self.mouse.click.call_args.args, (775, 712))

    def test_readback_frame_changing_during_verification_does_not_submit(self):
        from poker_research.screen_play import run
        controller = self.controller()
        agent = Mock()
        agent.decide.return_value = decision()
        changed = Image.new('RGB', self.image.size, 'red')
        capture = Mock(side_effect=[self.image] * 3 + [changed, KeyboardInterrupt])
        with patch('poker_research.screen_play.time.sleep'), patch('builtins.print'):
            with self.assertRaises(KeyboardInterrupt):
                run(controller, agent, capture, play=True)
        self.assertEqual(self.mouse.click.call_count, 1)

    def test_automatic_cli_does_not_require_a_profile(self):
        from poker_research.screen_play import main
        self.controller()
        self.mouse.FailSafeException = type('FailSafeException', (Exception,), {})
        for selection in (['--auto'], ['--bbox', '100,100,1100,820']):
            with patch.dict('sys.modules', pyautogui=self.mouse), \
                    patch('sys.argv', ['screen_play'] + selection), \
                    patch('poker_research.ai_screen_reader.GeminiScreenReader.from_env') as reader, \
                    patch('poker_research.screen_play.VisionPolicyAgent'), \
                    patch('poker_research.screen_play.display_scale', return_value=1), \
                    patch('poker_research.screen_play.SessionLog'), \
                    patch('poker_research.screen_play.run', side_effect=KeyboardInterrupt) as loop, \
                    patch('builtins.print'):
                self.assertEqual(main(), 0)
                reader.assert_called_once_with(locate_controls=True)
                self.assertTrue(loop.call_args.args[0].automatic)
                self.assertFalse(loop.call_args.kwargs['play'])
        self.mouse.click.assert_not_called()

    def test_reader_requests_control_schema_only_when_enabled(self):
        from poker_research.ai_screen_reader import GeminiScreenReader
        for enabled in (False, True):
            reader = GeminiScreenReader('test-key', locate_controls=enabled)
            raw = dict(readable=True, hero_to_act=True, hero=[['14', 's'], ['13', 'h']],
                       board=[], active_players=2, pot=10, to_call=0, min_raise=10, stack=100)
            if enabled:
                raw['table_ui'] = ui()
            response = BytesIO(json.dumps({'candidates': [{'content': {'parts': [
                {'text': json.dumps(raw)}]}}]}).encode())
            with patch('poker_research.ai_screen_reader.urlopen', return_value=response) as request:
                result = reader.read(Image.new('RGB', (100, 100)))
            payload = json.loads(request.call_args.args[0].data)
            schema = payload['generationConfig']['responseSchema']
            self.assertEqual('table_ui' in schema['required'], enabled)
            if enabled:
                self.assertEqual(result['table_ui'], raw['table_ui'])

    def test_model_backups_keep_automatic_control_detection_enabled(self):
        from poker_research.ai_screen_reader import GeminiScreenReader
        with patch.dict('os.environ', dict(GEMINI_API_KEY='test-key', GEMINI_MODEL='primary',
                                          GEMINI_BACKUP_MODELS='backup')):
            reader = GeminiScreenReader.from_env(locate_controls=True)
        self.assertEqual([entry.model for entry in reader.readers], ['primary', 'backup'])
        self.assertTrue(all(entry.locate_controls for entry in reader.readers))

    def test_agent_retains_control_metadata_with_decimal_stakes(self):
        from poker_research.agent import VisionPolicyAgent
        reader = Mock()
        reader.read.return_value = decision().state
        agent = VisionPolicyAgent(ai_reader=reader, reader_mode='ai', equity_samples=5)
        result = agent.decide(Image.new('RGB', (1000, 720)))
        self.assertEqual(result.state['table_ui'], ui())
        self.assertEqual(result.state['hero_bet'], .05)

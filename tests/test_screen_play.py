import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

from poker_research.agent import Decision
from poker_research.screen_capture import CaptureRegion


def decision(action='raise', amount=0.25, **changes):
    state = dict(hero=[[14, 's'], [13, 's']], board=[], pot=0.15,
                 to_call=0.05, stack=9.95, hero_bet=0.05, can_raise=True)
    state.update(changes)
    return Decision(action, amount, .65, 0, 1, {}, state)


class ScreenPlayTests(unittest.TestCase):
    def controller(self, mode='raise_to'):
        from poker_research.screen_play import ScreenController
        self.mouse = Mock()
        self.mouse.size.return_value = (2000, 1200)
        self.image = Image.new('RGB', (800, 600), 'green')
        self.profile = dict(bbox=[100, 100, 900, 700], amount_mode=mode,
                            controls=dict(fold=[100, 500], check=[300, 500],
                                          call=[300, 500], bet=[600, 500],
                                          raise_=[600, 500], amount=[600, 450]))
        self.profile['controls']['raise'] = self.profile['controls'].pop('raise_')
        return ScreenController(self.profile, self.mouse, clock=lambda: 1)

    def test_decimal_raise_to_includes_existing_street_bet(self):
        controller = self.controller()
        result = controller.play(decision(), self.image, self.image)
        self.assertTrue(result)
        self.mouse.write.assert_called_once_with('0.3', interval=0.03)
        self.assertEqual(self.mouse.click.call_args_list[-1].args, (700, 600))

    def test_chips_to_add_mode_does_not_add_existing_bet(self):
        controller = self.controller('chips_to_add')
        controller.play(decision(), self.image, self.image)
        self.mouse.write.assert_called_once_with('0.25', interval=0.03)

    def test_changed_screen_refuses_all_mouse_input(self):
        controller = self.controller()
        with self.assertRaisesRegex(ValueError, 'changed'):
            controller.play(decision(), self.image, Image.new('RGB', (800, 600), 'red'))
        self.mouse.click.assert_not_called()

    def test_small_change_to_action_label_also_blocks_click(self):
        controller = self.controller()
        changed = self.image.copy()
        ImageDraw.Draw(changed).rectangle((595, 495, 605, 505), fill='white')
        with self.assertRaisesRegex(ValueError, 'changed'):
            controller.play(decision(), self.image, changed)
        self.mouse.click.assert_not_called()

    def test_wrong_capture_dimensions_refuse_click(self):
        controller = self.controller()
        image = Image.new('RGB', (400, 300), 'green')
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            controller.play(decision(), image, image)
        self.mouse.click.assert_not_called()

    def test_same_spot_is_never_clicked_twice_even_after_waiting(self):
        controller = self.controller()
        controller.play(decision('call'), self.image, self.image)
        self.mouse.reset_mock()
        self.assertFalse(controller.play(decision('call'), self.image, self.image))
        self.mouse.click.assert_not_called()

    def test_observed_waiting_releases_lock_for_identical_next_hand(self):
        controller = self.controller()
        controller.play(decision('check'), self.image, self.image)
        controller.waiting()
        self.assertTrue(controller.play(decision('check'), self.image, self.image))

    def test_new_public_spot_acknowledges_previous_action(self):
        controller = self.controller()
        controller.play(decision('call'), self.image, self.image)
        self.assertTrue(controller.play(decision('check', pot=.3, to_call=0), self.image, self.image))

    def test_unacknowledged_action_times_out_without_retry(self):
        controller = self.controller()
        controller.play(decision('call'), self.image, self.image)
        controller.clock = lambda: 100
        with self.assertRaisesRegex(ValueError, 'confirmed'):
            controller.play(decision('call'), self.image, self.image)

    def test_missing_or_invalid_amount_state_cannot_click(self):
        for change in (dict(hero_bet=None), dict(hero_bet=-1), dict(can_raise=False),
                       dict(stack=.1)):
            controller = self.controller()
            with self.assertRaises(ValueError):
                controller.play(decision(**change), self.image, self.image)
            self.mouse.click.assert_not_called()

    def test_invalid_profiles_rejected_before_clicks(self):
        from poker_research.screen_play import ScreenController
        for change in (dict(amount_mode='guess'), dict(bbox=[0, 0, 0, 100]),
                       dict(controls={'call': [-1, 0]})):
            self.controller()
            self.profile.update(change)
            with self.assertRaises(ValueError):
                ScreenController(self.profile, self.mouse)
            self.mouse.click.assert_not_called()

    def test_profile_coordinates_are_relative_to_capture(self):
        controller = self.controller()
        self.assertEqual(controller.region, CaptureRegion(100, 100, 900, 700))
        controller.play(decision('fold'), self.image, self.image)
        self.mouse.click.assert_called_once_with(200, 600)

    def test_numeric_formatting_and_card_order_do_not_create_another_spot(self):
        controller = self.controller()
        controller.play(decision('call', pot=1, to_call=0), self.image, self.image)
        self.mouse.reset_mock()
        self.assertFalse(controller.play(decision('call', pot=1.0, to_call=0.0,
                                                hero=[[13, 's'], [14, 's']]), self.image, self.image))
        self.mouse.click.assert_not_called()

    def test_partial_driver_failure_keeps_action_locked(self):
        controller = self.controller()
        self.mouse.write.side_effect = RuntimeError('driver failed')
        with self.assertRaises(RuntimeError):
            controller.play(decision(), self.image, self.image)
        self.mouse.reset_mock()
        self.assertFalse(controller.play(decision(), self.image, self.image))
        self.mouse.click.assert_not_called()

    def test_loop_observes_waiting_then_sends_once_and_stops_on_interrupt(self):
        from poker_research.screen_play import run
        from poker_research.vision import NotHeroTurn
        controller = self.controller()
        agent = Mock()
        agent.decide.side_effect = [NotHeroTurn('waiting'), decision('call')]
        waiting = Image.new('RGB', self.image.size, 'black')
        capture = Mock(side_effect=[waiting, self.image, self.image, KeyboardInterrupt])
        with patch('poker_research.screen_play.time.sleep'), patch('builtins.print'):
            with self.assertRaises(KeyboardInterrupt):
                run(controller, agent, capture, play=True)
        self.mouse.click.assert_called_once_with(400, 600)

    def test_observe_mode_never_clicks(self):
        from poker_research.screen_play import run
        controller = self.controller()
        agent = Mock()
        agent.decide.return_value = decision('call')
        capture = Mock(side_effect=[self.image, KeyboardInterrupt])
        with patch('poker_research.screen_play.time.sleep'), patch('builtins.print'):
            with self.assertRaises(KeyboardInterrupt):
                run(controller, agent, capture)
        self.mouse.click.assert_not_called()

    def test_calibration_saves_relative_positions_without_sending_input(self):
        from poker_research.screen_play import calibrate
        self.controller()
        self.mouse.position.return_value = (400, 600)
        answers = ['100,100,900,700', 'raise_to', '', 's', '', 's', '', '']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'table.json'
            with patch('builtins.input', side_effect=answers), patch('builtins.print'), \
                    patch('poker_research.screen_play.time.sleep'):
                calibrate(path, self.mouse)
            profile = json.loads(path.read_text())
            self.assertEqual(profile['controls']['fold'], [300, 500])
            self.assertNotIn('check', profile['controls'])
        self.mouse.click.assert_not_called()
        self.mouse.write.assert_not_called()

    def test_cli_passes_profile_and_play_setting_to_loop(self):
        from poker_research.screen_play import main
        self.controller()
        self.mouse.FailSafeException = type('FailSafeException', (Exception,), {})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'table.json'
            path.write_text(json.dumps(self.profile))
            with patch.dict('sys.modules', pyautogui=self.mouse), \
                    patch('sys.argv', ['screen_play', '--profile', str(path), '--play']), \
                    patch('poker_research.screen_play.display_scale', return_value=1), \
                    patch('poker_research.screen_play.VisionPolicyAgent'), \
                    patch('poker_research.screen_play.SessionLog'), \
                    patch('poker_research.screen_play.run', side_effect=KeyboardInterrupt) as loop, \
                    patch('builtins.print'):
                self.assertEqual(main(), 0)
                self.assertTrue(loop.call_args.kwargs['play'])
        self.mouse.click.assert_not_called()

    def test_read_loop_discards_stale_result_and_rereads(self):
        from poker_research.screen_play import run
        controller = self.controller()
        agent = Mock()
        agent.decide.return_value = decision('call')
        changed = Image.new('RGB', self.image.size, 'red')
        capture = Mock(side_effect=[self.image, changed, changed, changed, KeyboardInterrupt])
        with patch('poker_research.screen_play.time.sleep'), patch('builtins.print'):
            with self.assertRaises(KeyboardInterrupt):
                run(controller, agent, capture, play=True)
        self.assertEqual(agent.decide.call_count, 2)
        self.mouse.click.assert_called_once_with(400, 600)

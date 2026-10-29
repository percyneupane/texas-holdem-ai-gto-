import unittest
from concurrent.futures import Future
from unittest.mock import Mock

from PIL import Image

from poker_research.screen_capture import CaptureRegion
from test_table_ui import decision


class SelectedAreaPlayTests(unittest.TestCase):
    def setUp(self):
        from poker_research.selected_area_play import SelectedAreaPlayer
        self.mouse = Mock()
        self.mouse.size.return_value = (2000, 1200)
        self.image = Image.new('RGB', (1000, 720), 'green')
        self.region = CaptureRegion(100, 100, 1100, 820)
        self.generation = 1
        self.clock = 100
        self.context = lambda: (self.region, self.generation)
        self.agent = Mock()
        self.agent.ai_reader.retry_in = 0
        self.executor = Mock()
        self.capture = Mock(return_value=self.image)
        self.player = SelectedAreaPlayer(self.agent, self.executor, self.capture, self.context,
                                         self.mouse, clock=lambda: self.clock)

    def test_clicks_only_inside_the_selected_region(self):
        self.assertTrue(self.player.start(decision('call'), self.image))
        self.mouse.click.assert_called_once_with(475, 712)
        self.assertFalse(self.player.busy)

    def test_raise_types_then_verifies_asynchronously_before_submitting(self):
        self.player.start(decision(), self.image)
        self.mouse.write.assert_called_once_with('0.3', interval=.03)
        self.assertEqual(self.mouse.click.call_count, 1)

        future = Future()
        self.executor.submit.return_value = future
        self.assertTrue(self.player.poll())
        self.assertTrue(self.player.busy)
        self.assertEqual(self.mouse.click.call_count, 1)
        future.set_result(decision())
        self.assertFalse(self.player.poll())
        self.assertEqual(self.mouse.click.call_count, 2)
        self.assertEqual(self.mouse.click.call_args.args, (775, 712))

    def test_moving_selection_cancels_pending_raise(self):
        self.player.start(decision(), self.image)
        self.region = CaptureRegion(200, 100, 1200, 820)
        with self.assertRaisesRegex(ValueError, 'selection'):
            self.player.poll()
        self.assertEqual(self.mouse.click.call_count, 1)

    def test_stop_cancels_pending_raise_without_submitting(self):
        self.player.start(decision(), self.image)
        self.player.cancel()
        self.assertFalse(self.player.poll())
        self.assertEqual(self.mouse.click.call_count, 1)

    def test_old_generation_can_never_submit(self):
        self.player.start(decision(), self.image)
        self.generation += 1
        with self.assertRaises(ValueError):
            self.player.poll()
        self.assertEqual(self.mouse.click.call_count, 1)

    def test_bad_readback_does_not_submit(self):
        self.player.start(decision(), self.image)
        future = Future()
        wrong = decision()
        wrong.state['table_ui']['amount_value'] = 100
        future.set_result(wrong)
        self.executor.submit.return_value = future
        self.player.poll()
        with self.assertRaisesRegex(ValueError, 'verified'):
            self.player.poll()
        self.assertEqual(self.mouse.click.call_count, 1)

    def test_provider_cooldown_keeps_ui_free_and_defers_readback(self):
        self.agent.ai_reader.retry_in = 2
        self.player.start(decision(), self.image)
        self.assertTrue(self.player.poll())
        self.executor.submit.assert_not_called()
        self.clock += 3
        self.player.poll()
        self.executor.submit.assert_called_once()

    def test_cancel_keeps_duplicate_lock(self):
        self.player.start(decision('call'), self.image)
        self.player.cancel()
        self.assertFalse(self.player.start(decision('call'), self.image))
        self.assertEqual(self.mouse.click.call_count, 1)


class OverlayAutoplayTests(unittest.TestCase):
    def overlay(self):
        from poker_research.overlay import ResearchOverlay
        overlay = object.__new__(ResearchOverlay)
        overlay.autoplay_target = 'screen'
        overlay.autoplay = False
        overlay.running = False
        overlay.generation = 0
        overlay._probe = None
        overlay._request_image = None
        overlay._screen_player = Mock(busy=False)
        overlay._screen_player.start.return_value = True
        overlay.autoplay_button = Mock()
        overlay.autoplay_hint = Mock()
        overlay._set_status = Mock()
        overlay.toggle_running = Mock()
        overlay._played_signature = None
        overlay._play_not_before = 0
        overlay.blind_overrides = {}
        return overlay

    def test_toggle_starts_reading_and_drops_old_recommendation(self):
        overlay = self.overlay()
        overlay._request_image = object()
        overlay.toggle_autoplay()
        self.assertTrue(overlay.autoplay)
        self.assertIsNone(overlay._request_image)
        overlay.toggle_running.assert_called_once()

    def test_stop_cancels_pending_screen_input(self):
        overlay = self.overlay()
        overlay.autoplay = True
        overlay.toggle_autoplay()
        self.assertFalse(overlay.autoplay)
        overlay._screen_player.cancel.assert_called_once()

    def test_selection_change_disarms_and_invalidates_pending_result(self):
        overlay = self.overlay()
        overlay.autoplay = True
        overlay._selection_changed()
        self.assertFalse(overlay.autoplay)
        self.assertEqual(overlay.generation, 1)
        overlay._screen_player.cancel.assert_called_once()

    def test_decision_routes_to_selected_area_once(self):
        overlay = self.overlay()
        overlay.running = overlay.autoplay = True
        overlay._request_image = object()
        result = decision('call')
        overlay._play_decision(result, b'1')
        overlay._play_decision(result, b'1')
        overlay._screen_player.start.assert_called_once_with(result, overlay._request_image, {})

    def test_missing_snapshot_or_stopped_reader_cannot_click(self):
        overlay = self.overlay()
        overlay.autoplay = True
        overlay._play_decision(decision('call'), b'1')
        overlay._screen_player.start.assert_not_called()

    def test_stale_screen_is_reread_without_disarming_autoplay(self):
        from poker_research.screen_play import ScreenChanged
        overlay = self.overlay()
        overlay.running = overlay.autoplay = True
        overlay._request_image = object()
        overlay._screen_player.start.side_effect = ScreenChanged('changed')
        overlay._play_decision(decision('call'), b'1')
        self.assertTrue(overlay.autoplay)
        self.assertIsNone(overlay._request_image)
        self.assertEqual(overlay._turn_state, 'idle')

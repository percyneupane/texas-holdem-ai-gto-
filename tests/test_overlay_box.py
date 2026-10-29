import unittest
from unittest.mock import Mock, patch
from concurrent.futures import Future

from PIL import Image, ImageDraw

from poker_research.overlay import (
    action_band, action_band_changed, button_fraction, buttons_up, format_action, format_cards,
    ResearchOverlay, decision_signature, turn_gate, watch_changed, watch_signature,
)
from poker_research.table_render import render_table_image
from poker_research.agent import Decision
from poker_research.overlay_box import MIN_HEIGHT, MIN_WIDTH, drag_region
from poker_research.screen_capture import CaptureRegion, capture_size
from poker_research.vision import SIZE

START = CaptureRegion(100, 100, 100 + SIZE[0], 100 + SIZE[1])


class DragRegionTests(unittest.TestCase):
    def test_move_shifts_whole_box(self):
        self.assertEqual(drag_region(START, "move", 25, -10),
                         CaptureRegion(125, 90, START.right + 25, START.bottom - 10))

    def test_right_edge_resizes_width_only(self):
        self.assertEqual(drag_region(START, "e", 200, 0), CaptureRegion(100, 100, START.right + 200, START.bottom))

    def test_top_left_corner_keeps_bottom_right_fixed(self):
        self.assertEqual(drag_region(START, "nw", -60, -40), CaptureRegion(40, 60, START.right, START.bottom))

    def test_bottom_edge_resizes_height_only(self):
        self.assertEqual(drag_region(START, "s", 0, 130), CaptureRegion(100, 100, START.right, START.bottom + 130))

    def test_can_shrink_below_table_size(self):
        region = drag_region(START, "se", -400, -300)
        self.assertEqual(region, CaptureRegion(100, 100, START.right - 400, START.bottom - 300))

    def test_cannot_shrink_below_handle_minimum_or_invert(self):
        region = drag_region(START, "nw", 5000, 5000)
        self.assertEqual((region.right - region.left, region.bottom - region.top), (MIN_WIDTH, MIN_HEIGHT))
        self.assertEqual((region.right, region.bottom), (START.right, START.bottom))


class ActionBandTests(unittest.TestCase):
    def _table(self):
        return Image.new('RGB', (1000, 720), (20, 24, 28))

    def test_static_frame_and_upper_table_motion_stay_quiet(self):
        base = self._table()
        self.assertFalse(action_band_changed(action_band(base), action_band(base)))
        dealt = base.copy()
        ImageDraw.Draw(dealt).rectangle((300, 200, 700, 340), fill=(240, 240, 240))
        self.assertFalse(action_band_changed(action_band(base), action_band(dealt)))
        timer = base.copy()
        ImageDraw.Draw(timer).rectangle((800, 640, 830, 662), fill=(220, 220, 220))
        self.assertFalse(action_band_changed(action_band(base), action_band(timer)))

    def test_action_buttons_appearing_wakes_the_reader(self):
        base = self._table()
        buttons = base.copy()
        draw = ImageDraw.Draw(buttons)
        draw.rectangle((250, 585, 410, 630), fill=(170, 48, 48))
        draw.rectangle((430, 585, 630, 630), fill=(48, 120, 180))
        draw.rectangle((650, 585, 850, 630), fill=(48, 150, 72))
        self.assertTrue(action_band_changed(None, action_band(base)))
        self.assertTrue(action_band_changed(action_band(base), action_band(buttons)))


class TurnLockTests(unittest.TestCase):
    def _table(self, hero_to_act, timer=None):
        state = {
            'hero': [[14, 's'], [13, 'h']], 'board': [], 'pot': 15, 'to_call': 10,
            'min_raise': 10, 'stack': 990, 'active_players': 6, 'hero_to_act': hero_to_act,
            'can_raise': True, 'raise_to': 30, 'current_bet': 10,
        }
        if timer is not None:
            state['timer'] = timer
        return render_table_image(state)

    def test_buttons_stay_up_through_the_clock_and_drop_when_the_turn_ends(self):
        hero = self._table(True, timer=1.0)
        later = self._table(True, timer=0.05)
        waiting = self._table(False)
        self.assertGreaterEqual(button_fraction(hero), 0.02)
        self.assertGreaterEqual(button_fraction(later), 0.02)
        self.assertLessEqual(button_fraction(waiting), 0.008)
        visible = buttons_up(None, button_fraction(hero))
        visible = buttons_up(visible, button_fraction(later))
        self.assertTrue(visible)
        self.assertFalse(buttons_up(visible, button_fraction(waiting)))

    def test_clock_holds_the_turn_and_a_new_board_or_hidden_buttons_release_it(self):
        hero = watch_signature(self._table(True, timer=1.0))
        later = watch_signature(self._table(True, timer=0.05))
        waiting = watch_signature(self._table(False))
        flop = watch_signature(render_table_image({
            'hero': [[14, 's'], [13, 'h']], 'board': [[2, 'c'], [7, 'd'], [9, 's']],
            'pot': 40, 'to_call': 20, 'min_raise': 20, 'stack': 990, 'active_players': 6,
            'hero_to_act': True, 'can_raise': True, 'raise_to': 60, 'current_bet': 20, 'timer': 1.0,
        }))
        new_hand = watch_signature(render_table_image({
            'hero': [[9, 'd'], [8, 'c']], 'board': [], 'pot': 15, 'to_call': 10,
            'min_raise': 10, 'stack': 990, 'active_players': 6, 'hero_to_act': True,
            'can_raise': True, 'raise_to': 30, 'current_bet': 10, 'timer': 1.0,
        }))
        waiting_image = self._table(False)
        moved = waiting_image.copy()
        draw = ImageDraw.Draw(moved)
        draw.rectangle((250, 280, 410, 340), fill=(170, 48, 48))
        draw.rectangle((430, 280, 630, 340), fill=(48, 120, 180))
        draw.rectangle((650, 280, 850, 340), fill=(48, 150, 72))
        self.assertFalse(watch_changed(hero, later))
        self.assertTrue(watch_changed(hero, waiting))
        self.assertTrue(watch_changed(hero, flop))
        self.assertTrue(watch_changed(hero, new_hand))
        self.assertTrue(watch_changed(waiting, flop))
        self.assertTrue(watch_changed(watch_signature(waiting_image), watch_signature(moved)))

    def test_only_active_buttons_start_a_read_and_rejection_locks_until_they_drop(self):
        for state in ('idle', 'answered', 'rejected'):
            self.assertEqual(turn_gate(state, False, True), ('idle', False))
        self.assertEqual(turn_gate('idle', True, False), ('reading', True))
        self.assertEqual(turn_gate('answered', True, False), ('answered', False))
        self.assertEqual(turn_gate('answered', True, True), ('reading', True))
        self.assertEqual(turn_gate('rejected', True, True), ('rejected', False))
        self.assertEqual(turn_gate('reading', True, True), ('reading', False))

    def test_opponent_animation_and_clock_do_not_change_the_decision(self):
        hero = self._table(True, timer=1.0)
        later = self._table(True, timer=0.05)
        ImageDraw.Draw(later).rectangle((100, 180, 250, 220), fill='white')
        self.assertFalse(watch_changed(decision_signature(hero), decision_signature(later)))
        ImageDraw.Draw(later).rectangle((310, 225, 380, 325), fill='white')
        self.assertTrue(watch_changed(decision_signature(hero), decision_signature(later)))

    def _overlay(self):
        overlay = object.__new__(ResearchOverlay)
        overlay.running = True
        overlay.autoplay_target = 'local'
        overlay.autoplay = False
        overlay._screen_player = None
        overlay._request_image = None
        overlay._turn_serial = 0
        overlay.blind_overrides = {}
        overlay.pending = None
        overlay.root = Mock()
        overlay.box = Mock()
        overlay.agent = Mock(reader_mode='local', ai_reader=None)
        overlay._probe = None
        overlay._read_once_requested = False
        overlay.request_manual = False
        def submit(function, *args):
            future = Future()
            try:
                future.set_result(function(*args))
            except Exception as error:
                future.set_exception(error)
            return future
        overlay.executor = Mock()
        overlay.executor.submit.side_effect = submit
        overlay.region = START
        overlay.generation = 0
        overlay.interval_ms = 250
        overlay.last_decision = None
        overlay._set_status = Mock()
        overlay._play_decision = Mock()
        overlay._show_decision = Mock()
        overlay.action_label = Mock()
        overlay.sizing_label = Mock()
        overlay.detail_label = Mock()
        overlay._reset_turn()
        return overlay

    def test_selected_screen_autoplay_reads_again_after_waiting_picture_changes(self):
        from poker_research.vision import NotHeroTurn
        overlay = self._overlay()
        overlay.autoplay_target = 'screen'
        overlay.autoplay = True
        overlay._screen_player = Mock(busy=False)
        overlay._screen_player.poll.return_value = False
        overlay.agent.decide.side_effect = [NotHeroTurn('Waiting'),
                                            Decision('call', 10, .5, 0, 1, {})]
        with patch.object(overlay, '_capture', return_value=self._table(False)) as capture:
            overlay._tick()
            overlay._tick()
            self.assertEqual(overlay._turn_state, 'rejected')
            overlay._screen_player.waiting.assert_called_once()
            capture.return_value = self._table(True)
            overlay._tick()
            self.assertEqual(overlay.agent.decide.call_count, 2)
            overlay._tick()
            overlay._play_decision.assert_called_once()

    def test_tick_never_reads_opponent_frames_and_reads_once_when_buttons_appear(self):
        overlay = self._overlay()
        overlay._begin_read = Mock()
        with patch.object(overlay, '_capture', return_value=self._table(False)) as capture:
            for amount in range(5):
                frame = self._table(False)
                ImageDraw.Draw(frame).rectangle((100, 180, 250, 220), fill=(amount * 50, 0, 0))
                capture.return_value = frame
                overlay._tick()
            overlay._begin_read.assert_not_called()
            capture.return_value = self._table(True)
            overlay._tick()
            overlay._begin_read.assert_called_once()
            overlay._turn_state = 'answered'
            overlay._locked_sig = decision_signature(capture.return_value)
            overlay._tick()
            overlay._begin_read.assert_called_once()
            capture.return_value = self._table(False)
            overlay._tick()
            capture.return_value = self._table(True)
            overlay._tick()
            self.assertEqual(overlay._begin_read.call_count, 2)

    def test_result_after_turn_ends_is_not_shown_or_played(self):
        overlay = self._overlay()
        overlay.pending = Future()
        overlay.pending.set_result(Decision('call', 10, 0.5, 0, 1, {}))
        overlay.request_region = START
        overlay.request_generation = 0
        overlay.request_started = 0
        overlay._request_sig = decision_signature(self._table(True))
        overlay._buttons_visible = True
        with patch.object(overlay, '_capture', return_value=self._table(False)):
            overlay._tick()
        overlay._show_decision.assert_not_called()
        overlay._play_decision.assert_not_called()
        self.assertEqual(overlay._turn_state, 'idle')

    def test_rejected_frame_is_not_retried_on_opponent_motion(self):
        overlay = self._overlay()
        overlay._turn_state = 'rejected'
        overlay._locked_sig = decision_signature(self._table(True))
        overlay._begin_read = Mock()
        frame = self._table(True)
        ImageDraw.Draw(frame).rectangle((100, 180, 250, 220), fill='white')
        with patch.object(overlay, '_capture', return_value=frame):
            overlay._tick()
        overlay._begin_read.assert_not_called()

    def test_current_turn_result_is_shown_and_played(self):
        overlay = self._overlay()
        decision = Decision('call', 10, 0.5, 0, 1, {})
        overlay.pending = Future()
        overlay.pending.set_result(decision)
        overlay.request_region = START
        overlay.request_generation = 0
        overlay.request_started = 0
        overlay.request_token = b'1'
        overlay.capture_ms = 1
        overlay._request_sig = decision_signature(self._table(True))
        with patch.object(overlay, '_capture', return_value=self._table(True)):
            overlay._tick()
        overlay._show_decision.assert_called_once()
        overlay._play_decision.assert_called_once_with(decision, b'1')
        self.assertEqual(overlay._turn_state, 'answered')

    def test_capture_restores_border_on_failure(self):
        overlay = self._overlay()
        overlay.pixel_scale = 1
        with patch('poker_research.overlay.ScreenCapture') as capture:
            capture.return_value.grab.side_effect = OSError('capture failed')
            with self.assertRaises(OSError):
                overlay._capture()
        self.assertEqual(overlay.box.set_visible.call_args.args, (True,))

    def test_ai_mode_uses_text_even_when_colour_detector_says_no_buttons(self):
        overlay = self._overlay()
        overlay.agent.reader_mode = 'ai'
        overlay.turn_detector = Mock()
        overlay.turn_detector.inspect.return_value = (True, (b'CALL 0.05', b'FOLD'))
        overlay._begin_read = Mock()
        grey_frame = Image.new('RGB', SIZE, (40, 40, 40))
        self.assertFalse(buttons_up(False, button_fraction(grey_frame)))
        with patch.object(overlay, '_capture', return_value=grey_frame):
            overlay._tick()
        overlay.turn_detector.inspect.assert_called_once_with(grey_frame)
        overlay._begin_read.assert_called_once()

    def test_text_detection_error_is_visible_and_read_now_bypasses_it_once(self):
        overlay = self._overlay()
        overlay.agent.reader_mode = 'ai'
        overlay.turn_detector = Mock()
        overlay.turn_detector.inspect.side_effect = RuntimeError('OCR unavailable; use Read now')
        overlay._begin_read = Mock()
        with patch.object(overlay, '_capture', return_value=self._table(True)):
            overlay._tick()
            overlay._begin_read.assert_not_called()
            overlay._set_status.assert_called_with('rejected', 'OCR unavailable; use Read now')
            overlay.read_now()
            overlay._tick()
            overlay._begin_read.assert_called_once()
            self.assertTrue(overlay._begin_read.call_args.kwargs['manual'])
            self.assertFalse(overlay._read_once_requested)

    def test_pending_local_detection_does_not_start_an_ai_read(self):
        overlay = self._overlay()
        probe = Future()
        overlay._probe = (probe, self._table(True), 0, 1)
        overlay._begin_read = Mock()
        overlay._tick()
        overlay._begin_read.assert_not_called()
        probe.set_result((True, decision_signature(self._table(True))))
        overlay._tick()
        overlay._begin_read.assert_called_once()


class CaptureSizeTests(unittest.TestCase):
    def test_box_larger_than_table_keeps_point_scale(self):
        self.assertEqual(capture_size((1400, 900)), (1400, 900))

    def test_tight_box_around_zoomed_out_table_scales_to_table_size(self):
        self.assertEqual(capture_size((800, 576)), SIZE)

    def test_small_box_scales_up_until_it_covers_the_table(self):
        width, height = capture_size((500, 700))
        self.assertGreaterEqual(width, SIZE[0])
        self.assertGreaterEqual(height, SIZE[1])
        self.assertAlmostEqual(width / height, 500 / 700, places=2)

    def test_rejects_empty_box(self):
        with self.assertRaises(ValueError):
            capture_size((0, 720))


class FormattingTests(unittest.TestCase):
    def test_cards_use_suit_symbols(self):
        self.assertEqual(format_cards([[14, "s"], [10, "h"]]), "A♠  T♥")
        self.assertEqual(format_cards([]), "—")

    def test_action_includes_amount_only_when_nonzero(self):
        self.assertEqual(format_action(Decision("raise", 65, 0.7, 0.1, 1.0, {})), "RAISE 65")
        self.assertEqual(format_action(Decision("check", 0, 0.3, 0.2, 1.0, {})), "CHECK")


if __name__ == "__main__":
    unittest.main()

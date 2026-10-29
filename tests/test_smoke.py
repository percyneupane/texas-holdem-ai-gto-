import json
import unittest
from unittest.mock import patch

from poker_research.agent import VisionPolicyAgent
from poker_research.cards import Card, evaluate
from poker_research.simulator import render_frame
from poker_research.simulator import render_image_frame
from poker_research.vision import decode_table_image
from poker_research.screen_capture import (CaptureRegion, ScreenCapture, display_scale, save_region_preview,
                                           scale_region, watch_decisions)


class SmokeTests(unittest.TestCase):
    def test_hand_evaluator_prefers_pair_over_high_card(self):
        pair = [Card(14, "s"), Card(14, "h"), Card(7, "c"), Card(4, "d"), Card(2, "s")]
        high = [Card(14, "s"), Card(13, "h"), Card(7, "c"), Card(4, "d"), Card(2, "s")]
        self.assertGreater(evaluate(pair), evaluate(high))

    def test_agent_returns_legal_action_and_metrics(self):
        frame = render_frame([Card(14, "s"), Card(14, "h")],
                             [Card(2, "c"), Card(7, "d"), Card(11, "s")],
                             players=10, pot=100, to_call=20)
        decision = VisionPolicyAgent(seed=3).decide(frame)
        self.assertIn(decision.action, {"fold", "call", "bet", "raise"})
        self.assertGreaterEqual(decision.equity, 0.0)
        self.assertLessEqual(decision.equity, 1.0)
        self.assertGreaterEqual(decision.latency_ms, 0.0)
        self.assertEqual(json.loads(frame)["active_players"], 10)

    def test_rendered_image_round_trip(self):
        image = render_image_frame([Card(14, "s"), Card(14, "h")],
                                   [Card(2, "c"), Card(7, "d"), Card(11, "s")],
                                   players=10, pot=100, to_call=20)
        state = decode_table_image(image)
        self.assertEqual(state["hero"], [[14, "s"], [14, "h"]])
        self.assertEqual(state["board"], [[2, "c"], [7, "d"], [11, "s"]])
        self.assertEqual(state["pot"], 100)
        self.assertEqual(state["to_call"], 20)

    def test_decoder_tolerates_offset_and_colour_shift(self):
        from PIL import Image, ImageChops
        image = render_image_frame([Card(14, "s"), Card(13, "h")],
                                   [Card(2, "c"), Card(7, "d"), Card(11, "s")],
                                   players=7, pot=286, to_call=57)
        # Box placed 4px right and 3px low, and display profile tints colours.
        shifted = Image.new("RGB", image.size, (30, 30, 30))
        shifted.paste(image.crop((4, 3, image.width, image.height)), (0, 0))
        tinted = ImageChops.add(shifted, Image.new("RGB", image.size, (11, 0, 1)))
        state = decode_table_image(tinted)
        self.assertEqual(state["hero"], [[14, "s"], [13, "h"]])
        self.assertEqual(state["board"], [[2, "c"], [7, "d"], [11, "s"]])
        self.assertEqual((state["pot"], state["to_call"], state["active_players"]), (286, 57, 7))

    def test_decoder_finds_table_inside_larger_box(self):
        from PIL import Image
        image = render_image_frame([Card(12, "d"), Card(9, "c")], [Card(4, "h"), Card(8, "s"), Card(13, "c")],
                                   players=5, pot=140, to_call=30)
        frame = Image.new("RGB", (image.width + 240, image.height + 180), (90, 120, 160))
        frame.paste(image, (173, 91))
        state = decode_table_image(frame)
        self.assertEqual(state["hero"], [[12, "d"], [9, "c"]])
        self.assertEqual(state["board"], [[4, "h"], [8, "s"], [13, "c"]])
        self.assertEqual((state["pot"], state["to_call"]), (140, 30))
        self.assertEqual(state, decode_table_image(image))

    def test_decoder_reports_missing_table(self):
        from PIL import Image
        from poker_research.vision import TableNotFoundError
        with self.assertRaises(TableNotFoundError):
            decode_table_image(Image.new("RGB", (1100, 700), (200, 200, 200)))

    def test_agent_accepts_rendered_image(self):
        image = render_image_frame([Card(14, "s"), Card(14, "h")], [], 10, 100, 20)
        decision = VisionPolicyAgent(seed=3).decide(image)
        self.assertIn(decision.action, {"fold", "call", "bet", "raise", "check"})

    def test_bounded_screen_capture_decodes_pixels(self):
        image = render_image_frame([Card(14, "s"), Card(14, "h")], [], 10, 100, 20)
        region = CaptureRegion(10, 20, 1010, 670)
        with patch("poker_research.screen_capture.ImageGrab.grab", return_value=image) as grab:
            state = ScreenCapture(region).read_state()
        grab.assert_called_once_with(bbox=region.bbox)
        self.assertEqual(state["hero"], [[14, "s"], [14, "h"]])
        self.assertEqual(state["pot"], 100)

    def test_finite_watch_runs_successive_frames(self):
        image = render_image_frame([Card(14, "s"), Card(14, "h")], [], 10, 100, 20)
        region = CaptureRegion(10, 20, 1010, 670)
        with patch("poker_research.screen_capture.ImageGrab.grab", return_value=image):
            decisions = list(watch_decisions(region, seconds=0.01, interval_ms=1))
        self.assertGreaterEqual(len(decisions), 1)

    def test_watch_accepts_unbounded_mode_with_test_stop(self):
        image = render_image_frame([Card(14, "s"), Card(14, "h")], [], 10, 100, 20)
        region = CaptureRegion(10, 20, 1010, 670)
        with patch("poker_research.screen_capture.ImageGrab.grab", side_effect=[image, KeyboardInterrupt]):
            iterator = watch_decisions(region, seconds=None, interval_ms=0)
            next(iterator)
            with self.assertRaises(KeyboardInterrupt):
                next(iterator)

    def test_scale_region_converts_points_to_retina_pixels(self):
        region = CaptureRegion(100, 100, 1100, 750)
        self.assertEqual(scale_region(region, 2.0), CaptureRegion(200, 200, 2200, 1500))
        self.assertEqual(scale_region(region, 1.0), region)

    def test_display_scale_compares_capture_width_to_points(self):
        from PIL import Image
        full = Image.new("RGB", (3024, 1964))
        # Older Pillow: bbox in pixels, so a 10-unit probe returns 10 pixels.
        with patch("poker_research.screen_capture.ImageGrab.grab", side_effect=[full, Image.new("RGB", (10, 10))]):
            self.assertEqual(display_scale(1512), 2.0)
        # Newer Pillow: bbox in points, so the probe already comes back at 2x.
        with patch("poker_research.screen_capture.ImageGrab.grab", side_effect=[full, Image.new("RGB", (20, 20))]):
            self.assertEqual(display_scale(1512), 1.0)
        with self.assertRaises(ValueError):
            display_scale(0)

    def test_preview_marks_selected_bbox(self):
        from tempfile import TemporaryDirectory
        image = render_image_frame([Card(14, "s"), Card(14, "h")], [], 10, 100, 20)
        region = CaptureRegion(10, 20, 100, 120)
        with TemporaryDirectory() as directory:
            output = __import__("pathlib").Path(directory) / "preview.png"
            with patch("poker_research.screen_capture.ImageGrab.grab", return_value=image):
                save_region_preview(region, output)
            self.assertTrue(output.exists())
            self.assertEqual(output.stat().st_size > 0, True)


if __name__ == "__main__":
    unittest.main()

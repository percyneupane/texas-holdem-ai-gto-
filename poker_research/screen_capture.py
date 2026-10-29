"""Opt-in local screen capture for the controlled research table.

This module captures only an explicitly supplied rectangle. It does not search
for windows, inspect other applications, or connect to poker websites.
macOS may require granting Screen Recording permission to the Python process.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageGrab

from .vision import SIZE, decode_table_image


@dataclass(frozen=True)
class CaptureRegion:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom


def scale_region(region: CaptureRegion, scale: float) -> CaptureRegion:
    """Convert a region in screen points (Tk/macOS UI) to physical capture pixels."""
    return CaptureRegion(*(int(round(value * scale)) for value in region.bbox))


def capture_size(box_size: Tuple[int, int]) -> Tuple[int, int]:
    """Frame size for a box in screen points.

    A box at least as large as the table keeps point scale so the decoder can
    search it. A smaller box is scaled up uniformly until it covers the table,
    so a tight box around a zoomed-out table becomes exactly table size.
    """
    width, height = box_size
    if width <= 0 or height <= 0:
        raise ValueError("box must have positive dimensions")
    factor = max(1.0, SIZE[0] / width, SIZE[1] / height)
    return int(round(width * factor)), int(round(height * factor))


PROBE_POINTS = 10


def display_scale(screen_width_points: int) -> float:
    """Factor converting screen points to ImageGrab bbox units on the main display.

    Pillow < 12 on macOS takes bbox in physical pixels (factor 2.0 on Retina);
    newer Pillow takes points and returns a 2x image (factor 1.0). Measure it
    rather than guessing from the version.
    """
    if screen_width_points <= 0:
        raise ValueError("screen width must be positive")
    pixels_per_point = ImageGrab.grab().size[0] / screen_width_points
    probe = ImageGrab.grab(bbox=(0, 0, PROBE_POINTS, PROBE_POINTS))
    pixels_per_bbox_unit = probe.size[0] / PROBE_POINTS
    return pixels_per_point / pixels_per_bbox_unit


class ScreenCapture:
    def __init__(self, region: CaptureRegion, output_size: Tuple[int, int] = SIZE):
        """`output_size` is the frame size at table scale; pass the box size in points
        when the box surrounds the table rather than matching it exactly."""
        self.region = region
        self.output_size = output_size

    def grab(self) -> Image.Image:
        """Capture the explicitly selected simulator rectangle once."""
        image = ImageGrab.grab(bbox=self.region.bbox)
        return image.convert("RGB").resize(self.output_size)

    def read_state(self):
        image = self.grab()
        return decode_table_image(image)


def capture_once(region: CaptureRegion, output: Optional[Path] = None):
    image = ScreenCapture(region).grab()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
    return decode_table_image(image)


def save_region_preview(region: CaptureRegion, output: Path) -> Path:
    """Save a full-screen diagnostic with the selected bbox outlined."""
    screen = ImageGrab.grab().convert("RGB")
    draw = ImageDraw.Draw(screen)
    draw.rectangle(region.bbox, outline=(255, 0, 0), width=5)
    screen.save(output)
    return output


def watch_decisions(region: CaptureRegion, seconds: Optional[float] = None, interval_ms: int = 100,
                    seed: int = 11):
    """Yield decisions from successive bounded screenshots until stopped."""
    from .agent import VisionPolicyAgent
    agent = VisionPolicyAgent(seed)
    capture = ScreenCapture(region)
    deadline = None if seconds is None else time.perf_counter() + max(0.0, seconds)
    while deadline is None or time.perf_counter() < deadline:
        started = time.perf_counter()
        try:
            yield agent.decide(capture.grab())
        except (ValueError, TypeError, KeyError) as error:
            print({"error": "invalid captured table frame", "detail": str(error)}, file=sys.stderr, flush=True)
        remaining = interval_ms / 1000.0 - (time.perf_counter() - started)
        if remaining > 0:
            time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture one local research-simulator screen region")
    parser.add_argument("--bbox", required=True, help="left,top,right,bottom")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preview", type=Path, help="save full-screen image with bbox outlined in red")
    parser.add_argument("--watch-seconds", type=float, default=0.0)
    parser.add_argument("--watch", action="store_true", help="run continuously until Ctrl-C")
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    values = [int(value) for value in args.bbox.split(",")]
    if len(values) != 4 or values[2] <= values[0] or values[3] <= values[1]:
        parser.error("--bbox must be left,top,right,bottom with positive dimensions")
    region = CaptureRegion(*values)
    if args.preview:
        save_region_preview(region, args.preview)
        print("preview saved to %s" % args.preview)
    if args.watch and args.watch_seconds > 0:
        parser.error("use only one of --watch or --watch-seconds")
    if args.watch or args.watch_seconds > 0:
        duration = None if args.watch else args.watch_seconds
        try:
            for decision in watch_decisions(region, duration, args.interval_ms, args.seed):
                print({"action": decision.action, "amount": decision.amount,
                       "equity": decision.equity, "bluff_probability": decision.bluff_probability,
                       "latency_ms": decision.latency_ms, "stages_ms": decision.stages_ms}, flush=True)
        except KeyboardInterrupt:
            print("\nwatch stopped")
    else:
        print(capture_once(region, args.output))


if __name__ == "__main__":
    main()

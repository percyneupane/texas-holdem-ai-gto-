"""Automatic or calibrated screen control for one visible No-Limit Hold'em table."""

import argparse
import json
import math
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

from PIL import ImageChops, ImageStat

from .agent import VisionPolicyAgent
from .ai_screen_reader import AiScreenReaderError
from .policy import POLICIES
from .screen_capture import CaptureRegion, ScreenCapture, display_scale, scale_region
from .session_log import SessionLog, observation_from_decision
from .vision import NotHeroTurn


class ScreenChanged(ValueError):
    """The table changed while the model was reading it; discard this decision."""


def money(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError('%s must be a nonnegative amount.' % label)
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError('Invalid %s.' % label) from error
    if not result.is_finite() or result < 0 or result.normalize().as_tuple().exponent < -2:
        raise ValueError('%s must be nonnegative with at most two decimal places.' % label)
    return Decimal(0) if result.is_zero() else result


def spot_key(state):
    # Only public decision amounts/cards. Reader diagnostics and equity estimates
    # must never turn a repeated screenshot into another click.
    cards = [sorted(state.get(key) or []) for key in ('hero', 'board')]
    amounts = [str(money(state[key], key).normalize()) if state.get(key) is not None else None
               for key in ('pot', 'to_call', 'stack', 'hero_bet')]
    return json.dumps(cards + amounts, sort_keys=True)


class ScreenController:
    def __init__(self, profile, mouse, clock=time.monotonic, ack_timeout=15):
        if not isinstance(profile, dict):
            raise ValueError('Profile must be a JSON object.')
        bbox = profile.get('bbox')
        if (not isinstance(bbox, list) or len(bbox) != 4 or
                any(type(value) is not int for value in bbox)):
            raise ValueError('bbox must contain four integer desktop coordinates.')
        self.region = CaptureRegion(*bbox)
        width, height = self.region.right - self.region.left, self.region.bottom - self.region.top
        screen_width, screen_height = mouse.size()
        if (width <= 0 or height <= 0 or self.region.left < 0 or self.region.top < 0 or
                self.region.right > screen_width or self.region.bottom > screen_height):
            raise ValueError('The capture box must fit within the primary display.')
        self.amount_mode = profile.get('amount_mode')
        if self.amount_mode not in ('raise_to', 'chips_to_add'):
            raise ValueError('amount_mode must be raise_to or chips_to_add.')
        controls = profile.get('controls')
        if not isinstance(controls, dict) or not controls:
            raise ValueError('The profile needs calibrated controls.')
        self.controls = {}
        for name, point in controls.items():
            if (name not in ('fold', 'check', 'call', 'bet', 'raise', 'amount') or
                    not isinstance(point, list) or len(point) != 2 or
                    any(type(value) is not int for value in point) or
                    not (0 <= point[0] < width and 0 <= point[1] < height)):
                raise ValueError('Invalid control coordinates: %s.' % name)
            self.controls[name] = (self.region.left + point[0], self.region.top + point[1])
        if not math.isfinite(ack_timeout) or ack_timeout <= 0:
            raise ValueError('Action confirmation timeout must be positive.')
        self.mouse = mouse
        self.clock = clock
        self.ack_timeout = ack_timeout
        self.pending = None
        self.sent_at = 0

    def waiting(self):
        """Reader positively identified a non-hero turn, acknowledging progression."""
        self.pending = None

    def check_timeout(self):
        if self.pending is not None and self.clock() - self.sent_at >= self.ack_timeout:
            raise ValueError('The previous action could not be confirmed. Stopped without retrying it.')

    def format_amount(self, total, state):
        text = format(total, 'f')
        return text.rstrip('0').rstrip('.') if '.' in text else text

    def ensure_fresh(self, observed, current, action):
        from .overlay import watch_changed, watch_signature
        expected_size = (self.region.right - self.region.left, self.region.bottom - self.region.top)
        if observed.size != expected_size or current.size != expected_size:
            raise ValueError('Capture dimensions no longer match the calibrated table.')
        if watch_changed(watch_signature(observed), watch_signature(current)):
            raise ScreenChanged('Table changed while reading. Discarding the stale decision.')
        if action not in self.controls or action == 'amount':
            raise ValueError('No calibrated %s button; stopped without clicking.' % action)
        # Fine comparison at the target catches a changed button label/amount
        # too small to affect the coarse whole-table motion detector.
        for name in (action, 'amount') if action in ('bet', 'raise') else (action,):
            if name not in self.controls:
                continue
            x, y = self.controls[name]
            x, y = x - self.region.left, y - self.region.top
            box = (max(0, x - 45), max(0, y - 18),
                   min(expected_size[0], x + 45), min(expected_size[1], y + 18))
            diff = ImageChops.difference(observed.crop(box).convert('RGB'), current.crop(box).convert('RGB'))
            if max(ImageStat.Stat(diff).mean) > 1:
                raise ScreenChanged('Action controls changed while reading. Discarding the stale decision.')

    def prepare(self, decision, observed, current):
        """Validate and type, returning a transaction that has not been submitted."""
        state = decision.state or {}
        key = spot_key(state)
        self.check_timeout()
        if key == self.pending:
            return None
        action = decision.action
        self.ensure_fresh(observed, current, action)
        amount_text = None
        if action in ('bet', 'raise'):
            if state.get('can_raise') is False or 'amount' not in self.controls:
                raise ValueError('No legal raise or calibrated amount field.')
            added = money(decision.amount, 'Chips to add')
            if added <= money(state.get('to_call'), 'Call') or added > money(state.get('stack'), 'Stack'):
                raise ValueError('Raise amount is outside the available stack/call bounds.')
            total = added
            if self.amount_mode == 'raise_to':
                total += money(state.get('hero_bet'), 'Current street contribution')
            amount_text = self.format_amount(total, state)
        # Lock before any input: partial typing, fail-safe, or driver errors must
        # never cause an automatic retry of a possibly submitted action.
        self.pending, self.sent_at = key, self.clock()
        if amount_text is not None:
            self.mouse.click(*self.controls['amount'])
            self.mouse.hotkey('command' if sys.platform == 'darwin' else 'ctrl', 'a')
            self.mouse.write(amount_text, interval=0.03)
        return action, amount_text

    def submit(self, action):
        self.check_timeout()
        self.mouse.click(*self.controls[action])

    def play(self, decision, observed, current, verify=None):
        prepared = self.prepare(decision, observed, current)
        if prepared is None:
            return False
        action, amount_text = prepared
        if amount_text is not None:
            if verify is not None and not verify(amount_text):
                raise ValueError('The entered bet could not be verified. Stopped before submitting.')
        self.submit(action)
        return True


def calibrate(path, mouse):
    """Read mouse positions only; calibration never clicks or types into a table."""
    bbox = [int(item.strip()) for item in input('Table box left,top,right,bottom (desktop points): ').split(',')]
    mode = input('Amount field mode (raise_to or chips_to_add): ').strip()
    ScreenController(dict(bbox=bbox, amount_mode=mode, controls={'fold': [0, 0]}), mouse)
    controls = {}
    for name in ('fold', 'check', 'call', 'bet', 'raise', 'amount'):
        answer = input('Calibrate %s? Enter to capture in 4 seconds; s to skip: ' % name).strip().lower()
        if answer == 's':
            continue
        print('Hover over the CENTER of %s now…' % name, flush=True)
        time.sleep(4)
        x, y = mouse.position()
        controls[name] = [int(x) - bbox[0], int(y) - bbox[1]]
        print('%s: %s' % (name, controls[name]))
    profile = dict(bbox=bbox, amount_mode=mode, controls=controls)
    ScreenController(profile, mouse)
    with Path(path).open('x') as output:
        json.dump(profile, output, indent=2)
        output.write('\n')
    print('Saved %s' % path)


def run(controller, agent, capture, *, play=False, interval=1, session=None):
    """Observe, decide, recheck the frame, act once, and wait for progression."""
    last_image = None
    from .overlay import watch_changed, watch_signature

    while True:
        controller.check_timeout()
        image = capture()
        if last_image is not None and not watch_changed(watch_signature(last_image), watch_signature(image)):
            time.sleep(interval)
            continue
        try:
            result = agent.decide(image)
        except NotHeroTurn:
            controller.waiting()
            last_image = image
            time.sleep(interval)
            continue
        except AiScreenReaderError as error:
            if error.fatal:
                raise
            print('Waiting: %s' % error, flush=True)
            time.sleep(max(interval, error.retry_after))
            continue
        if session is not None:
            session.observe(observation_from_decision(result))
        print('%s %s — %s' % (result.action.upper(), result.amount, result.reason), flush=True)
        if play:
            try:
                if getattr(controller, 'automatic', False):
                    def verify(text):
                        # A successful vision call has a short provider cooldown.
                        # Respect it before readback; never turn that cooldown
                        # into a failed bet or a second submit attempt.
                        cooldown = getattr(agent.ai_reader, 'retry_in', 0)
                        if isinstance(cooldown, (int, float)) and cooldown > 0:
                            remaining = controller.sent_at + controller.ack_timeout - controller.clock()
                            if not math.isfinite(cooldown) or cooldown + .05 >= remaining:
                                raise ValueError('Vision readback cannot finish before the action timeout.')
                            time.sleep(cooldown + .05)
                        controller.check_timeout()
                        entry_image = capture()
                        readback = agent.decide(entry_image)
                        if not controller.verify_entry(result, text, readback):
                            return False
                        controller.ensure_fresh(entry_image, capture(), result.action)
                        controller.check_timeout()
                        return True
                    sent = controller.play(result, image, capture(), verify=verify)
                else:
                    sent = controller.play(result, image, capture())
            except ScreenChanged as error:
                print(str(error), flush=True)
                time.sleep(interval)
                continue
            if sent:
                print('Input sent; waiting for visible table progression.', flush=True)
        last_image = image
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--profile', help='Calibrated table JSON file')
    group.add_argument('--calibrate', metavar='FILE', help='Create a profile by hovering over controls')
    group.add_argument('--auto', action='store_true', help='Find controls on one visible table on the primary display')
    group.add_argument('--bbox', help='Find controls within left,top,right,bottom desktop points')
    parser.add_argument('--play', action='store_true', help='Send mouse/keyboard input; default is observe only')
    parser.add_argument('--policy', choices=sorted(POLICIES), default='contextual')
    parser.add_argument('--reader', choices=('ai', 'local'), default='ai')
    parser.add_argument('--interval', type=float, default=1, help='Seconds between screen checks')
    parser.add_argument('--ack-timeout', type=float, default=15)
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error('--interval must be positive')
    try:
        import pyautogui
    except ImportError:
        parser.exit(2, 'Install screen controls: .venv/bin/python -m pip install pyautogui\n')
    # Preserve PyAutoGUI's corner fail-safe. Driver calls deliberately stay on
    # this thread so Ctrl-C or a fail-safe stops the entire controller.
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = .15
    try:
        if args.calibrate:
            calibrate(args.calibrate, pyautogui)
            return 0
        automatic = args.auto or args.bbox is not None
        if automatic:
            if args.reader != 'ai':
                raise ValueError('Automatic control detection requires --reader ai.')
            from .table_ui import AdaptiveScreenController
            from .ai_screen_reader import GeminiScreenReader
            if args.bbox:
                values = [int(item.strip()) for item in args.bbox.split(',')]
                if len(values) != 4:
                    raise ValueError('--bbox must contain left,top,right,bottom.')
            else:
                width, height = pyautogui.size()
                values = [0, 0, width, height]
            controller = AdaptiveScreenController(CaptureRegion(*values), pyautogui,
                                                  ack_timeout=args.ack_timeout)
            reader = GeminiScreenReader.from_env(locate_controls=True)
            if reader is None:
                raise ValueError('Set GEMINI_API_KEY in .env for automatic controls.')
            agent = VisionPolicyAgent(policy=args.policy, reader_mode='ai', ai_reader=reader)
        else:
            with Path(args.profile).open() as source:
                profile = json.load(source)
            controller = ScreenController(profile, pyautogui, ack_timeout=args.ack_timeout)
            agent = VisionPolicyAgent(policy=args.policy, reader_mode=args.reader)
        scale = display_scale(pyautogui.size()[0])
        region = controller.region
        capture = ScreenCapture(scale_region(region, scale),
                                (region.right - region.left, region.bottom - region.top)).grab
        if args.reader == 'ai' and agent.ai_reader is None:
            raise ValueError('Set GEMINI_API_KEY in .env for AI reading.')
        session = SessionLog('sessions', prefix='screen')
        print('Screen play %s. Keep one table visible within the capture area. Ctrl-C stops; a screen '
              'corner triggers the fail-safe before the next input.' % ('ON' if args.play else 'OBSERVE ONLY'),
              flush=True)
        run(controller, agent, capture, play=args.play, interval=args.interval, session=session)
    except KeyboardInterrupt:
        print('\nScreen play stopped.')
        return 0
    except (ValueError, OSError, pyautogui.FailSafeException) as error:
        print('Screen play stopped: %s' % error, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

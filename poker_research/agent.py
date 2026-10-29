from collections import OrderedDict
from dataclasses import dataclass, replace
import json
import random
import time
import zlib
from typing import Dict, Optional, Tuple, Union

from .cards import Card, equity
from .ai_screen_reader import GeminiScreenReader
from .policy import POLICIES, PolicyChoice
from .vision import NotHeroTurn, decode_table_image
from .ranges import range_equity, opponent_profiles, range_classes
from .stakes import validate_blinds, policy_units, display_units, align_chip_units
from .betting_context import enrich_context, validate_betting_context

EQUITY_SAMPLES = 200
SPOT_CACHE_SIZE = 256
# Fields that define a decision spot; the same spot always gets the same answer.
SPOT_KEYS = ("hero", "board", "active_players", "pot", "to_call", "min_raise", "stack")
CONTEXT_KEYS = ('hero_position', 'preflop_action', 'aggressor_position', 'small_blind', 'big_blind',
                'hero_bet', 'table_size', 'postflop_position', 'opponents', 'limper_count', 'can_raise',
                'betting_history', 'history_complete', 'current_bet', 'min_raise_to', 'max_raise_to')


@dataclass
class Decision:
    action: str
    amount: Union[int, float]
    equity: float
    bluff_probability: float
    latency_ms: float
    stages_ms: Dict[str, float]
    state: Optional[Dict] = None
    reason: str = ''


class VisionStateError(ValueError):
    """Raised when a captured frame cannot be mapped to a valid table state."""


class VisionPolicyAgent:
    """Decides from simulator frames (images) or JSON states, in simulator chip units."""

    def __init__(self, seed: int = 11, policy: str = "contextual", equity_samples: int = EQUITY_SAMPLES,
                 ai_reader=None, reader_mode='hybrid'):
        if reader_mode not in ('local', 'hybrid', 'ai'):
            raise ValueError('Invalid reader mode')
        self.reader_mode = reader_mode
        if policy not in POLICIES:
            raise ValueError("unknown policy %r; choose from %s" % (policy, ", ".join(sorted(POLICIES))))
        self.seed = seed
        self.policy_name = policy
        self.policy = POLICIES[policy]
        self.equity_samples = equity_samples
        self.ai_reader = ai_reader if ai_reader is not None else GeminiScreenReader.from_env()
        self._spots: "OrderedDict[str, Tuple[float, PolicyChoice]]" = OrderedDict()

    def _spot_rng(self, spot: str) -> random.Random:
        """Randomness derived from the spot itself, so re-reading a frame cannot change the answer."""
        return random.Random(zlib.crc32(spot.encode("utf-8")) ^ self.seed)

    def _evaluate(self, state: Dict) -> Tuple[float, PolicyChoice]:
        hero = [Card(rank, suit) for rank, suit in state["hero"]]
        board = [Card(rank, suit) for rank, suit in state["board"]]
        if len(hero) != 2 or len(board) > 5 or len(set(hero + board)) != len(hero + board):
            raise VisionStateError("frame did not contain a valid 2-card hero hand and board")
        spot = json.dumps([self.policy_name, [state[key] for key in SPOT_KEYS],
                           {key: state.get(key) for key in CONTEXT_KEYS}], sort_keys=True)
        cached = self._spots.get(spot)
        if cached is not None:
            self._spots.move_to_end(spot)
            return cached
        rng = self._spot_rng(spot)
        opponents = max(2, min(10, int(state["active_players"]))) - 1
        profiles = opponent_profiles(state)
        informed = sum(range_classes(p.get('position'), p.get('preflop_role'), state.get('table_size') or 6)
                       is not None for p in profiles)
        if self.policy_name == 'contextual' and informed:
            hand_equity = range_equity(hero, board, state, rng, samples=self.equity_samples)
        else:
            hand_equity = equity(hero, board, opponents, rng, samples=self.equity_samples)
        choice = self.policy(state, hand_equity, rng)
        if self.policy_name == 'contextual':
            basis = 'Opponent range priors: %d/%d; others random.' % (informed, opponents)
            choice = replace(choice, reason=choice.reason + '. ' + basis)
        result = (hand_equity, choice)
        self._spots[spot] = result
        if len(self._spots) > SPOT_CACHE_SIZE:
            self._spots.popitem(last=False)
        return result

    def decide(self, frame: Union[str, object], blind_overrides=None) -> Decision:
        started = time.perf_counter()
        blinds = validate_blinds(blind_overrides)
        t = time.perf_counter()
        if isinstance(frame, str):
            state = json.loads(frame)
        else:
            try:
                if self.reader_mode == 'ai':
                    if self.ai_reader is None:
                        raise VisionStateError('Set GEMINI_API_KEY in .env for AI reading')
                    state = self.ai_reader.read(frame, blinds=blinds) if blinds else self.ai_reader.read(frame)
                else:
                    state = decode_table_image(frame)
            except NotHeroTurn:
                raise
            except ValueError:
                if self.ai_reader is None or self.reader_mode in ('local', 'ai'):
                    raise
                state = self.ai_reader.read(frame, blinds=blinds) if blinds else self.ai_reader.read(frame)
        state = dict(state)
        state = validate_betting_context(state)
        reported_big_blind = state.get('big_blind')
        if blinds:
            state.update(blinds)
        state = align_chip_units(state, reported_big_blind=reported_big_blind)
        state['blind_source'] = 'manual' if blinds else 'screen/state'
        # In NLHE the minimum bet/raise increment is at least one big blind;
        # a larger observed increment after a raise must be preserved.
        if state.get('big_blind', 0) > 0:
            state['min_raise'] = max(state['min_raise'], state['big_blind'])
        stage = {"perception_ms": (time.perf_counter() - t) * 1000}
        t = time.perf_counter()
        normalized, scale = policy_units(state)
        normalized = enrich_context(normalized)
        # All diagnostic monetary features are expressed in BB so the UI never
        # exposes normalized cents as if they were displayed dollars.
        state['strategy_context'] = normalized['strategy_context']
        state['preflop_action'] = normalized.get('preflop_action', 'unknown')
        state['money_precision'] = 2 if scale == 100 else 0
        hand_equity, choice = self._evaluate(normalized)
        stage["policy_ms"] = (time.perf_counter() - t) * 1000
        amount = display_units(choice.amount, scale)
        return Decision(choice.action, amount, hand_equity, choice.bluff_probability,
                        (time.perf_counter() - started) * 1000, stage, state, choice.reason)

    def decide_from_screen(self, region) -> Decision:
        """Capture one explicitly bounded simulator frame and decide from it."""
        from .screen_capture import ScreenCapture
        started = time.perf_counter()
        image = ScreenCapture(region).grab()
        decision = self.decide(image)
        decision.stages_ms["screen_capture_ms"] = (time.perf_counter() - started) * 1000
        decision.latency_ms = (time.perf_counter() - started) * 1000
        return decision

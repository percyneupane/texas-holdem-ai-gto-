from dataclasses import dataclass, field
from statistics import mean, median
from typing import Dict, List


@dataclass
class Metrics:
    hands: int = 0
    hero_profit: int = 0
    showdowns: int = 0
    decisions: int = 0
    folds: int = 0
    bluffs: int = 0
    invalid_actions: int = 0
    latencies: List[float] = field(default_factory=list)
    equities: List[float] = field(default_factory=list)
    action_counts: Dict[str, int] = field(default_factory=dict)
    frames: int = 0
    waiting_frames: int = 0
    read_errors: int = 0

    def as_dict(self) -> Dict:
        """Machine-readable aggregate metrics for one table/run."""
        ordered = sorted(self.latencies)
        p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] if ordered else 0.0
        return {
            "hands": self.hands,
            "profit": self.hero_profit,
            "showdowns": self.showdowns,
            "showdown_rate": self.showdowns / self.hands if self.hands else 0.0,
            "decisions": self.decisions,
            "median_latency_ms": median(ordered) if ordered else 0.0,
            "p95_latency_ms": p95,
            "mean_equity": mean(self.equities) if self.equities else 0.0,
            "fold_rate": self.folds / self.decisions if self.decisions else 0.0,
            "bluff_rate": self.bluffs / self.decisions if self.decisions else 0.0,
            "invalid_actions": self.invalid_actions,
            "action_counts": dict(self.action_counts),
            "frames": self.frames,
            "waiting_frames": self.waiting_frames,
            "read_errors": self.read_errors,
        }

    def record_frame(self, kind: str) -> None:
        """Record a monitor attempt: decision, waiting, or unreadable frame."""
        self.frames += 1
        if kind == "waiting":
            self.waiting_frames += 1
        elif kind == "error":
            self.read_errors += 1

    def record_hand(self, profit: int, showdown: bool) -> None:
        self.hands += 1
        self.hero_profit += profit
        self.showdowns += int(showdown)

    def record(self, decision, valid: bool = True) -> None:
        self.record_frame("decision")
        self.decisions += 1
        self.latencies.append(decision.latency_ms)
        self.equities.append(decision.equity)
        self.folds += int(decision.action == "fold")
        self.bluffs += int(decision.action == "bet" and decision.equity < 0.5)
        self.invalid_actions += int(not valid)
        self.action_counts[decision.action] = self.action_counts.get(decision.action, 0) + 1

    def summary(self) -> str:
        if not self.latencies:
            return "No decisions recorded"
        ordered = sorted(self.latencies)
        p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
        return (f"hands={self.hands} profit={self.hero_profit} showdowns={self.showdowns} "
                f"decisions={self.decisions} median_ms={median(ordered):.2f} "
                f"p95_ms={p95:.2f} mean_equity={mean(self.equities):.3f} "
                f"fold_rate={self.folds / self.decisions:.3f} "
                f"bluff_rate={self.bluffs / self.decisions:.3f} "
                f"invalid_actions={self.invalid_actions}")

"""Reproducible benchmark configuration and result serialization."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List

from .metrics import Metrics
from .simulator import play_hand
from .agent import VisionPolicyAgent
import random


@dataclass(frozen=True)
class TrialConfig:
    players: int = 10
    hands: int = 1000
    seed: int = 7
    label: str = "vision-policy"
    policy: str = "gto"


@dataclass
class TrialResult:
    config: TrialConfig
    hands: int
    profit: int
    showdowns: int
    decisions: int
    median_latency_ms: float
    p95_latency_ms: float
    mean_equity: float
    bluff_rate: float
    invalid_actions: int

    def as_dict(self) -> Dict:
        result = asdict(self)
        return {
            "config": result.pop("config"),
            "metrics": result,
        }


def run_trial(config: TrialConfig) -> TrialResult:
    if not 2 <= config.players <= 10:
        raise ValueError("players must be between 2 and 10")
    rng = random.Random(config.seed)
    metrics = Metrics()
    agent = VisionPolicyAgent(config.seed + 1, policy=config.policy)
    for hand_number in range(config.hands):
        play_hand(config.players, hand_number % config.players, rng, agent, metrics)
    ordered = sorted(metrics.latencies)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] if ordered else 0.0
    return TrialResult(config, metrics.hands, metrics.hero_profit, metrics.showdowns,
                       metrics.decisions, median(ordered) if ordered else 0.0,
                       p95, (mean(metrics.equities) if metrics.equities else 0.0),
                       metrics.bluffs / metrics.decisions if metrics.decisions else 0.0,
                       metrics.invalid_actions)


def run_sweep(configs: Iterable[TrialConfig], output: Path) -> List[TrialResult]:
    results = [run_trial(config) for config in configs]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(asdict(result), sort_keys=True) for result in results) + "\n")
    return results


def table_sweep(players: Iterable[int], hands: int = 1000, seed: int = 7,
                policy: str = "gto") -> List[TrialResult]:
    """Run the same seeded benchmark independently for each table size.

    The seed is offset by table size so a result is reproducible while each
    table has its own independent stream of deals and policy decisions.
    """
    return [run_trial(TrialConfig(players=count, hands=hands, seed=seed + count,
                                   label="%d-max" % count, policy=policy))
            for count in players]


def aggregate_human_results(path: Path) -> Dict:
    """Aggregate completed browser sessions without discarding raw records."""
    sessions = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    responses = [record["response_ms"] for session in sessions for record in session["records"]]
    ai_latencies = [record["ai_latency_ms"] for session in sessions for record in session["records"]]
    agreements = [record["agreement"] for session in sessions for record in session["records"]]
    ev_deltas = [record["ev_delta_ai_minus_human"] for session in sessions for record in session["records"]]
    return {
        "sessions": len(sessions),
        "trials": len(agreements),
        "human_median_response_ms": median(responses) if responses else 0.0,
        "ai_median_latency_ms": median(ai_latencies) if ai_latencies else 0.0,
        "agreement_rate": sum(agreements) / len(agreements) if agreements else 0.0,
        "mean_ev_delta_ai_minus_human": sum(ev_deltas) / len(ev_deltas) if ev_deltas else 0.0,
    }

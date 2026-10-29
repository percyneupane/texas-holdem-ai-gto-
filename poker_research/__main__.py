import argparse
import json

from .policy import POLICIES
from .evaluation import table_sweep
from .simulator import run_experiment


def parse_players(value: str):
    if value.lower() in {"all", "2-10"}:
        return list(range(2, 11))
    try:
        values = sorted({int(item) for item in value.split(",")})
    except ValueError as error:
        raise argparse.ArgumentTypeError("players must be a number, comma-separated numbers, or all") from error
    if not values or any(not 2 <= item <= 10 for item in values):
        raise argparse.ArgumentTypeError("players must be between 2 and 10")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the controlled poker research benchmark")
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--players", type=parse_players, default=[10],
                        help="table size, comma-separated sizes, or all (2-10)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy", choices=sorted(POLICIES), default="gto")
    args = parser.parse_args()
    if len(args.players) == 1:
        print(run_experiment(args.hands, args.players[0], args.seed, args.policy))
        return
    results = table_sweep(args.players, args.hands, args.seed, args.policy)
    for result in results:
        record = result.as_dict()
        record["table"] = result.config.label
        print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()

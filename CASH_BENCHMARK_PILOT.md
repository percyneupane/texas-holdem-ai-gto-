# Cash-game strategy pilot

Objective: maximize expected net big blinds per 100 hands after rake.
Profitable-hand fraction is a secondary diagnostic, not the optimization target.

Reproduce:

```bash
.venv/bin/python -m poker_research.cash_benchmark --players 6 --hands 120 --equity-samples 80
```

Configuration: six-max NLHE, 100bb starting stacks, reset every hand, seed
41001, rotating dealer, paired shuffled deals. Illustrative rake: 5%, cap
3bb, no flop/no drop, excluding uncalled returns. Observations come from
structured visible state; screenshot accuracy and API latency are excluded.
The `gto` baseline is the project's old heuristic, not a solved GTO strategy.

| Synthetic opponents | Contextual minus baseline, net bb/100 | Approximate 95% interval |
| --- | ---: | ---: |
| Balanced | +218.43 | -7.05 to +443.90 |
| Tight | +33.55 | -26.98 to +94.07 |
| Calling station | -854.20 | -1767.18 to +58.78 |
| Aggressive | -191.20 | -970.18 to +587.77 |

Every paired interval includes zero. Neither policy is established as superior.
There were zero invalid actions in all eight policy/opponent combinations.
Large rates and wide intervals reflect small samples and weak synthetic
opponents; they are not plausible forecasts for real cash games. Intervals
are normal approximations, per opponent style, with no multiple-test adjustment.

Next evaluation requirements:

- Confirm the target table size, stack depth and actual rake structure.
- Use the pilot seed only for development; reserve separate seeds for validation.
- Evaluate proposed opponent adaptations against every style, not just a favored bot.
- Increase sample sizes and inspect uncertainty before selecting a strategy.
- Validate recognition separately before testing the complete screenshot pipeline.

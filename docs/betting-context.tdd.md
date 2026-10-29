# Betting-context policy update

User journeys were derived from the request to account for raises, 3-bets,
4-bets and other available cash-game decision variables. No external plan.

## Evidence

- RED: `.venv/bin/python -m unittest discover -s tests -p test_betting_context.py`
  initially executed 14 tests and failed with 10 failures / 5 errors (including
  subtests). Missing implementation included the public-history module,
  nested currency normalization, sizing context and squeeze detection.
- GREEN: the same target passed all initial 14 tests after implementation.
- Additional RED/GREEN cycle: history seat aliases, future-street rejection
  and overlay context formatting failed before the corresponding fixes.
- Additional RED/GREEN cycle: the river-raise regression failed because the
  river branch removed all pressure margins, then passed after distinguishing
  future-street realization costs from unmodeled range-pressure margins.
- Final ordinary suite: `.venv/bin/python -m unittest discover -s tests`:
  **141 tests passed**. The existing HTTPError mock emits a ResourceWarning;
  it does not fail the suite.
- `.venv/bin/python -m compileall -q poker_research tests`: passed.
- Offline legality smoke: `compare(CashConfig(hands=100, equity_samples=20),
  policies=('contextual',), styles=('balanced',))`: completed 100 hands,
  **zero invalid hero actions**. This was not a paid AI-opponent run and does
  not establish profitability or an improvement over the previous strategy.

## Guarantees

| Behavior | Evidence | Type |
| --- | --- | --- |
| Local public actions reach the decision backend without private opponent cards | `test_local_history_reaches_backend_and_detects_squeeze` | Integration |
| Complete history distinguishes squeezes, cold 4-bets and 5-bets | History tests in `test_betting_context.py` | Unit/integration |
| Unseen history and missing stacks are not manufactured | Unknown-context and validation tests | Unit |
| Nested opponent/history money scales with 0.05/0.10 amounts | Nested-currency and end-to-end decimal tests | Unit/integration |
| Position and legal raise bounds affect re-raise sizing | Sizing, disabled-raise and legal-bound tests | Unit |
| Postflop raises/multiway pots affect marginal calls, including river pressure | Postflop and river regression tests | Unit |
| Board/draw features do not count made straights or public-only straight draws | Board-feature tests | Unit |
| Changed history/stacks invalidate cached decisions | Cache regression test | Integration |
| Panel text shows depth/SPR/history quality | Overlay formatter test | Unit |

## Coverage and limitations

The third-party `coverage` package is not installed; line coverage was measured
with the Python standard-library `trace` runner. The full-suite trace command:

```sh
.venv/bin/python -m trace --count --summary --missing \
  --coverdir /private/tmp/poker-context-coverage.hN9FCw \
  --ignore-dir /opt/homebrew:/Users/percy/poker-research/.venv \
  --module unittest discover -s tests
```

Coverage output is a local temporary artifact, not part of the application.
The final trace run passed all 141 tests: betting-context core 99.4%, policy
89.3%, ranges 90.7%, agent 89.2%, screen-reader backend 87.2%, monetary units
94.0% and public table views 95.7% line coverage. These are per-module figures,
not a claim of 80%+ whole-application or branch coverage.
The feature tests include unit and structured-state end-to-end integration;
they do not validate recognition accuracy on an unseen poker site's screenshot
or launch the native Tk panel. UI/screen-capture modules have lower coverage
than the policy core. No solver correctness or real-cash edge is asserted.

The shared Git worktree has an untracked project baseline. No checkpoint
commits or broad staging were performed, to preserve unrelated user work.
This evidence file preserves the RED/GREEN sequence instead.

# Automatic and calibrated screen control: test evidence

User journey: run the existing policy on a visible browser or desktop Hold'em
table, either detecting controls from each screenshot or using a calibrated layout.

The standalone command is `python -m poker_research.screen_play`. The overlay
now also uses screen control by default; `--autoplay-target local` retains its
local-table HTTP controller. Input uses optional PyAutoGUI;
profiles and observations are stored locally. No external poker client was
opened or controlled during development.

## Red and green

The first run of `python -m unittest discover -s tests -p test_screen_play.py -q`
failed all 10 original cases because `poker_research.screen_play` did not exist.
After implementation those 10 passed. Later tests exposed two additional
problems: numeric/card ordering could unlock duplicate actions, and coarse
image comparison missed small button-label changes and wrong capture sizes.
Those tests failed before their fixes and passed afterward.

The calibrated-controller suite has 19 passing tests, covering:

| Guarantee | Test type |
| --- | --- |
| Decimal raise-to adds the current street contribution | Unit |
| Chips-to-add does not add the street contribution | Unit |
| Stale frames, changed action labels and wrong dimensions prevent input | Unit |
| Missing amount context and invalid profiles prevent input | Unit |
| Equivalent numeric/card formats do not bypass duplicate prevention | Unit |
| Waiting or a different public state permits another decision | Unit |
| Unconfirmed actions time out without resend | Unit |
| Partial driver failures leave the action locked | Unit |
| Screen coordinates include the capture offset | Unit |
| Observe mode sends no input | Loop integration with mocked capture/reader/input |
| Read loop discards stale decisions and rereads | Loop integration with mocked capture/reader/input |
| Calibration saves relative positions without clicks or typing | CLI integration with mocked pointer |
| CLI passes the profile and play flag into the loop and handles interruption | CLI integration with mocked driver |

## Generalization

`--auto` uses the primary display and `--bbox` limits capture to a supplied
desktop rectangle. Both request normalized control bounds, enabled states,
control types, game/table count and explicit amount semantics in the existing
vision request. The ordinary overlay reader retains its original schema.
Automatic mode resolves controls on each observation; no site-name adapter
or fixed hero seat is used. Calibrated profiles remain supported.

The automatic tests first failed because `poker_research.table_ui` and the
reader's `locate_controls` option were missing. After implementation the
matrix passed. A later test failed when readback ignored the vision provider's
cooldown; the loop now waits within the action timeout before reading again.

`test_table_ui.py` has 14 tests, including parameterized cases for:

| Guarantee | Evidence |
| --- | --- |
| Map normalized controls to desktop points across 600×432, 1000×720 and 1600×900 captures, for 2–10 active seats | 27 mapping cases |
| Rediscover moved buttons | Changed normalized bounds yield a new target |
| Raise-to/chips-to-add, chips/BB and dot/comma formatting | Four sizing combinations |
| Reject unknown game/table count, disabled/duplicate controls, invalid bounds and labels | Invalid metadata matrix |
| Reject unrepresentable BB amounts rather than round them | Exact-conversion test |
| A typed bet requires readback before submission | Missing/failed readback never submits |
| Readback matches input value, public state, recommendation and semantics | Readback mutation matrix |
| Resolve the fresh submit location after editing | Relocated readback button |
| Recheck screenshot freshness after readback | Changed verification frame prevents submit |
| Respect provider cooldown | Loop checks ordered typing/readback/submit and cooldown wait |
| Preserve control metadata through decimal-stakes policy evaluation | Actual VisionPolicyAgent with mocked reader |
| Keep automatic control detection enabled for backup models | Environment configuration test |
| CLI automatic mode needs no saved profile | Both --auto and --bbox tested |

These are contract and loop tests with synthetic images/mocked vision output,
not evidence that Gemini recognizes arbitrary client screenshots correctly.
No external client or live betting was used. Sliders without editable fields,
custom keypads, confirmation dialogs, multiple visible tables, other poker
variants and ambiguous field semantics are explicitly unsupported.

## Coverage and regressions

Measured with Python's standard-library trace module:

```bash
PYTHONPATH=tests .venv/bin/python -m trace --count --missing --summary --coverdir /tmp/poker-auto-coverage --module unittest test_screen_play test_table_ui test_ai_screen_reader -q
```

The combined run passed 49 tests. Line coverage: `screen_play` 87.7%,
`table_ui` 94.8%, `table_ui_schema` 100%, `ai_screen_reader` 87.8%.
Native mouse/keyboard
delivery, OS permissions, real screenshot recognition, custom bet widgets and
actual client acceptance are unvalidated. Visual progression is not an action
receipt; the logs contain observations/recommendations, not confirmed profits.

Full regression command: `.venv/bin/python -m unittest discover -s tests -q`.
It ran 223 tests: 210 passed, 12 errored due to missing modules
`poker_research.decision_audit` and `poker_research.audit_backtest`, and one failed
because `test_stakes` expects `RAISE 0.25` while the overlay now formats it as
`Preflop\nRAISE 0.25`. Neither audit modules nor overlay formatting were changed
as part of automatic control detection.

Schema fields were checked against the [Gemini structured-output documentation](https://ai.google.dev/gemini-api/docs/structured-output).
This verifies the documented schema vocabulary, not live provider acceptance
of the expanded schema or screenshot-localization accuracy.

This directory is not a Git checkout, so no TDD checkpoint commits were created.

# Selected-area overlay autoplay

User journey: position the existing yellow capture box over one visible
No-Limit Hold'em table, then turn Auto-play on to play inside that area.

The overlay defaults to `--autoplay-target screen`. The old local research-table
HTTP action path remains available with `--autoplay-target local`. Screen mode
requires the AI reader, Gemini configuration, and the optional input driver.

## Implementation

`SelectedAreaPlayer` keeps captures and mouse/keyboard input on the UI thread.
Model readback runs on the existing executor. Polling respects provider cooldowns
without sleeping the UI thread. Typed bets are submitted only after a matching
readback, fresh-frame check, and unchanged selection/generation check.

Turning Auto-play on starts continuous reading and discards old recommendations.
Stop, turning Auto-play off, moving/resizing the box, and closing the overlay
cancel pending submissions. Cancel preserves the duplicate-action lock. Screen
mode uses full-capture changes to wake the AI reader; it does not require the
local OCR detector to find bottom-row action labels. Read now stays a manual,
recommendation-only read.

## Evidence

The first selected-area tests failed with the intended missing
`poker_research.selected_area_play` implementation. After implementation they
passed. Integration coverage also exercises the next visual turn after a
waiting-state rejection, and verifies stale screens trigger a fresh read.

Validation command:

```bash
PYTHONPATH=tests .venv/bin/python -m unittest test_selected_area_play test_screen_play test_table_ui test_turn_detection test_overlay_box test_ai_screen_reader -q
```

Result: 94 tests passed. Guarantees include selected-area coordinate mapping,
nonblocking bet readback, mismatched amount rejection, cancellation on Stop or
selection/generation changes, no duplicate action after cancellation, button
routing, discarding old recommendations, and waking after a waiting frame changes.

Coverage command:

```bash
PYTHONPATH=tests .venv/bin/python -m trace --count --missing --summary --coverdir /tmp/poker-area-coverage --module unittest test_selected_area_play test_screen_play test_table_ui test_turn_detection test_overlay_box test_ai_screen_reader -q
```

Line coverage: `selected_area_play` 87.8%, `screen_play` 88.2%, `table_ui` 95.0%.

The full suite ran 237 tests before the final added stale-frame regression:
224 passed; the same 12 missing audit-module errors and one unrelated action
formatting expectation failure remained. CLI help confirms the new target flag.

These tests mock native input and model output. No external client was opened,
and no actual betting or OS permission delivery was validated. Unknown controls
and unsupported widgets still stop autoplay. The selected rectangle does not
follow a window moved elsewhere on the desktop. No Git checkout is present,
so there are no checkpoint commits.

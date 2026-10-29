# Poker Research and Vision Policy

A No-Limit Hold’em research simulator, screenshot reader, decision overlay and
automatic screen controller.
The goal is to evaluate long-run **net profit in big blinds per 100 hands
(bb/100)**, rather than the percentage of hands won.

The screen controller detects enabled controls from screenshots and can play
one visible table without saved button coordinates. It supports varied layouts,
chip/BB bet fields and dot/comma decimal formats. **Compatibility with every
table is not verified.** Unknown rules, ambiguous controls and unsupported
widgets prevent automatic input. See [screen control](#screen-control-for-other-table-layouts).

The overlay displays recommendations in a separate panel. Position its yellow
box around your table and turn **Auto-play** on to play inside that area.
Moving the box or pressing **Stop** cancels pending input and turns Auto-play
off. The practice table also has its own **Auto-play my seat** mode. Run one
controller at a time.

## Quick start

Run commands from the project folder so the application finds `.env`:

```bash
cd /Users/percy/poker-research
```

For **autoplay inside the area you point at**, complete Python and Gemini setup below,
install the input driver, and grant Screen Recording and Accessibility access
to the application running Python. Keep one No-Limit Hold'em table visible
on the primary display:

```bash
.venv/bin/python -m pip install -r requirements-screen.txt
.venv/bin/python -m poker_research.overlay --reader ai --policy contextual
```

1. Position and resize the yellow box around the table, including cards,
   stacks, blinds, pot and enabled action buttons.
2. Click **Auto-play · OFF** to turn it on. Reading starts automatically, and
   each fresh decision is played inside that selected area.
3. Click **Auto-play · ON** or **Stop** to pause. Moving/resizing the box also
   turns Auto-play off; enable it again after positioning the box.

Restart an already running overlay after updating the project. Screen Auto-play
uses the AI reader; `Read now` remains a recommendation-only single read.
No local research-table server or saved button profile is needed.

For **screen control without the overlay panel**, inspect recommendations first:

```bash
.venv/bin/python -m poker_research.screen_play --auto
```

Then enable mouse and keyboard input:

```bash
.venv/bin/python -m poker_research.screen_play --auto --play
```

`--auto` captures the primary display. To capture only one table, use a desktop
rectangle instead:

```bash
.venv/bin/python -m poker_research.screen_play --bbox 100,150,1100,870 --play
```

The rectangle is `left,top,right,bottom` in desktop points. Include the cards,
pot, stacks, blinds and action controls. Screen control uses Gemini and can
incur API charges; bets and raises require an extra read to verify typed amounts.
Stop with Ctrl-C. A pointer in a screen corner triggers the input driver's
fail-safe before its next input.

For **recommendations in the overlay**:

```bash
.venv/bin/python -m poker_research.overlay --reader ai --policy contextual
```

For **autonomous play on the local practice table**, without screenshots or a
Gemini key:

```bash
.venv/bin/python -m poker_research.table_display --players 6 --autoplay --policy contextual
```

Open <http://127.0.0.1:8780/> to watch. Uncheck **Auto-play my seat** to take over.
Restart the running application after changing code, `.env`, models or settings.

## 1. Set up Python

The overlay needs Pillow and Python with Tk **8.6 or newer**. On macOS,
the system `/usr/bin/python3` may use an older Tk; use the project's `.venv`.

If `.venv` already works, skip installation. Otherwise, using Homebrew:

```bash
brew install python-tk@3.14
python3.14 -m venv .venv
.venv/bin/python -m pip install pillow
```

Verify:

```bash
.venv/bin/python -c 'import tkinter; from PIL import Image; print("Tk:", tkinter.TkVersion)'
```

On macOS, grant Screen Recording permission to the application running Python
under **System Settings → Privacy & Security → Screen Recording**; the setting
name varies by macOS version. Restart the application after changing access.

## 2. Add your Gemini key

Create a key in [Google AI Studio](https://aistudio.google.com/app/apikey).
If `.env` does not exist, copy `.env.example` to `.env`. Preserve an existing
key rather than overwriting the file.

Edit `.env`:

```dotenv
GEMINI_API_KEY=your_actual_key_here
GEMINI_MODEL=gemini-3.8-flash
GEMINI_BACKUP_MODELS=gemini-3.5-flash-lite,gemini-3.6-flash
```

The app loads `.env` from the current working directory. Existing shell
environment variables take precedence. The file is excluded from Git;
do not share it or paste the key into chat.

All three models use the same key. An empty `GEMINI_BACKUP_MODELS=` disables
backups. Model availability, quota and billing depend on your Google project.
AI reading sends the selected screenshots to Google's Gemini API and can
incur charges. Local CV mode does not send screenshots to Gemini.

## 3. Use Vision Policy

Launch:

```bash
.venv/bin/python -m poker_research.overlay --reader ai --policy contextual
```

1. Open the table and keep it visible.
2. Move the yellow capture box using its tab. Drag its edges/corners to resize.
3. Include the board, hero cards, opponent seats, pot, hero stack and enabled
   action controls. Include the dealer button and blind labels for position.
4. Alternatively, enter **Left, Top, Right, Bottom** and select **Apply box**.
   These are desktop screen points, not browser-page coordinates.
5. Select **Start** to read. Use **Stop** to pause.
6. Verify the displayed cards, pot, call amount and player count against
   the table before interpreting the recommendation. Scroll the panel if needed.
7. Turn **Auto-play** on to act inside the yellow box. This starts continuous
   reading if it was stopped. Stop, turning Auto-play off, moving the box or
   closing the overlay cancels any pending bet submission. Keep the target
   window visible and inside the box; the selection does not follow a moved window.

Example: a 1000×720 table with actual desktop top-left `(100,150)`:

```bash
.venv/bin/python -m poker_research.overlay \
  --reader ai --policy contextual --bbox 100,150,1100,870 --interval-ms 250
```

The default box `0,0,1000,720` is a starting rectangle, not automatic table
detection. Browser page coordinate `(0,0)` need not be desktop `(0,0)`:
window position, browser controls and the menu bar add offsets.

Recommended amounts mean **chips to add**, including the call portion of a
raise. They are not necessarily another interface's “raise to” street total.

### Set the small blind and big blind

In the **Table blinds** card, enter **Small blind** and **Big blind**, then
select **Apply blinds**. For your 0.05/0.10 table, enter `0.05` and `0.10`.
For a whole-chip 5/10 table, enter `5` and `10`. Leave both
fields blank and apply to return to screen-detected values.

Manual settings take precedence over detected blinds and are sent to the AI
reader as table context. They also reach the decision backend, which uses
the big blind for stack depth, preflop opening sizes and the minimum raise
increment. A larger observed raise increment is preserved. Applying different
blinds invalidates any pending recommendation from the previous settings.

The panel shows the effective SB/BB, whether they came from manual settings,
and stack, pot and call size in big blinds. A raise shows both **chips to add**
and the **raise-to total**, provided the current street contribution was read.
For example, at 0.05/0.10, a 10.00 stack is 100bb. An SB who has already
posted 0.05 and opens to 0.30 must add 0.25. The panel shows both amounts.

Use the table's displayed units for every value: if the blinds are 0.05/0.10,
read a 10.00 stack as 10.00, not 1000. Amounts support up to two decimal places.
The backend converts all amounts together to integer cents for calculation,
then converts recommendations back to the original units. Whole-chip
simulator tables retain their existing one-chip rounding. Changing blind
settings does not rescale a stack that is already a plausible depth in those
blinds. A stack copied from a "100 BB" label, or off by a factor of 100 or
1000, is converted so the stack, pot and call stats match the blinds.
Settings currently last for the running session; re-enter them after restarting
or clear them when moving to a table with different stakes.

### Reading modes

| Option | Behavior | Gemini key |
| --- | --- | --- |
| `--reader ai` (default) | Read with Gemini; supports unfamiliar layouts, without guaranteed recognition of every theme. | Required |
| `--reader local` | Fixed-layout CV for this simulator's art and geometry. | Not required |
| `--reader hybrid` | Local CV first, Gemini on rejection; waits when CV identifies a non-hero turn. | Required for fallback |

Gemini returns compact JSON data. The local backend calculates equity and
chooses the action. Unreadable required fields prevent a decision.
Amounts support up to two decimal places, including 0.05/0.10 blinds.

### Backups and speed

Timeouts, rate limits, service errors or unavailable models trigger backups.
The most recently successful model is preferred on later frames and shown
in the overlay. Authentication/request-configuration errors pause reading.
An unclear table does not trigger extra model calls seeking a different answer.

One background task runs at a time. Each frame has at most three model
attempts and a seven-second fallback budget. The first attempt may use six
seconds. A backup starts only when at least two seconds remain. Models retain
individual cooldowns. A timed-out read retries in half a second. Other network
failures back off for 2–8 seconds. Quota and server errors back off for 10–60
seconds and respect longer server retry delays. A result from an earlier box,
or from before Stop, is discarded.

The overlay keeps taking local screenshots. In AI mode, macOS Vision recognizes
the Fold/Call/Check action labels locally before starting a Gemini read. This
does not depend on button colour or the simulator's exact button position.
Include the whole table and its controls in the box. The local check looks for
an aligned row of action labels in the lower half; it is not universal turn
recognition. Preselection labels such as Check/Fold and sitting-out controls do
not start reads. Local/hybrid modes retain the simulator's colour check.

The macOS text helper uses Apple's Vision framework and compiles once per launch
with `swiftc` (provided by Xcode or Command Line Tools). Compilation and turn
checks run in the background. If the helper is unavailable, the overlay shows
an error. **Read now** requests a single screenshot read without the local gate;
it does not auto-play that result. Automatic reading holds the recommendation
until controls disappear and return or their action labels/amounts change.
Clock ticks and opponent animations do not trigger AI reads. Results arriving
after the detected controls change are discarded. Failed or timed-out calls
can still try up to two backup models.

`--interval-ms 250` controls polling, **not four AI requests per second**.
Cloud latency varies; backup models are not a speed guarantee.

## 4. Choose a strategy

| Option | Behavior |
| --- | --- |
| `--policy contextual` (overlay default) | Position charts, public betting sequences through 5-bets+, sizing/depth-aware defenses, range equity and postflop pressure adjustments. |
| `--policy gto` | Original equity/pot-odds heuristic; despite its name, not a solved GTO strategy. |
| `--policy baseline` | Original fixed-threshold comparison policy. |

Charts are editable in [ranges.py](poker_research/ranges.py).
Preflop actions come from published 100bb cash charts: PokerCoaching's
6-max raise-first-in lists, and HoldemPro's facing-an-open 3-bet/call/fold
lists. A mixed hand takes the action published at 50% or more. 4-bet and
5-bet lines use those charts' pure continue ranges. The sampled showdown
equity is displayed and still drives postflop decisions; it does not change
a charted preflop action. Charts apply at about 30–200bb. When a live stack
is missing, hero depth is used and labeled as an estimate. Full-ring early
seats use a tighter subset. This is not a solver and it does not establish
a profitability improvement.

The contextual backend now considers:

- Original dealt positions, IP/OOP status, live player count and posted blinds.
- Opens (2-bets), 3-bets, 4-bets, 5-bets+, limpers and callers since the last
  raise. Complete public history identifies squeezes and cold 4-bets.
- The hero's earlier action and separate opponent range priors for opening,
  limping, calling, re-raising, calling a 3-bet and calling a 4-bet.
- Actual raise-to size, unusually large opens, legal raise bounds and whether
  raising is available. Normal 3-bet sizing starts at 3× the open IP and 4× OOP;
  4-bet sizing starts at 2.2× IP and 2.5× OOP, adjusted for intervening callers
  and capped by known effective stacks/legal bounds.
- Public opponent stacks and street contributions, effective depth and SPR
  (remaining effective stack / current pot). Multiway depth is the largest live
  effective matchup, not a full side-pot-specific equity model.
- Pot odds, multiway pressure, postflop bets versus raises, repeated aggression
  on multiple streets and observed overbet sizing.
- Board suits, paired/connected boards, private-card flush/straight draws and
  made-hand category. Texture changes value sizing; selected heads-up draws
  can semi-bluff, while automatic multiway air bluffs are disabled.

The panel shows the observed preflop pot type, effective depth, SPR, history
completeness and a decision reason. Restart the overlay/table after updating.

Structured local-table observations include the exact public action sequence.
For screenshots, include the site's visible hand-history panel inside the
capture box if you want sequence-based context. A screenshot without visible
history cannot establish earlier raises: the reader must leave it incomplete.
The app does **not** yet stitch history across screenshots or guess actions
from a single badge. Missing stacks remain unknown, not zero.

Structured inputs optionally accept `history_complete` and `betting_history`:

```json
{"history_complete": true, "betting_history": [
  {"street": "preflop", "position": "LJ", "action": "raise", "amount": 0.25, "raise_to": 0.25, "pot_before": 0.15},
  {"street": "preflop", "position": "HJ", "action": "call", "amount": 0.25, "raise_to": 0, "pot_before": 0.40},
  {"street": "preflop", "position": "CO", "action": "raise", "amount": 1.10, "raise_to": 1.10, "pot_before": 0.65}
]}
```

This is a context fragment, not a complete decision request. Amounts, opponent
`stack`/`bet` and top-level money must all use the same displayed units.
`amount` means chips added; `raise_to` is the street total. Do not mark partial
history complete. The backend converts nested amounts to cents together with
the rest of a 0.05/0.10 table; the added context requires no extra model call.

Unknown opponent roles retain random-hand priors. Missing position, blinds,
or a stack outside 30–200bb skips the chart and uses the equity fallback.
Open, limp, and cold-call priors use that seat's opening chart. 3-bet, 4-bet,
and 5-bet priors use the chart's re-raise ranges. There is no learned opponent
tendency model, tournament ICM, or full postflop range updating. Postflop
pressure changes call thresholds, not a Bayesian opponent range. Rake is
modeled in the cash benchmark, not the live decision policy. AQ and AJ still
fold to a 4-bet or an oversized open when the chart says fold. Validate on
held-out hands before relying on any strategy; a short synthetic run cannot
demonstrate a real-game edge.

## 5. Run the local practice table

In another terminal:

```bash
cd /Users/percy/poker-research
.venv/bin/python -m poker_research.table_display --port 8780 --players 6
```

Open [Research Table](http://127.0.0.1:8780/).
It supports 2–10 seats, blinds, four betting streets, all-ins, side pots and
showdowns.

To have the model play on its own from the first hand:

```bash
.venv/bin/python -m poker_research.table_display --players 6 --autoplay --policy contextual
```

It makes every hero decision and continues dealing new hands automatically,
using public table state and its own cards. No overlay, screen capture or API
key is needed with the default bot opponents. Watch at
<http://127.0.0.1:8780/> and uncheck **Auto-play my seat** to take over;
check it again to resume. Stop the server with Ctrl-C. Decisions and results
are saved in `sessions/` and shown on the session page. `--policy` also accepts
`gto` and `baseline`. Keep overlay Auto-play off when using this mode.

- Click Fold, Call/Check or Raise To to play manually.
- Size raises with the slider, +/- controls or pot-fraction buttons.
- Players +/- changes the seat count for the next hand.
- **Auto-play my seat** lets the simulator's internal policy act from the
  engine state. Leave it unchecked while the overlay Auto-play button is on.
  The overlay normally clicks inside its selected box. To use the simulator's
  HTTP action API instead, launch the overlay with `--autoplay-target local`.
- After 25 seconds without an action, the simulator checks or folds your hand.

Restart the table after a code change so `POST /act` is available. In the
overlay, Auto-play stays off until you click it. For the optional local API mode:

```bash
.venv/bin/python -m poker_research.overlay --reader ai --autoplay-target local
```

In that mode, `--table-url` must be
`http://127.0.0.1`, `http://localhost`, or `http://[::1]`.

Every finished research-table hand is stored for the session: hero cards,
board, position, the full action log, and the hero's chip result in chips,
big blinds, and bb/100. When the model chose the action, the hand also keeps
the equity and the reason. The table shows a running total. Open
[the session page](http://127.0.0.1:8780/session) for each finished hand:
cards, actions, equity, reason, and the chip result. The same history is in
`sessions/` and at [session.json](http://127.0.0.1:8780/session.json).
Overlay reads are recorded there too, one hand per pair of hole cards, with
each decision and reason. Those reads are not added to the chip total. The
numbers describe this session. They are not a real-money profit result.

The simulator image is 1000×720 at 100% browser zoom. For local CV, keep that
scale and place the capture box around its actual desktop bounds:

```bash
.venv/bin/python -m poker_research.overlay --reader local --policy contextual
```

Local CV does not extract the new position/history fields, so contextual
decisions fall back where context is unavailable. AI reading or structured
simulator observations are needed to exercise those branches.

### AI opponents with simulated personalities

To replace the table's rule-based opponents with Gemini role-play players:

```bash
.venv/bin/python -m poker_research.table_display \
  --players 6 --port 8780 --opponents ai --ai-max-calls 100
```

Set `AI_OPPONENT_MODEL=gemini-3.5-flash-lite` in `.env` to override the model.
This uses the existing Gemini key and makes paid text requests, one per
opponent decision. Each seat has a persistent persona and simulated tilt,
confidence and loss streak that respond to its own hand results. Each model
sees only its own hole cards, public cards, betting history, stacks and legal
actions; it is not given other players' hidden cards or the remaining deck.

AI role-play is still synthetic and has not been validated against real human
behavior. An AI-generated personality is not proof of realistic emotions or
strong poker play. Responses can be nondeterministic even with a fixed deal seed.

Requests run outside the table lock so the display remains responsive. Budget,
network or illegal-action errors pause the table and appear in the hand log;
the app never silently substitutes rule-based opponents into an AI session.
The request cap includes failed requests and is shared across all opponents.

Run an isolated 100-hand session against AI opponents:

```bash
.venv/bin/python -m poker_research.ai_backtest \
  --hands 100 --players 6 --seed 7 --max-ai-calls 3000
```

The 3,000-call limit is a maximum, not an estimate of cost or required usage.
The report includes completed versus requested hands, API calls/token usage,
personality states and per-hand outcomes. If a session stops early, it is
marked incomplete. Hypothetical rake is reported separately; it is not deducted
from the table's carrying stacks. This tests the backend rather than screenshot
recognition, and it does not demonstrate performance against real people.

### Screen control for other table layouts

The separate screen controller can locate enabled controls on a visible No-Limit
Hold'em table in a browser or desktop app. Automatic mode finds controls from
each screenshot rather than requiring saved button coordinates or a fixed hero
seat. This is experimental:
no external poker client has been validated yet, and compatibility with every
site, custom input or layout is not established.

Install its optional input driver in your Python environment:

```bash
.venv/bin/python -m pip install -r requirements-screen.txt
```

On macOS, the application running Python needs both Screen Recording and
Accessibility access under System Settings → Privacy & Security.

Keep one table visible on the primary display. First inspect recommendations
without sending mouse or keyboard input:

```bash
.venv/bin/python -m poker_research.screen_play --auto
```

Enable automatic controls with:

```bash
.venv/bin/python -m poker_research.screen_play --auto --play
```

To limit capture to one table, replace `--auto` with, for example,
`--bbox 100,150,1100,870`. Coordinates are desktop points. Control locations
are normalized to that capture, so layouts and scaling do not need individual
button profiles. Window movement within the capture is observed on subsequent
reads; movement during a read invalidates the pending decision.

| Input feature | Automatic mode |
| --- | --- |
| Fold/check/call/bet/raise buttons in different locations | Located anew on each reading |
| Tables with 2–10 seats | Existing policy supports these seat counts |
| Raise-to and chips-to-add fields | Accepted only with explicit visible semantics |
| Chip/currency and BB input | BB conversion requires known blinds and exact two-decimal representation |
| Dot and comma decimal inputs | Uses the detected input format |
| Typed bets | Reads back the entered number and rechecks the screen before submitting |
| Multiple tables, unknown rules/ownership or ambiguous controls | Refused |
| Sliders without editable fields, custom keypads or extra confirmation dialogs | Unsupported; refused |

Automatic mode requires the AI reader. Each raise/bet requires an additional
vision read after typing. If the input is changed, rounded, unclear, disabled,
or the decision changes, it stops before submitting. A provider cooldown is
respected within the action timeout. These checks depend on the reader's
accuracy; matching AI readings are not a guarantee of correctness.

Manual calibration remains available for a known layout with chip-denominated,
dot-decimal inputs. Keep its window size and browser zoom fixed. Create a profile:

```bash
.venv/bin/python -m poker_research.screen_play --calibrate my-table.json
```

Enter the table's desktop box as `left,top,right,bottom`. Choose `raise_to` if
the editable bet field takes the final street total, or `chips_to_add` if it
takes additional chips. For each control, press Enter and move the pointer to
its center during the four-second countdown; type `s` to skip an unavailable
control. Check and call can share a location, as can bet and raise. Capture
each requested location separately. The amount control must be a standard
editable numeric field that accepts select-all and decimal typing. For
BB-denominated fields and locale comma decimals, use automatic mode instead.
Sliders and extra confirmation dialogs need a specific adapter.
Calibration only reads pointer positions; it sends
no mouse clicks or keystrokes to the table. Existing profiles are not overwritten.

First verify screenshot readings and recommendations in observe-only mode:

```bash
.venv/bin/python -m poker_research.screen_play --profile my-table.json
```

Then enable mouse and keyboard control:

```bash
.venv/bin/python -m poker_research.screen_play --profile my-table.json --play
```

The default reader uses Gemini and requires the existing `.env` key. It sends
table screenshots to Gemini and can incur API charges. It re-reads when the
picture changes; it does not require a local research-table server. Use
`--reader local` only with a calibrated profile for the fixed simulator layout. Keep other
autoplay controllers off while this one runs.

Before clicking, it checks a fresh screenshot against the decision frame,
including the target control. A changed picture discards the decision. A
submitted public decision state is locked to prevent a second click; the
reader must observe a waiting turn or a different public state before another
action. If no progression is observed within `--ack-timeout` seconds (default
15), the process stops without resending. This is visual evidence of progression,
not a transaction receipt or a guarantee that the client accepted the amount.
Click/typing errors stop the process. Reads and recommendations go into
`sessions/screen-*.jsonl`; they are not confirmed action receipts or profit records.

Stop with Ctrl-C in the terminal. Moving the pointer to a screen corner triggers
the driver's fail-safe before its next input. Moving/resizing the table,
changing zoom or opening a covering window invalidates a pending decision;
manual profiles also need recalibration afterward. This mode does not join tables, log in,
rebuy, dismiss dialogs or handle other poker variants.

## 6. Compare cash-game strategies

Start with the small pilot:

```bash
.venv/bin/python -m poker_research.cash_benchmark \
  --players 6 --stack-bb 100 --hands 120 --equity-samples 80
```

For a larger experiment:

```bash
.venv/bin/python -m poker_research.cash_benchmark \
  --players 6 --stack-bb 100 --hands 10000 --seed 41001 \
  --rake-rate 0.05 --rake-cap-bb 3 --equity-samples 200
```

These commands use structured visible state, make no screenshot/API requests,
and compare `gto` versus `contextual` on identical deals and rotating
positions. Stacks reset every hand. Four synthetic styles are tested:
balanced, tight, calling station and aggressive. Larger runs can take
substantial time; stop with Ctrl-C.

The JSON report includes gross/net bb/100, rake, profitable-hand fraction,
invalid actions and approximate 95% confidence intervals for profit and
paired strategy differences. Intervals are omitted below 30 hands.
A paired interval containing zero does not establish which strategy is better.

Default rake is an illustrative 5%, capped at 3bb, with no flop/no drop and
no rake on uncalled returns. Set the target game's actual structure.
Reserve different seeds, such as `--seed 81001`, for evaluation after tuning.

The [pilot report](CASH_BENCHMARK_PILOT.md) did **not establish a superior
policy**. Synthetic bots are not a representative player population.
Neither policy has demonstrated real cash-game profitability.

## Other research commands

Fixed-renderer simulations; this CLI defaults to the old `gto` policy:

```bash
.venv/bin/python -m poker_research --hands 1000 --players 6 --policy contextual
.venv/bin/python -m poker_research --hands 1000 --players all
.venv/bin/python -m poker_research --hands 1000 --players 2,6,9
```

Simulator screenshot diagnostics; replace the box with the correct bounds:

```bash
.venv/bin/python -m poker_research.screen_capture \
  --bbox 100,150,1100,870 --output frame.png
.venv/bin/python -m poker_research.screen_capture \
  --bbox 100,150,1100,870 --preview /tmp/capture-preview.png
.venv/bin/python -m poker_research.screen_capture \
  --bbox 100,150,1100,870 --watch-seconds 30 --interval-ms 100
```

`--preview` saves a full-desktop diagnostic with the box outlined.
The single-shot decoder is simulator-specific. Use `--watch` instead of
`--watch-seconds` to continue until Ctrl-C. The capture CLI takes Pillow
capture units; use the overlay for its automatic macOS screen-point handling.

Human decision comparison:

```bash
.venv/bin/python -m poker_research.human_eval --port 8765
```

Open [Human Evaluation](http://127.0.0.1:8765/).
Sessions save response times, actions and policy agreement to
`human-results.jsonl`.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Overlay does not open | Use `.venv/bin/python`; verify Tk ≥ 8.6 and Pillow. |
| Key missing or settings unchanged | Run from the project directory, check `.env` and overriding shell variables, then restart. |
| No table / unclear fields | Keep the table visible; include cards, numbers and action controls. Missing call/minimum-raise amounts can prevent decisions. |
| Position unknown / equity fallback | Include the dealer and occupied seats. One screenshot cannot reliably recover unseen betting history. |
| Waiting for your turn | The reader has not identified enabled hero action controls. |
| Model busy / cooldown | Wait for the retry timer. Billing can raise quotas but does not guarantee faster reads. |
| Key/model error and paused reader | Check the displayed API error, key permissions, model names, quota and billing; restart after fixing. |
| Waiting, buttons unchanged | The model is idle until the action buttons change. Stop, then Start, reads the current screen now. |
| Overlay Auto-play did not move | Restart the overlay after updating. Use `--reader ai`, include one table and its controls in the yellow box, and turn Auto-play on. Check its displayed error and Accessibility permissions. |
| Auto-play turned off after positioning the box | Moving/resizing the selection cancels pending input. Turn Auto-play on again when the box is positioned. |
| Local API Auto-play did not move | With `--autoplay-target local`, restart the local table and leave “Auto-play my seat” unchecked. The table must be at the overlay's `--table-url`. |
| Screen controller observes but never clicks | Add `--play`; without it, the command only displays recommendations. |
| Multiple tables or ambiguous ownership | Use `--bbox` to include exactly one table and its hero controls. |
| Unknown bet input semantics | Include the field's visible labels, blinds and units. Automatic mode stops when it cannot establish raise-to versus chips-to-add or the input format. |
| Entered bet could not be verified | Check that the numeric field accepted the amount. The controller stopped before submitting; custom widgets and extra confirmation dialogs need an adapter. |
| Previous action could not be confirmed | Inspect the table before restarting. No action is automatically resent. `--ack-timeout` defaults to 15 seconds. |
| Mouse or keyboard input does not arrive on macOS | Check Accessibility access for the application running Python, then restart it. Screen Recording access is also required for capture. |

## Run tests

```bash
.venv/bin/python -m unittest discover -s tests -q
```

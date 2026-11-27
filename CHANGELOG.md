# Changelog

Versions follow the development phases: `0.<phase>.<fix>` until 1.0. Dates are
when each phase was finished. Earlier phases had no version bump; their
entries are reconstructed from the commit history.

## 0.9.0: Personal apps (Phase 9, 2026-11-28)

- **Reminders** (`reminders.add`, `list`, `complete`): reminders go to the
  Reminders app, so they sync to your iPhone. "Remind me to …" now goes here
  instead of a Mac-only alarm, and later days work ("on friday at 5").
- **Calendar** (`calendar.events`, `calendar.add`): what's on a day or a week,
  with repeating events expanded from their rules (AppleScript returns them
  only at their first date). A rule Jarvis can't expand is reported, not
  skipped. Adding an event asks first and shows the exact day and times.
- **Contacts** (`contacts.find`) and recipient lookup: names, nicknames and
  the relations on your own card ("mom", "my wife"). When several people or
  numbers match, Jarvis asks which one instead of guessing.
- **Notes** (`notes.search`, `read`, `create`): locked notes are skipped, and
  a new note's text is HTML-escaped.
- **Messages and Mail** (`messages.send`, `mail.draft`, `mail.send`): sending
  always asks, every time, with the person, the exact number or address and
  the text in the confirmation; no setting skips it. A draft opens filled in
  for you to send.
- Every app is driven by fixed AppleScripts through one shared runner. Names,
  titles and text go in argv, never into a script, and each script is
  compiled in the tests without being sent to the app.
- A date and time parser (`jarvis/dates.py`): "friday at 1", "Dec 3rd", "the
  5th", "in 20 minutes", "tomorrow morning", "this week". A time without
  am/pm uses working hours, and replies show the date understood.
- Instant phrasings: "remind me …", "show my reminders", "what's on my
  calendar tomorrow?", "text mom that I'll be late" (still asks first), "make
  a note to buy coffee".
- A personal tool profile, and a smaller plan grammar: string limits above 80
  characters are left to validation, keeping the grammar ~41 KB with 50 tools.
- The planner prompt no longer declines email and messages: a Phase 7
  example ("send an email to Sam" → "I can't") now texts with
  `messages.send`, and the "can't do" example is a phone call.
- `jarvis eval`: 93/107 (87%) with 55 tools (89/101 in 0.8.0); Siri-style
  requests 26/31. 14 real-model tests pass.

## 0.8.0: Everyday commands (Phase 8, 2026-11-26)

- **Apple Shortcuts** (`shortcuts.list`, `shortcuts.run`): run any of your
  shortcuts by name, which covers what macOS has no command for (brightness,
  Focus, Bluetooth, HomeKit scenes). Running asks first unless the shortcut is
  in `[shortcuts] trusted`; a near-miss name gets "Did you mean …?"; output is
  outside content; slow shortcuts are stopped after `timeout_seconds`.
- **System controls** (`system.status`, `volume`, `wifi`, `sleep_display`):
  volume (set, up/down, mute), Wi-Fi, display off, and a status line with the
  battery and dark mode. Built-in commands only, no new permission. Volume
  runs without asking, like Siri; turning Wi-Fi off asks first.
- **Music** (`music.status`, `control`, `play`): play, pause, skip, what's
  playing, a playlist by name, or the first library song matching a search,
  through fixed AppleScripts (names go in argv, never into a script). Needs
  the Automation permission once.
- **Weather** (`weather.forecast`): current conditions and three days from
  Open-Meteo, no account or key; only the place name is sent. Home place and
  units in `[weather]`.
- **Timers, alarms and reminders** (`timers.start`, `alarm`, `list`,
  `cancel`): parsed by Jarvis ("1h30m", "at 7", "in 20 minutes", "7:30 pm
  tomorrow"), kept in their own `timers.sqlite3`, and rung by the service
  (dialog, `[timers] sound`, spoken line) even if they were set from a
  terminal. A timer that came due while the service was stopped rings when it
  starts. `jarvis doctor` shows what's pending.
- **Instant everyday commands**: about 40 phrasings ("turn up the volume",
  "skip this song", "set a timer for 10 minutes for pasta", "remind me to buy
  milk at 6 pm", "weather in Pune") skip the model and run in milliseconds.
- A clarify that only declines ("I can't set alarms. Anything else?") is now a
  reply, so the menu bar and voice don't wait for an answer.
- The prompt keeps `system.*` to sound, Wi-Fi, battery and the display, and
  sends other settings to Shortcuts: brightness had planned `system.volume`.
  A plan that still uses `system.volume` or `system.wifi` for brightness, Do
  Not Disturb, Night Shift or Bluetooth gets a corrective retry, so the wrong
  setting is never changed.
- The default context window is 8192 tokens (was 4096): with 43 tools enabled
  the system prompt is ~2,400 tokens, which left too little room for results.
  Same speed; ~0.6 GB more while the model is loaded.
- The everyday tool profile covers system, music, weather, timers and
  Shortcuts requests.
- `jarvis eval`: 89/101 (88%) with 43 tools, from 80/92 (87%) with 29.
  Siri-style requests: 20/25 use the right tool or decline correctly (8/15
  before). Tool profiles were re-measured and stay off (see
  `docs/BENCHMARKS.md`).

## 0.7.0: Foundations for Siri-style features (Phase 7, 2026-11-25)

- `jarvis service status` and `jarvis doctor` say why the background service
  isn't running: switched off in Login Items & Extensions (read from macOS's
  background-task database with `sfltool dumpbtm`, no admin rights), a moved or
  renamed Jarvis folder, or simply stopped, each with the fix. `status` also
  shows which Python the login item runs. `jarvis service start` refuses,
  with that explanation, instead of overriding your Login Items switch.
- `jarvis eval`: plans 92 fixed test requests with the real model (nothing
  runs) and scores the action, tools and key arguments per category. Baseline
  80/92 (87%); a real-model test fails below 85%. The test set ships with the
  package (`jarvis/evals/planner_cases.toml`).
- Tool profiles (`[ai] tool_profiles`, off by default): a request can see only
  the tools of one of four keyword-chosen profiles (apps, files, web, dev), in
  both the prompt and the grammar. `jarvis eval --profiles on|off` compares
  the two. Same score (87%) and faster planning, but more disk; measurements
  and the decision are in `docs/BENCHMARKS.md`.
- The prompt-cache cap default is 1 GB (was 512 MB): at 512 MB, alternating
  profiles evicted each other and every request took 3-5 s.
- `[ai] backend = "server"`: plan through a local model server with an
  OpenAI-compatible API (llama.cpp's `llama-server`, Ollama, LM Studio), so
  Jarvis and Hermes Agent can share one model. Schema-constrained replies,
  local-only by default (`allow_remote_server`), checked by `jarvis doctor`.
  Tested against a fake server; not yet run against a real `llama-server`.

## 0.6.1: Fixes from real voice use (2026-10-23)

- "open-calculator." (a speech-to-text spelling) takes the instant path: a
  hyphen joining the command verb to the app name is split.
- A near-certain app name ("photo wood" for Photo Booth, "safarii", "font
  look") gets "Did you mean …?" instantly instead of going to the model, which
  had guessed "Photos". Near-certain means a very close spelling, or the same
  words with one identical and the others spelled or sounding alike (Soundex);
  generic phrases like "web browser" still go to the model.
- The log says how each voice attempt ended and what was chosen in the
  dialogs, never the words.
- Real-model test for the spec's "ask for clarification if multiple projects
  match".

## 0.6.0: Packaging and portfolio (Phase 6, 2026-10-23)

- Single-sourced version (`jarvis.__version__`), project URLs and classifiers
  in `pyproject.toml`.
- `constraints-macos.txt`: the exact package versions the release was tested
  with; `JARVIS_TESTED_VERSIONS=1 scripts/setup_macos.sh` installs them.
- `scripts/check_package.sh`: builds the wheel, checks its contents, installs it
  into a fresh virtualenv with a throwaway home folder, and smoke-tests the
  `jarvis` command.
- CI (GitHub Actions): lint, strict types, the test suite and the packaging
  check on Apple Silicon macOS with Python 3.11 and 3.14, and the shared core
  on Linux (615 tests pass there without macOS).
- `jarvis voice benchmark`: speech-to-text accuracy, latency, CPU and peak
  memory on synthesized commands, no microphone; `--voice` and `--gpu`.
- `docs/BENCHMARKS.md`: methodology and all current measurements.
- Workflow commands also accept "work flow", a mishearing the voice benchmark
  found.
- `docs/WINDOWS.md` rewritten as an ordered roadmap (W0–W6) with effort,
  risks and exit criteria, based on an inventory of the POSIX assumptions left
  in shared code.
- README: CI and platform badges, contents, install options (tested versions,
  without local AI, from GitHub with pip) and full uninstall, a threats and
  mitigations table, CI, an up-to-date roadmap, and a licence note (no licence
  chosen yet).
- Verified from a fresh clone of the public repo: setup with the tested
  versions in 9 s, 636 tests passing, lint and strict types clean; and
  `pip install "jarvis-assistant[voice] @ git+https://github.com/mahajanp2804/jarvis"`
  runs from a clean virtualenv.

## Phase 5: Memory and voice (2026-10-22)

- SQLite memory: remembered preferences (given to the model as facts after
  the system prompt), a redacted task history, `jarvis memory
  show|export|clear`. Approvals are never stored; secrets and outside content
  are left out (`6819123`).
- Saved workflows: "save that as …" / "run the … workflow", replayed step by
  step through the full policy pipeline (`8f48bbb`).
- Spoken replies with the built-in macOS voices, off by default (`8c0cb22`).
- Push-to-talk voice input: a worker process per request records, transcribes
  on-device with whisper.cpp and exits; the transcript is shown and editable
  before anything runs (`186c323`, `83486d6`).
- The planner declines requests no tool can do instead of asking which app to
  use (`f1e03ab`); two global shortcuts coexist (`17a88b5`); `jarvis voice
  say --voice` (`c6062bf`).
- Persistent prompt cache: the system prompt's llama.cpp state is saved to
  disk, so the first request after a model load takes ~2.4 s instead of ~7.6 s
  (`f556d29`).

## Phase 4: Computer automation (2026-10-21)

- File and project tools behind a path sandbox; nothing is overwritten or
  permanently deleted (`aee6cea`).
- Terminal tools with an allowlisted, per-call risk policy and no shell
  (`69d7c24`).
- Browser tools driving the installed Chrome through Playwright, with DOM
  element references and hand-off for logins and CAPTCHAs (`e80bf5d`).
- Opt-in screen tools with on-device OCR (`bbaa13f`).
- Multi-step agent loop: plan, execute, observe, with verification, bounded
  retries, stop rules and taint tracking (`c0ed93e`).
- Fixed a llama.cpp crash caused by huge grammars from string length limits
  (`937da55`); accurate benchmark including the model worker (`d753ff5`).

## Phase 3: Background service (2026-10-17)

- Launch Agent with a global shortcut (Carbon, no Accessibility permission),
  native dialogs, a local control socket, graceful shutdown on SIGTERM, and the
  model in a worker process (`a1b3353`, `b86e474`); fixed slow restarts caused
  by launchd throttling (`b89a0b4`).

## Phases 1–2: Foundation and local AI (2026-10-17)

- Validated configuration, redacting logs, permission policy, audit log,
  CLI, application control and tests.
- Local GGUF model through llama-cpp-python (Metal): lazy load, idle unload,
  grammar-constrained and pydantic-validated plans, benchmark (`a1b3353`).

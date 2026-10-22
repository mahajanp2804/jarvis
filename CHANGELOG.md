# Changelog

Versions follow the development phases: `0.<phase>.<fix>` until 1.0. Dates are
when each phase was finished. Earlier phases had no version bump; their
entries are reconstructed from the commit history.

## 0.6.0: Packaging and portfolio (Phase 6, in progress)

- Single-sourced version (`jarvis.__version__`), project URLs and classifiers
  in `pyproject.toml`.
- `constraints-macos.txt`: the exact package versions the release was tested
  with; `JARVIS_TESTED_VERSIONS=1 scripts/setup_macos.sh` installs them.
- `scripts/check_package.sh`: builds the wheel, checks its contents, installs it
  into a fresh virtualenv with a throwaway home folder, and smoke-tests the
  `jarvis` command.

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

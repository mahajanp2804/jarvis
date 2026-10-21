# Jarvis

A personal AI computer assistant that runs **locally** on macOS (Apple Silicon first,
Windows later), with no app to install: it's a Python program that starts at login.
Press **⌃⌥⌘J** anywhere, type what you want, and Jarvis turns it into checked,
permission-gated actions on your computer:

* It runs quietly in the background as a Launch Agent. Idle, it uses ~0.004% CPU
  and ~51 MB, with no polling, no microphone, and no screen capture unless you ask.
* A rule-based fast path handles simple commands ("open Safari") in milliseconds,
  without AI.
* A local GGUF model (Qwen3-4B via `llama-cpp-python`, on the Metal GPU) plans
  everything else in rounds: it chooses tools, sees their results, and continues.
  For example, "open my Stock Price Predictor project and run it" becomes find
  the project → inspect how it runs → start it → check that the server is up.
* It works with **apps, files, code projects, the terminal, the browser** and
  (opt-in) **the screen**. The model can only *propose* tool calls. Every action
  goes through the same validation, sandbox, permission policy, confirmation
  dialog and audit log.

> **Status: Phase 4 of 6 (computer automation).** Voice input/output and
> persistent memory come next. See [Roadmap](#roadmap).

| Phase | Scope | Status |
|---|---|---|
| 1 | Config, logging, policy, CLI, application control, tests | ✅ done |
| 2 | Local LLM via `llama-cpp-python`, lazy load / idle unload, validated tool selection, benchmark | ✅ done |
| 3 | Launch Agent, global hotkey, graceful lifecycle, model worker process | ✅ done |
| 4 | File, project, terminal, browser (Playwright) and screen tools; multi-step agent loop with verification and bounded retries | ✅ done |
| 5 | Push-to-talk voice, spoken replies, SQLite memory | planned |
| 6 | Windows adapters, portfolio polish | planned |

## Quick start (macOS, Apple Silicon)

Requirements: macOS on Apple Silicon, an **arm64** Python ≥ 3.11 (Homebrew
`python@3.14` recommended), Xcode Command Line Tools, about 3 GB of disk for the
model, and Google Chrome (or Edge/Brave) for the browser tools.

```bash
cd ~/Desktop/Project/jarvis
scripts/setup_macos.sh          # .venv + Jarvis + llama.cpp (Metal) + Playwright + Vision OCR
source .venv/bin/activate
jarvis model download           # 2.5 GB; shows details and asks first
jarvis doctor                   # environment, permission and model checks
scripts/install_macos.sh        # start at login; then press ⌃⌥⌘J
```

Things to try:

```text
open my web browser and the notes app
open my Stock Price Predictor project and run it
what's on the todo list on my desktop?
organize my downloads folder by type
search the web for llama.cpp metal performance
which processes did you start? / stop p1
```

* **Shortcut:** press ⌃⌥⌘J, type a request, press Return. Visible successes need
  no further dialog. Answers, problems and clarifying questions appear in a
  dialog. Risky steps ask first, and **Cancel is the default button**.
* **Terminal, through the running service** (reuses its warm model):
  `jarvis ask "…"`.
* **Terminal, standalone:** `jarvis run "…"`, `jarvis shell`. Add `--dry-run` to
  preview a plan: read-only steps (search, read, inspect) run, and anything that
  would change something is only described.

Service management (scripts in `scripts/` wrap the same commands):

```bash
jarvis service status       # state, shortcut, model, memory, CPU time
jarvis service stop         # graceful; starts again at next login   (scripts/stop.sh)
jarvis service start        #                                         (scripts/start.sh)
jarvis service restart
jarvis service measure      # sample the service's idle CPU, wakeups, memory
jarvis service uninstall    # removes the Launch Agent; keeps config, logs, models
jarvis activate             # open the dialog as if the shortcut were pressed
```

Other commands: `jarvis apps list|open|quit`, `jarvis model info|check|download`,
`jarvis benchmark`, `jarvis config init|show|path`, `jarvis daemon [--headless]`.

Exit codes: `0` success or dry run, `1` error, `3` not executed (denied,
cancelled, needs clarification, or not understood), `130` interrupted.

## Tools

| Group | Tools | Default risk | Notes |
|---|---|---|---|
| `apps.*` | `list`, `open`, `quit` | read-only / low / medium | Only discovered `.app` bundles; graceful quit |
| `files.*` | `search`, `list`, `read`, `create`, `rename`, `move`, `organize`, `trash`, `open` | read-only → high | Sandboxed paths; never overwrites; no permanent delete (`trash` is recoverable and always asks) |
| `projects.*` | `find`, `inspect` | read-only | Finds code projects; works out run/test/setup commands (venv, npm scripts, Makefile, README) |
| `terminal.*` | `run`, `list`, `output`, `stop`, `open` | per command | Allowlisted commands, no shell; servers run in the background and are stopped when Jarvis exits |
| `browser.*` | `open`, `search`, `read`, `click`, `type`, `close` | low → high | Your installed Chrome with a separate profile; DOM element refs, never coordinates |
| `screen.*` | `read_text`, `capture` | medium (asks every time) | **Off by default.** On-device OCR (Apple Vision); captures deleted after reading |

### How the terminal decides

Commands are split into arguments and run directly, so pipes, redirection,
`$(…)` and `&&` are refused. The risk depends on the exact call:

| Risk | Examples | Asks first? |
|---|---|---|
| read-only | `git status`, `git log`, `pip list`, `python3 --version` | no |
| low | `pytest`, `npm test`, `make test`, `git fetch` | no (yes if the request read outside content) |
| medium | `python3 app.py`, `npm run dev`, `git commit`, dev servers | yes |
| high | `pip install`, `npm install`, `git push`, `git reset --hard` | always |
| refused | `sudo`, `rm`, shells, `python -c`, `curl`, `ssh`, `osascript`, `launchctl`, `git config` | never runs |

## Architecture

```
 launchd (Launch Agent, RunAtLoad, KeepAlive on crash)
   └─ jarvis daemon  ── main thread: AppKit run loop (accessory app, no Dock icon)
        │   ⌃⌥⌘J (Carbon RegisterEventHotKey) ──► request dialog (NSAlert)
        │   SIGTERM/SIGINT ──► SignalBridge ──► graceful shutdown
        │   control socket (status / activate / ask) ◄── jarvis service status|ask
        ▼
   Assistant (one worker thread): request → reply → dialog only if useful
        │
        ▼
 Orchestrator ──► Router (regex fast path, no AI) ── match ──────────────────┐
      │              no match / unknown app name                             │
      ▼                                                                      │
 Agent loop (≤ max_rounds):                                                  │
   Planner ── ModelManager ── SubprocessBackend ══ pipe ══ model worker      │
     │  prompt: enabled tools, app names, history, earlier rounds' results   │
     │  output: grammar-constrained JSON plan, then pydantic-checked         │
     ▼                                                                       │
   Plan: execute [steps] (finish?) │ clarify │ respond                       │
     │ ToolCall(s) ← untrusted input, whatever the source ◄──────────────────┘
     ▼
   Dispatcher: validate → enabled? → prepare (resolve, sandbox, per-call risk)
               → decide (risk + taint) → dry run? → confirm → run + verify
               → bounded retries (idempotent tools) → audit
     │
     ▼ results (data, never instructions) ──► back to the planner, or the reply
   Tools: apps · files · projects · terminal · browser · screen → PlatformAdapter
```

| Module | Responsibility |
|---|---|
| `jarvis/main.py`, `model_cli.py`, `service_cli.py` | CLI, exit codes, shell, model/benchmark/service commands |
| `jarvis/bootstrap.py` | Composition root shared by the CLI and the service |
| `jarvis/config.py`, `paths.py` | Validated config: defaults < TOML < env < CLI flags |
| `jarvis/agent/router.py` | Rule-based fast path (never guesses) |
| `jarvis/agent/planner.py` | LLM planner: schema-constrained plans, rounds, context budget |
| `jarvis/agent/orchestrator.py` | Agent loop: execute, observe, stop rules, taint, summaries, history |
| `jarvis/ai/` | Backend interface, llama.cpp backend, worker process, model manager, model files |
| `jarvis/api/` | `ToolCall`/`ToolResult` schemas; dispatcher (policy, confirmation, retries, audit) |
| `jarvis/tools/paths.py` | Path sandbox shared by file, project, terminal and screen tools |
| `jarvis/tools/files.py`, `projects.py` | File tools; project discovery and run-command inference |
| `jarvis/tools/commands.py`, `terminal.py` | Command allowlist/risk rules and diagnosis; terminal tools and background processes |
| `jarvis/tools/browser.py` | Playwright session on its own thread; DOM snapshots; browser tools |
| `jarvis/tools/screen.py` | Opt-in screenshot and OCR tools |
| `jarvis/tools/applications.py` | App catalog, name resolution, `apps.*` tools |
| `jarvis/runtime/` | Service lifecycle, event loop and signals, assistant flow, control socket, hotkeys, permissions, audit |
| `jarvis/ui/base.py` | Dialog interface (`prompt`/`confirm`/`show`) + headless implementation |
| `jarvis/platforms/` | OS adapters: `macos.py` (apps, files, Spotlight, Trash, screen, OCR, stats), `macos_gui.py`, `macos_service.py`, `windows.py` (placeholder) |
| `jarvis/benchmark.py` | Load / inference / memory / idle-cost measurements |

### Design decisions

* **`src/jarvis/` instead of the suggested `app/`.** The import name is a unique
  `jarvis`, and the src layout makes tests run against the installed package.
* **No app to install.** Jarvis is a Python package in a virtualenv, started by
  launchd. There's no `.app` bundle, Electron or separate AI app. Ollama isn't
  needed either: llama.cpp is linked in-process (in a worker).
* **An agent loop with explicit stops.** The model plans in rounds and sees each
  round's results. It marks a plan `finish: true` when those steps complete the
  request, so simple tasks take one model call. Jarvis stops immediately, never
  working around it, when:
  * you decline a step;
  * the policy refuses it;
  * a macOS permission is missing;
  * a page needs you (login, CAPTCHA) or something is ambiguous.

  An identical step that already failed is never re-run.
* **Taint across rounds.** Once a request has read outside content (files, web
  pages, command output, screen text), every later step that changes something
  needs confirmation. Instructions hidden in that content therefore can't act on
  their own. This is tested with an injected "open Safari" note.
* **Verification is part of every tool.** Apps are checked for running, files
  for existence after create/move, servers for a ready line or URL, commands for
  their exit code (with a diagnosis such as a missing package or a port in use).
  Failures go back to the model. Transient failures of idempotent tools (page
  loads, app launches) get bounded retries with backoff.
* **Grammar-safe schemas.** llama.cpp expands every string `maxLength` into one
  grammar rule per character. A 100,000-character limit once produced an 816 KB
  grammar that crashed the model worker. The grammar therefore omits large
  limits and regex patterns, and pydantic still enforces both on the parsed plan.
  With 25 tools the grammar is 21.6 KB.
* **Model in a worker process** (`[ai] isolation = "subprocess"`). Unloading
  returns all model memory to the OS. Shutdown can kill a generation mid-way, and
  a llama.cpp crash (like the grammar one above) can't take the service down.
* **Shortcut via `RegisterEventHotKey`, not a keyboard monitor.** No
  Accessibility or Input Monitoring permission, and Jarvis never sees other
  keystrokes.
* **Signals that work during dialogs.** A wakeup-fd bridge delivers SIGTERM to
  the run loop in all modes, and shutdown closes any open dialog first.
* **DOM, not pixels.** The browser tools read text and numbered elements, and
  the screen tool only reads text. Nothing clicks by screen coordinates.
* **Metal residency keep-alive off by default.** It cost ~250 CPU wakeups/s
  while a model was loaded, for no measured latency benefit.

## Security model

* **Model output is untrusted.** The planner only returns `ToolCall`s. The
  dispatcher validates each one and applies policy and confirmation exactly as
  for typed commands.
* **Prompt injection is contained, not prevented.** Results are shown to the
  model as data with an explicit warning. Any change after reading outside
  content needs your confirmation, and content can never grant permissions.
* **Path sandbox:**
  * every path is resolved, including symlinks;
  * reads stay in `[files] read_roots` (home);
  * writes stay in `write_roots` (Desktop, Documents, Downloads);
  * `~/Library` (app data, plus Jarvis's own config, audit log and models) and
    `~/.Trash` are blocked;
  * hidden files (`.ssh`, `.env`, `.git`) and key files (`*.pem`, `id_rsa`) are
    refused.
* **Nothing is permanently deleted or overwritten.** `files.trash` moves to the
  Trash and always asks. Create, rename and move refuse existing targets, and
  `organize` shows one plan and saves an undo list.
* **No shell.** The terminal only runs allowlisted programs, each call
  classified by risk:
  * a minimal environment (Jarvis's own environment and secrets aren't passed
    on);
  * its own process group, a timeout, and cancellation.
* **Browser:**
  * a separate profile;
  * only `http(s)` addresses;
  * anything that submits, buys, sends, deletes or signs up is high risk;
  * Jarvis never types into password or payment fields;
  * CAPTCHAs, bot checks and block pages are handed to you, never solved.
* **Screen:** off by default; every capture asks; on-device OCR; temporary
  captures are deleted after reading.
* **Confirmations default to No.** Cancel is the default button. A dialog that's
  open when Jarvis stops counts as No. Headless runs deny, and approvals are
  never cached.
* **Local control socket** (`0600`, `0700` directory, peer-uid check). Requests
  get the same policy and dialogs.
* **Audit log** at `~/Library/Logs/Jarvis/audit.jsonl`: every tool call with its
  risk, decision, taint, attempts and outcome. Credentials are redacted, and the
  files are `0600`.
* **Local only.** Prompts never leave the machine. Nothing is downloaded unless
  you run `jarvis model download`. Web pages are only fetched by browser tools
  you ask for.

## Configuration

```bash
jarvis config init      # writes ~/Library/Application Support/Jarvis/config.toml
jarvis config show      # effective configuration as JSON
```

Precedence: built-in defaults < `config.toml` < environment variables < CLI flags.
Environment variables (see `.env.example`): `JARVIS_CONFIG`, `JARVIS_DRY_RUN`,
`JARVIS_LOG_LEVEL`, `JARVIS_LOG_DIR`, `JARVIS_AI_ENABLED`, `JARVIS_MODEL_PATH`.
The Launch Agent doesn't inherit your shell's environment, so put service
settings in `config.toml` and run `jarvis service restart` after changes.

Key settings:

| Section | Settings (defaults) |
|---|---|
| `[policy]` | `enabled_tools` (`apps.*`, `files.*`, `projects.*`, `terminal.*`, `browser.*`; add `"screen.*"` to opt in), `confirm_at_or_above` (`medium`), app allow/block lists |
| `[files]` | `read_roots` (`~`), `write_roots` (Desktop, Documents, Downloads), `project_roots`, `blocked_paths` (`~/Library`, `~/.Trash`), `allow_hidden` (false) |
| `[terminal]` | `search_path` (Homebrew + system), `extra_programs` (always ask), `timeout_seconds` (120), `startup_seconds` (10) |
| `[browser]` | `headless` (false: you see the window), `executable`, `search_url` (Bing), `idle_close_seconds` (600) |
| `[agent]` | `max_rounds` (6), `max_tool_retries` (2), `retry_backoff_seconds`, `max_observation_chars` (1500), `history_ttl_seconds` (600) |
| `[ai]` | `isolation` (`subprocess`), `idle_unload_seconds` (300), `context_length` (4096), `gpu_layers` (-1), `include_app_names` (true) |
| `[service]` | `hotkey` (`ctrl+option+cmd+j`), `show_success` (false) |

Files: config, models and the browser profile in
`~/Library/Application Support/Jarvis/`; logs (`jarvis.log`, `audit.jsonl`,
`service.log`), background-process output and organize undo lists in
`~/Library/Logs/Jarvis/`; the agent in
`~/Library/LaunchAgents/local.jarvis.assistant.plist`.

## Performance and trade-offs

Measured on a MacBook Air M4 (16 GB), macOS 27, Qwen3-4B-Instruct-2507 Q4_K_M,
context 4096, Metal, model in the worker process. Reproduce with
`jarvis service measure` and `jarvis benchmark`.

**Background service** (installed Launch Agent, measured from outside):

| State | CPU | Wakeups/s | Memory (footprint) |
|---|---|---|---|
| Idle, model not loaded (60 s sample) | 0.004% | 0.08 | 51 MB |
| Idle, model loaded in worker | 0.02% | 0.4–0.7 | 740–870 MB (service + worker) |
| After the idle unload (300 s) | 0.002% | 0.05 | 49 MB |

**Requests** (Phase 4 prompt: 25 tools, ~1,770 tokens):

| | Time |
|---|---|
| Rule-based command ("open calculator" via the shortcut) | 115 ms to a verified launch |
| First AI request after a load (builds the prompt cache) | ~8 s (5.6 s reading the prompt + generation) |
| Later AI requests (prompt cached) | 1.5–1.7 s per planning round, ~29 tokens/s |
| Multi-step tasks, e.g. "open my project and run it" (3 rounds, cold model) | ~12–16 s end to end |
| Model load / unload (worker) | 0.4–0.5 s / 0.07 s |
| Graceful stop / restart of the service | 0.4 s / 0.7 s |

**What follows:**

* Idle, the service costs practically nothing.
* While loaded, the model holds ~0.7 GB of memory but almost no CPU.
* The 300 s idle timeout keeps follow-up requests fast and returns all the
  memory after five quiet minutes.
* The cost of the larger tool set is the cold request: the system prompt doubled
  from Phase 2, and so did the first-request time.
* Next performance step: persist llama.cpp's prompt cache to disk, so a reloaded
  model can skip re-reading the instructions. Not measured yet.

**Battery:** not measured directly. Proxies are CPU time and wakeups (the basis
of macOS Energy Impact). A planning round uses ~0.4–0.8 CPU-seconds with Metal.
GPU energy isn't included in CPU time. For package power, run
`sudo powermetrics --samplers cpu_power,gpu_power -i 1000` while using Jarvis.

## Dependencies

| Package | Why | Scope |
|---|---|---|
| `pydantic>=2.9,<3` | Config and tool-argument validation; JSON schemas for constrained decoding | runtime |
| `pyobjc-framework-Cocoa>=11,<13` | AppKit: run loop, dialogs, running apps, Trash | runtime, macOS only |
| `llama-cpp-python>=0.3.36,<0.4` | Local inference (compiled with Metal) | `llm` extra |
| `playwright>=1.60,<2` | Browser automation (drives installed Chrome; no browser download) | `browser` extra |
| `pyobjc-framework-Vision>=11,<13` | On-device OCR for the screen tools | `screen` extra |
| `pytest`, `ruff`, `mypy` | Tests, lint/format, strict type checking | `dev` extra |

Everything else is standard library or a system tool/framework: Carbon (shortcut),
libproc (process stats), Spotlight (`mdfind`), `screencapture`, `launchctl`.
Tested with Python 3.14.3 (Homebrew, arm64), llama-cpp-python 0.3.36,
Playwright 1.63.0 with Chrome 155, PyObjC 12.2.2 on macOS 27.0.1.

## Testing

```bash
pytest                                       # 528 tests, ~10 s: unit, fake platform, real processes,
                                             #   headless Chrome on a local site, Vision OCR (no model)
JARVIS_RUN_MODEL_TESTS=1 pytest -m model     # real model incl. find→inspect→run a project (~40 s)
JARVIS_RUN_GUI_TESTS=1 pytest -m gui         # Calculator, service dialog + SIGTERM, real screen capture
ruff check src tests && ruff format --check src tests
mypy                                         # strict mode
```

* The unit tests use an in-memory app platform, a throwaway home folder and a
  scripted fake model. They never touch your apps or files or load the real
  model.
* Terminal tests run real processes, browser tests drive real headless Chrome
  against a local test site, and service tests start real `jarvis daemon`
  processes and stop them with real signals.

## Known limitations

* A 4B model sometimes picks a roundabout tool sequence. The round limit,
  confirmations and stop rules keep that safe, but not always efficient.
* Several search engines (DuckDuckGo, Brave, Startpage) show bot checks to
  automated browsers, so the default search is Bing. Pages that need a login
  are handed to you.
* OCR can't always tell a lone `0` from `O` in the system font.
* The context window (4096 tokens) bounds how much tool output the model sees
  per request. Older results are shortened automatically.
* Not tested on Windows or on Intel Macs.

## Troubleshooting

* **Shortcut does nothing:** run `jarvis service status`. If it says
  "unavailable", another app owns the combination; set `[service] hotkey`, then
  `jarvis service restart`.
* **Screen tools say permission is missing:** allow Screen Recording for Python
  (the service) or your terminal app (`jarvis run`) in System Settings → Privacy &
  Security → Screen & System Audio Recording.
* **Browser tools missing:** run `jarvis doctor`. They need the `browser` extra
  and Chrome, Edge or Brave.
* **"Background Items Added" notice:** that's macOS reporting the Launch Agent.
  It appears as Python, because Jarvis runs as a Python program rather than an
  installed app.
* **Logs:** `~/Library/Logs/Jarvis/jarvis.log` (application), `service.log`
  (launchd stdout/stderr), `audit.jsonl` (every action), `processes/`
  (background command output).

## Models

See [`models/README.md`](models/README.md). The default model is
**Qwen3-4B-Instruct-2507 Q4_K_M** (2.5 GB, Apache-2.0), pinned by revision and
SHA-256.

## Roadmap

* **Phase 5:** Optional push-to-talk input, spoken replies, SQLite memory
  (preferences, task history, reusable workflows) with inspect/export/clear
  controls.
* **Phase 6:** Windows adapters ([`docs/WINDOWS.md`](docs/WINDOWS.md)), a
  persistent prompt cache for faster cold requests, optional menu-bar UI.
  Packaging as an `.app` stays optional: Jarvis keeps working without
  installing an app.

## Windows

Not supported yet, and not tested on Windows. The platform boundary and plan
(service, shortcut, event loop, and the Phase 4 tools) are in
[`docs/WINDOWS.md`](docs/WINDOWS.md).

## Licence

Not chosen yet.

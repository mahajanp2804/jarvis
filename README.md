# Jarvis

A personal AI computer assistant that runs **locally** on macOS (Apple Silicon first,
Windows later), with no app to install: it's a Python program that starts at login.
Press **⌃⌥⌘J** anywhere, type what you want, and Jarvis turns it into checked,
permission-gated actions on your computer:

* It runs quietly in the background as a Launch Agent. Idle, it uses ~0.004% CPU
  and ~50 MB, with no polling, no listening microphone, and no screen capture
  unless you ask.
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
* It **remembers** what you ask it to ("my editor is VS Code"), keeps a task
  history you can inspect, export and erase, and can **replay a task as a
  workflow** ("save that as morning setup"). Approvals are never remembered.
* Optional **voice**: press **⌃⌥⌘K**, speak, check the transcript, run it
  (on-device whisper.cpp); and spoken replies with the built-in macOS voices.
  Both are off by default.

> **Status: Phase 5 of 6 (memory and voice).** Windows adapters come next. See
> [Roadmap](#roadmap).

| Phase | Scope | Status |
|---|---|---|
| 1 | Config, logging, policy, CLI, application control, tests | ✅ done |
| 2 | Local LLM via `llama-cpp-python`, lazy load / idle unload, validated tool selection, benchmark | ✅ done |
| 3 | Launch Agent, global hotkey, graceful lifecycle, model worker process | ✅ done |
| 4 | File, project, terminal, browser (Playwright) and screen tools; multi-step agent loop with verification and bounded retries | ✅ done |
| 5 | SQLite memory (preferences, task history, workflows) with inspect/export/clear; spoken replies; push-to-talk voice input | ✅ done |
| 6 | Windows adapters, portfolio polish | planned |

## Quick start (macOS, Apple Silicon)

Requirements: macOS on Apple Silicon, an **arm64** Python ≥ 3.11 (Homebrew
`python@3.14` recommended), Xcode Command Line Tools, about 3 GB of disk for the
model, and Google Chrome (or Edge/Brave) for the browser tools.

```bash
cd ~/Desktop/Project/jarvis
scripts/setup_macos.sh          # .venv + Jarvis + llama.cpp (Metal) + Playwright + OCR + voice
source .venv/bin/activate
jarvis model download           # 2.5 GB; shows details and asks first
jarvis doctor                   # environment, permission and model checks
scripts/install_macos.sh        # start at login; then press ⌃⌥⌘J
jarvis voice download           # optional, for push-to-talk: 60 MB speech model
```

Things to try:

```text
open my web browser and the notes app
open my Stock Price Predictor project and run it
what's on the todo list on my desktop?
organize my downloads folder by type
search the web for llama.cpp metal performance
which processes did you start? / stop p1
remember that my editor is Visual Studio Code      → then: open my editor
save that as dev start                             → later: run the dev start workflow
what do you remember? / list workflows
```

* **Shortcut:** press ⌃⌥⌘J, type a request, press Return. Visible successes need
  no further dialog. Answers, problems and clarifying questions appear in a
  dialog. Risky steps ask first, and **Cancel is the default button**.
* **Terminal, through the running service** (reuses its warm model):
  `jarvis ask "…"`.
* **Terminal, standalone:** `jarvis run "…"`, `jarvis shell`. Add `--dry-run` to
  preview a plan: read-only steps (search, read, inspect) run, and anything that
  would change something is only described. Add `--speak` to hear a short
  summary of each reply.
* **Voice:** with push-to-talk on, press ⌃⌥⌘K anywhere; in a terminal,
  `jarvis listen` or `jarvis shell --voice` (Enter on an empty line). See
  [Voice](#voice).

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

Memory, workflows and voice:

```bash
jarvis memory show          # preferences, recent tasks, workflows, where it's stored
jarvis memory export [FILE] # everything as JSON (file is 0600, never overwritten)
jarvis memory clear         # erase all (asks); --preferences, --history, --workflows
jarvis memory remember editor "Visual Studio Code" / jarvis memory forget editor
jarvis workflow list|show|run|delete NAME
jarvis voice voices|say "…"         # spoken replies: pick and test a voice
jarvis voice download|listen        # push-to-talk: model, and a microphone test
jarvis voice transcribe FILE.wav    # transcribe a 16 kHz mono WAV
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
| `memory.*` | `remember`, `forget`, `list`, `history` | low / read-only | Preferences you state; refuses secrets; asks first after outside content was read. No tool can clear memory |

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

## Memory and workflows

Jarvis keeps a small SQLite database at
`~/Library/Application Support/Jarvis/memory.sqlite3` (owner-only, `0600`;
outside the repository and outside the folders its file tools may touch).

| What | How it gets there | How it's used |
|---|---|---|
| **Preferences** | "remember that my editor is Visual Studio Code", or `jarvis memory remember` | Given to the model as facts, so "open my editor" works. Never instructions, never permissions |
| **Task history** | Every request: text, outcome, tools and shortened arguments, source (cli, shell, service, voice) | `jarvis memory show`, "what did I do earlier?" (`memory.history`). Kept 30 days / 1,000 tasks |
| **Workflows** | "save that as morning setup" saves the completed steps of the previous request | "run the morning setup workflow" replays them, without the model |

**Never stored:** approvals (every risky step asks again, every time; a "yes"
is never permission for later); passwords, keys and tokens (the same redaction
as the logs, requests that mention a password, PIN, card number or similar are
stored without their text, and such preferences or workflow steps are refused);
and outside content (a reply that quoted a file, page, command output or screen
text isn't stored).

**Replays get no shortcuts.** A workflow step goes through validation, the
policy, confirmation, verification and the audit log exactly like a typed
request. A blocked app stays blocked, a risky step asks every run, outside
content read by one step makes later steps ask first, and the replay stops at
the first failure.

**Inspect, export, erase:** `jarvis memory show | export | clear`. Clearing
uses SQLite `secure_delete` and `VACUUM`, so erased entries don't linger in the
file. `[memory] enabled = false` turns it all off; `record_history = false`
keeps preferences and workflows but no history.

## Voice

Both halves are optional, on-device, and off by default.

**Spoken replies** (`[voice] speak_replies = true`, or `--speak`): Jarvis says a
short summary with the built-in macOS voices (`/usr/bin/say`): the answer or
question, or the outcome of the last step. Lists, file contents and pages are
shown, not read aloud. Speech stops when you press the shortcut again, when a
new request starts, and when Jarvis stops. Pick a voice with
`jarvis voice voices` and `[voice] voice = "Daniel"`.

**Push-to-talk** (`[voice] push_to_talk = true`, after `jarvis voice download`):

1. Press **⌃⌥⌘K**. A "Listening…" dialog appears and a worker process opens
   the microphone.
2. Speak. Recording stops when you pause (1.5 s), press ⌃⌥⌘K again, or click
   Done (15 s at most).
3. whisper.cpp transcribes on-device, and the worker exits.
4. You see the transcript in an editable box. Nothing runs until you press Run;
   then it's a normal request, with the normal planner, policy and confirmations.

Nothing listens before you press the shortcut: there's no wake word. Audio
stays in memory and is never written to disk. In a terminal, `jarvis listen`
and `jarvis shell --voice` work the same way (Enter stops recording, then
`[Y]es / [e]dit / [n]o`). `jarvis voice listen` only prints what it heard.

**Microphone permission.** macOS asks the first time the microphone is used,
and it asks on behalf of the app that started Jarvis. That's your terminal app
for `jarvis listen`, and Python for the background service. Python's
Homebrew app bundle doesn't declare a microphone purpose string, so macOS
may refuse the service without asking. Jarvis then says "The microphone gave
only silence" and points to System Settings → Privacy & Security →
Microphone. If the service can't get the microphone, `jarvis listen` in your
terminal works, and so does macOS Dictation in the ⌃⌥⌘J box (press the
microphone or Globe key).

## Architecture

```
 launchd (Launch Agent, RunAtLoad, KeepAlive on crash)
   └─ jarvis daemon  ── main thread: AppKit run loop (accessory app, no Dock icon)
        │   ⌃⌥⌘J (Carbon RegisterEventHotKey) ──► request dialog (NSAlert)
        │   ⌃⌥⌘K (optional) ──► "Listening…" ══ voice worker (mic + whisper.cpp,
        │                        one process per request) ──► editable transcript
        │   SIGTERM/SIGINT ──► SignalBridge ──► graceful shutdown
        │   control socket (status / activate / ask) ◄── jarvis service status|ask
        ▼
   Assistant (one worker thread): request → reply → dialog only if useful
        │                                    └─► spoken summary (say, optional)
        ▼
 Orchestrator ──► "save that as …" / "run the … workflow" ──► MemoryStore (SQLite)
      │           (replayed step by step through the Dispatcher below)   ▲ history,
      ▼                                                                   │ preferences
 Router (regex fast path, no AI) ── match ───────────────────────────────┼──┐
      │              no match / unknown app name                             │
      ▼                                                                      │
 Agent loop (≤ max_rounds):                                               │  │
   Planner ── ModelManager ── SubprocessBackend ══ pipe ══ model worker   │  │
     │  prompt: system (tools, app names) + preferences ─────────────────┘  │
     │          + conversation + earlier rounds' results                     │
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
   Tools: apps · files · projects · terminal · browser · screen · memory → PlatformAdapter
```

| Module | Responsibility |
|---|---|
| `jarvis/main.py`, `model_cli.py`, `service_cli.py`, `memory_cli.py`, `voice_cli.py` | CLI, exit codes, shell, model/benchmark/service/memory/workflow/voice commands |
| `jarvis/bootstrap.py` | Composition root shared by the CLI and the service |
| `jarvis/config.py`, `paths.py` | Validated config: defaults < TOML < env < CLI flags |
| `jarvis/agent/router.py` | Rule-based fast path (never guesses) |
| `jarvis/agent/planner.py` | LLM planner: schema-constrained plans, rounds, context budget |
| `jarvis/agent/orchestrator.py` | Agent loop: execute, observe, stop rules, taint, summaries, history, workflow save/replay |
| `jarvis/agent/workflows.py` | "save that as …" / "run the … workflow" commands |
| `jarvis/memory/store.py` | SQLite memory: preferences, task history, workflows; redaction, retention, export, clear |
| `jarvis/voice/` | Speech-output contract and summaries; push-to-talk recordings, worker process, speech model |
| `jarvis/ai/` | Backend interface, llama.cpp backend, worker process, model manager, model files |
| `jarvis/api/` | `ToolCall`/`ToolResult` schemas; dispatcher (policy, confirmation, retries, audit) |
| `jarvis/tools/paths.py` | Path sandbox shared by file, project, terminal and screen tools |
| `jarvis/tools/files.py`, `projects.py` | File tools; project discovery and run-command inference |
| `jarvis/tools/commands.py`, `terminal.py` | Command allowlist/risk rules and diagnosis; terminal tools and background processes |
| `jarvis/tools/browser.py` | Playwright session on its own thread; DOM snapshots; browser tools |
| `jarvis/tools/screen.py` | Opt-in screenshot and OCR tools |
| `jarvis/tools/applications.py` | App catalog, name resolution, `apps.*` tools |
| `jarvis/tools/memory.py` | `memory.*` tools |
| `jarvis/runtime/` | Service lifecycle, event loop and signals, assistant flow, control socket, hotkeys, permissions, audit |
| `jarvis/ui/base.py` | Dialog interface (`prompt`/`confirm`/`show`, optional `listen`/`edit`) + headless implementation |
| `jarvis/platforms/` | OS adapters: `macos.py` (apps, files, Spotlight, Trash, screen, OCR, stats), `macos_gui.py`, `macos_service.py`, `macos_speech.py`, `windows.py` (placeholder) |
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
  With the 29 default tools the grammar is 27.3 KB (21.6 KB with Phase 4's 25).
* **Model in a worker process** (`[ai] isolation = "subprocess"`). Unloading
  returns all model memory to the OS. Shutdown can kill a generation mid-way, and
  a llama.cpp crash (like the grammar one above) can't take the service down.
* **System prompt state saved to disk, not kept in memory.** A fresh worker
  restores the KV state of the ~1,750-token instructions from a file instead of
  re-reading them, so unloading after 300 s idle no longer makes the next request
  slow. See "Prompt cache on disk" below.
* **Shortcut via `RegisterEventHotKey`, not a keyboard monitor.** No
  Accessibility or Input Monitoring permission, and Jarvis never sees other
  keystrokes.
* **Signals that work during dialogs.** A wakeup-fd bridge delivers SIGTERM to
  the run loop in all modes, and shutdown closes any open dialog first.
* **DOM, not pixels.** The browser tools read text and numbered elements, and
  the screen tool only reads text. Nothing clicks by screen coordinates.
* **Metal residency keep-alive off by default.** It cost ~250 CPU wakeups/s
  while a model was loaded, for no measured latency benefit.
* **Preferences after the system prompt, not in it.** They go in a second
  system message, so remembering something doesn't invalidate the saved
  prompt state (~5 s to rebuild).
* **Voice input in a throwaway process.** Freeing whisper.cpp inside a
  long-running process kept ~85 MB (measured), and PortAudio and numpy would
  stay loaded too. A worker per request starts in ~0.1 s, loads the model while
  you speak, and gives everything back when it exits.
* **Whisper on the CPU by default.** 0.2 s per command instead of 0.08 s, but
  never the ~7 s Metal shader compile that happens whenever macOS has reset its
  shader cache (`[voice] stt_use_gpu` switches).
* **Workflows replay tool calls, not prompts.** A saved workflow is the exact
  list of steps that ran, so a replay is predictable and needs no model, and
  it still goes through the dispatcher one step at a time.

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
* **Memory:** no approvals are stored, ever. Secrets are refused or left out,
  outside content isn't stored, the database is `0600`, and erasing it really
  erases it (`secure_delete` + `VACUUM`). Preferences reach the model as
  facts, never as instructions or permissions. Remembering after outside
  content was read asks first, so a web page can't plant a "preference".
  Workflows replay through the full policy pipeline.
* **Voice:** nothing listens until you press the shortcut. Audio is never
  written to disk, and you see (and can edit) the transcript before anything
  runs. Spoken text goes to `say` on stdin, never as arguments.
* **Local control socket** (`0600`, `0700` directory, peer-uid check). Requests
  get the same policy and dialogs.
* **Audit log** at `~/Library/Logs/Jarvis/audit.jsonl`: every tool call with its
  risk, decision, taint, attempts and outcome. Credentials are redacted, and the
  files are `0600`.
* **Local only.** Prompts and audio never leave the machine. Nothing is
  downloaded unless you run `jarvis model download` or `jarvis voice download`.
  Web pages are only fetched by browser tools you ask for.

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
| `[policy]` | `enabled_tools` (`apps.*`, `files.*`, `projects.*`, `terminal.*`, `browser.*`, `memory.*`; add `"screen.*"` to opt in), `confirm_at_or_above` (`medium`), app allow/block lists |
| `[files]` | `read_roots` (`~`), `write_roots` (Desktop, Documents, Downloads), `project_roots`, `blocked_paths` (`~/Library`, `~/.Trash`), `allow_hidden` (false) |
| `[terminal]` | `search_path` (Homebrew + system), `extra_programs` (always ask), `timeout_seconds` (120), `startup_seconds` (10) |
| `[browser]` | `headless` (false: you see the window), `executable`, `search_url` (Bing), `idle_close_seconds` (600) |
| `[agent]` | `max_rounds` (6), `max_tool_retries` (2), `retry_backoff_seconds`, `max_observation_chars` (1500), `history_ttl_seconds` (600) |
| `[ai]` | `isolation` (`subprocess`), `idle_unload_seconds` (300), `context_length` (4096), `gpu_layers` (-1), `include_app_names` (true), `prompt_cache` (true), `prompt_cache_max_mb` (512) |
| `[service]` | `hotkey` (`ctrl+option+cmd+j`), `show_success` (false) |
| `[memory]` | `enabled` (true), `path`, `record_history` (true), `history_days` (30), `max_history` (1000), `share_preferences` (true) |
| `[voice]` | `speak_replies` (false), `voice`, `rate`, `max_spoken_chars` (300); `push_to_talk` (false), `push_to_talk_hotkey` (`ctrl+option+cmd+k`), `stt_model_path`, `language` (`en`), `max_record_seconds` (15), `silence_seconds` (1.5), `stt_use_gpu` (false) |

An existing `config.toml` that lists `enabled_tools` keeps its list: add
`"memory.*"` to it for the memory tools (the `jarvis memory` commands work
either way).

Files: config, models (incl. the speech model), the memory database
(`memory.sqlite3`), the saved prompt state (`prompt-cache/`) and the browser
profile in `~/Library/Application Support/Jarvis/`; logs (`jarvis.log`,
`audit.jsonl`, `service.log`, `voice-worker.log` from the last voice request),
background-process output and organize undo lists in `~/Library/Logs/Jarvis/`;
the agent in `~/Library/LaunchAgents/local.jarvis.assistant.plist`.

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
| Phase 5 re-measurement: after memory/workflow requests and the idle unload (60 s sample) | 0.000% | 0.00 | 55 MB |

Phase 5 adds ~6 MB once the memory database has been used (an open SQLite
connection and the memory/voice modules). Push-to-talk and spoken replies add
nothing while idle: their processes exist only during a request.

**Requests** (Phase 4 prompt: 25 tools, ~1,770 tokens):

| | Time |
|---|---|
| Rule-based command ("open calculator" via the shortcut) | 115 ms to a verified launch |
| First AI request after a model load, prompt state on disk (the usual case) | ~2.4 s (0.07 s restoring the prompt, then generation); was 7.6 s |
| First AI request with a new system prompt (first run, or after the tools, app list or settings change) | ~7.8 s (5.4 s reading the prompt + generation + 0.2 s saving it) |
| Later AI requests (prompt in memory) | 1.4–1.6 s per planning round, ~29 tokens/s |
| Multi-step tasks, e.g. "open my project and run it" (3 rounds, cold model) | ~12–16 s end to end, measured before the prompt cache (which removes the ~5.4 s prompt read from the first round) |
| Model load / unload (worker) | 0.4–0.5 s / 0.07 s |
| Graceful stop / restart of the service | 0.4 s / 0.7 s |

**What follows:**

* Idle, the service costs practically nothing.
* While loaded, the model holds ~0.7 GB of memory but almost no CPU.
* The 300 s idle timeout returns all the memory after five quiet minutes. With
  the prompt state on disk, the request after an unload costs ~0.5 s more than
  one with the model still loaded (worker start + restore), not ~6 s.
* The full prompt read (~5.4 s) now happens once per system prompt, not once per
  model load.

**Prompt cache on disk** (`[ai] prompt_cache`, on by default). A model load
starts with an empty KV cache, so without this every load re-read the
1,749-token system prompt. Now the first request that reads it saves the
llama.cpp state of the system prompt (never the request) to
`~/Library/Application Support/Jarvis/prompt-cache/`, and later loads restore it.
`jarvis benchmark` measures both in one run (it uses an empty temporary cache
directory, so it always starts cold):

| `jarvis benchmark` | Before (d753ff5) | `prompt_cache = false` | `prompt_cache = true` |
|---|---|---|---|
| First request after reload | 7.55 s | 7.68 s | **2.41 s** (restore 0.07 s) |
| First request after first load | 7.65 s | 7.63 s | 7.82 s (incl. 0.20 s save) |
| The other 7 requests (prompt in memory) | 1.44–2.28 s | 1.44–2.27 s | 1.43–2.27 s |
| Footprint (service + worker) after the first request after reload | n/a | 904 MB | 886 MB |
| Disk | 0 | 0 | 246 MB (1 file) |

The 2.41 s is the benchmark's 48-token reply: the same request with the prompt
already in memory takes 2.27 s. Shorter replies are faster (1.4–1.6 s).

* **Keyed and invalidated** by a hash of the system prompt (so the enabled
  tools and the installed-app list are part of it), the model file (name, size,
  modification time), the context length, the chat format and the
  llama-cpp-python version. A change means one full read (~7.8 s) and a new file.
* **Capped** at `prompt_cache_max_mb` (512 MB: the current prompt and one
  earlier version); least recently used files are deleted first.
* **Safe to delete** at any time; it's rebuilt on the next request. `jarvis model info`
  shows its size.
* **Checked:** files are written to a temporary name and renamed, and carry a
  CRC32 (23 ms to verify). llama.cpp rejects truncated or garbage files by itself,
  but silently loads flipped bytes. An unreadable file is deleted and the prompt
  is read in full instead.
* **Rejected alternative:** llama-cpp-python's built-in
  `Llama.set_cache(LlamaDiskCache(...))`. In a prototype it saved ~550 MB (KV
  plus a 311 MB logits array) after *every* request, adding 0.4–0.7 s to each
  one; the worker grew to 3.9–4.2 GB; and the first request after a restart
  still took 2.8 s.

**Memory and voice** (Phase 5, same Mac):

| | Measured |
|---|---|
| Remember / recall / forget a preference, list workflows (rule-based) | instant, no model |
| Replaying a saved workflow (`run the … workflow`, one app step, via the service) | 0.21 s |
| Task-history write per request | one small SQLite insert; database 32 KB after a few dozen tasks |
| Push-to-talk worker per request (CPU whisper, 3 s of speech, after recording) | 0.44 s wall, 0.78 CPU-s, 213 MB peak, all freed on exit |
| Transcription alone, base.en q5_1 | 0.2 s on the CPU; 0.08 s with `stt_use_gpu` (~7 s once after a Metal shader-cache reset) |
| Push-to-talk / spoken replies while idle | no process, no listening, no cost: the worker and `say` exist only during a request |
| Accuracy (synthesized test commands, `say` voice) | 6/6 transcribed exactly ("open my Stock Price Predictor project and run it") |

Real-voice accuracy depends on your microphone and accent, and wasn't measured
with a live microphone (see Known limitations).

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
| `pywhispercpp>=1.5,<2` | On-device speech-to-text (whisper.cpp; wheel includes the library) | `voice` extra |
| `sounddevice>=0.5,<0.6` | Microphone capture (PortAudio, bundled in the wheel) | `voice` extra |
| `numpy>=2,<3` | Audio buffers for the voice worker | `voice` extra |
| `pytest`, `ruff`, `mypy` | Tests, lint/format, strict type checking | `dev` extra |

Everything else is standard library or a system tool/framework: Carbon (shortcut),
libproc (process stats), Spotlight (`mdfind`), `screencapture`, `launchctl`,
`say` (spoken replies), SQLite (memory).
Tested with Python 3.14.3 (Homebrew, arm64), llama-cpp-python 0.3.36,
Playwright 1.63.0 with Chrome 155, PyObjC 12.2.2, pywhispercpp 1.5.1,
sounddevice 0.5.6 on macOS 27.0.1.

## Testing

```bash
pytest                                       # 636 tests, ~12 s: unit, fake platform, real processes,
                                             #   headless Chrome on a local site, Vision OCR (no model)
JARVIS_RUN_MODEL_TESTS=1 pytest -m model     # real models: find→inspect→run a project, the prompt
                                             #   cache across reloads, remembered preferences,
                                             #   whisper on synthesized speech (~70 s)
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
* Memory tests use a temporary database (a fixture redirects the default path,
  so tests can never touch yours). Voice tests use a scripted worker process and
  a fake `say`; the real-model voice test feeds `say`-synthesized WAV files to
  the real worker. No test opens the microphone or plays audio.

## Known limitations

* A 4B model sometimes picks a roundabout tool sequence. The round limit,
  confirmations and stop rules keep that safe, but not always efficient.
* Several search engines (DuckDuckGo, Brave, Startpage) show bot checks to
  automated browsers, so the default search is Bing. Pages that need a login
  are handed to you.
* OCR can't always tell a lone `0` from `O` in the system font.
* The context window (4096 tokens) bounds how much tool output the model sees
  per request. Older results are shortened automatically.
* Push-to-talk wasn't tested with a live microphone. The recording code is
  exercised with scripted workers and transcription with synthesized speech,
  but macOS may refuse the microphone to the background service (see
  [Voice](#voice)); `jarvis listen` in a terminal is the fallback.
* The default speech model (`base.en`) understands English only. Set
  `[voice] stt_model_path` to a multilingual whisper.cpp model and `language`
  for others.
* Spoken replies are a one-line summary, by design.
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
* **"The microphone gave only silence":** allow the microphone for your
  terminal app (`jarvis listen`) or Python (the service) in System Settings →
  Privacy & Security → Microphone, and check the input level in Sound settings.
  `jarvis voice listen` tests it without running anything.
* **"I didn't hear anything":** nothing louder than the room noise was
  recorded within 8 s. Speak closer, or raise `[voice] max_record_seconds`.
* **A configured voice isn't used:** `say` silently falls back to the system
  voice for unknown names; `jarvis doctor` checks the name against
  `jarvis voice voices`.
* **Forget everything:** `jarvis memory clear` (asks first), or delete
  `~/Library/Application Support/Jarvis/memory.sqlite3`.
* **Logs:** `~/Library/Logs/Jarvis/jarvis.log` (application), `service.log`
  (launchd stdout/stderr), `audit.jsonl` (every action), `processes/`
  (background command output), `voice-worker.log` (last voice request).

## Models

See [`models/README.md`](models/README.md). The default model is
**Qwen3-4B-Instruct-2507 Q4_K_M** (2.5 GB, Apache-2.0); the optional speech
model is **Whisper base.en q5_1** (60 MB, MIT, whisper.cpp). Both are pinned by
revision and SHA-256.

## Roadmap

* **Phase 5 (done):** SQLite memory (preferences, task history, workflows)
  with inspect/export/clear, spoken replies, push-to-talk voice input.
* **Phase 6:** Windows adapters ([`docs/WINDOWS.md`](docs/WINDOWS.md)),
  optional menu-bar UI.
  Packaging as an `.app` stays optional: Jarvis keeps working without
  installing an app.

## Windows

Not supported yet, and not tested on Windows. The platform boundary and plan
(service, shortcut, event loop, the Phase 4 tools, and Phase 5 memory and
voice) are in [`docs/WINDOWS.md`](docs/WINDOWS.md).

## Licence

Not chosen yet.

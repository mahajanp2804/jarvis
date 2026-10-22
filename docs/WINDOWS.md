# Windows support roadmap

**Status: not implemented and not tested on Windows.** `jarvis/platforms/windows.py`
is a placeholder. Every operation raises `UnsupportedPlatformError` with a pointer
to this document, so Windows users get a clear message instead of a crash. The
README won't claim Windows support until the [definition of done](#definition-of-done)
is met.

## Where things stand

The shared core (agent loop, planner, dispatcher, policy, config, memory,
audit log, tool logic) imports no OS-specific libraries. Each OS supplies a
`PlatformAdapter` (`jarvis/platforms/base.py`). There is evidence this boundary
holds: **CI runs the test suite on Linux, where Jarvis has no adapter at all,
and 615 tests pass with only the macOS-specific ones skipped** (see
`.github/workflows/ci.yml`).

What's left besides the adapters themselves: a short list of POSIX
assumptions in shared code, found by searching the source (October 2026):

| Shared code | POSIX assumption | Windows replacement |
|---|---|---|
| `runtime/lifecycle.py` (`InstanceLock`) | `fcntl.flock`, imported at module load | Named mutex (`CreateMutexW`), or `msvcrt.locking`; import lazily per OS |
| `runtime/control.py`, `runtime/lifecycle.py` | Unix socket, peer-uid check (`LOCAL_PEERCRED`/`SO_PEERCRED`), `os.getuid()` in the socket path | Named pipe with a current-user-only ACL (Python's `AF_UNIX` on Windows has no peer credentials) |
| `runtime/event_loop.py` (`SignalBridge`), `runtime/lifecycle.py` | `signal.set_wakeup_fd` + SIGTERM/SIGINT | `SetConsoleCtrlHandler` in a console; `WM_CLOSE`/`WM_ENDSESSION` to a hidden window for the background service |
| `tools/terminal.py` | `start_new_session=True`, `os.killpg` | `CREATE_NEW_PROCESS_GROUP` + `CTRL_BREAK_EVENT`, then `TerminateProcess`; a job object so child trees die with Jarvis |
| `voice_cli.py` (terminal push-to-talk) | `select()` on stdin | `msvcrt.kbhit()` polling (`select` only takes sockets on Windows) |
| logs, memory, prompt cache, control dir, browser profile, undo lists | POSIX modes `0600`/`0700` (silently ignored on Windows) | Rely on per-user `%APPDATA%`/`%LOCALAPPDATA%` ACLs, and verify them in `jarvis doctor` |
| `config.py` defaults | `terminal.search_path` (`/opt/homebrew/bin`, …), `files.blocked_paths` (`~/Library`) | Per-OS defaults: block `%APPDATA%`, `%LOCALAPPDATA%`, `C:\Windows`; case-insensitive path checks |
| `service_cli.py` | `PATH` for the launchd job | Task Scheduler action environment |

## Milestones

In order. Each milestone ends with tests passing on GitHub Actions'
`windows-latest`. Effort is a rough guess for one developer.

| # | Milestone | Done when | Effort |
|---|---|---|---|
| W0 | **Core runs on Windows.** Lazy-import `fcntl`; add a `windows-latest` CI job; skip OS-specific tests there like on Linux | The core test suite passes on Windows CI | 1–2 days |
| W1 | **Applications.** Start Menu `.lnk` and Store-app (`Get-StartApps`) discovery, launch via `ShellExecuteW` / `shell:AppsFolder`, running check, graceful quit with `WM_CLOSE`, process stats (`GetProcessMemoryInfo`) | `apps.*` integration tests open and quit Notepad on CI | 3–5 days |
| W2 | **Files and terminal.** Recycle Bin (`IFileOperation`), open/reveal, Windows Terminal, search fallback, Windows sandbox defaults, process groups and job objects, `.exe`/`.cmd` resolution, venv `Scripts\` | File, project and terminal tests pass on Windows CI | 3–5 days |
| W3 | **Background service.** Win32 message loop, `RegisterHotKey`, dialogs, named mutex, named-pipe control channel, shutdown messages, start at logon with Task Scheduler | Headless service tests pass on CI; shortcut and dialogs checked on a real Windows PC | 1–2 weeks |
| W4 | **Screen, speech and voice.** `Windows.Graphics.Capture` + `Windows.Media.Ocr`; SAPI speech output; the voice worker as-is (both libraries ship Windows wheels); `kbhit` push-to-talk | OCR and speech tests on CI; microphone checked on a real PC | 3–5 days |
| W5 | **Local AI.** llama-cpp-python on Windows (CPU wheel; CUDA or Vulkan optional), prompt cache file replacement under Windows file locking, model download paths | Real-model tests pass on a Windows PC | 2–4 days |
| W6 | **Install and docs.** `scripts/setup_windows.ps1`, install/uninstall, README section with tested Windows versions | A fresh Windows PC goes from clone to working shortcut by following the README | 2–3 days |

**Risks.** GitHub's Windows runners have a desktop session, but global
shortcuts, dialogs and the microphone still need a real machine (W3, W4).
Building llama-cpp-python needs Visual Studio Build Tools (W5). Windows
Defender may slow first runs of new executables. SmartScreen doesn't apply to
pip-installed scripts.

**Test strategy.** Keep the macOS rule: unit tests on fakes everywhere,
integration tests that only touch the real OS behind markers, and GUI tests
opt-in. CI gains a `windows-latest` job at W0 that grows with each milestone.

## Requirements (expected)

* Windows 10 22H2 or Windows 11, x64 or ARM64.
* Python 3.11+ from python.org (matching the CPU architecture).
* `llama-cpp-python`: needs Visual Studio Build Tools (C++ workload) and CMake to
  build from source, or a prebuilt CPU wheel. GPU options are CUDA (NVIDIA) or
  Vulkan. Metal is macOS-only, so the `GGML_METAL_NO_RESIDENCY` default doesn't
  apply. The model manager, planner and benchmark are platform-neutral.
  Process stats (`process_stats()`) need a Windows implementation, for example
  `GetProcessMemoryInfo`.
* Likely extra dependencies, installed only on Windows through environment markers
  (`; sys_platform == "win32"`): `psutil`, and possibly `pywin32` or plain
  `ctypes` for window messages.

## Adapter reference

| Interface | macOS (implemented) | Windows (plan) |
|---|---|---|
| Config / log / model paths | `~/Library/Application Support/Jarvis` (+ `models/`), `~/Library/Logs/Jarvis` | `%APPDATA%\Jarvis`, `%LOCALAPPDATA%\Jarvis\Logs`, `%LOCALAPPDATA%\Jarvis\models` (non-roaming; already in `paths.py`) |
| App discovery | Scan `.app` bundles and read `Info.plist` | Resolve Start Menu `.lnk` shortcuts (`%ProgramData%` and `%APPDATA%` Start Menu\Programs); Store/UWP apps via `Get-StartApps` (AppUserModelID) |
| Launch | `/usr/bin/open -a <bundle>` | `ShellExecuteW` / `os.startfile` on the shortcut, or `explorer.exe shell:AppsFolder\<AUMID>` for Store apps. Never `shell=True`. |
| Is running | `NSRunningApplication` by bundle ID | `psutil` processes matched by executable path |
| Graceful quit | `NSRunningApplication.terminate()` | Post `WM_CLOSE` to the app's top-level windows so it can prompt to save. `TerminateProcess` only as a separate HIGH-risk, always-confirmed action. |
| Diagnostics | arm64 / Rosetta, PyObjC | Architecture, Python bitness, Build Tools present |

### Background service (Phase 3 interfaces)

| Interface | macOS (implemented) | Windows (plan) |
|---|---|---|
| `EventLoop` | AppKit run loop (accessory app) | Win32 message loop (`GetMessage`) on the main thread; `call_soon` via `PostThreadMessage` |
| `HotkeyListener` | Carbon `RegisterEventHotKey` (no extra permission) | `RegisterHotKey` via `ctypes` (also needs no special permission); `WM_HOTKEY` in the message loop |
| `UserInterface` | `NSAlert` dialogs | `MessageBoxW` for confirm/show; a small Win32 or Tk input box for prompts |
| `ServiceManager` | launchd Launch Agent (`launchctl`) | Task Scheduler task "at log on" for the current user (`schtasks` or COM). Not a Windows Service: services run in session 0 and can't show dialogs or receive hotkeys. |
| Shutdown signal | SIGTERM via `SignalBridge` | `WM_CLOSE`/`WM_ENDSESSION` to a hidden window, plus `SetConsoleCtrlHandler` when run in a console |
| Control channel | Unix socket (0600, peer uid check) | Named pipe with an ACL for the current user only (AF_UNIX on Windows lacks peer-credential checks) |
| Single instance | `flock` | Named mutex (`CreateMutexW`) |

The headless mode (`jarvis daemon --headless`) uses only the portable
`QueueEventLoop`, so it is the natural first target on Windows. `SignalBridge`
relies on POSIX `set_wakeup_fd` semantics, and Windows needs the console
control handler above instead.

### Computer automation (Phase 4 interfaces)

| Interface | macOS (implemented) | Windows (plan) |
|---|---|---|
| `find_by_name` | Spotlight (`mdfind`) | Windows Search (`SystemIndex` via OLE DB/ADO) or a time-limited scan (the shared fallback already exists) |
| `move_to_trash` | `NSFileManager.trashItemAtURL` | `SHFileOperationW` / `IFileOperation` with `FOF_ALLOWUNDO` (Recycle Bin) |
| `open_path` / `open_terminal` | `/usr/bin/open` | `ShellExecuteW`; Windows Terminal (`wt.exe -d <folder>`) |
| Path sandbox defaults | `~/Library` blocked | Block `%APPDATA%`, `%LOCALAPPDATA%`, `C:\Windows`; case-insensitive comparisons |
| Terminal tools | POSIX process groups, `killpg` | `CREATE_NEW_PROCESS_GROUP` + `CTRL_BREAK_EVENT`, then `TerminateProcess`; `.exe`/`.cmd` resolution; venv in `Scripts\` |
| `capture_screen` / `recognize_text` | `screencapture` + Apple Vision | `Windows.Graphics.Capture` + `Windows.Media.Ocr` (on-device) |
| Browser tools | Playwright with installed Chrome/Edge | Same code; Edge is preinstalled (`channel="msedge"`) |

### Memory and voice (Phase 5 interfaces)

| Interface | macOS (implemented) | Windows (plan) |
|---|---|---|
| Memory database | `~/Library/Application Support/Jarvis/memory.sqlite3`, `0600` | `%APPDATA%\Jarvis\memory.sqlite3` (already in `paths.py`); restrict with an ACL for the current user, since POSIX modes don't apply |
| `SpeechOutput` | `SaySpeaker`: `/usr/bin/say -f -` (text on stdin), one child process per reply | `System.Speech` / SAPI `SpVoice` through `comtypes` or `pywin32`, or WinRT `Windows.Media.SpeechSynthesis`; same non-blocking `speak`/`stop` contract |
| Voice worker | `python -m jarvis.voice.stt_worker`: `sounddevice` (PortAudio) + `pywhispercpp` | Same code: both have Windows wheels. PortAudio uses WASAPI. Whisper on the CPU by default (CUDA/Vulkan builds optional) |
| Push-to-talk shortcut | Second Carbon hotkey, filtered by hotkey id | Second `RegisterHotKey` id on the same message window |
| `UserInterface.listen` / `edit` | `NSAlert` closed by a modal-mode timer; prefilled text field | A small Win32/Tk dialog with Done/Cancel closed from a timer; an edit box prefilled with the transcript |
| Terminal push-to-talk | `select()` on stdin while recording | `msvcrt.kbhit()` polling (Windows `select` only works on sockets) |
| Microphone permission | macOS asks per responsible app (terminal, or Python for the service) | Settings → Privacy → Microphone → "Let desktop apps access your microphone" |

## Definition of done

1. All unit tests pass on Windows (GitHub Actions `windows-latest`).
2. Windows integration tests exist (read-only discovery; opt-in GUI open/quit of
   Notepad) and pass on a real machine.
3. `jarvis doctor` reports Windows-specific checks.
4. The README states the tested Windows versions. Until then, the README says
   Windows is unsupported.

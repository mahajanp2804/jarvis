# Windows support plan

**Status: not implemented and not tested on Windows.** `jarvis/platforms/windows.py`
is a placeholder. Every operation raises `UnsupportedPlatformError` with a pointer
to this document, so Windows users get a clear message instead of a crash.

The shared core (agent, dispatcher, policy, config, audit log, tools) imports no
OS-specific libraries. Adding Windows means implementing the adapters below. The
core does not need rewriting.

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

## Adapter plan

| Interface | macOS (implemented) | Windows (plan) |
|---|---|---|
| Config / log / model paths | `~/Library/Application Support/Jarvis` (+ `models/`), `~/Library/Logs/Jarvis` | `%APPDATA%\Jarvis`, `%LOCALAPPDATA%\Jarvis\Logs`, `%LOCALAPPDATA%\Jarvis\models` (non-roaming; already in `paths.py`) |
| App discovery | Scan `.app` bundles and read `Info.plist` | Resolve Start Menu `.lnk` shortcuts (`%ProgramData%` and `%APPDATA%` Start Menu\Programs); Store/UWP apps via `Get-StartApps` (AppUserModelID) |
| Launch | `/usr/bin/open -a <bundle>` | `ShellExecuteW` / `os.startfile` on the shortcut, or `explorer.exe shell:AppsFolder\<AUMID>` for Store apps. Never `shell=True`. |
| Is running | `NSRunningApplication` by bundle ID | `psutil` processes matched by executable path |
| Graceful quit | `NSRunningApplication.terminate()` | Post `WM_CLOSE` to the app's top-level windows so it can prompt to save. `TerminateProcess` only as a separate HIGH-risk, always-confirmed action. |
| Diagnostics | arm64 / Rosetta, PyObjC | Architecture, Python bitness, Build Tools present |

### Later phases

* **Hotkeys (Phase 3):** `RegisterHotKey` via `ctypes` on a message-loop thread
  (or `pynput`, if it is chosen for macOS too).
* **Start at login (Phase 3):** a Task Scheduler task triggered "at log on" for
  the current user, or a Startup-folder shortcut. Not a Windows Service: services
  run in session 0 and cannot interact with the user's desktop.
* **Speech (Phase 5):** Windows SAPI / `Windows.Media.SpeechSynthesis` for output.
  The local speech-to-text engine chosen in Phase 5 for input.
* **Browser automation (Phase 4):** Playwright is cross-platform and needs no
  adapter beyond browser paths.
* **Accessibility (Phase 4+):** UI Automation (UIA) instead of the macOS AX API.

## Definition of done for Windows

1. All unit tests pass on Windows (GitHub Actions `windows-latest`).
2. Windows integration tests exist (read-only discovery; opt-in GUI open/quit of
   Notepad) and pass on a real machine.
3. `jarvis doctor` reports Windows-specific checks.
4. The README states the tested Windows versions. Until then, the README says
   Windows is unsupported.

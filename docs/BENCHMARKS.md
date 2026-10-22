# Benchmarks

Every number here was measured on one machine and can be reproduced with the
commands shown. Absolute times will differ on other Macs; the comparisons
(loaded vs unloaded, CPU vs GPU, cached vs not) should hold.

**Machine:** MacBook Air M4, 16 GB, macOS 27.0.1, on mains power.
**Software:** Jarvis 0.6.0, Python 3.14.3 (Homebrew, arm64), llama-cpp-python
0.3.36 (Metal), Qwen3-4B-Instruct-2507 Q4_K_M, context 4096; pywhispercpp 1.5.1
with Whisper base.en q5_1. Exact package versions: `constraints-macos.txt`.
**Date:** 22–23 October 2026.

## How to reproduce

```bash
jarvis service measure --seconds 60   # the running service, sampled from outside
jarvis benchmark --json bench.json    # model: load, unload, planning requests, idle cost
jarvis voice benchmark                # speech-to-text: accuracy and latency, no microphone
jarvis voice benchmark --gpu          # the same on the GPU (Metal)
jarvis voice benchmark --voice Karen  # the same with another accent
pytest                                # the test suite (timing below)
```

`jarvis benchmark` only plans (dry run, all confirmations denied), so nothing on
your Mac changes. It uses an empty temporary prompt cache, so it always starts
cold. `jarvis voice benchmark` renders test sentences to files with `say`, so
nothing is played aloud.

## What the numbers mean

* **CPU %** is CPU time divided by wall time, from the kernel's per-process
  accounting (`proc_pid_rusage`). It excludes GPU work.
* **Wakeups/s** counts interrupt and idle wakeups (`ri_pkg_idle_wkups` +
  `ri_interrupt_wkups`). This is the main input to macOS's Energy Impact.
* **Footprint** is macOS's `phys_footprint`: memory the process owns. **RSS**
  also counts memory-mapped file pages, such as the 2.4 GB of model weights.
  macOS can drop those pages at any time, so RSS overstates the real cost of a
  loaded model.
* **Battery** isn't measured directly. Use the proxies above, or
  `sudo powermetrics --samplers cpu_power,gpu_power -i 1000` while using Jarvis.

## Background service

`jarvis service measure --seconds 60`, the installed Launch Agent, sampled from
outside:

| State | CPU | Wakeups/s | Footprint |
|---|---|---|---|
| Just started (nothing used yet) | n/a | n/a | 48–49 MB |
| Idle, model loaded in its worker (service + worker) | 0.002% | 0.60 | 893 MB (RSS 3.3 GB, mostly mapped weights) |
| Idle, after the 300 s idle unload, memory database in use | 0.000% | 0.00 | 55 MB |

Push-to-talk and spoken replies add no idle cost: the voice worker and `say`
exist only during a request.

## Language model

`jarvis benchmark` (2 rounds over 4 prompts; system prompt 1,905 tokens with
the 29 default tools):

| | Result |
|---|---|
| Rule-based command (no model) | 28.5 ms |
| Model load, first in the process / reload | 1.59 s / 0.41 s |
| First request with a new system prompt (reads all 1,905 tokens, then saves the state to disk) | 8.73 s (6.17 s reading the prompt, 1.73 s generating, 0.24 s saving) |
| First request after a reload (1,887 prompt tokens restored from disk in 0.07 s) | 2.53 s |
| Later requests (prompt already in memory) | 1.51–1.67 s; 2.40 s for a two-step plan |
| Generation speed | 28 tokens/s |
| Unload | 0.09 s; footprint back to 41 MB |
| Idle with the model loaded (benchmark process + worker) | 0.04% CPU, 0.8 wakeups/s, 743 MB footprint |
| Saved prompt state on disk | 265 MB (1 file) |

A "new system prompt" happens once after installing, and again whenever the
enabled tools, the installed-app list, the model or llama-cpp-python change.
Every later model load restores the saved state. Phase 4 measured 2.41 s for
the first request after a reload with 25 tools and a 1,749-token prompt; the
four memory tools and one rule account for the difference.

The prompt cache's own before/after comparison is in the README, under
[Prompt cache on disk](../README.md#performance-and-trade-offs).

## Speech-to-text (push-to-talk)

`jarvis voice benchmark`: 6 test commands, each synthesized with a system voice
and transcribed by a fresh worker process. Wall time covers the whole worker:
process start, imports, model load and transcription. Recording time isn't
included.

| Voice (accent) | Whisper on | Exact | Wall / request | Transcription | CPU / request | Peak worker memory |
|---|---|---|---|---|---|---|
| Rishi (Indian English; this Mac's system voice) | CPU | 6/6 | 0.44 s | 0.20 s | 0.83 s | 242 MB |
| Rishi | GPU (Metal) | 6/6 | 0.34 s | 0.10 s | 0.18 s | 222 MB |
| Samantha (US English) | CPU | 6/6 | 0.43 s | 0.20 s | 0.83 s | 242 MB |
| Karen (Australian English) | CPU | 4/6 | 0.44 s | 0.20 s | 0.84 s | 242 MB |

Karen's two misses: "open my stock price, predict a project and run it" and
"run the morning setup work flow". Workflow commands now accept "work flow".
The first miss still reaches the planner as an ordinary request, and you see
and can correct every transcript before it runs.

**Why the CPU is the default:** Metal is 0.1 s faster per request, but it
compiles its shaders again (about 7 s, measured once) whenever macOS has reset
its shader cache. `[voice] stt_use_gpu = true` switches.

**Limits of this benchmark:** synthetic speech is clean and evenly paced, so
these accuracies are an upper bound for a real microphone in a real room. No
live-microphone measurements yet.

## Memory and workflows

| | Result |
|---|---|
| Remember / recall / forget / list workflows (rule-based) | instant, no model |
| Replaying a saved one-step workflow through the service | 0.21 s |
| Database after a few dozen tasks | 32 KB |

## Test suite

| Where | Tests | Time |
|---|---|---|
| This Mac (`pytest`) | 637 passed, 16 skipped (opt-in model/GUI tests) | ~13 s |
| CI, macOS 15 arm64, Python 3.11 / 3.14 | full suite + lint + types + packaging | ~1.7 / 2.0 min per job |
| CI, Linux, Python 3.12 (shared core only) | 614+ passed; macOS-only tests skipped | ~1 min |
| Real-model tests (`JARVIS_RUN_MODEL_TESTS=1 pytest -m model`) | 12 | ~70 s |

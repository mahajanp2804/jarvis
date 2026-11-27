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

## Planner accuracy

`jarvis eval` plans fixed test requests with the real model (first round
only; nothing is executed) and checks the action, tools and key arguments.
Siri-style requests Jarvis has no tool for yet are expected to be declined;
when a phase adds the tool, its cases expect the tool instead. The test set
lives in `src/jarvis/evals/planner_cases.toml`. It plans with the model only:
phrasings the rule-based router handles never reach it in real use.

| Category | Baseline (0.6.1, 29 tools, 92 requests) | 0.8.0 (43 tools, 101 requests) | 0.9.0 (55 tools, 107 requests) |
|---|---|---|---|
| apps | 13/13 | 12/13 | 12/13: lists apps for "open system settings" |
| files | 15/15 | 15/15 | 14/15: lists the folder before organizing it |
| memory | 5/5 | 5/5 | 5/5 |
| chat (general questions) | 8/8 | 8/8 | 8/8 |
| vague (must ask) | 7/7 | 6/7 | 5/7: looks for "my project"; plans a quit for a bare "quit" (which asks before quitting) |
| safety (secrets, `sudo rm`, injected instructions) | 5/5 | 5/5 | 5/5 |
| projects and terminal | 11/14 | 11/14 | 11/14: plans `projects.find` or `files.list` before plain commands such as `git status` |
| browser | 8/10 | 7/9 | 7/9: declines YouTube and the bitcoin price |
| siri | 8/15 (no tools: should decline) | 20/25 | 26/31: texts, reminders, the calendar, notes and contacts use their tools. Fails: looks the manager up before emailing, checks the calendar before adding lunch, asks "text or call?" for "call dad", plans `files.search` for "mark buy milk as done", asks what "what song is this?" means |
| **overall** | **80/92 (87%)** | **89/101 (88%)** | **93/107 (87%)** |

Planning time (0.9.0): 10.6 s for the first request (it reads the system
prompt), then a median of 1.8 s. A real-model test fails if the score drops
below 85%. "skip this song", "what song is this?" and "mark buy milk as done"
are also router phrasings, so they run correctly without the model.

The first 0.9.0 run scored 91/107 (85%): a Phase 7 example declining "send
an email to Sam" and a rule naming email and messages as impossible taught
the model to ask "which messaging app?". Replacing them (an example that
texts with `messages.send`, a declined phone call) gave 93/107.

## Tool profiles (`[ai] tool_profiles`, off by default)

With profiles on, a request sees only the tools of one of four fixed profiles
(apps, files, web, dev), chosen by keywords, or every tool when nothing
matches (see `src/jarvis/agent/tool_groups.py`). Measured on the 92-request
test set and on requests that alternate between profiles:

| | Profiles off (today) | Profiles on |
|---|---|---|
| `jarvis eval` score | 80/92 (87%) | 80/92 (87%): browser +1, projects -1 |
| `jarvis eval` planning time | first 9.6 s, median 2.4 s | first 5.0 s, median 1.6 s |
| System prompt (19 test apps) | 1,670 tokens | 755 (apps) / 1,143 (files) / 1,301 (web) / 1,528 (dev) / 1,670 (all) |
| Requests alternating profiles, 512 MB cache cap | n/a | 3.0-5.3 s each: the saved prompts evict each other, so every request re-reads its prompt |
| The same with a 1.2 GB cap | 1.5-2.4 s | first use of each profile 4.3-5.3 s; afterwards 1.3-2.1 s (state restored from disk) |
| Saved prompt state on disk | ~250 MB (1 prompt) | 887 MB (5 prompts) |

**Decision (Nov 2026): off for now.** With 29 tools the gain is small (no
accuracy change, ~0.3 s faster per request), and it costs about 640 MB more
disk plus a slow first request per profile whenever the prompt changes (for
example after installing an app). It's the plan for Phases 8-9, when new tools
would push the full prompt towards the 4,096-token context: switch it on and
re-run `jarvis eval --profiles on`. The cache cap default is now 1 GB so the
profiles never evict each other.

**Phase 8 re-measurement (0.8.0, 43 tools, 101 requests, Nov 2026):** the
everyday profile (apps, system, music, weather, timers, Shortcuts, memory)
was added, and profiles were compared again before the wrong-setting guard:

| | Profiles off | Profiles on |
|---|---|---|
| `jarvis eval` score | 88/101 (87%) | 89/101 (88%): apps +1, browser +1, vague +1, projects -1, siri -1 |
| Planning time | first 10.2 s, median 2.0 s | first 6.2 s, median 2.9 s, slowest 15.3 s (each profile's first use reads its prompt) |

**Decision: still off.** The score difference is one request, and switching
between six saved prompts made the median slower on a mixed workload. The
prompt-size problem profiles were meant to solve is handled by the larger
context window instead (below).

## Context window (`[ai] context_length`)

With 43 tools and the installed-app list the system prompt is 2,353 tokens.
At 4,096 tokens that left about 1,300 for the conversation and tool results,
so one long result no longer fitted in full. `jarvis benchmark` on the same
Mac (M4, 16 GB), 4,096 vs 8,192:

| | 4,096 | 8,192 |
|---|---|---|
| Planning, prompt cached | 1.58-1.77 s | 1.54-1.74 s |
| First request after a reload | 2.69 s | 2.63 s |
| Generation | 27 tokens/s | 27-28 tokens/s |
| Footprint, model loaded | 744 MB | 1,318 MB |
| Footprint, model unloaded | 38 MB | 38 MB |

**Decision: 8,192 by default** (0.8.0): same speed, about 0.6 GB more only
while the model is loaded; it's freed after the idle timeout. Set 4,096 to
save that memory.

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
| This Mac (`pytest`) | 867 passed, 18 skipped (opt-in model/GUI tests) | ~21 s |
| CI, macOS 15 arm64, Python 3.11 / 3.14 | 864 passed + lint + types + packaging | ~1.5 / 2 min per job |
| CI, Linux, Python 3.12 (shared core only) | 836 passed; macOS-only tests skipped | ~45 s |
| Real-model tests (`JARVIS_RUN_MODEL_TESTS=1 pytest -m model`) | 14 (incl. the 101-request eval) | ~7 min |

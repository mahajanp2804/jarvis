# Models

Jarvis runs GGUF language models locally through
[`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python), with no
Ollama, no server and no cloud. The model only plans. Every action it proposes
goes through the same validation, permission policy and confirmations as any
other command.

* Models are **never downloaded automatically**. You run `jarvis model download`,
  which shows the size, licence and destination and asks first.
* The default location is `~/Library/Application Support/Jarvis/models/`, which is
  outside the git checkout and outside any iCloud-synced Desktop/Documents folder.
  To use another file, set `[ai] model_path` in `config.toml` (or
  `JARVIS_MODEL_PATH`).
* This repository's `models/` folder is git-ignored apart from this README, if
  you prefer to keep files here. Never commit weights.

## Default model: Qwen3-4B-Instruct-2507 (Q4_K_M)

| | |
|---|---|
| File | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`, 2,497,281,120 bytes |
| Source | [`unsloth/Qwen3-4B-Instruct-2507-GGUF`](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF), pinned to revision `a06e946b` |
| SHA-256 | `3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597` |
| Licence | Apache-2.0 |
| Why | Reliable instruction following and JSON tool selection for its size. The non-"thinking" variant answers directly, so latency stays low. |

```bash
jarvis model download          # resumable; verifies SHA-256 before use
jarvis model check --verify    # re-check an existing file
jarvis model info
```

Manual alternative: download the file in a browser from the repository page,
place it in `~/Library/Application Support/Jarvis/models/`, then run
`jarvis model check --verify`.

### Measured on a MacBook Air M4 (16 GB), context 4096, Metal

| | Value |
|---|---|
| Memory while loaded | 0.72 GB footprint (KV cache + buffers) + 2.4 GB memory-mapped weights (clean file cache macOS can reclaim). RSS 3.15 GB. |
| Memory after unloading | about 130 MB above the pre-AI baseline (Metal and llama.cpp runtime state stays in the process) |
| Load / reload | 0.31 s / 0.19 s (weights are memory-mapped, so the real cost moves to the first request) |
| First request after loading | ~4.1 s (2.3 s to evaluate the ~800-token instructions, then generation) |
| Later requests (prompt cached) | 1.2–1.8 s, ~32 tokens/s generation |

Run `jarvis benchmark` to measure on your machine. See the main README for the
idle-cost and battery trade-offs.

## Speech model (optional, for push-to-talk): Whisper base.en (q5_1)

| | |
|---|---|
| File | `ggml-base.en-q5_1.bin`, 59,721,011 bytes |
| Source | [`ggerganov/whisper.cpp`](https://huggingface.co/ggerganov/whisper.cpp), pinned to revision `5359861c` |
| SHA-256 | `4baf70dd0d7c4247ba2b81fafd9c01005ac77c2f9ef064e00dcf195d0e2fdd2f` |
| Licence | MIT |
| Why | Small (60 MB), English, accurate on short commands: 6/6 synthesized test commands transcribed exactly, ~0.2 s each on an M4 CPU |

```bash
jarvis voice download          # asks first; verifies SHA-256
jarvis voice transcribe x.wav  # check it on a 16 kHz mono WAV
```

It runs in a worker process that exists only during a voice request. For
other languages, set `[voice] stt_model_path` to a multilingual whisper.cpp
model (e.g. `ggml-base-q5_1.bin`) and `[voice] language`. Other models aren't
verified by Jarvis. `ggml-small.en-q5_1.bin` (190 MB) is a step up in accuracy
if base.en mishears you.

## Other models (untested with Jarvis)

Any instruction-tuned GGUF model with a chat template should work through
`[ai] model_path`, but planning quality varies. These were *not* tested:

| Model (Q4_K_M) | File | Licence |
|---|---|---|
| Llama-3.2-3B-Instruct | 2.02 GB | Llama 3.2 Community Licence |
| Qwen2.5-3B-Instruct | 2.10 GB | Qwen Research Licence (non-commercial) |

If a model's built-in chat template doesn't work with llama-cpp-python, set
`[ai] chat_format` (for example `"chatml"` for Qwen models).

## Installing llama-cpp-python

`scripts/setup_macos.sh` installs it. PyPI ships it as source only, so pip
compiles llama.cpp locally (1–4 minutes on an M4). You need Xcode Command Line
Tools (`xcode-select --install`) and an **arm64** Python (an x86_64/Rosetta
Python builds without Metal). To install it by hand:

```bash
CMAKE_ARGS="-DGGML_METAL=on" python -m pip install -e ".[llm]"
```

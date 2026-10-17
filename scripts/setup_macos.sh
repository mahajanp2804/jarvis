#!/usr/bin/env bash
# Create the Jarvis virtual environment on Apple Silicon macOS and install
# Jarvis in editable mode with development tools and local AI (llama-cpp-python,
# compiled from source with Metal; needs Xcode Command Line Tools, ~1-4 min).
#
# Usage: scripts/setup_macos.sh            (picks a suitable Python)
#        JARVIS_PYTHON=/path/to/python3 scripts/setup_macos.sh
#        JARVIS_SKIP_LLM=1 scripts/setup_macos.sh   (rule-based commands only)
#
# This script never downloads a model. Run `jarvis model download` for that.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_DIR}/.venv"

die() { echo "error: $*" >&2; exit 1; }

[[ "$(uname -s)" == "Darwin" ]] || die "this script is for macOS. See docs/WINDOWS.md for Windows."
if [[ "$(/usr/sbin/sysctl -in sysctl.proc_translated 2>/dev/null || echo 0)" == "1" ]]; then
    die "this shell is running under Rosetta (x86_64). Open a native arm64 terminal."
fi
[[ "$(uname -m)" == "arm64" ]] || echo "warning: not Apple Silicon ($(uname -m)); untested." >&2

is_suitable() {
    # Python >= 3.11 whose interpreter runs natively as arm64.
    "$1" -c 'import platform, sys; sys.exit(0 if sys.version_info >= (3, 11) and platform.machine() == "arm64" else 1)' 2>/dev/null
}

PYTHON="${JARVIS_PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
    # Homebrew's Pythons are arm64-only; prefer them over universal2 or conda builds.
    for candidate in /opt/homebrew/bin/python3.14 /opt/homebrew/bin/python3.13 \
                     /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 \
                     /opt/homebrew/bin/python3 python3; do
        if command -v "${candidate}" >/dev/null 2>&1 && is_suitable "${candidate}"; then
            PYTHON="$(command -v "${candidate}")"
            break
        fi
    done
fi
[[ -n "${PYTHON}" ]] || die "no arm64 Python >= 3.11 found. Install one with: brew install python@3.14"
is_suitable "${PYTHON}" || die "${PYTHON} is not an arm64 Python >= 3.11."

EXTRAS="dev,llm"
if [[ "${JARVIS_SKIP_LLM:-0}" == "1" ]]; then
    EXTRAS="dev"
elif ! xcode-select -p >/dev/null 2>&1; then
    die "Xcode Command Line Tools are needed to build llama-cpp-python. Run: xcode-select --install
       (or set JARVIS_SKIP_LLM=1 to install without local AI)"
fi

echo "Using ${PYTHON} ($("${PYTHON}" --version))"
if [[ ! -d "${VENV_DIR}" ]]; then
    "${PYTHON}" -m venv "${VENV_DIR}"
fi
"${VENV_DIR}/bin/python" -m pip install --upgrade pip >/dev/null
# Metal is llama.cpp's default on Apple Silicon; stated explicitly for clarity.
CMAKE_ARGS="${CMAKE_ARGS:--DGGML_METAL=on}" \
    "${VENV_DIR}/bin/python" -m pip install -e "${REPO_DIR}[${EXTRAS}]"

echo
echo "Installed. Next:"
echo "  source .venv/bin/activate"
echo "  jarvis doctor"
if [[ "${EXTRAS}" == *llm* ]]; then
    echo "  jarvis model download      # 2.5 GB, asks before downloading"
fi

#!/usr/bin/env bash
# Install Jarvis as a per-user Launch Agent: it starts at login and runs in the
# background. Press ⌃⌥⌘J (configurable) to open it.
#
# Usage: scripts/install_macos.sh [--config /path/to/config.toml]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [[ ! -x "${JARVIS}" ]]; then
    echo "No virtualenv yet; running scripts/setup_macos.sh first."
    "${REPO_DIR}/scripts/setup_macos.sh"
fi
require_jarvis
"${JARVIS}" "$@" doctor || echo "warning: doctor reported problems (see above); installing anyway." >&2
"${JARVIS}" "$@" service install

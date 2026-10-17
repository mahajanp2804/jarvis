#!/usr/bin/env bash
# Stop Jarvis and remove the Launch Agent. Configuration, logs and downloaded
# models are kept; the command prints where they are if you want to delete them.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_jarvis
"${JARVIS}" "$@" service uninstall

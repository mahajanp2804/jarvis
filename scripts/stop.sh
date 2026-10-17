#!/usr/bin/env bash
# Stop the Jarvis service gracefully (it starts again at next login).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_jarvis
"${JARVIS}" "$@" service stop

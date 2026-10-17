#!/usr/bin/env bash
# Start the installed Jarvis service.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_jarvis
"${JARVIS}" "$@" service start

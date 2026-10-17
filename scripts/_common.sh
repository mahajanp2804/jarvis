# Shared helpers for the service scripts (sourced, not executed).
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JARVIS="${REPO_DIR}/.venv/bin/jarvis"

die() { echo "error: $*" >&2; exit 1; }

require_jarvis() {
    [[ "$(uname -s)" == "Darwin" ]] || die "these scripts are for macOS (see docs/WINDOWS.md)."
    [[ -x "${JARVIS}" ]] || die "Jarvis isn't installed in ${REPO_DIR}/.venv. Run scripts/setup_macos.sh first."
}

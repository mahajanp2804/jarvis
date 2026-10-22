#!/usr/bin/env bash
# Build the Jarvis wheel and check it works outside this checkout:
#   1. build the wheel (and an sdist) from the current tree;
#   2. check what's inside: the jarvis package and nothing else (no tests, no
#      local data), and the version matches jarvis.__version__;
#   3. install it into a fresh virtualenv with a throwaway HOME, so your real
#      config, memory and logs are never read or written;
#   4. smoke-test the installed `jarvis` command.
#
# Usage: scripts/check_package.sh [python]     (default: the project's .venv Python)
# Used by CI on macOS and Linux. Downloads only the runtime dependencies (pydantic,
# and PyObjC on macOS) from PyPI; never a model.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${1:-${REPO_DIR}/.venv/bin/python}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/jarvis-package.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

die() { echo "error: $*" >&2; exit 1; }

echo "== build"
"${PYTHON}" -m pip wheel --no-deps --quiet --wheel-dir "${WORK}/dist" "${REPO_DIR}"
WHEEL="$(ls "${WORK}"/dist/jarvis_assistant-*.whl)"
echo "built $(basename "${WHEEL}")"

echo "== contents"
"${PYTHON}" - "${WHEEL}" "${REPO_DIR}" <<'EOF'
import re
import sys
import zipfile
from pathlib import Path

wheel, repo = sys.argv[1], Path(sys.argv[2])
names = zipfile.ZipFile(wheel).namelist()
expected = re.search(r'__version__ = "([^"]+)"', (repo / "src/jarvis/__init__.py").read_text())[1]
assert f"jarvis_assistant-{expected}" in wheel, f"wheel name doesn't carry version {expected}"
code = sorted(n for n in names if not n.startswith("jarvis_assistant-"))
outside = [n for n in code if not n.startswith("jarvis/")]
assert not outside, f"unexpected files in the wheel: {outside}"
sources = sorted(str(p.relative_to(repo / "src")) for p in (repo / "src/jarvis").rglob("*.py"))
missing = sorted(set(sources) - set(code))
assert not missing, f"modules missing from the wheel: {missing}"
banned = [n for n in names if re.search(r"\.(gguf|bin|sqlite3?|db|jsonl|log)$|__pycache__", n)]
assert not banned, f"data or caches in the wheel: {banned}"
print(f"{len(code)} files, all under jarvis/; version {expected}")
EOF

echo "== install into a fresh virtualenv"
"${PYTHON}" -m venv "${WORK}/venv"
"${WORK}/venv/bin/python" -m pip install --quiet --upgrade pip
"${WORK}/venv/bin/python" -m pip install --quiet "${WHEEL}"
mkdir -p "${WORK}/home"
run() {  # isolated: throwaway HOME, no inherited Jarvis settings, not from the repo folder
    (cd "${WORK}" && env -u JARVIS_CONFIG -u JARVIS_MODEL_PATH HOME="${WORK}/home" \
        JARVIS_LOG_DIR="${WORK}/logs" "${WORK}/venv/bin/jarvis" "$@")
}

echo "== smoke test"
version="$(run --version)"
echo "${version}"
[[ "${version}" == "jarvis $("${PYTHON}" -c 'import re,sys; print(re.search(r"__version__ = \"([^\"]+)\"", open(sys.argv[1]).read())[1])' "${REPO_DIR}/src/jarvis/__init__.py")" ]] \
    || die "installed version doesn't match the source"
run config show >/dev/null 2>&1 || die "jarvis config show failed"
run memory show | grep -q "not created yet" || die "jarvis memory show failed"
run doctor || true  # the model and optional extras are absent here; doctor must still run
if [[ "$(uname -s)" == "Darwin" ]]; then
    # Rule-based and read-only: lists real installed apps, no model needed.
    JARVIS_AI_ENABLED=false run run "list apps matching calculator" | grep -qi "calculator" \
        || die "jarvis run (rule-based) failed"
fi
echo "OK: the wheel installs and runs outside the checkout."

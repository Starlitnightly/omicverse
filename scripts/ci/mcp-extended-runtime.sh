#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

INSTALL=false
PYTEST_EXTRA_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--install" ]]; then
        INSTALL=true
    else
        PYTEST_EXTRA_ARGS+=("$arg")
    fi
done

echo "=== MCP CI Profile: extended-runtime ==="
echo "Scope: extended-tier tests (requires SEACells, pertpy, mira)"

if [[ "$INSTALL" == true ]]; then
    echo "--- Installing dependencies ---"
    pip install -e "$REPO_ROOT[tests,mcp]"
    pip install -r "$REPO_ROOT/requirements/mcp-extended-runtime.txt"
fi

python -m pytest tests/mcp/ -m "extended" -v --tb=short "${PYTEST_EXTRA_ARGS[@]}"

# --- Version snapshot (non-blocking) ---
echo "--- Generating version snapshot ---"
mkdir -p "$REPO_ROOT/.ci-artifacts"
SOURCE_FLAG="local"
if [[ "${CI:-}" == "true" ]]; then
    SOURCE_FLAG="ci"
fi
python "$REPO_ROOT/scripts/ci/mcp-report-versions.py" \
    --profile extended-runtime \
    --source "$SOURCE_FLAG" \
    --output "$REPO_ROOT/.ci-artifacts/mcp-extended-runtime-versions.json" \
    || true

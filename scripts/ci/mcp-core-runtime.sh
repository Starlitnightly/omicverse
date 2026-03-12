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

echo "=== MCP CI Profile: core-runtime ==="
echo "Scope: core real-runtime tests (requires anndata + scanpy)"

if [[ "$INSTALL" == true ]]; then
    echo "--- Installing dependencies ---"
    pip install -e "$REPO_ROOT[tests,mcp]"
fi

PYTEST_EXIT=0
python -m pytest tests/mcp/ -m "core" -v --tb=short "${PYTEST_EXTRA_ARGS[@]}" || PYTEST_EXIT=$?

# --- Version snapshot (always runs, even if tests fail) ---
echo "--- Generating version snapshot ---"
mkdir -p "$REPO_ROOT/.ci-artifacts"
SOURCE_FLAG="local"
if [[ "${CI:-}" == "true" ]]; then
    SOURCE_FLAG="ci"
fi
python "$REPO_ROOT/scripts/ci/mcp-report-versions.py" \
    --profile core-runtime \
    --source "$SOURCE_FLAG" \
    --output "$REPO_ROOT/.ci-artifacts/mcp-core-runtime-versions.json" \
    || true

exit $PYTEST_EXIT

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

echo "=== MCP CI Profile: fast-mock ==="
echo "Scope: mock tests only (no real dependencies)"

if [[ "$INSTALL" == true ]]; then
    echo "--- Installing dependencies ---"
    pip install -e "$REPO_ROOT[tests,mcp]"
fi

python -m pytest tests/mcp/ -m "not real_runtime" -v --tb=short "${PYTEST_EXTRA_ARGS[@]}"

# --- Version snapshot (non-blocking) ---
echo "--- Generating version snapshot ---"
mkdir -p "$REPO_ROOT/.ci-artifacts"
SOURCE_FLAG="local"
if [[ "${CI:-}" == "true" ]]; then
    SOURCE_FLAG="ci"
fi
python "$REPO_ROOT/scripts/ci/mcp-report-versions.py" \
    --profile fast-mock \
    --source "$SOURCE_FLAG" \
    --output "$REPO_ROOT/.ci-artifacts/mcp-fast-mock-versions.json" \
    || true

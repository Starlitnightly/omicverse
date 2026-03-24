#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
task_root="${TAIWAN_REMOTE_TASK_ROOT:-$(cd -- "${repo_root}/.." && pwd)}"
env_dir="${OV_NGAGENT_REMOTE_ENV_DIR:-${task_root}/.ngagent-venv}"
bootstrap_stamp="${env_dir}/.bootstrap-fingerprint"

if [[ -n "${OV_NGAGENT_REMOTE_BASE_PYTHON:-}" ]]; then
  base_python="${OV_NGAGENT_REMOTE_BASE_PYTHON}"
  use_system_site_packages=0
elif [[ -x "${HOME}/micromamba/envs/aliyawak/bin/python" ]]; then
  base_python="${HOME}/micromamba/envs/aliyawak/bin/python"
  use_system_site_packages=1
else
  base_python="${OV_NGAGENT_REMOTE_FALLBACK_PYTHON:-python3}"
  use_system_site_packages=0
fi

if ! command -v "${base_python}" >/dev/null 2>&1 && [[ ! -x "${base_python}" ]]; then
  echo "Base Python for Taiwan review is unavailable: ${base_python}" >&2
  exit 1
fi

current_fingerprint="$("${base_python}" - <<'PY'
from pathlib import Path
import hashlib

paths = [
    Path("pyproject.toml"),
]
h = hashlib.sha256()
for path in paths:
    if path.exists():
        h.update(path.as_posix().encode("utf-8"))
        h.update(path.read_bytes())
print(h.hexdigest())
PY
)"

needs_bootstrap=0
if [[ ! -x "${env_dir}/bin/python" ]]; then
  needs_bootstrap=1
elif [[ ! -f "${bootstrap_stamp}" ]]; then
  needs_bootstrap=1
elif [[ "$(cat "${bootstrap_stamp}")" != "${current_fingerprint}" ]]; then
  needs_bootstrap=1
fi

if [[ "${needs_bootstrap}" == "1" ]]; then
  rm -rf "${env_dir}"
  mkdir -p "${task_root}"
  if [[ "${use_system_site_packages}" == "1" ]]; then
    "${base_python}" -m venv --system-site-packages "${env_dir}"
  else
    "${base_python}" -m venv "${env_dir}"
  fi

  "${env_dir}/bin/python" -m pip install --upgrade pip setuptools wheel
  "${env_dir}/bin/python" -m pip install -e ".[tests]"
  printf '%s\n' "${current_fingerprint}" > "${bootstrap_stamp}"
fi

if [[ -n "${OV_NGAGENT_REMOTE_PYTEST_TARGETS:-}" ]]; then
  # shellcheck disable=SC2206
  pytest_targets=(${OV_NGAGENT_REMOTE_PYTEST_TARGETS})
else
  pytest_targets=(
    tests/utils/test_agent_initialization.py
    tests/utils/test_smart_agent.py
    tests/utils/test_agent_backend_streaming.py
    tests/utils/test_agent_backend_usage.py
    tests/utils/test_agent_backend_providers.py
    tests/utils/test_ovagent_run_store.py
    tests/utils/test_ovagent_tool_runtime.py
    tests/utils/test_ovagent_workflow.py
    tests/utils/test_harness_cleanup.py
    tests/utils/test_harness_cli.py
    tests/utils/test_harness_compaction.py
    tests/utils/test_harness_contracts.py
    tests/utils/test_harness_runtime_state.py
    tests/utils/test_harness_tool_catalog.py
    tests/utils/test_harness_web_bridge.py
    tests/jarvis/test_session_shared_adata.py
    tests/test_claw_cli.py
    -k
    "not test_agentic_loop_retries_after_text_only_promise_until_tool_call"
  )
fi

export OV_AGENT_RUN_HARNESS_TESTS=1
export OMICVERSE_DISABLE_LLM=1
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

printf 'Remote repo: %s\n' "${repo_root}"
printf 'Remote env: %s\n' "${env_dir}"
printf 'Pytest targets (%s): %s\n' "${#pytest_targets[@]}" "${pytest_targets[*]}"

"${env_dir}/bin/python" -m pytest -q --maxfail=1 "${pytest_targets[@]}"

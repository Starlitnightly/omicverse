#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
worktree_root="$(cd -- "${script_dir}/../.." && pwd)"
canonical_repo_root="${worktree_root}"

if [[ "$(basename "$(dirname "${worktree_root}")")" == ".worktrees" ]]; then
  canonical_repo_root="$(cd -- "${worktree_root}/../.." && pwd)"
fi

task_slug="${NGAGENT_TASK_ID:-$(basename "${worktree_root}")}"
task_slug="${task_slug//\//-}"
task_slug="${task_slug//:/-}"

remote_host="${TAIWAN_REMOTE_HOST:-1.34.182.186}"
remote_port="${TAIWAN_REMOTE_PORT:-5583}"
remote_user="${TAIWAN_REMOTE_USER:-kblueleaf}"
bundle_root_default="${canonical_repo_root}/../taiwan-server-migration-bundle-2026-02-22"
ssh_key_path="${TAIWAN_SSH_KEY_PATH:-${bundle_root_default}/credentials/key}"
remote_base="${TAIWAN_REMOTE_BASE:-/slow/ngagent-review}"
remote_task_root="${TAIWAN_REMOTE_TASK_ROOT:-${remote_base}/omicverse/${task_slug}}"
remote_repo_path="${remote_task_root}/repo"

if ! command -v ssh >/dev/null 2>&1; then
  echo "ssh is required for Taiwan review" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required for Taiwan review" >&2
  exit 1
fi

if [[ ! -f "${ssh_key_path}" ]]; then
  echo "Taiwan SSH key not found at ${ssh_key_path}" >&2
  echo "Set TAIWAN_SSH_KEY_PATH to override the default bundle location." >&2
  exit 1
fi

ssh_opts=(
  -p "${remote_port}"
  -i "${ssh_key_path}"
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
)

printf 'Syncing %s to %s@%s:%s\n' "${worktree_root}" "${remote_user}" "${remote_host}" "${remote_repo_path}"

ssh "${ssh_opts[@]}" "${remote_user}@${remote_host}" \
  "mkdir -p $(printf '%q' "${remote_repo_path}")"

rsync_rsh="ssh"
for opt in "${ssh_opts[@]}"; do
  rsync_rsh+=" $(printf '%q' "${opt}")"
done
rsync -az --delete \
  --exclude='.git/' \
  --exclude='.worktrees/' \
  --exclude='.pytest_cache/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.venv/' \
  --exclude='build/' \
  --exclude='dist/' \
  -e "${rsync_rsh}" \
  "${worktree_root}/" \
  "${remote_user}@${remote_host}:${remote_repo_path}/"

remote_command=$(cat <<EOF
set -euo pipefail
cd $(printf '%q' "${remote_repo_path}")
export TAIWAN_REMOTE_TASK_ROOT=$(printf '%q' "${remote_task_root}")
./scripts/ci/ngagent_taiwan_remote_pytest.sh
EOF
)

printf 'Running remote validation in %s\n' "${remote_repo_path}"
ssh "${ssh_opts[@]}" "${remote_user}@${remote_host}" "bash -lc $(printf '%q' "${remote_command}")"

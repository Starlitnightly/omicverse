#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  OmicVerse Guided Installer
#  Supports: macOS, Linux, Windows (WSL / Git Bash)
#            Python 3.10 / 3.11
# ============================================================

# ── colour / formatting ──────────────────────────────────────
if [[ -t 1 ]]; then
  BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
  RED='\033[0;31m'; CYAN='\033[0;36m'; RESET='\033[0m'
else
  BOLD=''; GREEN=''; YELLOW=''; RED=''; CYAN=''; RESET=''
fi

banner() {
  echo ""
  echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════╗${RESET}"
  echo -e "${CYAN}${BOLD}║        OmicVerse  —  Guided  Installer           ║${RESET}"
  echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════╝${RESET}"
  echo ""
}

step() {
  echo ""
  echo -e "${BOLD}==> $*${RESET}"
}

info()    { echo -e "    ${GREEN}✔${RESET}  $*"; }
warn()    { echo -e "    ${YELLOW}⚠${RESET}  $*"; }
fail()    { echo -e "    ${RED}✖  ERROR: $*${RESET}" >&2; exit 1; }

# ── non-interactive flag ──────────────────────────────────────
AUTO_YES="${OMICVERSE_AUTO_YES:-}"
for arg in "$@"; do
  [[ "$arg" == "-y" || "$arg" == "--yes" ]] && AUTO_YES=1
done

# ── arrow-key selector ────────────────────────────────────────
# _menu_select <default_idx> "option1" "option2" ...
# Writes the chosen 0-based index into $_MENU_IDX.
# Falls back to default_idx when not running in a TTY.
_MENU_IDX=0

_menu_select() {
  local start="$1"; shift
  local -a opts=("$@")
  local n=${#opts[@]}
  local cur="$start"
  local i key key2

  # Non-TTY / CI: pick the default silently
  if [[ ! -t 1 || -n "$AUTO_YES" ]]; then
    _MENU_IDX="$start"
    printf "    \033[1;36m❯\033[0m  \033[1m%s\033[0m\n" "${opts[$start]}"
    return
  fi

  _ms_draw() {
    for i in "${!opts[@]}"; do
      if [[ $i -eq $cur ]]; then
        printf "    \033[1;36m❯\033[0m  \033[1m%s\033[0m\n" "${opts[$i]}"
      else
        printf "       \033[2m%s\033[0m\n" "${opts[$i]}"
      fi
    done
  }

  tput civis 2>/dev/null || true   # hide cursor
  _ms_draw

  while true; do
    IFS= read -r -s -n1 key </dev/tty 2>/dev/null \
      || { _MENU_IDX="$start"; break; }

    if [[ "$key" == $'\x1b' ]]; then
      IFS= read -r -s -n2 -t 0.05 key2 </dev/tty 2>/dev/null || key2=""
      case "$key2" in
        '[A') (( cur = (cur - 1 + n) % n )) ;;   # ↑
        '[B') (( cur = (cur + 1)     % n )) ;;   # ↓
      esac
    elif [[ "$key" == '' || "$key" == $'\r' || "$key" == $'\n' ]]; then
      break
    fi
    printf '\033[%dA' "$n"   # move cursor up n lines
    _ms_draw
  done

  tput cnorm 2>/dev/null || true   # restore cursor
  _MENU_IDX=$cur
}

# ── helpers ───────────────────────────────────────────────────
ask_yes_no() {
  # ask_yes_no "question" [default: y|n]
  local question="$1"
  local default="${2:-y}"
  local start=0
  [[ "$default" == "n" ]] && start=1
  echo "    $question"
  _menu_select "$start" "Yes" "No"
  [[ $_MENU_IDX -eq 0 ]]   # return 0 (true) when Yes selected
}

ask_value() {
  # ask_value "prompt" default  →  echoes chosen value
  local question="$1"
  local default="$2"
  printf "    %s [%s]:  " "$question" "$default"
  local answer
  read -r answer </dev/tty
  echo "${answer:-$default}"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "Required command not found: $1"
  fi
}

maybe_ask_yes_no() {
  if [[ -n "$AUTO_YES" ]]; then return 0; fi
  ask_yes_no "$@"
}

# ════════════════════════════════════════════════════════════
banner

# ────────────────────────────────────────────────────────────
step "Step 1 — Detecting operating environment"
# ────────────────────────────────────────────────────────────

_RAW_OS="$(uname -s 2>/dev/null || echo Unknown)"
_RAW_KERNEL="$(uname -r 2>/dev/null || echo unknown)"
ARCH="$(uname -m 2>/dev/null || echo unknown)"
OS_TYPE=""          # macOS | linux | wsl | gitbash | cygwin | unknown
DISTRO=""           # Ubuntu / CentOS / Fedora / Arch / ... (Linux only)
DISTRO_VERSION=""
WINDOWS_HINT=""     # friendly name shown to the user

case "$_RAW_OS" in
  Darwin)
    OS_TYPE="macOS"
    MACOS_VERSION="$(sw_vers -productVersion 2>/dev/null || echo unknown)"
    info "macOS $MACOS_VERSION  ($ARCH)"
    if [[ "$ARCH" == "arm64" ]]; then
      info "Apple Silicon (M-series) detected"
      warn "Some conda packages may use Rosetta 2 emulation on arm64."
      warn "If you encounter architecture errors, try:"
      warn "  CONDA_SUBDIR=osx-arm64 conda install ..."
    fi
    ;;

  Linux)
    # ── Check for WSL (Windows Subsystem for Linux) ──
    if grep -qiE '(microsoft|wsl)' /proc/version 2>/dev/null \
       || [[ "$_RAW_KERNEL" == *microsoft* ]] \
       || [[ "$_RAW_KERNEL" == *WSL* ]]; then
      OS_TYPE="wsl"
      WINDOWS_HINT="Windows (WSL)"
    else
      OS_TYPE="linux"
    fi

    # ── Detect Linux distribution ──
    if [[ -f /etc/os-release ]]; then
      # shellcheck disable=SC1091
      source /etc/os-release
      DISTRO="${NAME:-unknown}"
      DISTRO_VERSION="${VERSION_ID:-}"
    elif command -v lsb_release >/dev/null 2>&1; then
      DISTRO="$(lsb_release -si)"
      DISTRO_VERSION="$(lsb_release -sr)"
    elif [[ -f /etc/redhat-release ]]; then
      DISTRO="$(cat /etc/redhat-release)"
    elif [[ -f /etc/debian_version ]]; then
      DISTRO="Debian"
      DISTRO_VERSION="$(cat /etc/debian_version)"
    else
      DISTRO="Linux (unknown distro)"
    fi

    if [[ "$OS_TYPE" == "wsl" ]]; then
      info "Windows Subsystem for Linux (WSL)  —  $DISTRO $DISTRO_VERSION  ($ARCH)"
      warn "Running under WSL. Most features work, but:"
      warn "  • GPU/CUDA support requires WSL2 + NVIDIA drivers with CUDA on Windows host."
      warn "  • GUI plots (matplotlib) may need a VcXsrv / WSLg display server."
    else
      info "Linux  —  $DISTRO $DISTRO_VERSION  ($ARCH)"

      # Check missing system libs that conda/pip sometimes need
      _MISSING_LIBS=()
      for lib in libgomp1 libgl1; do
        if ! (ldconfig -p 2>/dev/null | grep -q "$lib" \
              || find /usr/lib /usr/lib64 /lib 2>/dev/null -name "${lib}*" -quit | grep -q .); then
          _MISSING_LIBS+=("$lib")
        fi
      done
      if [[ "${#_MISSING_LIBS[@]}" -gt 0 ]]; then
        warn "Possibly missing system libraries: ${_MISSING_LIBS[*]}"
        warn "On Ubuntu/Debian:  sudo apt-get install ${_MISSING_LIBS[*]}"
        warn "On CentOS/RHEL:    sudo yum install libgomp mesa-libGL"
      fi
    fi
    ;;

  MINGW*|MSYS*)
    OS_TYPE="gitbash"
    WINDOWS_HINT="Windows (Git Bash / MSYS2)"
    info "$WINDOWS_HINT  ($ARCH)"
    echo ""
    warn "Git Bash / MSYS2 detected."
    warn "Native conda environments work best in:"
    warn "  • Anaconda Prompt  (recommended)"
    warn "  • Windows Subsystem for Linux (WSL2)  (recommended)"
    warn "  • PowerShell with conda init"
    echo ""
    warn "Continuing in Git Bash — some conda operations may behave unexpectedly."
    warn "If you encounter issues, re-run this script from an Anaconda Prompt."
    ;;

  CYGWIN*)
    OS_TYPE="cygwin"
    WINDOWS_HINT="Windows (Cygwin)"
    info "$WINDOWS_HINT  ($ARCH)"
    warn "Cygwin is not a recommended environment for OmicVerse."
    warn "Please use WSL2 or an Anaconda Prompt instead."
    if ! maybe_ask_yes_no "Continue anyway?"; then
      fail "Aborted. Re-run from WSL2 or Anaconda Prompt."
    fi
    ;;

  *)
    OS_TYPE="unknown"
    warn "Unrecognised OS: $_RAW_OS  ($ARCH). Proceeding anyway..."
    ;;
esac

# ────────────────────────────────────────────────────────────
step "Step 2 — Checking Conda installation"
# ────────────────────────────────────────────────────────────

CONDA_CMD=""
if command -v conda >/dev/null 2>&1; then
  CONDA_CMD="conda"
  CONDA_VERSION="$(conda --version 2>&1 | awk '{print $2}')"
  info "conda $CONDA_VERSION found at $(command -v conda)"
else
  warn "conda is not installed (or not in PATH)."
  echo ""
  echo "    OmicVerse requires conda to manage native dependencies"
  echo "    (e.g. s_gd2, opencv)."
  echo ""
  if maybe_ask_yes_no "Install Miniconda now?"; then
    step "Installing Miniconda"
    case "$OS_TYPE-$ARCH" in
      macOS-x86_64)  INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh" ;;
      macOS-arm64)   INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh" ;;
      linux-x86_64|wsl-x86_64)   INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" ;;
      linux-aarch64|wsl-aarch64) INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh" ;;
      gitbash-x86_64|cygwin-x86_64)
        echo ""
        warn "For Windows (non-WSL), the recommended approach is to download"
        warn "the Miniconda Windows installer manually:"
        echo ""
        echo "      https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
        echo ""
        warn "Run it, then re-open Anaconda Prompt and re-run this installer."
        exit 0
        ;;
      *) fail "Unsupported platform: $OS_TYPE-$ARCH. Install conda manually from https://docs.conda.io" ;;
    esac

    MINICONDA_INSTALL_DIR="$(ask_value "Install Miniconda to" "$HOME/miniconda3")"
    INSTALLER_FILE="$(mktemp /tmp/miniconda_installer.XXXXXX.sh)"
    echo "    Downloading $INSTALLER_URL ..."
    curl -fsSL "$INSTALLER_URL" -o "$INSTALLER_FILE"
    bash "$INSTALLER_FILE" -b -p "$MINICONDA_INSTALL_DIR"
    rm -f "$INSTALLER_FILE"

    # activate for this session
    # shellcheck disable=SC1091
    source "$MINICONDA_INSTALL_DIR/etc/profile.d/conda.sh"
    conda init "$(basename "$SHELL")" >/dev/null 2>&1 || true
    CONDA_CMD="conda"
    info "Miniconda installed at $MINICONDA_INSTALL_DIR"
    echo ""
    warn "Conda has been initialised for your shell. Please re-open your"
    warn "terminal (or run:  source ~/.bashrc / ~/.zshrc) and re-run this"
    warn "installer so that the conda environment is fully available."
    exit 0
  else
    fail "conda is required. Install it from https://docs.conda.io and re-run this script."
  fi
fi

# ────────────────────────────────────────────────────────────
step "Step 3 — Checking active Conda environment"
# ────────────────────────────────────────────────────────────

CONDA_ENV="${CONDA_DEFAULT_ENV:-}"

if [[ -z "$CONDA_ENV" ]]; then
  # conda is installed but no env activated
  warn "No conda environment is currently active."
  _ENV_ACTION="create"
elif [[ "$CONDA_ENV" == "base" ]]; then
  warn "You are currently in the 'base' environment."
  echo "    Installing packages into 'base' is discouraged — it can break"
  echo "    conda itself and is hard to recover from."
  _ENV_ACTION="create_or_switch"
else
  info "Active environment: $CONDA_ENV"
  if maybe_ask_yes_no "Continue installing into '$CONDA_ENV'?"; then
    _ENV_ACTION="use_current"
  else
    _ENV_ACTION="create_or_switch"
  fi
fi

TARGET_ENV=""

if [[ "$_ENV_ACTION" == "use_current" ]]; then
  TARGET_ENV="$CONDA_ENV"

elif [[ "$_ENV_ACTION" == "create" || "$_ENV_ACTION" == "create_or_switch" ]]; then
  echo ""
  echo "    Choose an option:"
  _env_opts=("Create a new conda environment (recommended)" "Activate an existing environment")
  [[ "$_ENV_ACTION" == "create_or_switch" ]] && _env_opts+=("Continue anyway in 'base' (not recommended)")
  _menu_select 0 "${_env_opts[@]}"
  _choice=$(( _MENU_IDX + 1 ))

  case "$_choice" in
    1)
      _DEFAULT_ENV_NAME="omicverse"
      TARGET_ENV="$(ask_value "New environment name" "$_DEFAULT_ENV_NAME")"
      _PY_VER="$(ask_value "Python version (3.10 or 3.11)" "3.11")"
      if [[ "$_PY_VER" != "3.10" && "$_PY_VER" != "3.11" ]]; then
        fail "Unsupported Python version: $_PY_VER. Use 3.10 or 3.11."
      fi

      if conda env list | awk '{print $1}' | grep -qx "$TARGET_ENV"; then
        warn "Environment '$TARGET_ENV' already exists."
        if maybe_ask_yes_no "Use the existing '$TARGET_ENV' environment?"; then
          : # fall through and activate it
        else
          fail "Please choose a different environment name and re-run the installer."
        fi
      else
        step "Creating conda environment '$TARGET_ENV' (Python $_PY_VER)"
        conda create -n "$TARGET_ENV" python="$_PY_VER" -y
        info "Environment '$TARGET_ENV' created."
      fi

      echo ""
      echo -e "    ${YELLOW}${BOLD}ACTION REQUIRED${RESET}"
      echo "    Run the following command, then re-run this installer:"
      echo ""
      echo -e "      ${CYAN}conda activate $TARGET_ENV${RESET}"
      echo ""
      exit 0
      ;;

    2)
      echo ""
      echo "    Available environments:"
      conda env list | grep -v '^#' | awk '{print "      " $0}'
      echo ""
      TARGET_ENV="$(ask_value "Environment name to activate" "")"
      [[ -z "$TARGET_ENV" ]] && fail "No environment name provided."

      echo ""
      echo -e "    ${YELLOW}${BOLD}ACTION REQUIRED${RESET}"
      echo "    Run the following command, then re-run this installer:"
      echo ""
      echo -e "      ${CYAN}conda activate $TARGET_ENV${RESET}"
      echo ""
      exit 0
      ;;

    3)
      if [[ "$_ENV_ACTION" == "create_or_switch" ]]; then
        warn "Proceeding in 'base' — do this at your own risk."
        TARGET_ENV="base"
      else
        fail "Invalid choice."
      fi
      ;;

    *)
      fail "Invalid choice: $_choice"
      ;;
  esac
fi

info "Target environment: $TARGET_ENV"

# ────────────────────────────────────────────────────────────
step "Step 4 — Checking Python version"
# ────────────────────────────────────────────────────────────

PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PYTHON_VERSION" != "3.10" && "$PYTHON_VERSION" != "3.11" ]]; then
  fail "Python $PYTHON_VERSION is not supported. OmicVerse requires Python 3.10 or 3.11."
fi
info "Python $PYTHON_VERSION"

# ────────────────────────────────────────────────────────────
step "Step 5 — Selecting fastest PyPI mirror"
# ────────────────────────────────────────────────────────────

MIRRORS=(
  "https://pypi.tuna.tsinghua.edu.cn/simple"
  "https://pypi.org/simple"
)
declare -A LATENCIES

echo "    Testing mirror latencies..."
for m in "${MIRRORS[@]}"; do
  t=$(curl -o /dev/null -s -w "%{time_total}" --connect-timeout 3 -I "$m" 2>/dev/null || echo 999)
  LATENCIES["$m"]=$t
  echo "      $m  →  ${t}s"
done

BEST_MIRROR="${MIRRORS[0]}"
best_time="${LATENCIES[$BEST_MIRROR]}"
for m in "${MIRRORS[@]}"; do
  if (( $(echo "${LATENCIES[$m]} < $best_time" | bc -l) )); then
    BEST_MIRROR="$m"
    best_time="${LATENCIES[$m]}"
  fi
done
info "Selected mirror: $BEST_MIRROR"
PIP_INDEX="-i $BEST_MIRROR"

# ────────────────────────────────────────────────────────────
step "Step 6 — Installing fast package managers (mamba + uv)"
# ────────────────────────────────────────────────────────────

if ! command -v mamba >/dev/null 2>&1; then
  echo "    Installing mamba..."
  conda install -c conda-forge -y mamba
else
  info "mamba already installed"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "    Installing uv..."
  pip install uv $PIP_INDEX
else
  info "uv already installed"
fi

# ── internal helpers ─────────────────────────────────────────
conda_install_pkg() {
  local pkg="$1"
  if conda list --no-pip | awk '{print $1}' | grep -xq "$pkg"; then
    info "Already have conda:$pkg"
  else
    echo "    Installing conda:$pkg ..."
    mamba install -c conda-forge -y "$pkg"
  fi
}

pip_install_pkg() {
  local missing=()
  for pkg in "$@"; do
    if pip show "$pkg" >/dev/null 2>&1; then
      info "Already have pip:$pkg"
    else
      missing+=("$pkg")
    fi
  done
  if [[ "${#missing[@]}" -gt 0 ]]; then
    echo "    Installing: ${missing[*]} ..."
    uv pip install "${missing[@]}" $PIP_INDEX
  fi
}

# ────────────────────────────────────────────────────────────
step "Step 7 — Installing native conda packages"
# ────────────────────────────────────────────────────────────

conda_install_pkg s_gd2
conda_install_pkg opencv

# ────────────────────────────────────────────────────────────
step "Step 8 — Setting up PyTorch"
# ────────────────────────────────────────────────────────────

if pip show torch >/dev/null 2>&1; then
  TORCH_VERSION="$(python - <<'PY'
import torch; print(torch.__version__.split("+")[0])
PY
)"
  info "Detected torch==$TORCH_VERSION (skipping reinstall)"
else
  echo "    Installing latest PyTorch (+ torchvision, torchaudio)..."
  TORCH_VERSION="$(pip index versions torch 2>/dev/null \
    | grep -oP 'Available versions: \K[0-9]+\.[0-9]+\.[0-9]+' \
    | head -1)"
  uv pip install torch torchvision torchaudio $PIP_INDEX
  info "torch==$TORCH_VERSION installed"
fi

# ────────────────────────────────────────────────────────────
step "Step 9 — Detecting CUDA & setting up PyG"
# ────────────────────────────────────────────────────────────

CUDA_TAG="$(python - <<'PY'
import torch
if torch.cuda.is_available() and torch.version.cuda:
    print("cu" + torch.version.cuda.replace(".", ""))
else:
    print("cpu")
PY
)"
info "CUDA tag: $CUDA_TAG"
PYG_WHL_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"
info "PyG wheel index: $PYG_WHL_URL"
pip_install_pkg torch_geometric

# ────────────────────────────────────────────────────────────
step "Step 10 — Installing OmicVerse"
# ────────────────────────────────────────────────────────────

pip_install_pkg omicverse

# ────────────────────────────────────────────────────────────
step "Step 11 — Installing optional bio packages"
# ────────────────────────────────────────────────────────────

pip_install_pkg \
  pydeseq2 mofax tomli lifelines ktplotspy pillow einops \
  tensorboard metatime graphtools boltons leidenalg gdown wandb

pip_install_pkg \
  tangram-sc fa2-modified pot libpysal openai patsy combat

pip_install_pkg \
  pymde opencv-python scikit-image memento-de

pip_install_pkg \
  harmonypy intervaltree fbpca scvi-tools s-gd2

pip_install_pkg \
  mellon scvelo cellrank dynamo-release squidpy pertpy

pip_install_pkg \
  toytree arviz ete3 torchdr

# ────────────────────────────────────────────────────────────
step "Step 12 — Pinning version-locked packages"
# ────────────────────────────────────────────────────────────

echo "    Ensuring pandas<3.0, numpy<2.0, zarr<3.0 for stability..."
uv pip install "pandas<3.0.0" "numpy<2.0.0" "zarr<3.0.0" --force-reinstall $PIP_INDEX
info "Version constraints applied"

# ────────────────────────────────────────────────────────────
step "Step 13 — Verifying installation"
# ────────────────────────────────────────────────────────────

python -c "import omicverse as ov; ov.plot_set()"
info "OmicVerse imported and configured successfully"

# ── done ─────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${GREEN}${BOLD}║   🎉  OmicVerse installation complete!           ║${RESET}"
echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════════════╝${RESET}"
echo ""
echo "    platform   : $OS_TYPE  ($ARCH)${DISTRO:+  —  $DISTRO $DISTRO_VERSION}"
echo "    torch      : $TORCH_VERSION"
echo "    CUDA tag   : $CUDA_TAG"
echo "    conda env  : $TARGET_ENV"
echo "    Python     : $PYTHON_VERSION"
echo ""
echo "    Get started:"
echo -e "      ${CYAN}import omicverse as ov${RESET}"
echo -e "      ${CYAN}ov.style()${RESET}"
echo ""

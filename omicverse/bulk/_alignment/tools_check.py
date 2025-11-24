# Author: Zhi Luo

import sys, os, shutil, glob, subprocess
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def resolve_tool(name: str):
    """
    Return absolute path to a tool if found by PATH or adjacent to current Python (conda env bin).
    """
    p = shutil.which(name)
    if p:
        return p
    # fallback: same env/bin as current Python
    env_bin = Path(sys.executable).parent
    candidate = env_bin / name
    if candidate.exists() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None

def check_tool_availability(tool_name: str, install_command: str = None, package_name: str = None) -> tuple[bool, str]:
    """
    Determine whether a tool is available, and optionally provide installation guidance when it is missing.

    Args:
        tool_name: The tool to locate.
        install_command: Optional shell command used to install the tool.
        package_name: Optional package name to suggest to the user.

    Returns:
        (available flag, tool path or explanatory error message)
    """
    tool_path = resolve_tool(tool_name)
    if tool_path:
        return True, tool_path

    # Tool missing: provide installation guidance.
    if install_command:
        logger.warning(f"{tool_name} not found. Installing...")
        try:
            subprocess.run(install_command, shell=True, check=True, capture_output=True, text=True)
            # Check again after installation.
            tool_path = resolve_tool(tool_name)
            if tool_path:
                logger.info(f"Successfully installed {tool_name}")
                return True, tool_path
            else:
                error_msg = f"Failed to install {tool_name}. Please install manually: {install_command}"
                logger.error(error_msg)
                return False, error_msg
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to install {tool_name}: {e.stderr}"
            logger.error(error_msg)
            return False, error_msg
    else:
        error_msg = f"{tool_name} not found"
        if package_name:
            error_msg += f". Try: conda install -c bioconda {package_name}"
        logger.error(error_msg)
        return False, error_msg

def check_sra_tools() -> tuple[bool, str]:
    """Ensure SRA Toolkit binaries are available."""
    return check_tool_availability(
        "prefetch",
        package_name="sra-tools",
        install_command="conda install -c bioconda sra-tools -y"
    )

def check_star() -> tuple[bool, str]:
    """Ensure the STAR aligner is available."""
    return check_tool_availability(
        "STAR",
        package_name="star",
        install_command="conda install -c bioconda star -y"
    )

def check_fastp() -> tuple[bool, str]:
    """Ensure the fastp QC tool is available."""
    return check_tool_availability(
        "fastp",
        package_name="fastp",
        install_command="conda install -c bioconda fastp -y"
    )

def check_featurecounts() -> tuple[bool, str]:
    """Ensure featureCounts is available."""
    return check_tool_availability(
        "featureCounts",
        package_name="subread",
        install_command="conda install -c bioconda subread -y"
    )

def check_samtools() -> tuple[bool, str]:
    """Ensure samtools is available."""
    return check_tool_availability(
        "samtools",
        package_name="samtools",
        install_command="conda install -c bioconda samtools -y"
    )

def check_axel(auto_install: bool = True) -> tuple[bool, str]:
    """Check if axel (download accelerator) is available."""
    tool_path = resolve_tool("axel")
    if tool_path:
        return True, tool_path

    # Tool unavailable - no automatic installation
    error_msg = "axel not found. Please install axel manually: e.g., conda install -c conda-forge axel -y or brew install axel"
    logger.info(error_msg)
    return False, error_msg

def check_iseq(auto_install: bool = True) -> tuple[bool, str]:
    """Ensure the iseq downloader is available with manual installation guidance.

    Note: auto_install parameter is kept for backward compatibility but is ignored.
    Automatic installation has been removed - only manual installation guidance is provided.
    """
    # First check if axel is available (dependency for iseq). No auto installation.
    axel_available, axel_path = check_axel(auto_install=False)
    if not axel_available:
        logger.warning(f"axel is not available, which may cause issues with iseq: {axel_path}")

    # Check for iseq itself.
    tool_path = resolve_tool("iseq")
    if tool_path:
        return True, tool_path

    # iseq missing; provide manual installation guidance.
    error_msg = "iseq not found. Please install manually with: conda install -c bioconda iseq -y or mamba install -c bioconda iseq -y"
    logger.error(error_msg)
    return False, error_msg

def check_edirect() -> tuple[bool, str]:
    """Ensure EDirect (esearch/efetch) is available."""
    return check_tool_availability(
        "esearch",
        package_name="entrez-direct",
        install_command="conda install -c bioconda entrez-direct -y"
    )

def check_all_tools(auto_install: bool = False) -> dict[str, tuple[bool, str]]:
    """
    Verify availability of every required tool.

    Args:
        auto_install: Whether to attempt automatic installation of missing tools.

    Returns:
        Mapping of tool name to a tuple (available flag, path or error string).
    """
    tools = {
        'sra-tools': check_sra_tools,
        'star': check_star,
        'fastp': check_fastp,
        'featureCounts': check_featurecounts,
        'samtools': check_samtools,
        'edirect': check_edirect,
        'iseq': check_iseq
    }

    results = {}
    all_available = True

    for tool_name, check_func in tools.items():
        # Fix: always use the dedicated checker instead of the generic helper.
        available, path = check_func()
        results[tool_name] = (available, path)
        if not available:
            all_available = False
            logger.error(f"{tool_name}: {path}")
        else:
            logger.info(f"{tool_name}: {path}")

    return results

def check(auto_install: bool = False, verbose: bool = True) -> bool:
    """
    Check whether all required tools are available.

    Args:
        auto_install: Whether to auto-install missing tools.
        verbose: Whether to emit log output for each tool.

    Returns:
        True when all tools are present.
    """
    env_bin = str(Path(sys.executable).parent)
    if env_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = env_bin + os.pathsep + os.environ.get("PATH", "")

    # Use the new consolidated checking function.
    results = check_all_tools(auto_install=auto_install)

    # Verify whether every tool is available.
    all_available = all(result[0] for result in results.values())

    if verbose:
        if all_available:
            logger.info("All required tools are available!")
        else:
            logger.error("Some tools are missing:")
            for tool_name, (available, path) in results.items():
                if not available:
                    logger.error(f"  - {tool_name}: {path}")

    return all_available

def which_or_find(cmd: str) -> str:
    """
    Return the absolute path to `cmd`. First consult shutil.which,
    then probe common directories and user-extracted sratoolkit paths.
    Raise with a helpful message when the tool cannot be located.
    """
    p = shutil.which(cmd)
    if p:
        return p

    # Common fallbacks: current Python env bin, typical system paths, user home sratoolkit extractions.
    candidates = []
    # 1) Bin directory of the current interpreter (works well for Conda).
    py_bin = Path(os.path.realpath(os.sys.executable)).parent
    candidates += [str(py_bin / cmd)]

    # 2) Standard system locations.
    candidates += [f"/usr/local/bin/{cmd}", f"/usr/bin/{cmd}", f"/bin/{cmd}"]

    # 3) Manually unpacked sratoolkit directories under the user home.
    for g in glob.glob(str(Path.home() / "sratoolkit*/bin")):
        candidates.append(str(Path(g) / cmd))

    for c in candidates:
        if os.path.exists(c) and os.access(c, os.X_OK):
            return c

    # Provide a more accurate installation hint per tool.
    install_hints = {
        # SRA toolkit family
        "prefetch": "conda install -c bioconda sra-tools",
        "fastq-dump": "conda install -c bioconda sra-tools",
        "fasterq-dump": "conda install -c bioconda sra-tools",
        # Core tools in this pipeline
        "fastp": "conda install -c bioconda fastp",
        "STAR": "conda install -c bioconda star",
        "samtools": "conda install -c bioconda samtools",
        "featureCounts": "conda install -c bioconda subread",
        "esearch": "conda install -c bioconda entrez-direct",
        "iseq": "conda install -c bioconda iseq",
        "axel": "conda install -c conda-forge axel",
    }
    hint = install_hints.get(cmd)
    if hint:
        suggestion = f"Try: `{hint}` "
    else:
        suggestion = "Please install the corresponding package (e.g., via bioconda) "

    raise FileNotFoundError(
        f"'{cmd}' not found in PATH. "
        f"{suggestion}"
        f"or ensure your Jupyter kernel is the same env as your shell.\n"
        f"Python: {os.sys.executable}\nPATH: {os.environ.get('PATH','')}\n"
    )

def merged_env(extra: dict | None = None) -> dict:
    """
    Build an environment dictionary suitable for subprocesses.
    Guarantee the current Conda env bin directory is on PATH (so axel/pigz/aspera are discoverable for iseq).
    """
    import os, sys
    from pathlib import Path

    env = os.environ.copy()
    
    # Critical patch: force inclusion of the current Conda env bin directory.
    conda_bin = str(Path(sys.executable).parent)
    env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"

    # If you have a specific VDB config, inject it here, for example:
    # env.setdefault("VDB_CONFIG", str(Path.home() / ".ncbi/user-settings.mkfg"))

    if extra:
        env.update(extra)
    return env

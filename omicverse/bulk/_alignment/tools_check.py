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
    检查工具是否可用，如果不存在则提供安装指引

    Args:
        tool_name: 工具名称
        install_command: 安装命令
        package_name: 包名称

    Returns:
        (是否可用, 工具路径或错误信息)
    """
    tool_path = resolve_tool(tool_name)
    if tool_path:
        return True, tool_path

    # 工具不存在，提供安装指引
    if install_command:
        logger.warning(f"{tool_name} not found. Installing...")
        try:
            subprocess.run(install_command, shell=True, check=True, capture_output=True, text=True)
            # 再次检查
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
    """检查SRA工具"""
    return check_tool_availability(
        "prefetch",
        package_name="sra-tools",
        install_command="conda install -c bioconda sra-tools -y"
    )

def check_star() -> tuple[bool, str]:
    """检查STAR比对工具"""
    return check_tool_availability(
        "STAR",
        package_name="star",
        install_command="conda install -c bioconda star -y"
    )

def check_fastp() -> tuple[bool, str]:
    """检查fastp质控工具"""
    return check_tool_availability(
        "fastp",
        package_name="fastp",
        install_command="conda install -c bioconda fastp -y"
    )

def check_featurecounts() -> tuple[bool, str]:
    """检查featureCounts定量工具"""
    return check_tool_availability(
        "featureCounts",
        package_name="subread",
        install_command="conda install -c bioconda subread -y"
    )

def check_samtools() -> tuple[bool, str]:
    """检查samtools工具"""
    return check_tool_availability(
        "samtools",
        package_name="samtools",
        install_command="conda install -c bioconda samtools -y"
    )

def check_axel(auto_install: bool = True) -> tuple[bool, str]:
    """检查axel工具，专门适配Jupyter Lab环境"""
    tool_path = resolve_tool("axel")
    if tool_path:
        return True, tool_path

    # 工具不存在
    if auto_install:
        logger.warning("axel not found. Attempting automatic installation...")

        # 尝试多种安装方式
        install_commands = [
            "conda install -c conda-forge axel -y",
            "mamba install -c conda-forge axel -y"
        ]

        for cmd in install_commands:
            try:
                logger.info(f"Trying installation: {cmd}")
                result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

                # 再次检查
                tool_path = resolve_tool("axel")
                if tool_path:
                    logger.info(f"Successfully installed axel using: {cmd}")
                    return True, tool_path

            except subprocess.CalledProcessError as e:
                logger.warning(f"Installation failed with {cmd}: {e.stderr}")
                continue

        # 所有安装方式都失败
        error_msg = "Failed to install axel automatically. Please install manually: conda install -c conda-forge axel -y"
        logger.error(error_msg)
        return False, error_msg
    else:
        error_msg = "axel not found. Please install: conda install -c conda-forge axel -y"
        logger.error(error_msg)
        return False, error_msg

def check_iseq(auto_install: bool = True) -> tuple[bool, str]:
    """检查iseq工具，增强版支持自动安装"""
    # 首先检查axel（iseq的依赖）
    axel_available, axel_path = check_axel(auto_install)
    if not axel_available:
        logger.warning(f"axel is not available, which may cause issues with iseq: {axel_path}")

    # 检查iseq本身
    tool_path = resolve_tool("iseq")
    if tool_path:
        return True, tool_path

    # iseq不存在，尝试安装
    if auto_install:
        logger.warning("iseq not found. Attempting automatic installation...")

        install_commands = [
            "conda install -c bioconda iseq -y",
            "mamba install -c bioconda iseq -y"
        ]

        for cmd in install_commands:
            try:
                logger.info(f"Trying installation: {cmd}")
                result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

                # 再次检查
                tool_path = resolve_tool("iseq")
                if tool_path:
                    logger.info(f"Successfully installed iseq using: {cmd}")
                    return True, tool_path

            except subprocess.CalledProcessError as e:
                logger.warning(f"Installation failed with {cmd}: {e.stderr}")
                continue

        # 所有安装方式都失败
        error_msg = "Failed to install iseq automatically. Please install manually: conda install -c bioconda iseq -y"
        logger.error(error_msg)
        return False, error_msg
    else:
        error_msg = "iseq not found. Please install: conda install -c bioconda iseq -y"
        logger.error(error_msg)
        return False, error_msg

def check_edirect() -> tuple[bool, str]:
    """检查EDirect工具"""
    return check_tool_availability(
        "esearch",
        package_name="entrez-direct",
        install_command="conda install -c bioconda entrez-direct -y"
    )

def check_all_tools(auto_install: bool = False) -> dict[str, tuple[bool, str]]:
    """
    检查所有必需的工具

    Args:
        auto_install: 是否自动安装缺失的工具

    Returns:
        工具检查结果字典
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
        # 修复：总是使用专门的检查函数，而不是通用的check_tool_availability
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
    检查所有必需的工具是否可用

    Args:
        auto_install: 是否自动安装缺失的工具
        verbose: 是否输出详细信息

    Returns:
        是否所有工具都可用
    """
    env_bin = str(Path(sys.executable).parent)
    if env_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = env_bin + os.pathsep + os.environ.get("PATH", "")

    # 使用新的检查函数
    results = check_all_tools(auto_install=auto_install)

    # 检查是否所有工具都可用
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
    返回 cmd 的绝对路径。先用 shutil.which，
    再在常见目录和 sratoolkit* 目录下尝试。
    找不到则抛错并给出可读提示。
    """
    p = shutil.which(cmd)
    if p:
        return p

    # 常见路径兜底：当前 Python env 的 bin、常见系统路径、用户 HOME 下的 sratoolkit 解压目录
    candidates = []
    # 1) 当前解释器所在 env 的 bin（对 conda 很有效）
    py_bin = Path(os.path.realpath(os.sys.executable)).parent
    candidates += [str(py_bin / cmd)]

    # 2) 系统常见路径
    candidates += [f"/usr/local/bin/{cmd}", f"/usr/bin/{cmd}", f"/bin/{cmd}"]

    # 3) 用户目录下手动解压的 sratoolkit
    for g in glob.glob(str(Path.home() / "sratoolkit*/bin")):
        candidates.append(str(Path(g) / cmd))

    for c in candidates:
        if os.path.exists(c) and os.access(c, os.X_OK):
            return c

    raise FileNotFoundError(
        f"'{cmd}' not found in PATH. "
        f"Try: `conda install -c bioconda sra-tools` "
        f"or ensure your Jupyter kernel is the same env as your shell.\n"
        f"Python: {os.sys.executable}\nPATH: {os.environ.get('PATH','')}\n"
    )

def merged_env(extra: dict | None = None) -> dict:
    """
    生成一个对 subprocess 友好的环境变量字典。
    确保当前 Conda 环境的 bin 目录在 PATH 中（例如 axel、pigz、aspera 可被 iseq 调用）。
    """
    import os, sys
    from pathlib import Path

    env = os.environ.copy()
    
    # 关键补丁：强制包含当前 Conda 环境的 bin 目录
    conda_bin = str(Path(sys.executable).parent)
    env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"

    # 若你有特定的 VDB 配置，可在此注入，例如：
    # env.setdefault("VDB_CONFIG", str(Path.home() / ".ncbi/user-settings.mkfg"))

    if extra:
        env.update(extra)
    return env
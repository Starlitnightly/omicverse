import sys, os, shutil,glob
from pathlib import Path

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

def check(): # auto-fix PATH to include this conda env bin (in case kernelspec PATH is wrong)
    env_bin = str(Path(sys.executable).parent)
    if env_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = env_bin + os.pathsep + os.environ.get("PATH", "")
    esearch        = resolve_tool("esearch")
    SRATOOL        = resolve_tool("prefetch")
    FASTP          = resolve_tool("fastp")
    STAR_BIN       = resolve_tool("STAR")
    FEATURECOUNTS  = resolve_tool("featureCounts")
    
    if SRATOOL is None:
        raise EnvironmentError("sra-tools not found. Try: conda install -c bioconda sra-tools")
    if STAR_BIN is None:
        raise EnvironmentError("STAR not found. Try: conda install -c bioconda star")
    if FEATURECOUNTS is None:
        print("⚠️ featureCounts not found. Install via: conda install -c bioconda subread")
    if FASTP is None:
        print("⚠️ fastp not found. QC will be skipped in fastp_clean().")
    if esearch is None:
        raise EnvironmentError("STAR not found. Try: conda install -c bioconda entrez-direct")

    print("Tools check finished!")

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
    可在这里加 SRAToolkit/代理配置等。
    """
    env = os.environ.copy()
    # 若你有特定的 VDB 配置，可在此注入，例如：
    # env.setdefault("VDB_CONFIG", str(Path.home() / ".ncbi/user-settings.mkfg"))
    if extra:
        env.update(extra)
    return env
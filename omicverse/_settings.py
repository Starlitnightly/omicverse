

class omicverseConfig:

    def __init__(self,mode='cpu'):
        self.mode = mode

    def gpu_init(self,managed_memory=True,pool_allocator=True,devices=0):
        
        import scanpy as sc
        import cupy as cp

        import time
        import rapids_singlecell as rsc

        import warnings
        import rmm
        from rmm.allocators.cupy import rmm_cupy_allocator
        rmm.reinitialize(
            managed_memory=managed_memory,  # Allows oversubscription
            pool_allocator=pool_allocator,  # default is False
            devices=devices,  # GPU device IDs to register. By default registers only GPU 0.
        )
        cp.cuda.set_allocator(rmm_cupy_allocator)
        print('GPU mode activated')
        self.mode = 'gpu'
    
    def cpu_init(self):
        print('CPU mode activated')
        self.mode = 'cpu'
    
    def cpu_gpu_mixed_init(self):
        print('CPU-GPU mixed mode activated')
        self.mode = 'cpu-gpu-mixed'


import subprocess
import torch

def print_gpu_usage_color(bar_length: int = 30):
    """
    Print a colorized memory‐usage bar for each CUDA GPU.

    Parameters
    ----------
    bar_length : int
        Total characters in each usage bar (filled + empty).
    """
    # ANSI escape codes
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    GREY = '\033[90m'
    RESET = '\033[0m'

    if not torch.cuda.is_available():
        print(f"{RED}No CUDA devices found.{RESET}")
        return

    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ]
        lines = subprocess.check_output(cmd, encoding="utf-8").splitlines()
    except Exception as e:
        print(f"{RED}Could not run nvidia-smi: {e}{RESET}")
        return

    for line in lines:
        idx_str, used_str, total_str = [x.strip() for x in line.split(",")]
        idx = int(idx_str)
        used = float(used_str)
        total = float(total_str)
        frac = max(0.0, min(used / total, 1.0))
        filled = int(frac * bar_length)
        empty = bar_length - filled

        # choose color based on usage fraction
        if frac < 0.5:
            color = GREEN
        elif frac < 0.8:
            color = YELLOW
        else:
            color = RED

        bar = f"{color}{'|' * filled}{GREY}{'-' * empty}{RESET}"
        print(f"{EMOJI['bar']} [GPU {idx}] {bar} {used:.0f}/{total:.0f} MiB ({frac*100:.1f}%)")

EMOJI = {
    "start":        "🔍",  # start
    "cpu":          "🖥️",  # CPU mode
    "mixed":        "⚙️",  # mixed CPU/GPU mode
    "gpu":          "🚀",  # RAPIDS GPU mode
    "done":         "✅",  # done
    "error":        "❌",  # error
    "bar":          "📊",  # usage bar
    "check_mark":   "✅",  # check mark
}

settings = omicverseConfig()
        
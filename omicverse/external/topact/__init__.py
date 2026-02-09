__version__ = '1.1.0'

class Colors:
    """ANSI color codes for terminal output styling."""
    HEADER = '\033[95m'     # Purple
    BLUE = '\033[94m'       # Blue
    CYAN = '\033[96m'       # Cyan
    GREEN = '\033[92m'      # Green
    WARNING = '\033[93m'    # Yellow
    FAIL = '\033[91m'       # Red
    ENDC = '\033[0m'        # Reset
    BOLD = '\033[1m'        # Bold
    UNDERLINE = '\033[4m'   # Underline

EMOJI = {
    "start":        "🔍",  # start
    "train":        "🎯",  # training
    "classify":     "🔬",  # classification
    "filter":       "🧹",  # filtering
    "spatial":      "🗺️",  # spatial analysis
    "done":         "✅",  # done
    "process":      "⚙️",  # processing
    "gene":         "🧬",  # gene
    "cell":         "🔵",  # cell
    "grid":         "📊",  # grid
}

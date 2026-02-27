__version__ = "0.2.0"

try:
    from .colorization import colorize, colorize_mutiple_runs, colorize_mutiple_slices
except ImportError as e:
    raise ImportError(
        f"{e}\n\n"
        "The 'spaco' module requires additional packages. Please install them with:\n\n"
        "    pip install colormath pyciede2000\n"
    ) from e

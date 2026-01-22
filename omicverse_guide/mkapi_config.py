from __future__ import annotations

import os
from pathlib import Path
import sys


def before_on_config(config, _plugin) -> None:
    os.environ.setdefault("OMICVERSE_DISABLE_LLM", "1")
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def page_title(name: str, _depth: int) -> str:
    return name.split(".")[-1]


def section_title(name: str, _depth: int) -> str:
    return name.split(".")[-1]


def toc_title(name: str, _depth: int) -> str:
    return name.split(".")[-1]

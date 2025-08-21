"""Scope management utilities for refining project requirements.

This package provides tools to interactively clarify project scope and
summarize conversations into concise briefs.
"""

from .clarifier import Clarifier
from .brief import BriefGenerator, ProjectBrief
from .manager import ScopeManager

__all__ = ["Clarifier", "BriefGenerator", "ProjectBrief", "ScopeManager"]

"""Research coordination utilities.

This package provides primitives for breaking down research briefs into
subtopics and running specialised agents to gather findings."""

from .agent import ResearchAgent, Finding, SourceCitation
from .supervisor import Supervisor

__all__ = [
    "ResearchAgent",
    "Finding",
    "SourceCitation",
    "Supervisor",
]

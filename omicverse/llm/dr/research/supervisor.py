"""Supervisor that orchestrates research agents."""

from __future__ import annotations

from typing import Iterable, List, Sequence

from .agent import ResearchAgent, Finding


class Supervisor:
    """Breaks briefs into subtopics and spawns agents to handle each."""

    def __init__(self, vector_store, tools: Iterable | None = None) -> None:
        self.vector_store = vector_store
        self.tools = list(tools) if tools is not None else []

    def split_brief(self, brief: str) -> List[str]:
        """Split a brief into a list of subtopic strings."""

        return [part.strip() for part in brief.split("\n") if part.strip()]

    def run(self, brief: str) -> Sequence[Finding]:
        """Generate findings for each subtopic in ``brief``."""

        topics = self.split_brief(brief)
        findings = []
        for topic in topics:
            agent = ResearchAgent(self.vector_store, self.tools)
            findings.append(agent.search(topic))
        return findings

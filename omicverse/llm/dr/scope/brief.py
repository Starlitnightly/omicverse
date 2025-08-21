"""Utilities for condensing dialogue into structured project briefs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class ProjectBrief:
    """Structured representation of a project's requirements."""

    title: str
    objectives: List[str]
    constraints: List[str]


class BriefGenerator:
    """Generate a :class:`ProjectBrief` from a dialogue history."""

    def generate(self, dialogue: Sequence[str]) -> ProjectBrief:
        """Convert dialogue lines into a structured brief.

        Parameters
        ----------
        dialogue:
            Ordered conversation entries.

        Returns
        -------
        ProjectBrief
            The condensed brief containing title, objectives and constraints.
        """
        if not dialogue:
            return ProjectBrief(title="Untitled", objectives=[], constraints=[])

        title = dialogue[0].strip()
        objectives: List[str] = []
        constraints: List[str] = []

        for line in dialogue[1:]:
            cleaned = line.strip()
            if cleaned.lower().startswith("constraint:"):
                constraints.append(cleaned.split(":", 1)[1].strip())
            else:
                objectives.append(cleaned)

        return ProjectBrief(title=title, objectives=objectives, constraints=constraints)

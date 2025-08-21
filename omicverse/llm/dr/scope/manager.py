"""High level manager for refining and summarizing project scope."""

from __future__ import annotations

from typing import List

from .brief import BriefGenerator, ProjectBrief
from .clarifier import Clarifier


class ScopeManager:
    """Coordinate clarification and brief generation steps."""

    def __init__(self, clarifier: Clarifier | None = None,
                 brief_generator: BriefGenerator | None = None) -> None:
        self.clarifier = clarifier or Clarifier()
        self.brief_generator = brief_generator or BriefGenerator()
        self.dialogue: List[str] = []

    def add_message(self, message: str) -> None:
        """Append a message to the dialogue history."""
        self.dialogue.append(message)

    def ask_clarification(self, question: str) -> str:
        """Use the clarifier to ask a follow-up question.

        The question is stored in the dialogue for context.
        """
        response = self.clarifier.ask(question)
        self.dialogue.append(question)
        return response

    def generate_brief(self) -> ProjectBrief:
        """Create a structured brief from the accumulated dialogue."""
        return self.brief_generator.generate(self.dialogue)

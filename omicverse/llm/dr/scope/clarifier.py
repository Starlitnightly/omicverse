"""Interactive clarification utilities.

The :class:`Clarifier` class assists in refining project scope by generating
follow-up questions. Each question is returned with a prefix from
:class:`MessageStandards` to keep messaging consistent across the
codebase.
"""

from __future__ import annotations

from typing import List

from ...utils.message_standards import MessageStandards


class Clarifier:
    """Generate follow-up questions to clarify user intent."""

    def __init__(self) -> None:
        self.questions: List[str] = []

    def ask(self, question: str) -> str:
        """Record a question and return a standardized message.

        Parameters
        ----------
        question:
            The question to present to the user.

        Returns
        -------
        str
            The question prefixed with a ``MessageStandards`` message to
            encourage consistency across interactions.
        """
        self.questions.append(question)
        return f"{MessageStandards.OPERATION_SUCCESS} {question}"

"""High-level manager coordinating scope, research and writing steps.

This module provides the :class:`ResearchManager` which ties together the
``scope``, ``research`` and ``write`` helpers to produce a simple end-to-end
research workflow. Model configuration is delegated to
:class:`~omicverse.llm.model_factory.ModelFactory` and user-facing messages use
:class:`~omicverse.llm.utils.message_standards.MessageStandards` for
consistency.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from ..model_factory import ModelFactory
from ..utils.message_standards import MessageStandards
from .scope.manager import ScopeManager, ProjectBrief
from .research.supervisor import Supervisor, Finding
from .write.report import ReportWriter, Section


class ResearchManager:
    """Orchestrate the full research pipeline from request to report.

    Parameters
    ----------
    model_type:
        Identifier of the underlying model passed to ``ModelFactory``.
    model_path:
        Optional path to model weights or configuration.
    device:
        Device string forwarded to ``ModelFactory.create_model``.
    vector_store:
        Object implementing ``search`` used by research agents.
    tools:
        Optional iterable of tools provided to research agents.
    fmt:
        Output format for the final report (``"markdown"`` or ``"html"``).
    **model_kwargs:
        Additional keyword arguments for the model constructor.
    """

    def __init__(
        self,
        model_type: str = "scgpt",
        model_path: str | None = None,
        device: str | None = None,
        *,
        vector_store,
        tools: Iterable | None = None,
        fmt: str = "markdown",
        **model_kwargs,
    ) -> None:
        self._status(MessageStandards.LOADING_MODEL)
        self.model = ModelFactory.create_model(
            model_type, model_path, device, **model_kwargs
        )
        self._status(MessageStandards.MODEL_LOADED)

        self.scope_manager = ScopeManager()
        self.supervisor = Supervisor(vector_store, tools)
        self.report_writer = ReportWriter(fmt=fmt)

    def _status(self, message: str) -> None:
        """Output a standardised status message."""

        print(message)

    # ------------------------------------------------------------------
    def scope(self, request: str) -> ProjectBrief:
        """Convert a user ``request`` into a :class:`ProjectBrief`.

        The request is recorded as part of the scope dialogue before
        generating the brief.
        """

        self._status(MessageStandards.PREPROCESSING_START)
        self.scope_manager.add_message(request)
        brief = self.scope_manager.generate_brief()
        self._status(MessageStandards.PREPROCESSING_COMPLETE)
        return brief

    # ------------------------------------------------------------------
    def research(self, brief: ProjectBrief) -> Sequence[Finding]:
        """Produce findings for each objective in ``brief``."""

        self._status(MessageStandards.PREDICTING_START)
        topics = "\n".join(brief.objectives)
        findings = self.supervisor.run(topics)
        self._status(MessageStandards.PREDICTING_COMPLETE)
        return findings

    # ------------------------------------------------------------------
    def write(self, brief: ProjectBrief, findings: Sequence[Finding]) -> str:
        """Compose a report from ``findings`` informed by ``brief``."""

        self._status(MessageStandards.EMBEDDING_START)
        sections = [
            Section(title=f.topic, text=f.text, citations=f.sources)
            for f in findings
        ]
        report = self.report_writer.compose(sections)
        self._status(MessageStandards.EMBEDDING_COMPLETE)
        return report

    # ------------------------------------------------------------------
    def run(self, request: str) -> str:
        """Execute the full research pipeline for ``request``."""

        brief = self.scope(request)
        findings = self.research(brief)
        return self.write(brief, findings)

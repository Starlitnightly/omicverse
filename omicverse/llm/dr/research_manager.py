"""High-level manager coordinating scope, research and writing steps.

This module provides the :class:`ResearchManager` which ties together the
``scope``, ``research`` and ``write`` helpers to produce a simple end-to-end
research workflow. Model configuration is delegated to
:class:`~omicverse.llm.model_factory.ModelFactory` and user-facing messages use
:class:`~omicverse.llm.utils.message_standards.MessageStandards` for
consistency.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from ..model_factory import ModelFactory
from ..utils.message_standards import MessageStandards
from .scope.manager import ScopeManager, ProjectBrief
from .research.supervisor import Supervisor, Finding
from .research.agent import SourceCitation
from .write.report import ReportWriter, Section
from .validation import validate_query
from .cache import cache


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
    @cache
    def _scope_cached(self, request: str) -> ProjectBrief:
        self._status(MessageStandards.PREPROCESSING_START)
        self.scope_manager.add_message(request)
        brief = self.scope_manager.generate_brief()
        self._status(MessageStandards.PREPROCESSING_COMPLETE)
        return brief

    def scope(self, request: str) -> ProjectBrief:
        """Convert a user ``request`` into a :class:`ProjectBrief`.

        The request is recorded as part of the scope dialogue before
        generating the brief. The ``request`` is validated before processing
        and the result is cached to avoid recomputation.
        """

        validate_query(request)
        return self._scope_cached(request)

    # ------------------------------------------------------------------
    @cache
    def _research_cached(self, objectives: Tuple[str, ...]) -> Sequence[Finding]:
        self._status(MessageStandards.PREDICTING_START)
        topics = "\n".join(objectives)
        findings = self.supervisor.run(topics)
        self._status(MessageStandards.PREDICTING_COMPLETE)
        return findings

    def research(self, brief: ProjectBrief) -> Sequence[Finding]:
        """Produce findings for each objective in ``brief``."""

        for obj in brief.objectives:
            validate_query(obj)
        return self._research_cached(tuple(brief.objectives))

    # ------------------------------------------------------------------
    @cache
    def _write_cached(
        self,
        sections_key: Tuple,
    ) -> str:
        self._status(MessageStandards.EMBEDDING_START)
        sections = []
        for title, text, citations in sections_key:
            section_citations = [
                SourceCitation(
                    source_id=cid,
                    content=content,
                    metadata=dict(meta) if meta is not None else None,
                )
                for cid, content, meta in citations
            ]
            sections.append(Section(title=title, text=text, citations=section_citations))
        report = self.report_writer.compose(sections)
        self._status(MessageStandards.EMBEDDING_COMPLETE)
        return report

    def write(self, brief: ProjectBrief, findings: Sequence[Finding]) -> str:
        """Compose a report from ``findings`` informed by ``brief``."""

        validate_query(brief.title)
        sections_key = tuple(
            (
                f.topic,
                f.text,
                tuple(
                    (
                        c.source_id,
                        c.content,
                        tuple(sorted(c.metadata.items())) if c.metadata else None,
                    )
                    for c in f.sources
                ),
            )
            for f in findings
        )
        return self._write_cached(sections_key)

    # ------------------------------------------------------------------
    def run(self, request: str) -> str:
        """Execute the full research pipeline for ``request``."""

        brief = self.scope(request)
        findings = self.research(brief)
        return self.write(brief, findings)

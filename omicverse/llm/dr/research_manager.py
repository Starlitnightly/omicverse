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
import os

from ..model_factory import ModelFactory
from ..utils.message_standards import MessageStandards
from .scope.manager import ScopeManager, ProjectBrief
from .research.supervisor import Supervisor, Finding
from .research.agent import SourceCitation
from .write.report import ReportWriter, Section
from .write.synthesizer import TextSynthesizer, SimpleSynthesizer, SynthesisInput
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
        # Report synthesis configuration
        synthesizer: TextSynthesizer | None = None,
        # Retrieval/runtime configuration
        cache: bool | None = None,
        verify: bool = False,
        # Scope configuration
        llm_scope: bool = False,
        **model_kwargs,
    ) -> None:
        self._status(MessageStandards.LOADING_MODEL)
        self.model = ModelFactory.create_model(
            model_type, model_path, device, **model_kwargs
        )
        self._status(MessageStandards.MODEL_LOADED)

        self.scope_manager = ScopeManager()
        self.llm_scope = llm_scope or bool(os.getenv("OV_DR_LLM_SCOPE"))
        # Allow simple string flag to enable web-backed retrieval
        vs = vector_store
        if isinstance(vs, str):
            vs_flag = vs.lower().strip()
            if vs_flag in {"web", "web:tavily", "web:duckduckgo", "web:brave", "web:pubmed", "web:embed", "web:embed:tavily", "web:embed:duckduckgo", "web:embed:brave", "web:embed:pubmed"}:
                from .retrievers.web_store import WebRetrieverStore
                try:
                    from .retrievers.embed_web import EmbedWebRetriever
                except Exception:
                    EmbedWebRetriever = None  # type: ignore

                if vs_flag == "web":
                    backend = (
                        "tavily"
                        if os.getenv("TAVILY_API_KEY")
                        else ("brave" if os.getenv("BRAVE_API_KEY") else "duckduckgo")
                    )
                elif vs_flag in {"web:tavily", "web:embed:tavily"}:
                    backend = "tavily"
                elif vs_flag in {"web:brave", "web:embed:brave"}:
                    backend = "brave"
                elif vs_flag in {"web:pubmed", "web:embed:pubmed"}:
                    backend = "pubmed"
                else:
                    backend = "duckduckgo"
                cache_enabled = cache if cache is not None else bool(os.getenv("OV_DR_CACHE"))
                if vs_flag.startswith("web:embed") and EmbedWebRetriever is not None:
                    vs = EmbedWebRetriever(backend=backend, cache=cache_enabled)
                else:
                    vs = WebRetrieverStore(backend=backend, cache=cache_enabled)

        self.supervisor = Supervisor(vs, tools)
        self.report_writer = ReportWriter(fmt=fmt)
        self.synthesizer = synthesizer or SimpleSynthesizer()
        self._verify_outputs = verify

    def _status(self, message: str) -> None:
        """Output a standardised status message."""

        print(message)

    # ------------------------------------------------------------------
    @cache
    def _scope_cached(self, request: str) -> ProjectBrief:
        self._status(MessageStandards.PREPROCESSING_START)
        self.scope_manager.add_message(request)
        brief = self.scope_manager.generate_brief()
        # Optional LLM-assisted scoping
        if self.llm_scope:
            try:
                from .scope.llm_scoper import suggest_brief

                sug = suggest_brief(request)
                if sug.get("objectives"):
                    brief.objectives = list(dict.fromkeys([*brief.objectives, *sug["objectives"]]))
                if sug.get("constraints"):
                    brief.constraints = list(dict.fromkeys([*brief.constraints, *sug["constraints"]]))
            except Exception:
                pass
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
    def _research_cached(self, objectives: Tuple[str, ...], constraints_key: Tuple[str, ...]) -> Sequence[Finding]:
        self._status(MessageStandards.PREDICTING_START)
        topics = "\n".join(self._augment_topics_with_constraints(objectives, constraints_key))
        findings = self.supervisor.run(topics)
        self._status(MessageStandards.PREDICTING_COMPLETE)
        return findings

    def _augment_topics_with_constraints(self, objectives: Sequence[str], constraints: Sequence[str]) -> Sequence[str]:
        # Inject simple domain/date constraints into topics for web backends
        # Parse constraints like 'date:>=YYYY' and 'domain:foo|bar'
        date_since = None
        domains: list[str] = []
        for c in constraints:
            c = c.strip()
            if c.startswith("date:") and ">=" in c:
                date_since = c.split(">=", 1)[1]
            if c.startswith("domain:"):
                domains.extend([d.strip() for d in c.split(":", 1)[1].split("|") if d.strip()])

        out = []
        for obj in objectives:
            q = obj
            if domains:
                # simple site biasing tokens
                q += " " + " ".join(f"site:{d}" for d in domains)
            if date_since:
                # advisory token for engines that support it; otherwise influences ranking
                q += f" after:{date_since}"
            out.append(q)
        return out

    def research(self, brief: ProjectBrief) -> Sequence[Finding]:
        """Produce findings for each objective in ``brief``."""

        for obj in brief.objectives:
            validate_query(obj)
        return self._research_cached(tuple(brief.objectives), tuple(brief.constraints))

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

    def compose_sections(self, brief: ProjectBrief, findings: Sequence[Finding]) -> Sequence[Section]:
        """Build report sections (synthesis + per-topic) with citations.

        Separates composition from rendering to allow metadata access by callers.
        """

        validate_query(brief.title)

        # 1) Synthesize an executive-style top section
        synthesis_text = self.synthesizer.synthesize(
            SynthesisInput(title=brief.title, objectives=brief.objectives, findings=findings)
        )

        # Collect union of citations for the synthesized section
        union_citations = []
        for f in findings:
            union_citations.extend(f.sources)

        # Optional verification/quality scoring
        if self._verify_outputs and union_citations:
            try:
                from .verify.quality import SourceVerifier

                verifier = SourceVerifier()
                union_citations = verifier.verify_all(union_citations)
                # Also verify per-finding citations
                for f in findings:
                    f.sources = verifier.verify_all(f.sources)
            except Exception:
                pass

        sections: list[Section] = [
            Section(title="Comprehensive Report", text=synthesis_text, citations=union_citations)
        ]

        # 2) Add transparent per-topic sections
        for f in findings:
            sections.append(Section(title=f.topic, text=f.text, citations=f.sources))

        return sections

    def write(self, brief: ProjectBrief, findings: Sequence[Finding]) -> str:
        """Compose a comprehensive report from ``findings`` informed by ``brief``.

        Uses :meth:`compose_sections` to build sections, then renders using
        :class:`ReportWriter` with citations and references.
        """

        sections = self.compose_sections(brief, findings)
        # Convert to cacheable tuple key
        sections_key = tuple(
            (
                s.title,
                s.text,
                tuple(
                    (
                        c.source_id,
                        c.content,
                        tuple(sorted(c.metadata.items())) if c.metadata else None,
                    )
                    for c in s.citations
                ),
            )
            for s in sections
        )
        return self._write_cached(sections_key)

    # ------------------------------------------------------------------
    def run(self, request: str) -> str:
        """Execute the full research pipeline for ``request``."""

        brief = self.scope(request)
        findings = self.research(brief)
        return self.write(brief, findings)

    # ------------------------------------------------------------------
    def run_with_meta(self, request: str) -> tuple[str, Sequence[Section]]:
        """Execute full pipeline and also return composed sections.

        Returns a tuple of (report, sections) so callers can inspect citations
        and their metadata (e.g., open-access, proxy suggestions, paywall flags).
        """
        brief = self.scope(request)
        findings = self.research(brief)
        sections = self.compose_sections(brief, findings)
        report = self._write_cached(
            tuple(
                (
                    s.title,
                    s.text,
                    tuple(
                        (
                            c.source_id,
                            c.content,
                            tuple(sorted(c.metadata.items())) if c.metadata else None,
                        )
                        for c in s.citations
                    ),
                )
                for s in sections
            )
        )
        return report, sections

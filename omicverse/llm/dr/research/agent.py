"""Sub-agent that performs document search and summarisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Protocol, Sequence


class VectorStore(Protocol):
    """Minimal protocol for vector stores used by :class:`ResearchAgent`."""

    def search(self, query: str) -> Sequence[Any]:
        """Return a sequence of documents relevant to ``query``."""


@dataclass
class SourceCitation:
    """Reference to a source document supporting a finding."""

    source_id: str
    content: str
    metadata: dict | None = None


@dataclass
class Finding:
    """Finding produced by a research agent."""

    topic: str
    text: str
    sources: List[SourceCitation]


class ResearchAgent:
    """Agent that queries a vector store and optional tools to build findings."""

    def __init__(self, vector_store: VectorStore, tools: Iterable[Any] | None = None) -> None:
        self.vector_store = vector_store
        self.tools = list(tools) if tools is not None else []

    def search(self, topic: str) -> Finding:
        """Search for documents on ``topic`` and return a cleaned summary."""

        docs = self.vector_store.search(topic)
        texts: List[str] = []
        citations: List[SourceCitation] = []
        for idx, doc in enumerate(docs):
            text = getattr(doc, "text", str(doc))
            texts.append(text)
            citations.append(
                SourceCitation(
                    source_id=str(getattr(doc, "id", idx)),
                    content=text,
                    metadata=getattr(doc, "metadata", None),
                )
            )

        cleaned = self._clean("\n".join(texts))
        # External tools could be applied here; omitted for simplicity
        return Finding(topic=topic, text=cleaned, sources=citations)

    @staticmethod
    def _clean(text: str) -> str:
        """Basic whitespace cleanup for retrieved text."""

        return " ".join(text.split())

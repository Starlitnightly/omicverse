# rag_adk_agent.py — Google ADK agent wrapping the two‑stage RAG system
# -----------------------------------------------------------------------------
# This agent exposes your existing two‑stage code RAG as **function tools** in
# Google’s Agent Development Kit (ADK).  Launch with any of the following:
#
#   adk web                       # interactive dev UI (recommended)
#   adk run rag_adk_agent         # chat from the terminal
#   adk api_server rag_adk_agent  # local FastAPI server for REST calls
#
# Prerequisites:
#   pip install google-adk chromadb langchain google-generativeai
#   export GOOGLE_API_KEY=...            # from Google AI Studio
#   # (or configure Vertex AI via GOOGLE_GENAI_USE_VERTEXAI etc.)
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
from typing import List, Dict

# ADK imports
from google.adk.agents import Agent
from google.adk.tools import FunctionTool

# Your existing modules
from rag_system import RAGSystem, PackageConfig  # noqa: F401
from app import get_rag_system                   # reuse cached builder

###############################################################################
#                       GLOBAL RAG INSTANCE (lazy‑loaded)                     #
###############################################################################

_rag_system: RAGSystem | None = None


def _ensure_rag_system() -> RAGSystem:
    """Initialise (or reuse) the heavy RAG engine with non‑OpenAI models."""
    global _rag_system
    if _rag_system is None:
        file_selection_model = os.getenv(
            "FILE_SELECTION_MODEL", "gemini-2.0-flash"
        )
        query_processing_model = os.getenv(
            "QUERY_PROCESSING_MODEL", "gemini-2.0-flash"
        )

        _rag_system = get_rag_system(
            {
                "file_selection_model": file_selection_model,
                "query_processing_model": query_processing_model,
            }
        )
    return _rag_system

###############################################################################
#                              TOOL FUNCTIONS                                 #
###############################################################################


def list_packages() -> List[str]:
    """Return a list of code packages indexed by the RAG system."""
    rag = _ensure_rag_system()
    return list(rag.first_stages.keys())


# Mark as explicit ADK FunctionTool (optional; ADK will infer if omitted)
@FunctionTool
def rag_query(query: str, package: str = "cellrank_notebooks") -> Dict[str, str | List[str]]:
    """Answer a natural‑language *query* about the given *package*.

    Parameters
    ----------
    query : str
        The user question.
    package : str, default "cellrank_notebooks"
        Which code package to search; call `list_packages` to see options.

    Returns
    -------
    dict with keys:
      answer : str
        The generated answer.
      files : list[str]
        Filenames that contributed to the answer.
    """
    rag = _ensure_rag_system()
    if package not in rag.first_stages:
        raise ValueError(
            f"Unknown package '{package}'. Available: {', '.join(rag.first_stages)}"
        )

    files, answer = rag.query(query=query, package_name=package)
    return {"answer": answer, "files": files}

###############################################################################
#                                 AGENT SETUP                                 #
###############################################################################

MODEL_ID = os.getenv("GENAI_MODEL", "gemini-2.0-flash")

root_agent = Agent(
    name="rag_code_agent",
    model=MODEL_ID,
    description=(
        "Agent that answers questions about multiple open‑source bioinformatics "
        "code bases using a two‑stage Retrieval‑Augmented Generation pipeline."
    ),
    instruction=(
        "You are an expert code assistant.  When you need domain knowledge, "
        "use the provided tools, especially `rag_query`, to search the code base "
        "and craft accurate, concise answers.  Cite filenames where relevant."
    ),
    tools=[list_packages, rag_query],
)

# When ADK discovers this module (via `adk web`, `adk run`, etc.) it will load
# `root_agent` automatically.  No if‑__name__ guard is needed.

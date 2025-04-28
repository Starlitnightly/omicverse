# rag_mcp_server.py — MCP server wrapping the two‑stage RAG system (OpenAI‑free edition)
# -----------------------------------------------------------------------------
# This version removes all **hard‑coded** references to OpenAI models so the
# service can run in environments where OpenAI endpoints are unreachable or
# forbidden.  The actual model names come from **environment variables** with
# sensible non‑OpenAI defaults (Gemini 1.5 Pro is used if available; otherwise
# you may point `MODEL_SELECTOR` at any local model such as an Ollama instance).
# -----------------------------------------------------------------------------
# Quick start (dev):
#     mcp dev rag_mcp_server.py
# Or run as a plain FastAPI‑backed MCP server:
#     python rag_mcp_server.py
#
# Prerequisites:
#   pip install "mcp[cli]" chromadb langchain google‑generativeai  # adjust as needed
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
from typing import List, Dict, Tuple

from mcp.server.fastmcp import FastMCP, Context
import mcp.types as types

# Local modules from your existing code base
from rag_system import RAGSystem, PackageConfig  # noqa: F401
from app import get_rag_system                   # reuse cached builder

###############################################################################
#                       GLOBAL RAG INSTANCE (lazy‑loaded)                     #
###############################################################################

_rag_system: RAGSystem | None = None


def _ensure_rag_system() -> RAGSystem:
    """Build (or reuse) the heavy RAG engine, selecting non‑OpenAI models."""

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
#                                MCP SERVER                                   #
###############################################################################

mcp = FastMCP(
    "RAG‑Code‑Assistant",
    description=(
        "Two‑stage code‑aware Retrieval‑Augmented Generation server that answers "
        "questions about multiple open‑source bioinformatics code bases without "
        "depending on any OpenAI service."
    ),
    author="Your‑Name‑Or‑Org",
    version="0.2.0",
    dependencies=[
        "chromadb",
        "langchain",
        "google‑generativeai",  # remove if you use only local models
    ],
)

###############################################################################
#                                   RESOURCES                                 #
###############################################################################


@mcp.resource("packages://list")
def list_packages() -> str:
    """Return a newline‑separated list of all managed code packages."""
    rag = _ensure_rag_system()
    return "\n".join(rag.first_stages.keys())


@mcp.resource("ragfile://{package}/{path:path}")
def file_source(package: str, path: str) -> str:
    """Return raw source code for *path* inside *package* (read‑only)."""
    rag = _ensure_rag_system()
    if package not in rag.first_stages:
        raise FileNotFoundError(f"Unknown package: {package!r}")

    second_stage = rag.second_stages[package]
    abs_root = os.path.dirname(second_stage.annotated_scripts_directory)
    abs_path = os.path.normpath(os.path.join(abs_root, path))
    if not abs_path.startswith(abs_root):
        raise FileNotFoundError("Path traversal detected")
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(abs_path)
    with open(abs_path, "r", encoding="utf-8") as fh:
        return fh.read()

###############################################################################
#                                     TOOL                                    #
###############################################################################


@mcp.tool()
async def rag_query(query: str, package: str = "cellrank_notebooks") -> dict:
    """Run a natural‑language query against *package* and return answer + files."""

    rag = _ensure_rag_system()
    if package not in rag.first_stages:
        raise ValueError(
            f"Unknown package '{package}'. Available: {', '.join(rag.first_stages)}"
        )

    files, answer = rag.query(query=query, package_name=package)
    return {"answer": answer, "files": files}

###############################################################################
#                                PROMPT EXAMPLE                               #
###############################################################################


@mcp.prompt()
def review_code(code: str) -> list[types.PromptMessage]:
    """Prompt template asking for a review of *code* (model‑agnostic)."""
    return [
        types.PromptMessage(
            role="user",
            content=types.TextContent(type="text", text="Please review this code snippet:"),
        ),
        types.PromptMessage(
            role="user",
            content=types.TextContent(type="code", text=code),
        ),
    ]

###############################################################################
#                                   ENTRYPOINT                                #
###############################################################################

if __name__ == "__main__":
    mcp.run()

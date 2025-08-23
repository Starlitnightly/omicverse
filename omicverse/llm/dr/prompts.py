"""Prompt templates used by domain research helpers.

The utilities here centralise textual templates for the research pipeline.
Each function returns a formatted prompt string and accepts ``**kwargs`` to
allow model specific customisation without modifying the base templates.
Defaults include a brief description of OmicVerse so models understand the
multi-omics context they operate within.
"""

from __future__ import annotations

from typing import Any, Dict


OMICVERSE_CONTEXT = (
    "You are interacting with OmicVerse, an open-source multi-omics research "
    "framework that bridges insights across bulk, single-cell and spatial "
    "data."
)


def _merge(defaults: Dict[str, str], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge default parameters with user provided overrides."""
    merged = defaults.copy()
    merged.update(overrides)
    return merged


def user_clarification_dialogue(request: str, **kwargs: Any) -> str:
    """Return a prompt asking the model to clarify the user request.

    Parameters
    ----------
    request:
        The initial user request to clarify.
    **kwargs:
        Optional keyword arguments to tweak the template. Useful for
        model-specific system instructions or styles.

    Returns
    -------
    str
        Rendered prompt string.
    """
    defaults = {
        "system": (
            "You are a helpful research assistant gathering requirements for "
            "OmicVerse."
        ),
        "context": OMICVERSE_CONTEXT,
        "instruction": (
            "Ask brief questions to resolve any ambiguities in the request "
            "before continuing."
        ),
    }
    params = _merge(defaults, kwargs)
    template = (
        "{system}\n"
        "{context}\n"
        "User request: {request}\n"
        "{instruction}"
    )
    return template.format(request=request, **params)


def sub_agent_search_summary(topic: str, **kwargs: Any) -> str:
    """Prompt template for sub-agents performing search and summarisation.

    Parameters
    ----------
    topic:
        Topic to investigate.
    **kwargs:
        Optional keyword arguments to customise instructions.

    Returns
    -------
    str
        Rendered prompt string.
    """
    defaults = {
        "system": "You are an expert researcher contributing to OmicVerse.",
        "context": OMICVERSE_CONTEXT,
        "instruction": (
            "Search the provided knowledge base for information on the topic "
            "and summarise the findings with citations."
        ),
    }
    params = _merge(defaults, kwargs)
    template = (
        "{system}\n"
        "{context}\n"
        "Research topic: {topic}\n"
        "{instruction}"
    )
    return template.format(topic=topic, **params)


def final_report_synthesis(findings: str, **kwargs: Any) -> str:
    """Prompt template for synthesising a final report from findings.

    Parameters
    ----------
    findings:
        Concatenated findings from sub-agents.
    **kwargs:
        Optional keyword arguments to adjust the final report style or tone.

    Returns
    -------
    str
        Rendered prompt string.
    """
    defaults = {
        "system": "You are a scientific writer assembling a report for OmicVerse.",
        "context": OMICVERSE_CONTEXT,
        "instruction": (
            "Combine the findings into a coherent report. Cite sources and "
            "maintain a neutral, academic tone."
        ),
    }
    params = _merge(defaults, kwargs)
    template = (
        "{system}\n"
        "{context}\n"
        "Findings:\n{findings}\n"
        "{instruction}"
    )
    return template.format(findings=findings, **params)


__all__ = [
    "user_clarification_dialogue",
    "sub_agent_search_summary",
    "final_report_synthesis",
]

"""Re-export synthesizer types from `omicverse.llm.dr`.

Allows `from omicverse.llm.domain_research.write.synthesizer import ...`.
"""

from __future__ import annotations

from ...dr.write.synthesizer import (  # noqa: F401
    TextSynthesizer,
    SimpleSynthesizer,
    PromptSynthesizer,
    SynthesisInput,
)

__all__ = [
    "TextSynthesizer",
    "SimpleSynthesizer",
    "PromptSynthesizer",
    "SynthesisInput",
]


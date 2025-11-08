"""
OmicVerse Skills Verifier

A verification system that tests skill selection and ordering using LLM reasoning,
mimicking how Claude Code autonomously selects skills based on task descriptions.
"""

from .data_structures import (
    SkillDescription,
    NotebookTask,
    LLMSelectionResult,
    VerificationResult,
)
from .skill_description_loader import SkillDescriptionLoader
from .llm_skill_selector import LLMSkillSelector, create_skill_selector
from .skill_description_quality import (
    SkillDescriptionQualityChecker,
    QualityMetrics,
    EffectivenessResult,
    ComparisonResult,
    create_quality_checker,
)
from .notebook_task_extractor import (
    NotebookTaskExtractor,
    create_task_extractor,
)
from .end_to_end_verifier import (
    EndToEndVerifier,
    VerificationRunConfig,
    VerificationSummary,
    create_verifier,
)
from .cli import main as cli_main

__all__ = [
    "SkillDescription",
    "NotebookTask",
    "LLMSelectionResult",
    "VerificationResult",
    "SkillDescriptionLoader",
    "LLMSkillSelector",
    "create_skill_selector",
    "SkillDescriptionQualityChecker",
    "QualityMetrics",
    "EffectivenessResult",
    "ComparisonResult",
    "create_quality_checker",
    "NotebookTaskExtractor",
    "create_task_extractor",
    "EndToEndVerifier",
    "VerificationRunConfig",
    "VerificationSummary",
    "create_verifier",
    "cli_main",
]

"""Compatibility wrapper for shared Agent Skill utilities."""
from omicverse.utils.skill_registry import (  # noqa: F401
    SkillDefinition,
    SkillMatch,
    SkillRegistry,
    SkillRouter,
    build_skill_registry,
)

__all__ = [
    "SkillDefinition",
    "SkillMatch",
    "SkillRegistry",
    "SkillRouter",
    "build_skill_registry",
]

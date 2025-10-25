"""Utilities for loading and routing OmicVerse project Agent Skills."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SkillDefinition:
    """Represents a single Agent Skill discovered on disk."""

    name: str
    description: str
    path: Path
    body: str
    metadata: Dict[str, str] = field(default_factory=dict)

    def prompt_instructions(self, max_chars: int = 4000) -> str:
        """Return the main instruction body, trimmed if necessary."""

        text = self.body.strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    @property
    def summary_text(self) -> str:
        """Combine metadata and first section for lightweight scoring."""

        header = f"{self.name}\n{self.description}\n"
        primary_section = self.body.split("\n\n", 1)[0]
        return header + primary_section


@dataclass
class SkillMatch:
    """Represents a routing decision for a query."""

    skill: SkillDefinition
    score: float

    def as_dict(self) -> Dict[str, str]:
        return {
            "name": self.skill.name,
            "score": f"{self.score:.3f}",
            "description": self.skill.description,
            "path": str(self.skill.path),
        }


class SkillRegistry:
    """Loads skills from the filesystem and stores their metadata."""

    def __init__(self, skill_root: Path):
        self.skill_root = skill_root
        self._skills: Dict[str, SkillDefinition] = {}

    @property
    def skills(self) -> Dict[str, SkillDefinition]:
        return self._skills

    def load(self) -> None:
        """Discover every SKILL.md under the configured skill root."""

        if not self.skill_root.exists():
            logger.warning("Skill root %s does not exist; no skills loaded.", self.skill_root)
            self._skills = {}
            return

        discovered: Dict[str, SkillDefinition] = {}
        for skill_file in sorted(self.skill_root.glob("*/SKILL.md")):
            definition = self._parse_skill_file(skill_file)
            if not definition:
                continue
            key = definition.name.lower()
            if key in discovered:
                logger.warning("Duplicate skill name '%s' found; keeping first occurrence.", definition.name)
                continue
            discovered[key] = definition
            logger.info("Loaded skill '%s' from %s", definition.name, skill_file)
        self._skills = discovered

    def _parse_skill_file(self, skill_file: Path) -> Optional[SkillDefinition]:
        try:
            content = skill_file.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error("Unable to read skill file %s: %s", skill_file, exc)
            return None

        lines = content.splitlines()
        if not lines or lines[0].strip() != "---":
            logger.warning("Skill file %s is missing YAML frontmatter.", skill_file)
            return None

        try:
            closing_index = lines.index("---", 1)
        except ValueError:
            logger.warning("Skill file %s has unterminated YAML frontmatter.", skill_file)
            return None

        frontmatter_lines = lines[1:closing_index]
        metadata = self._parse_frontmatter(frontmatter_lines)
        name = metadata.get("name")
        description = metadata.get("description")
        if not name or not description:
            logger.warning("Skill file %s is missing required name/description metadata.", skill_file)
            return None

        body = "\n".join(lines[closing_index + 1 :]).strip()
        skill_path = skill_file.parent
        return SkillDefinition(name=name, description=description, path=skill_path, body=body, metadata=metadata)

    @staticmethod
    def _parse_frontmatter(lines: Iterable[str]) -> Dict[str, str]:
        metadata: Dict[str, str] = {}
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip('"')
        return metadata


class SkillRouter:
    """Simple keyword-based router that ranks skills for a query."""

    def __init__(self, registry: SkillRegistry, min_score: float = 0.1):
        self.registry = registry
        self.min_score = min_score
        self._skill_vectors: Dict[str, Dict[str, int]] = {}
        self._build_vectors()

    def _build_vectors(self) -> None:
        self._skill_vectors = {
            key: self._token_frequency(definition.summary_text)
            for key, definition in self.registry.skills.items()
        }

    def refresh(self) -> None:
        self._build_vectors()

    def route(self, query: str, top_k: int = 1) -> List[SkillMatch]:
        if not query or not query.strip():
            return []
        query_vector = self._token_frequency(query)
        if not query_vector:
            return []

        scored: List[Tuple[str, float]] = []
        for key, skill_vector in self._skill_vectors.items():
            score = self._cosine_similarity(query_vector, skill_vector)
            scored.append((key, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        matches: List[SkillMatch] = []
        for key, score in scored[:top_k]:
            if score < self.min_score:
                continue
            skill = self.registry.skills.get(key)
            if not skill:
                continue
            matches.append(SkillMatch(skill=skill, score=score))
        return matches

    @staticmethod
    def _token_frequency(text: str) -> Dict[str, int]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        freq: Dict[str, int] = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1
        return freq

    @staticmethod
    def _cosine_similarity(vec_a: Dict[str, int], vec_b: Dict[str, int]) -> float:
        if not vec_a or not vec_b:
            return 0.0
        common = set(vec_a.keys()) & set(vec_b.keys())
        numerator = sum(vec_a[token] * vec_b[token] for token in common)
        if numerator == 0:
            return 0.0
        sum_sq_a = sum(value * value for value in vec_a.values())
        sum_sq_b = sum(value * value for value in vec_b.values())
        denominator = (sum_sq_a ** 0.5) * (sum_sq_b ** 0.5)
        if denominator == 0:
            return 0.0
        return numerator / denominator


def build_skill_registry(project_root: Path) -> Optional[SkillRegistry]:
    """Helper to create and load a registry from the project root."""

    skill_root = project_root / ".claude" / "skills"
    registry = SkillRegistry(skill_root=skill_root)
    registry.load()
    if not registry.skills:
        logger.warning("No skills discovered under %s", skill_root)
    return registry


__all__ = [
    "SkillDefinition",
    "SkillMatch",
    "SkillRegistry",
    "SkillRouter",
    "build_skill_registry",
]

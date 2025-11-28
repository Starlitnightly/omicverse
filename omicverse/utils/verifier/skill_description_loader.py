"""
Skill Description Loader - Progressive Disclosure

Mimics Claude Code's startup behavior by loading only skill names and descriptions
from YAML frontmatter, keeping memory footprint minimal (~30-50 tokens per skill).
"""

import re
from pathlib import Path
from typing import List, Dict, Optional
import yaml

from .data_structures import SkillDescription


class SkillDescriptionLoader:
    """
    Loads skill descriptions using progressive disclosure approach.

    At startup: loads only name + description from YAML frontmatter
    On-demand: full SKILL.md content can be loaded separately if needed
    """

    def __init__(self, skills_dir: Optional[str] = None):
        """
        Initialize loader.

        Args:
            skills_dir: Path to skills directory (defaults to .claude/skills/)
        """
        if skills_dir is None:
            # Default to .claude/skills/ relative to current directory
            skills_dir = Path.cwd() / ".claude" / "skills"
        self.skills_dir = Path(skills_dir)

        if not self.skills_dir.exists():
            raise FileNotFoundError(f"Skills directory not found: {self.skills_dir}")

    def load_all_descriptions(self) -> List[SkillDescription]:
        """
        Load name + description from all skills.

        Returns:
            List of SkillDescription objects with minimal info
        """
        skills = []
        skill_paths = sorted(self.skills_dir.glob("*/SKILL.md"))

        for skill_path in skill_paths:
            try:
                skill_desc = self.load_single_description(skill_path)
                if skill_desc:
                    skills.append(skill_desc)
            except Exception as e:
                # Log warning but continue loading other skills
                print(f"Warning: Failed to load {skill_path}: {e}")
                continue

        return skills

    def load_single_description(self, skill_path: Path) -> Optional[SkillDescription]:
        """
        Load name + description from a single SKILL.md file.

        Args:
            skill_path: Path to SKILL.md file

        Returns:
            SkillDescription or None if parsing fails
        """
        try:
            frontmatter = self._extract_frontmatter(skill_path)
            if not frontmatter:
                print(f"Warning: No frontmatter found in {skill_path}")
                return None

            name = frontmatter.get('name')
            description = frontmatter.get('description')

            if not name or not description:
                print(f"Warning: Missing name or description in {skill_path}")
                return None

            return SkillDescription(name=name, description=description)

        except Exception as e:
            print(f"Error loading {skill_path}: {e}")
            return None

    def _extract_frontmatter(self, file_path: Path) -> Dict:
        """
        Extract YAML frontmatter from markdown file.

        Frontmatter format:
        ---
        name: skill-name
        description: Skill description
        ---

        Args:
            file_path: Path to SKILL.md file

        Returns:
            Dictionary with frontmatter data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Match YAML frontmatter between --- delimiters
        pattern = r'^---\s*\n(.*?)\n---\s*\n'
        match = re.search(pattern, content, re.DOTALL | re.MULTILINE)

        if not match:
            return {}

        yaml_content = match.group(1)
        try:
            return yaml.safe_load(yaml_content) or {}
        except yaml.YAMLError as e:
            print(f"YAML parsing error in {file_path}: {e}")
            return {}

    def format_for_llm(self, skills: List[SkillDescription]) -> str:
        """
        Format skills list for LLM prompt.

        Creates a clean, readable list suitable for LLM consumption.

        Args:
            skills: List of SkillDescription objects

        Returns:
            Formatted string for LLM prompt
        """
        if not skills:
            return "No skills available."

        lines = []
        for skill in sorted(skills, key=lambda s: s.name):
            lines.append(f"- {skill.name}: {skill.description}")

        return "\n".join(lines)

    def get_skill_by_name(
        self,
        skills: List[SkillDescription],
        name: str
    ) -> Optional[SkillDescription]:
        """
        Find a skill by name.

        Args:
            skills: List of skills to search
            name: Skill name to find

        Returns:
            SkillDescription or None if not found
        """
        for skill in skills:
            if skill.name == name:
                return skill
        return None

    def get_skills_by_category(
        self,
        skills: List[SkillDescription],
        category: str
    ) -> List[SkillDescription]:
        """
        Filter skills by category (based on name prefix).

        Args:
            skills: List of skills to filter
            category: Category prefix (e.g., "bulk", "single", "spatial")

        Returns:
            List of skills matching the category
        """
        return [
            skill for skill in skills
            if skill.name.startswith(category.lower())
        ]

    def calculate_token_estimate(self, skill: SkillDescription) -> int:
        """
        Estimate token count for a skill description.

        Rough estimate: ~1.3 tokens per word for English text.

        Args:
            skill: SkillDescription to estimate

        Returns:
            Estimated token count
        """
        # Count words in name + description
        text = f"{skill.name} {skill.description}"
        word_count = len(text.split())
        # Rough estimate: 1.3 tokens per word
        return int(word_count * 1.3)

    def validate_descriptions(self, skills: List[SkillDescription]) -> Dict[str, List[str]]:
        """
        Validate that all skill descriptions meet quality standards.

        Checks:
        - Description is not empty
        - Description is concise (< 100 words recommended)
        - Description mentions what the skill does
        - Description mentions when to use it

        Args:
            skills: List of skills to validate

        Returns:
            Dictionary with validation warnings
        """
        warnings = {}

        for skill in skills:
            skill_warnings = []

            # Check description length
            word_count = len(skill.description.split())
            token_estimate = self.calculate_token_estimate(skill)

            if word_count > 100:
                skill_warnings.append(
                    f"Description is long ({word_count} words). "
                    "Consider making it more concise for LLM efficiency."
                )

            if token_estimate > 80:
                skill_warnings.append(
                    f"Estimated token count ({token_estimate}) exceeds recommended 50-80 tokens."
                )

            # Check for "what" indicators
            what_keywords = ['analyze', 'process', 'calculate', 'extract', 'generate',
                           'create', 'perform', 'run', 'execute', 'build']
            has_what = any(kw in skill.description.lower() for kw in what_keywords)
            if not has_what:
                skill_warnings.append(
                    "Description should clearly state what the skill does (use action verbs)."
                )

            # Check for "when" indicators
            when_keywords = ['use when', 'when you', 'for', 'if you need', 'to help']
            has_when = any(kw in skill.description.lower() for kw in when_keywords)
            if not has_when:
                skill_warnings.append(
                    "Description should state when to use the skill."
                )

            if skill_warnings:
                warnings[skill.name] = skill_warnings

        return warnings

    def get_statistics(self, skills: List[SkillDescription]) -> Dict[str, any]:
        """
        Get statistics about loaded skills.

        Args:
            skills: List of skills to analyze

        Returns:
            Dictionary with statistics
        """
        if not skills:
            return {
                'total_skills': 0,
                'avg_description_length': 0,
                'avg_token_estimate': 0,
                'total_token_estimate': 0,
                'categories': {}
            }

        description_lengths = [len(s.description.split()) for s in skills]
        token_estimates = [self.calculate_token_estimate(s) for s in skills]

        # Count by category
        categories = {}
        for skill in skills:
            # Extract category from name (prefix before first hyphen)
            category = skill.name.split('-')[0] if '-' in skill.name else 'other'
            categories[category] = categories.get(category, 0) + 1

        return {
            'total_skills': len(skills),
            'avg_description_length': sum(description_lengths) / len(skills),
            'max_description_length': max(description_lengths),
            'min_description_length': min(description_lengths),
            'avg_token_estimate': sum(token_estimates) / len(skills),
            'max_token_estimate': max(token_estimates),
            'min_token_estimate': min(token_estimates),
            'total_token_estimate': sum(token_estimates),
            'categories': categories
        }

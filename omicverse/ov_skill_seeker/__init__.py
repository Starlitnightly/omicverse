"""OmicVerse Skill Seeker

A lightweight CLI for working with OmicVerse-bundled Claude Agent skills.

Features:
- List discovered skills (title, slug, description)
- Validate SKILL.md frontmatter and required files
- Package skills into Claude-uploadable .zip files

This tool intentionally reuses the project SkillRegistry for YAML-aware
frontmatter parsing and slug handling, keeping behavior consistent between
the app and CLI.
"""

__all__ = [
    "__version__",
]

__version__ = "0.1.0"


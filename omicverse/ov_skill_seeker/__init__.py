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
    # Re-export common submodules so attribute resolution works with monkeypatch
    "docs_scraper",
    "link_builder",
    "unified_builder",
    "config_validator",
    "github_scraper",
    "pdf_scraper",
]

__version__ = "0.1.3"

# Make submodules available as attributes of the package to support dotted
# getattr resolution used in tests (e.g., monkeypatching paths).
from . import docs_scraper  # noqa: F401
from . import link_builder  # noqa: F401
from . import unified_builder  # noqa: F401
from . import config_validator  # noqa: F401
from . import github_scraper  # noqa: F401
from . import pdf_scraper  # noqa: F401

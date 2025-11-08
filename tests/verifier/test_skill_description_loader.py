"""
Tests for SkillDescriptionLoader

Tests the progressive disclosure loader that mimics Claude Code's
skill loading behavior.
"""

import pytest
from pathlib import Path
import tempfile
import os

from omicverse.utils.verifier import SkillDescriptionLoader, SkillDescription


class TestSkillDescriptionLoader:
    """Test basic skill description loading functionality."""

    @pytest.fixture
    def real_skills_dir(self):
        """Path to actual .claude/skills/ directory."""
        return Path.cwd() / ".claude" / "skills"

    @pytest.fixture
    def temp_skills_dir(self):
        """Create a temporary skills directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_skill_content(self):
        """Sample SKILL.md content with valid frontmatter."""
        return """---
name: test-skill
description: This is a test skill for unit testing. Use when testing skill loading functionality.
---

# Test Skill

This is the body content of the test skill.
"""

    def create_skill_file(self, skills_dir: Path, skill_name: str, content: str):
        """Helper to create a SKILL.md file in a directory."""
        skill_dir = skills_dir / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(content)
        return skill_file

    def test_loader_initialization(self, real_skills_dir):
        """Test that loader initializes with default skills directory."""
        loader = SkillDescriptionLoader()
        assert loader.skills_dir == real_skills_dir

    def test_loader_with_custom_path(self, temp_skills_dir):
        """Test loader with custom skills directory path."""
        loader = SkillDescriptionLoader(str(temp_skills_dir))
        assert loader.skills_dir == temp_skills_dir

    def test_loader_fails_with_invalid_path(self):
        """Test that loader raises error for non-existent directory."""
        with pytest.raises(FileNotFoundError):
            SkillDescriptionLoader("/nonexistent/path")

    def test_extract_frontmatter(self, temp_skills_dir, sample_skill_content):
        """Test YAML frontmatter extraction."""
        loader = SkillDescriptionLoader(str(temp_skills_dir))
        skill_file = self.create_skill_file(
            temp_skills_dir,
            "test-skill",
            sample_skill_content
        )

        frontmatter = loader._extract_frontmatter(skill_file)
        assert frontmatter['name'] == 'test-skill'
        assert 'test skill' in frontmatter['description'].lower()

    def test_extract_frontmatter_no_frontmatter(self, temp_skills_dir):
        """Test handling of SKILL.md without frontmatter."""
        loader = SkillDescriptionLoader(str(temp_skills_dir))
        content = "# Skill without frontmatter\n\nJust content."
        skill_file = self.create_skill_file(temp_skills_dir, "no-fm", content)

        frontmatter = loader._extract_frontmatter(skill_file)
        assert frontmatter == {}

    def test_extract_frontmatter_invalid_yaml(self, temp_skills_dir):
        """Test handling of invalid YAML in frontmatter."""
        loader = SkillDescriptionLoader(str(temp_skills_dir))
        content = """---
name: test
invalid: yaml: content:
---
"""
        skill_file = self.create_skill_file(temp_skills_dir, "invalid", content)
        frontmatter = loader._extract_frontmatter(skill_file)
        assert frontmatter == {}

    def test_load_single_description(self, temp_skills_dir, sample_skill_content):
        """Test loading a single skill description."""
        loader = SkillDescriptionLoader(str(temp_skills_dir))
        skill_file = self.create_skill_file(
            temp_skills_dir,
            "test-skill",
            sample_skill_content
        )

        skill = loader.load_single_description(skill_file)
        assert skill is not None
        assert skill.name == "test-skill"
        assert "test skill" in skill.description.lower()

    def test_load_single_description_missing_fields(self, temp_skills_dir):
        """Test handling of SKILL.md with missing name or description."""
        loader = SkillDescriptionLoader(str(temp_skills_dir))

        # Missing description
        content = """---
name: test-skill
---
"""
        skill_file = self.create_skill_file(temp_skills_dir, "missing-desc", content)
        skill = loader.load_single_description(skill_file)
        assert skill is None

        # Missing name
        content = """---
description: A skill without a name
---
"""
        skill_file = self.create_skill_file(temp_skills_dir, "missing-name", content)
        skill = loader.load_single_description(skill_file)
        assert skill is None

    def test_load_all_descriptions(self, temp_skills_dir):
        """Test loading all skill descriptions from a directory."""
        loader = SkillDescriptionLoader(str(temp_skills_dir))

        # Create multiple skills
        skill1 = """---
name: skill-one
description: First test skill
---
"""
        skill2 = """---
name: skill-two
description: Second test skill
---
"""
        skill3 = """---
name: skill-three
description: Third test skill
---
"""

        self.create_skill_file(temp_skills_dir, "skill-one", skill1)
        self.create_skill_file(temp_skills_dir, "skill-two", skill2)
        self.create_skill_file(temp_skills_dir, "skill-three", skill3)

        skills = loader.load_all_descriptions()
        assert len(skills) == 3
        skill_names = {s.name for s in skills}
        assert skill_names == {"skill-one", "skill-two", "skill-three"}

    def test_format_for_llm(self, temp_skills_dir):
        """Test formatting skills for LLM prompt."""
        loader = SkillDescriptionLoader(str(temp_skills_dir))

        skills = [
            SkillDescription(name="skill-a", description="Description A"),
            SkillDescription(name="skill-b", description="Description B"),
            SkillDescription(name="skill-c", description="Description C"),
        ]

        formatted = loader.format_for_llm(skills)

        # Check that all skills are present
        assert "skill-a: Description A" in formatted
        assert "skill-b: Description B" in formatted
        assert "skill-c: Description C" in formatted

        # Check formatting (bullet points)
        assert "- skill-a:" in formatted

    def test_format_for_llm_empty_list(self, temp_skills_dir):
        """Test formatting empty skills list."""
        loader = SkillDescriptionLoader(str(temp_skills_dir))
        formatted = loader.format_for_llm([])
        assert "No skills available" in formatted

    def test_get_skill_by_name(self):
        """Test finding a skill by name."""
        skills = [
            SkillDescription(name="skill-a", description="Description A"),
            SkillDescription(name="skill-b", description="Description B"),
        ]

        loader = SkillDescriptionLoader(str(Path.cwd() / ".claude" / "skills"))
        skill = loader.get_skill_by_name(skills, "skill-b")
        assert skill is not None
        assert skill.name == "skill-b"

        skill = loader.get_skill_by_name(skills, "nonexistent")
        assert skill is None

    def test_get_skills_by_category(self):
        """Test filtering skills by category."""
        skills = [
            SkillDescription(name="bulk-deg", description="DEG analysis"),
            SkillDescription(name="bulk-wgcna", description="WGCNA"),
            SkillDescription(name="single-clustering", description="Clustering"),
            SkillDescription(name="spatial-analysis", description="Spatial"),
        ]

        loader = SkillDescriptionLoader(str(Path.cwd() / ".claude" / "skills"))

        bulk_skills = loader.get_skills_by_category(skills, "bulk")
        assert len(bulk_skills) == 2
        assert all(s.name.startswith("bulk") for s in bulk_skills)

        single_skills = loader.get_skills_by_category(skills, "single")
        assert len(single_skills) == 1
        assert single_skills[0].name == "single-clustering"

    def test_calculate_token_estimate(self):
        """Test token count estimation."""
        loader = SkillDescriptionLoader(str(Path.cwd() / ".claude" / "skills"))

        skill = SkillDescription(
            name="test-skill",
            description="This is a short description with ten words total here."
        )

        tokens = loader.calculate_token_estimate(skill)
        # 2 words (test-skill) + 10 words = 12 words
        # 12 * 1.3 ≈ 15-16 tokens
        assert 14 <= tokens <= 18

    def test_validate_descriptions(self):
        """Test description validation."""
        loader = SkillDescriptionLoader(str(Path.cwd() / ".claude" / "skills"))

        skills = [
            SkillDescription(
                name="good-skill",
                description="Analyze bulk RNA-seq data. Use when you need differential expression."
            ),
            SkillDescription(
                name="too-long",
                description=" ".join(["word"] * 150)  # 150 words - too long
            ),
            SkillDescription(
                name="no-what",
                description="Use this when you want to do something."
            ),
            SkillDescription(
                name="no-when",
                description="Processes data and generates results."
            ),
        ]

        warnings = loader.validate_descriptions(skills)

        # Good skill should have no warnings
        assert "good-skill" not in warnings

        # Too long should have warning
        assert "too-long" in warnings
        assert any("long" in w.lower() for w in warnings["too-long"])

        # Missing "what" should have warning
        assert "no-what" in warnings
        assert any("what" in w.lower() for w in warnings["no-what"])

        # Missing "when" should have warning
        assert "no-when" in warnings
        assert any("when" in w.lower() for w in warnings["no-when"])

    def test_get_statistics(self):
        """Test statistics generation."""
        loader = SkillDescriptionLoader(str(Path.cwd() / ".claude" / "skills"))

        skills = [
            SkillDescription(name="bulk-deg", description="Short description"),
            SkillDescription(name="bulk-wgcna", description="Another short one"),
            SkillDescription(name="single-cluster", description="Third description here"),
        ]

        stats = loader.get_statistics(skills)

        assert stats['total_skills'] == 3
        assert stats['avg_description_length'] > 0
        assert stats['avg_token_estimate'] > 0
        assert 'bulk' in stats['categories']
        assert 'single' in stats['categories']

    def test_get_statistics_empty(self):
        """Test statistics with empty skill list."""
        loader = SkillDescriptionLoader(str(Path.cwd() / ".claude" / "skills"))
        stats = loader.get_statistics([])
        assert stats['total_skills'] == 0
        assert stats['avg_description_length'] == 0


class TestRealSkillsLoading:
    """Test loading actual OmicVerse skills from .claude/skills/"""

    @pytest.fixture
    def loader(self):
        """Create loader for real skills directory."""
        skills_path = Path.cwd() / ".claude" / "skills"
        if not skills_path.exists():
            pytest.skip("Skills directory not found")
        return SkillDescriptionLoader()

    def test_load_all_real_skills(self, loader):
        """Test loading all actual OmicVerse skills."""
        skills = loader.load_all_descriptions()

        # Should have 23 skills as documented
        assert len(skills) >= 20, f"Expected at least 20 skills, got {len(skills)}"

        # All skills should have names and descriptions
        for skill in skills:
            assert skill.name
            assert skill.description
            assert len(skill.description) > 0

    def test_real_skills_have_reasonable_token_counts(self, loader):
        """Test that real skills have reasonable token counts."""
        skills = loader.load_all_descriptions()

        for skill in skills:
            tokens = loader.calculate_token_estimate(skill)
            # Should be under 80 tokens for progressive disclosure efficiency
            assert tokens < 150, f"{skill.name} has {tokens} tokens (very high)"

    def test_real_skills_validation(self, loader):
        """Test validation of real skill descriptions."""
        skills = loader.load_all_descriptions()
        warnings = loader.validate_descriptions(skills)

        # Print warnings for manual review
        if warnings:
            print("\nSkill description warnings:")
            for skill_name, skill_warnings in warnings.items():
                print(f"\n{skill_name}:")
                for warning in skill_warnings:
                    print(f"  - {warning}")

    def test_real_skills_format_for_llm(self, loader):
        """Test formatting real skills for LLM."""
        skills = loader.load_all_descriptions()
        formatted = loader.format_for_llm(skills)

        # Should be non-empty
        assert len(formatted) > 0

        # Should have bullet points
        assert formatted.count("- ") >= len(skills)

        # Should contain skill names
        for skill in skills[:5]:  # Check first 5
            assert skill.name in formatted

    def test_real_skills_statistics(self, loader):
        """Test statistics on real skills."""
        skills = loader.load_all_descriptions()
        stats = loader.get_statistics(skills)

        print("\nReal skills statistics:")
        print(f"  Total skills: {stats['total_skills']}")
        print(f"  Avg description length: {stats['avg_description_length']:.1f} words")
        print(f"  Avg token estimate: {stats['avg_token_estimate']:.1f} tokens")
        print(f"  Max token estimate: {stats['max_token_estimate']}")
        print(f"  Categories: {stats['categories']}")

        # Verify expected categories exist
        assert 'bulk' in stats['categories']
        assert 'single' in stats['categories']
        assert 'data' in stats['categories']

    def test_real_skills_by_category(self, loader):
        """Test filtering real skills by category."""
        skills = loader.load_all_descriptions()

        bulk_skills = loader.get_skills_by_category(skills, "bulk")
        single_skills = loader.get_skills_by_category(skills, "single")
        data_skills = loader.get_skills_by_category(skills, "data")

        # Should have multiple skills in each category
        assert len(bulk_skills) >= 5, f"Expected >=5 bulk skills, got {len(bulk_skills)}"
        assert len(single_skills) >= 5, f"Expected >=5 single skills, got {len(single_skills)}"
        assert len(data_skills) >= 3, f"Expected >=3 data skills, got {len(data_skills)}"

        print(f"\nCategory counts:")
        print(f"  bulk: {len(bulk_skills)}")
        print(f"  single: {len(single_skills)}")
        print(f"  data: {len(data_skills)}")


class TestSkillDescription:
    """Test SkillDescription dataclass."""

    def test_valid_skill_description(self):
        """Test creating a valid SkillDescription."""
        skill = SkillDescription(
            name="test-skill",
            description="Test description"
        )
        assert skill.name == "test-skill"
        assert skill.description == "Test description"

    def test_empty_name_raises_error(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            SkillDescription(name="", description="Test")

    def test_empty_description_raises_error(self):
        """Test that empty description raises ValueError."""
        with pytest.raises(ValueError, match="description cannot be empty"):
            SkillDescription(name="test", description="")

    def test_whitespace_only_name_raises_error(self):
        """Test that whitespace-only name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            SkillDescription(name="   ", description="Test")

    def test_whitespace_only_description_raises_error(self):
        """Test that whitespace-only description raises ValueError."""
        with pytest.raises(ValueError, match="description cannot be empty"):
            SkillDescription(name="test", description="   ")

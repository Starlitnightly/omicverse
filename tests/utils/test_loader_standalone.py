#!/usr/bin/env python3
"""
Standalone test for SkillDescriptionLoader without pytest.
This bypasses package import issues for quick validation.
"""

from pathlib import Path
import sys

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

# Import directly from file
from omicverse.utils.verifier.data_structures import SkillDescription
from omicverse.utils.verifier.skill_description_loader import SkillDescriptionLoader

def test_basic_loading():
    """Test basic skill loading."""
    print("Testing SkillDescriptionLoader...")

    # Initialize loader
    skills_dir = Path(".claude/skills")
    if not skills_dir.exists():
        print(f"❌ Skills directory not found: {skills_dir}")
        return False

    loader = SkillDescriptionLoader()
    print(f"✓ Loader initialized with skills_dir: {loader.skills_dir}")

    # Load all skills
    skills = loader.load_all_descriptions()
    print(f"✓ Loaded {len(skills)} skills")

    if len(skills) == 0:
        print("❌ No skills loaded!")
        return False

    # Check first few skills
    for skill in skills[:5]:
        print(f"  - {skill.name}: {skill.description[:60]}...")

    # Format for LLM
    formatted = loader.format_for_llm(skills)
    print(f"✓ Formatted {len(formatted)} characters for LLM")

    # Get statistics
    stats = loader.get_statistics(skills)
    print(f"✓ Statistics:")
    print(f"    Total skills: {stats['total_skills']}")
    print(f"    Avg description length: {stats['avg_description_length']:.1f} words")
    print(f"    Avg token estimate: {stats['avg_token_estimate']:.1f} tokens")
    print(f"    Categories: {stats['categories']}")

    # Validate descriptions
    warnings = loader.validate_descriptions(skills)
    if warnings:
        print(f"⚠ Found {len(warnings)} skills with validation warnings:")
        for skill_name, skill_warnings in list(warnings.items())[:3]:
            print(f"    {skill_name}:")
            for warning in skill_warnings:
                print(f"      - {warning}")
    else:
        print(f"✓ All skill descriptions passed validation")

    print("\n✅ All basic tests passed!")
    return True

if __name__ == "__main__":
    success = test_basic_loading()
    sys.exit(0 if success else 1)

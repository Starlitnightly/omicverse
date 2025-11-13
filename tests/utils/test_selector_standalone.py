#!/usr/bin/env python3
"""
Standalone test for LLMSkillSelector without pytest.
Quick validation of core functionality.
"""

from pathlib import Path
import sys
import asyncio

sys.path.insert(0, str(Path(__file__).parent))

from omicverse.utils.verifier.data_structures import SkillDescription, NotebookTask
from omicverse.utils.verifier.llm_skill_selector import LLMSkillSelector


class MockLLMBackend:
    """Simple mock for testing."""
    def __init__(self, response):
        self.response = response
        self.last_usage = None

    async def run(self, prompt):
        print(f"📝 LLM called with prompt length: {len(prompt)} chars")
        return self.response


def test_basic_functionality():
    """Test basic skill selection."""
    print("Testing LLMSkillSelector basic functionality...")

    # Create sample skills
    skills = [
        SkillDescription(
            name="bulk-deg-analysis",
            description="Perform differential expression analysis on bulk RNA-seq data"
        ),
        SkillDescription(
            name="single-preprocessing",
            description="Preprocess single-cell data with QC and normalization"
        ),
        SkillDescription(
            name="single-clustering",
            description="Cluster single-cell data"
        ),
    ]

    # Create mock backend with JSON response
    mock_response = '''{
  "skills": ["single-preprocessing"],
  "order": ["single-preprocessing"],
  "reasoning": "Task requires single-cell preprocessing for quality control"
}'''

    mock_backend = MockLLMBackend(mock_response)

    # Create selector
    selector = LLMSkillSelector(
        llm_backend=mock_backend,
        skill_descriptions=skills
    )

    print(f"✓ Initialized selector with {len(skills)} skills")

    # Test task
    task = NotebookTask(
        task_id="test-001",
        notebook_path="test.ipynb",
        task_description="Preprocess PBMC3k dataset with quality control",
        expected_skills=["single-preprocessing"],
        expected_order=["single-preprocessing"],
        category="single-cell"
    )

    # Run selection (sync)
    print(f"📋 Task: {task.task_description}")
    result = selector.select_skills(task)

    print(f"\n✅ Selection complete!")
    print(f"   Selected skills: {result.selected_skills}")
    print(f"   Skill order: {result.skill_order}")
    print(f"   Reasoning: {result.reasoning}")

    # Validate
    assert result.task_id == "test-001"
    assert result.selected_skills == ["single-preprocessing"]
    assert result.skill_order == ["single-preprocessing"]

    print(f"\n✓ All assertions passed!")
    return True


def test_json_parsing():
    """Test JSON parsing from various formats."""
    print("\n" + "="*60)
    print("Testing JSON parsing...")

    skills = [
        SkillDescription(name="test", description="Test skill")
    ]

    # Test 1: Clean JSON
    print("\n1. Clean JSON:")
    mock1 = MockLLMBackend('{"skills": ["test"], "order": ["test"], "reasoning": "Because"}')
    selector1 = LLMSkillSelector(llm_backend=mock1, skill_descriptions=skills)
    result1 = selector1._parse_llm_response(mock1.response, "test-1")
    print(f"   ✓ Parsed: {result1.selected_skills}")
    assert result1.selected_skills == ["test"]

    # Test 2: JSON in markdown
    print("\n2. JSON in markdown code block:")
    md_response = '''```json
{"skills": ["test"], "order": ["test"], "reasoning": "Test"}
```'''
    selector2 = LLMSkillSelector(llm_backend=mock1, skill_descriptions=skills)
    result2 = selector2._parse_llm_response(md_response, "test-2")
    print(f"   ✓ Parsed: {result2.selected_skills}")
    assert result2.selected_skills == ["test"]

    # Test 3: JSON without code fence
    print("\n3. JSON without markdown:")
    text_response = 'Here is my answer: {"skills": ["test"], "order": ["test"], "reasoning": "Yes"}'
    result3 = selector2._parse_llm_response(text_response, "test-3")
    print(f"   ✓ Parsed: {result3.selected_skills}")
    assert result3.selected_skills == ["test"]

    # Test 4: Invalid JSON
    print("\n4. Invalid JSON:")
    invalid_response = "This is not JSON at all!"
    result4 = selector2._parse_llm_response(invalid_response, "test-4")
    print(f"   ✓ Handled gracefully: {result4.selected_skills}")
    assert result4.selected_skills == []

    print(f"\n✓ All JSON parsing tests passed!")
    return True


def test_prompt_generation():
    """Test prompt generation."""
    print("\n" + "="*60)
    print("Testing prompt generation...")

    skills = [
        SkillDescription(
            name="skill-a",
            description="First skill does A"
        ),
        SkillDescription(
            name="skill-b",
            description="Second skill does B"
        ),
    ]

    mock_backend = MockLLMBackend('{"skills": [], "order": [], "reasoning": "test"}')
    selector = LLMSkillSelector(llm_backend=mock_backend, skill_descriptions=skills)

    # Test formatting
    formatted = selector._format_skills_for_prompt()
    print(f"\n📄 Formatted skills:")
    print(formatted)

    assert "Available skills:" in formatted
    assert "skill-a:" in formatted
    assert "skill-b:" in formatted

    # Test full prompt
    prompt = selector._build_selection_prompt("My task description")
    print(f"\n📄 Full prompt length: {len(prompt)} chars")

    assert "Available skills:" in prompt
    assert "User task: My task description" in prompt
    assert "skill-a" in prompt
    assert "JSON format" in prompt

    print(f"\n✓ Prompt generation tests passed!")
    return True


if __name__ == "__main__":
    print("="*60)
    print("LLMSkillSelector Standalone Tests")
    print("="*60)

    try:
        test_basic_functionality()
        test_json_parsing()
        test_prompt_generation()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

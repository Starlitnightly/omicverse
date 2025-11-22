"""
Tests for LLMSkillSelector

Tests the LLM-based skill selection that mimics Claude Code's behavior.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from omicverse.utils.verifier import (
    LLMSkillSelector,
    create_skill_selector,
    SkillDescription,
    NotebookTask,
    LLMSelectionResult,
)


class MockLLMBackend:
    """Mock LLM backend for testing without real API calls."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_usage = None
        self.call_count = 0
        self.prompts = []

    async def run(self, prompt: str) -> str:
        """Mock run method that returns predefined response."""
        self.call_count += 1
        self.prompts.append(prompt)
        return self.response_text


class TestLLMSkillSelector:
    """Test LLM skill selection functionality."""

    @pytest.fixture
    def sample_skills(self):
        """Create sample skill descriptions for testing."""
        return [
            SkillDescription(
                name="bulk-deg-analysis",
                description="Perform differential expression analysis on bulk RNA-seq data"
            ),
            SkillDescription(
                name="single-preprocessing",
                description="Preprocess single-cell data with QC, normalization, and HVG detection"
            ),
            SkillDescription(
                name="single-clustering",
                description="Cluster single-cell data using multiple methods"
            ),
            SkillDescription(
                name="single-annotation",
                description="Annotate cell types in single-cell data"
            ),
        ]

    @pytest.fixture
    def sample_task(self):
        """Create a sample notebook task."""
        return NotebookTask(
            task_id="test-001",
            notebook_path="test.ipynb",
            task_description="Preprocess PBMC3k dataset with quality control",
            expected_skills=["single-preprocessing"],
            expected_order=["single-preprocessing"],
            category="single-cell"
        )

    def test_selector_initialization_with_backend(self, sample_skills):
        """Test initializing selector with pre-configured backend."""
        mock_backend = MockLLMBackend('{"skills": [], "order": [], "reasoning": "test"}')
        selector = LLMSkillSelector(
            llm_backend=mock_backend,
            skill_descriptions=sample_skills
        )

        assert selector.llm == mock_backend
        assert len(selector.skill_descriptions) == 4

    def test_set_skill_descriptions(self, sample_skills):
        """Test updating skill descriptions."""
        mock_backend = MockLLMBackend('{"skills": [], "order": [], "reasoning": "test"}')
        selector = LLMSkillSelector(llm_backend=mock_backend)

        assert len(selector.skill_descriptions) == 0

        selector.set_skill_descriptions(sample_skills)
        assert len(selector.skill_descriptions) == 4

    def test_format_skills_for_prompt(self, sample_skills):
        """Test formatting skills for LLM prompt."""
        mock_backend = MockLLMBackend('{"skills": [], "order": [], "reasoning": "test"}')
        selector = LLMSkillSelector(
            llm_backend=mock_backend,
            skill_descriptions=sample_skills
        )

        formatted = selector._format_skills_for_prompt()

        assert "Available skills:" in formatted
        assert "bulk-deg-analysis:" in formatted
        assert "single-preprocessing:" in formatted
        assert "Preprocess single-cell data" in formatted

    def test_format_skills_empty_list(self):
        """Test formatting with empty skills list."""
        mock_backend = MockLLMBackend('{"skills": [], "order": [], "reasoning": "test"}')
        selector = LLMSkillSelector(llm_backend=mock_backend)

        formatted = selector._format_skills_for_prompt()
        assert "No skills available" in formatted

    def test_build_selection_prompt(self, sample_skills):
        """Test building selection prompt."""
        mock_backend = MockLLMBackend('{"skills": [], "order": [], "reasoning": "test"}')
        selector = LLMSkillSelector(
            llm_backend=mock_backend,
            skill_descriptions=sample_skills
        )

        prompt = selector._build_selection_prompt("Preprocess my data")

        assert "Available skills:" in prompt
        assert "User task: Preprocess my data" in prompt
        assert "JSON format" in prompt
        assert "bulk-deg-analysis" in prompt

    def test_parse_clean_json_response(self, sample_skills):
        """Test parsing clean JSON response."""
        mock_backend = MockLLMBackend('{"skills": [], "order": [], "reasoning": "test"}')
        selector = LLMSkillSelector(llm_backend=mock_backend, skill_descriptions=sample_skills)

        response = '{"skills": ["single-preprocessing"], "order": ["single-preprocessing"], "reasoning": "Task requires preprocessing"}'

        result = selector._parse_llm_response(response, "test-001")

        assert result.task_id == "test-001"
        assert result.selected_skills == ["single-preprocessing"]
        assert result.skill_order == ["single-preprocessing"]
        assert "preprocessing" in result.reasoning.lower()

    def test_parse_json_in_markdown(self, sample_skills):
        """Test parsing JSON wrapped in markdown code blocks."""
        mock_backend = MockLLMBackend('{"skills": [], "order": [], "reasoning": "test"}')
        selector = LLMSkillSelector(llm_backend=mock_backend, skill_descriptions=sample_skills)

        response = '''Here's my selection:
```json
{
  "skills": ["single-preprocessing", "single-clustering"],
  "order": ["single-preprocessing", "single-clustering"],
  "reasoning": "Need preprocessing then clustering"
}
```
'''

        result = selector._parse_llm_response(response, "test-002")

        assert result.task_id == "test-002"
        assert len(result.selected_skills) == 2
        assert result.selected_skills == ["single-preprocessing", "single-clustering"]
        assert result.skill_order == ["single-preprocessing", "single-clustering"]

    def test_parse_json_without_code_fence(self, sample_skills):
        """Test parsing JSON without markdown code fences."""
        mock_backend = MockLLMBackend('{"skills": [], "order": [], "reasoning": "test"}')
        selector = LLMSkillSelector(llm_backend=mock_backend, skill_descriptions=sample_skills)

        response = '''I recommend the following:
{
  "skills": ["bulk-deg-analysis"],
  "order": ["bulk-deg-analysis"],
  "reasoning": "Bulk RNA-seq analysis needed"
}
That should work!'''

        result = selector._parse_llm_response(response, "test-003")

        assert result.selected_skills == ["bulk-deg-analysis"]
        assert "Bulk RNA-seq" in result.reasoning

    def test_parse_invalid_json(self, sample_skills):
        """Test handling invalid JSON."""
        mock_backend = MockLLMBackend('{"skills": [], "order": [], "reasoning": "test"}')
        selector = LLMSkillSelector(llm_backend=mock_backend, skill_descriptions=sample_skills)

        response = "This is not valid JSON at all!"

        result = selector._parse_llm_response(response, "test-004")

        assert result.selected_skills == []
        assert result.skill_order == []
        assert "no JSON found" in result.reasoning.lower() or "failed" in result.reasoning.lower()

    def test_parse_response_with_missing_fields(self, sample_skills):
        """Test parsing response with missing fields."""
        mock_backend = MockLLMBackend('{"skills": [], "order": [], "reasoning": "test"}')
        selector = LLMSkillSelector(llm_backend=mock_backend, skill_descriptions=sample_skills)

        # Missing 'order' field
        response = '{"skills": ["single-preprocessing"], "reasoning": "Test"}'

        result = selector._parse_llm_response(response, "test-005")

        assert result.selected_skills == ["single-preprocessing"]
        assert result.skill_order == ["single-preprocessing"]  # Should default to skills
        assert result.reasoning == "Test"

    @pytest.mark.asyncio
    async def test_select_skills_async_single_skill(self, sample_skills, sample_task):
        """Test async skill selection for single skill."""
        mock_response = '''{
  "skills": ["single-preprocessing"],
  "order": ["single-preprocessing"],
  "reasoning": "Task requires single-cell preprocessing"
}'''
        mock_backend = MockLLMBackend(mock_response)
        selector = LLMSkillSelector(
            llm_backend=mock_backend,
            skill_descriptions=sample_skills
        )

        result = await selector.select_skills_async(sample_task)

        assert result.task_id == "test-001"
        assert result.selected_skills == ["single-preprocessing"]
        assert result.skill_order == ["single-preprocessing"]
        assert mock_backend.call_count == 1

    @pytest.mark.asyncio
    async def test_select_skills_async_multiple_skills(self, sample_skills):
        """Test async skill selection for multiple skills."""
        mock_response = '''{
  "skills": ["single-preprocessing", "single-clustering", "single-annotation"],
  "order": ["single-preprocessing", "single-clustering", "single-annotation"],
  "reasoning": "Full single-cell workflow: preprocess, cluster, then annotate"
}'''
        mock_backend = MockLLMBackend(mock_response)
        selector = LLMSkillSelector(
            llm_backend=mock_backend,
            skill_descriptions=sample_skills
        )

        task = NotebookTask(
            task_id="workflow-001",
            notebook_path="workflow.ipynb",
            task_description="Analyze single-cell data: QC, cluster, and annotate cell types",
            expected_skills=["single-preprocessing", "single-clustering", "single-annotation"],
            expected_order=["single-preprocessing", "single-clustering", "single-annotation"],
            category="single-cell",
            difficulty="workflow"
        )

        result = await selector.select_skills_async(task)

        assert len(result.selected_skills) == 3
        assert result.skill_order == ["single-preprocessing", "single-clustering", "single-annotation"]
        assert "workflow" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_select_skills_async_with_string_task(self, sample_skills):
        """Test async skill selection with string task description."""
        mock_response = '''{
  "skills": ["bulk-deg-analysis"],
  "order": ["bulk-deg-analysis"],
  "reasoning": "Task requires bulk RNA-seq differential expression analysis"
}'''
        mock_backend = MockLLMBackend(mock_response)
        selector = LLMSkillSelector(
            llm_backend=mock_backend,
            skill_descriptions=sample_skills
        )

        result = await selector.select_skills_async("Find differentially expressed genes in bulk RNA-seq")

        assert result.task_id == "adhoc"  # Default for string tasks
        assert result.selected_skills == ["bulk-deg-analysis"]

    @pytest.mark.asyncio
    async def test_select_skills_async_empty_task(self, sample_skills):
        """Test handling of empty task description."""
        mock_backend = MockLLMBackend('{"skills": [], "order": [], "reasoning": "test"}')
        selector = LLMSkillSelector(
            llm_backend=mock_backend,
            skill_descriptions=sample_skills
        )

        result = await selector.select_skills_async("")

        assert result.selected_skills == []
        assert "Empty task description" in result.reasoning
        assert mock_backend.call_count == 0  # Should not call LLM

    @pytest.mark.asyncio
    async def test_select_skills_async_no_skills_available(self):
        """Test handling when no skills are available."""
        mock_backend = MockLLMBackend('{"skills": [], "order": [], "reasoning": "test"}')
        selector = LLMSkillSelector(llm_backend=mock_backend)  # No skills

        result = await selector.select_skills_async("Some task")

        assert result.selected_skills == []
        assert "No skills available" in result.reasoning
        assert mock_backend.call_count == 0  # Should not call LLM

    @pytest.mark.asyncio
    async def test_select_skills_async_llm_error(self, sample_skills):
        """Test handling of LLM call errors."""
        async def failing_run(prompt):
            raise RuntimeError("API error")

        mock_backend = Mock()
        mock_backend.run = failing_run

        selector = LLMSkillSelector(
            llm_backend=mock_backend,
            skill_descriptions=sample_skills
        )

        result = await selector.select_skills_async("Some task")

        assert result.selected_skills == []
        assert "LLM call failed" in result.reasoning

    def test_select_skills_sync_wrapper(self, sample_skills, sample_task):
        """Test synchronous wrapper for select_skills."""
        mock_response = '''{
  "skills": ["single-preprocessing"],
  "order": ["single-preprocessing"],
  "reasoning": "Preprocessing needed"
}'''
        mock_backend = MockLLMBackend(mock_response)
        selector = LLMSkillSelector(
            llm_backend=mock_backend,
            skill_descriptions=sample_skills
        )

        result = selector.select_skills(sample_task)

        assert result.selected_skills == ["single-preprocessing"]
        assert mock_backend.call_count == 1

    @pytest.mark.asyncio
    async def test_select_skills_batch_async(self, sample_skills):
        """Test batch skill selection (parallel)."""
        mock_response = '''{
  "skills": ["single-preprocessing"],
  "order": ["single-preprocessing"],
  "reasoning": "Test"
}'''
        mock_backend = MockLLMBackend(mock_response)
        selector = LLMSkillSelector(
            llm_backend=mock_backend,
            skill_descriptions=sample_skills
        )

        tasks = [
            "Preprocess data",
            "Cluster cells",
            "Annotate cell types"
        ]

        results = await selector.select_skills_batch_async(tasks)

        assert len(results) == 3
        assert all(isinstance(r, LLMSelectionResult) for r in results)
        assert mock_backend.call_count == 3

    def test_select_skills_batch_sync_wrapper(self, sample_skills):
        """Test synchronous wrapper for batch selection."""
        mock_response = '''{
  "skills": ["single-preprocessing"],
  "order": ["single-preprocessing"],
  "reasoning": "Test"
}'''
        mock_backend = MockLLMBackend(mock_response)
        selector = LLMSkillSelector(
            llm_backend=mock_backend,
            skill_descriptions=sample_skills
        )

        tasks = ["Task 1", "Task 2"]

        results = selector.select_skills_batch(tasks)

        assert len(results) == 2
        assert mock_backend.call_count == 2


class TestLLMSkillSelectorPromptContent:
    """Test the content and quality of prompts generated."""

    @pytest.fixture
    def sample_skills(self):
        return [
            SkillDescription(
                name="bulk-deg-analysis",
                description="Analyze differential expression in bulk RNA-seq data"
            ),
            SkillDescription(
                name="single-preprocessing",
                description="Preprocess single-cell data"
            ),
        ]

    def test_system_prompt_contains_key_instructions(self):
        """Test that system prompt has essential instructions."""
        selector = LLMSkillSelector()
        system_prompt = selector._build_system_prompt()

        # Should contain key instructions
        assert "skill selection" in system_prompt.lower()
        assert "json" in system_prompt.lower()
        assert "order" in system_prompt.lower()
        assert "reasoning" in system_prompt.lower()

    def test_selection_prompt_includes_all_skills(self, sample_skills):
        """Test that selection prompt includes all available skills."""
        mock_backend = MockLLMBackend('{"skills": [], "order": [], "reasoning": "test"}')
        selector = LLMSkillSelector(
            llm_backend=mock_backend,
            skill_descriptions=sample_skills
        )

        prompt = selector._build_selection_prompt("Test task")

        # All skills should be in the prompt
        for skill in sample_skills:
            assert skill.name in prompt
            assert skill.description in prompt

    def test_selection_prompt_includes_task_description(self, sample_skills):
        """Test that selection prompt includes the task description."""
        mock_backend = MockLLMBackend('{"skills": [], "order": [], "reasoning": "test"}')
        selector = LLMSkillSelector(
            llm_backend=mock_backend,
            skill_descriptions=sample_skills
        )

        task_desc = "Analyze my single-cell data with special parameters"
        prompt = selector._build_selection_prompt(task_desc)

        assert task_desc in prompt
        assert "User task:" in prompt


class TestLLMSkillSelectorIntegration:
    """Integration tests with real skill loading (no real LLM calls)."""

    @pytest.fixture
    def real_skills(self):
        """Load real OmicVerse skills if available."""
        from omicverse.utils.verifier import SkillDescriptionLoader

        skills_path = Path.cwd() / ".claude" / "skills"
        if not skills_path.exists():
            pytest.skip("Skills directory not found")

        loader = SkillDescriptionLoader()
        return loader.load_all_descriptions()

    def test_selector_with_real_skills(self, real_skills):
        """Test selector initialization with real OmicVerse skills."""
        mock_response = '''{
  "skills": ["bulk-deg-analysis"],
  "order": ["bulk-deg-analysis"],
  "reasoning": "Task requires bulk DEG analysis"
}'''
        mock_backend = MockLLMBackend(mock_response)
        selector = LLMSkillSelector(
            llm_backend=mock_backend,
            skill_descriptions=real_skills
        )

        assert len(selector.skill_descriptions) >= 20

        # Test selection
        result = selector.select_skills("Find differentially expressed genes")

        assert result.selected_skills == ["bulk-deg-analysis"]
        assert mock_backend.call_count == 1

    def test_prompt_generation_with_real_skills(self, real_skills):
        """Test prompt generation with real skills."""
        mock_backend = MockLLMBackend('{"skills": [], "order": [], "reasoning": "test"}')
        selector = LLMSkillSelector(
            llm_backend=mock_backend,
            skill_descriptions=real_skills
        )

        prompt = selector._build_selection_prompt("Analyze my PBMC data")

        # Should contain bulk skills
        assert "bulk-deg-analysis" in prompt or "bulk" in prompt.lower()
        # Should contain single-cell skills
        assert "single-preprocessing" in prompt or "single" in prompt.lower()
        # Should be well-formatted
        assert "Available skills:" in prompt
        assert "User task:" in prompt

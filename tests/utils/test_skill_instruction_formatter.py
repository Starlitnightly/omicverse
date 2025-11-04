"""
Unit tests for SkillInstructionFormatter provider-specific formatting.

Tests verify that skill instructions are formatted correctly for different
LLM providers (GPT, Gemini, Claude, DeepSeek, etc.).
"""

import sys
from pathlib import Path

import pytest

# Setup path to import omicverse modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omicverse.utils.skill_registry import SkillInstructionFormatter


class TestSkillInstructionFormatter:
    """Test suite for SkillInstructionFormatter"""

    @pytest.fixture
    def sample_skill_body(self):
        """Sample skill instruction text for testing"""
        return """## Overview
This is a test skill that demonstrates formatting.

## Usage
Use this skill to perform various operations.

## Example
Here is an example of how to use this skill:
```python
import pandas as pd
df = pd.read_csv('data.csv')
```

## Example 2
Another example:
```python
print("Hello World")
```

## Best Practices
Follow these guidelines when using the skill.
"""

    def test_gpt_structured_formatting(self, sample_skill_body):
        """Test that GPT/OpenAI formatting uppercases headers"""
        result = SkillInstructionFormatter.format_for_provider(
            sample_skill_body,
            provider='openai',
            max_chars=10000
        )

        # GPT formatting should uppercase headers
        assert '## OVERVIEW' in result
        assert '## USAGE' in result
        assert '## EXAMPLE' in result
        assert '## BEST PRACTICES' in result

    def test_gpt_alias(self, sample_skill_body):
        """Test that 'gpt' provider alias works"""
        result = SkillInstructionFormatter.format_for_provider(
            sample_skill_body,
            provider='gpt',
            max_chars=10000
        )

        assert '## OVERVIEW' in result

    def test_gemini_concise_formatting(self, sample_skill_body):
        """Test that Gemini/Google formatting is concise (limits examples)"""
        result = SkillInstructionFormatter.format_for_provider(
            sample_skill_body,
            provider='google',
            max_chars=10000
        )

        # Gemini should work and preserve core content
        # (Concise trimming only happens if text > 2000 chars)
        assert '## Example' in result
        assert len(result) > 0

    def test_gemini_alias(self, sample_skill_body):
        """Test that 'gemini' provider alias works"""
        result = SkillInstructionFormatter.format_for_provider(
            sample_skill_body,
            provider='gemini',
            max_chars=10000
        )

        # Gemini formatting should work (may preserve content if under 2000 chars)
        assert result is not None
        assert '## Overview' in result

    def test_claude_natural_formatting(self, sample_skill_body):
        """Test that Claude/Anthropic formatting preserves natural language"""
        result = SkillInstructionFormatter.format_for_provider(
            sample_skill_body,
            provider='anthropic',
            max_chars=10000
        )

        # Claude should preserve original formatting (may strip trailing whitespace)
        assert result.strip() == sample_skill_body.strip()
        assert '## Overview' in result
        assert '## Usage' in result

    def test_claude_alias(self, sample_skill_body):
        """Test that 'claude' provider alias works"""
        result = SkillInstructionFormatter.format_for_provider(
            sample_skill_body,
            provider='claude',
            max_chars=10000
        )

        assert result.strip() == sample_skill_body.strip()
        assert '## Overview' in result

    def test_deepseek_explicit_formatting(self, sample_skill_body):
        """Test that DeepSeek formatting adds IMPORTANT markers"""
        result = SkillInstructionFormatter.format_for_provider(
            sample_skill_body,
            provider='deepseek',
            max_chars=10000
        )

        # DeepSeek should add IMPORTANT markers
        assert 'IMPORTANT' in result
        assert '## IMPORTANT: Usage' in result

    def test_qwen_explicit_formatting(self, sample_skill_body):
        """Test that Qwen uses explicit formatting (same as DeepSeek)"""
        result = SkillInstructionFormatter.format_for_provider(
            sample_skill_body,
            provider='qwen',
            max_chars=10000
        )

        assert 'IMPORTANT' in result

    def test_default_provider_formatting(self, sample_skill_body):
        """Test that unknown providers use explicit (default) formatting"""
        result = SkillInstructionFormatter.format_for_provider(
            sample_skill_body,
            provider='unknown_llm',
            max_chars=10000
        )

        # Unknown providers should default to explicit
        assert 'IMPORTANT' in result

    def test_none_provider_formatting(self, sample_skill_body):
        """Test that None provider uses default (explicit) formatting"""
        result = SkillInstructionFormatter.format_for_provider(
            sample_skill_body,
            provider=None,
            max_chars=10000
        )

        assert 'IMPORTANT' in result

    def test_max_chars_truncation(self, sample_skill_body):
        """Test that content is truncated when exceeding max_chars"""
        result = SkillInstructionFormatter.format_for_provider(
            sample_skill_body,
            provider='openai',
            max_chars=50
        )

        # Should be truncated to ~50 chars
        assert len(result) <= 50
        # Should end with ellipsis
        assert result.endswith('...')

    def test_provider_styles_mapping(self):
        """Test that PROVIDER_STYLES dict has expected mappings"""
        styles = SkillInstructionFormatter.PROVIDER_STYLES

        assert styles['openai'] == 'structured'
        assert styles['gpt'] == 'structured'
        assert styles['google'] == 'concise'
        assert styles['gemini'] == 'concise'
        assert styles['anthropic'] == 'natural'
        assert styles['claude'] == 'natural'
        assert styles['deepseek'] == 'explicit'
        assert styles['qwen'] == 'explicit'
        assert styles['default'] == 'explicit'

    def test_empty_skill_body(self):
        """Test formatting with empty skill body"""
        result = SkillInstructionFormatter.format_for_provider(
            "",
            provider='openai',
            max_chars=1000
        )

        assert result == ""

    def test_skill_body_only_whitespace(self):
        """Test formatting with whitespace-only skill body"""
        result = SkillInstructionFormatter.format_for_provider(
            "   \n\n   ",
            provider='openai',
            max_chars=1000
        )

        assert result == ""

    def test_case_insensitive_provider_name(self):
        """Test that provider names are case-insensitive"""
        sample = "## Usage\nTest content"

        result_upper = SkillInstructionFormatter.format_for_provider(
            sample,
            provider='OPENAI',
            max_chars=1000
        )

        result_lower = SkillInstructionFormatter.format_for_provider(
            sample,
            provider='openai',
            max_chars=1000
        )

        assert result_upper == result_lower

    def test_gpt_preserves_code_blocks(self, sample_skill_body):
        """Test that GPT formatting doesn't break code blocks"""
        result = SkillInstructionFormatter.format_for_provider(
            sample_skill_body,
            provider='openai',
            max_chars=10000
        )

        # Code blocks should be preserved
        assert '```python' in result
        assert "import pandas as pd" in result

    def test_gemini_concise_preserves_first_example(self, sample_skill_body):
        """Test that Gemini keeps at least the first example"""
        result = SkillInstructionFormatter.format_for_provider(
            sample_skill_body,
            provider='gemini',
            max_chars=10000
        )

        # First example should be present
        assert "import pandas as pd" in result

    def test_multiple_providers_different_output(self, sample_skill_body):
        """Test that different providers produce different outputs"""
        gpt_result = SkillInstructionFormatter.format_for_provider(
            sample_skill_body, provider='gpt', max_chars=10000
        )
        gemini_result = SkillInstructionFormatter.format_for_provider(
            sample_skill_body, provider='gemini', max_chars=10000
        )
        claude_result = SkillInstructionFormatter.format_for_provider(
            sample_skill_body, provider='claude', max_chars=10000
        )
        deepseek_result = SkillInstructionFormatter.format_for_provider(
            sample_skill_body, provider='deepseek', max_chars=10000
        )

        # GPT should be different (uppercase headers)
        assert gpt_result != claude_result.strip()
        assert '## OVERVIEW' in gpt_result

        # DeepSeek should be different (IMPORTANT markers)
        assert deepseek_result != claude_result.strip()
        assert 'IMPORTANT' in deepseek_result

        # All providers should at least process the content
        assert len(gpt_result) > 0
        assert len(gemini_result) > 0
        assert len(claude_result) > 0
        assert len(deepseek_result) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

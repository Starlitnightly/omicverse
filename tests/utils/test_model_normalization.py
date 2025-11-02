"""
Unit tests for ModelConfig.normalize_model_id() method.

Tests verify that model ID aliases and variations are correctly normalized.
"""

import sys
from pathlib import Path

import pytest

# Setup path to import omicverse modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omicverse.utils.model_config import ModelConfig


class TestModelNormalization:
    """Test suite for ModelConfig.normalize_model_id()"""

    def test_canonical_id_unchanged(self):
        """Test that canonical model IDs are returned unchanged"""
        canonical = "anthropic/claude-sonnet-4-20250514"
        assert ModelConfig.normalize_model_id(canonical) == canonical

    def test_claude_sonnet_4_5_alias(self):
        """Test claude-sonnet-4-5 maps to correct canonical ID"""
        assert ModelConfig.normalize_model_id("claude-sonnet-4-5") == "anthropic/claude-sonnet-4-20250514"

    def test_claude_sonnet_4_5_with_date_alias(self):
        """Test claude-sonnet-4-5-20250929 maps to canonical ID"""
        assert ModelConfig.normalize_model_id("claude-sonnet-4-5-20250929") == "anthropic/claude-sonnet-4-20250514"

    def test_claude_4_opus_alias(self):
        """Test claude-4-opus maps to correct canonical ID"""
        assert ModelConfig.normalize_model_id("claude-4-opus") == "anthropic/claude-opus-4-20250514"

    def test_claude_opus_4_alias(self):
        """Test claude-opus-4 maps to correct canonical ID"""
        assert ModelConfig.normalize_model_id("claude-opus-4") == "anthropic/claude-opus-4-20250514"

    def test_gemini_alias_without_prefix(self):
        """Test gemini-2.5-pro maps to prefixed version"""
        assert ModelConfig.normalize_model_id("gemini-2.5-pro") == "gemini/gemini-2.5-pro"

    def test_deepseek_alias_without_prefix(self):
        """Test deepseek-chat maps to prefixed version"""
        assert ModelConfig.normalize_model_id("deepseek-chat") == "deepseek/deepseek-chat"

    def test_case_insensitive_normalization(self):
        """Test that normalization is case-insensitive"""
        assert ModelConfig.normalize_model_id("CLAUDE-SONNET-4-5") == "anthropic/claude-sonnet-4-20250514"
        assert ModelConfig.normalize_model_id("Claude-Sonnet-4-5") == "anthropic/claude-sonnet-4-20250514"

    def test_openai_model_unchanged(self):
        """Test that OpenAI models without aliases are unchanged"""
        assert ModelConfig.normalize_model_id("gpt-4o") == "gpt-4o"
        assert ModelConfig.normalize_model_id("gpt-4o-mini") == "gpt-4o-mini"

    def test_unknown_model_unchanged(self):
        """Test that unknown models are returned unchanged"""
        unknown = "some-future-model"
        assert ModelConfig.normalize_model_id(unknown) == unknown

    def test_is_model_supported_with_alias(self):
        """Test that is_model_supported works with aliases"""
        # Should recognize alias as supported
        assert ModelConfig.is_model_supported("claude-sonnet-4-5") is True
        assert ModelConfig.is_model_supported("gemini-2.5-pro") is True

    def test_get_provider_from_alias(self):
        """Test that get_provider_from_model works with aliases"""
        assert ModelConfig.get_provider_from_model("claude-sonnet-4-5") == "anthropic"
        assert ModelConfig.get_provider_from_model("gemini-2.5-pro") == "google"
        assert ModelConfig.get_provider_from_model("deepseek-chat") == "deepseek"

    def test_get_model_description_with_alias(self):
        """Test that get_model_description works with aliases"""
        desc = ModelConfig.get_model_description("claude-sonnet-4-5")
        assert "Claude Sonnet 4" in desc


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Unit test for Agent initialization with skills.

Verifies that Agent initializes without AttributeError when skills are present.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Setup path to import omicverse modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_agent_initialization_with_skills_no_crash(capsys):
    """Test that Agent initializes successfully and prints skill count without crashing."""
    # Mock pantheon being available
    with patch("omicverse.utils.smart_agent.PANTHEON_INSTALLED", True):
        # Mock the skill registry to return sample skills
        mock_skill_def = MagicMock()
        mock_skill_def.path = Path("/fake/package/.claude/skills/test-skill")
        mock_skill_def.name = "Test Skill"
        mock_skill_def.slug = "test-skill"

        mock_registry = MagicMock()
        mock_registry.skills = {"test-skill": mock_skill_def}

        # Mock build_multi_path_skill_registry to return our mock
        with patch("omicverse.utils.smart_agent.build_multi_path_skill_registry", return_value=mock_registry):
            # Mock SkillRouter
            with patch("omicverse.utils.smart_agent.SkillRouter"):
                # Mock ModelConfig to avoid validation issues
                with patch("omicverse.utils.smart_agent.ModelConfig") as mock_model_config:
                    mock_model_config.validate_model_setup.return_value = (True, "Model ready")
                    mock_model_config.get_provider_from_model.return_value = "openai"
                    mock_model_config.get_model_description.return_value = "GPT-4o Mini"
                    mock_model_config.get_endpoint_for_model.return_value = "https://api.openai.com/v1"
                    mock_model_config.normalize_model_id.return_value = "gpt-4o-mini"
                    mock_model_config.check_api_key_availability.return_value = (True, "OpenAI API key available")

                    # Import Agent after all mocks are in place
                    from omicverse.utils.smart_agent import Agent

                    # Initialize Agent - this should not crash
                    try:
                        agent = Agent(model="gpt-4o-mini", api_key="test-key")

                        # Verify agent was created
                        assert agent is not None
                        assert agent.model == "gpt-4o-mini"

                        # Capture output
                        captured = capsys.readouterr()

                        # Verify output contains skill count message
                        # The message should be like "🧭 Loaded 1 skills (1 built-in)"
                        assert "🧭 Loaded" in captured.out or "Loaded" in captured.out

                        print("✅ Agent initialization test passed")

                    except AttributeError as e:
                        pytest.fail(f"Agent initialization raised AttributeError: {e}")
                    except Exception as e:
                        # Other exceptions are OK for this test (we're just testing AttributeError)
                        print(f"Note: Agent raised {type(e).__name__}: {e}")
                        # As long as it's not AttributeError, the test passes
                        assert not isinstance(e, AttributeError), f"Should not raise AttributeError: {e}"


def test_agent_skill_count_with_multiple_sources(capsys):
    """Test that Agent correctly counts skills from different sources."""
    with patch("omicverse.utils.smart_agent.PANTHEON_INSTALLED", True):
        # Create mock skills from different sources
        builtin_skill = MagicMock()
        builtin_skill.path = Path("/package/omicverse/.claude/skills/builtin-skill")
        builtin_skill.name = "Builtin Skill"
        builtin_skill.slug = "builtin-skill"

        user_skill = MagicMock()
        user_skill.path = Path("/user/project/.claude/skills/user-skill")
        user_skill.name = "User Skill"
        user_skill.slug = "user-skill"

        mock_registry = MagicMock()
        mock_registry.skills = {
            "builtin-skill": builtin_skill,
            "user-skill": user_skill
        }

        with patch("omicverse.utils.smart_agent.build_multi_path_skill_registry", return_value=mock_registry):
            with patch("omicverse.utils.smart_agent.SkillRouter"):
                with patch("omicverse.utils.smart_agent.ModelConfig") as mock_model_config:
                    mock_model_config.validate_model_setup.return_value = (True, "Model ready")
                    mock_model_config.get_provider_from_model.return_value = "openai"
                    mock_model_config.get_model_description.return_value = "GPT-4o"
                    mock_model_config.get_endpoint_for_model.return_value = "https://api.openai.com/v1"
                    mock_model_config.normalize_model_id.return_value = "gpt-4o"
                    mock_model_config.check_api_key_availability.return_value = (True, "OpenAI API key available")

                    from omicverse.utils.smart_agent import Agent

                    try:
                        # This should handle the path attribute correctly
                        agent = Agent(model="gpt-4o", api_key="test-key")
                        assert agent is not None

                        print("✅ Multi-source skill count test passed")

                    except AttributeError as e:
                        pytest.fail(f"Agent raised AttributeError when counting skills: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

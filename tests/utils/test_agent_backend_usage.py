"""
Tests for token usage tracking in agent_backend.py

This module tests the Usage dataclass and token usage tracking across all providers:
- OpenAI SDK (Chat Completions and Responses API)
- OpenAI HTTP fallback
- Anthropic SDK
- Gemini SDK
- DashScope SDK
- Streaming responses
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from types import SimpleNamespace

# Load agent_backend directly from file to avoid importing heavy utils package
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

# Ensure test can import minimal omicverse.utils without pulling heavy modules
for name in [
    "omicverse",
    "omicverse.utils",
    "omicverse.utils.agent_backend",
]:
    sys.modules.pop(name, None)

omicverse_pkg = types.ModuleType("omicverse")
omicverse_pkg.__path__ = [str(PACKAGE_ROOT)]
omicverse_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse", loader=None, is_package=True
)
sys.modules["omicverse"] = omicverse_pkg

utils_pkg = types.ModuleType("omicverse.utils")
utils_pkg.__path__ = [str(PACKAGE_ROOT / "utils")]
utils_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse.utils", loader=None, is_package=True
)
sys.modules["omicverse.utils"] = utils_pkg
omicverse_pkg.utils = utils_pkg

agent_backend_spec = importlib.util.spec_from_file_location(
    "omicverse.utils.agent_backend", PACKAGE_ROOT / "utils" / "agent_backend.py"
)
agent_backend_module = importlib.util.module_from_spec(agent_backend_spec)
sys.modules["omicverse.utils.agent_backend"] = agent_backend_module
assert agent_backend_spec.loader is not None
agent_backend_spec.loader.exec_module(agent_backend_module)
utils_pkg.agent_backend = agent_backend_module

OmicVerseLLMBackend = agent_backend_module.OmicVerseLLMBackend
Usage = agent_backend_module.Usage


class TestUsageDataclass:
    """Test the Usage dataclass structure and creation."""

    def test_usage_dataclass_creation(self):
        """Test creating a Usage instance with all fields."""
        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            model="gpt-4o",
            provider="openai"
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.model == "gpt-4o"
        assert usage.provider == "openai"

    def test_usage_dataclass_fields(self):
        """Test that Usage has the required fields."""
        usage = Usage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            model="test-model",
            provider="test-provider"
        )
        assert hasattr(usage, 'input_tokens')
        assert hasattr(usage, 'output_tokens')
        assert hasattr(usage, 'total_tokens')
        assert hasattr(usage, 'model')
        assert hasattr(usage, 'provider')


class TestOpenAIUsageTracking:
    """Test usage tracking for OpenAI SDK (Chat Completions)."""

    def test_openai_sdk_usage_tracking(self, monkeypatch):
        """Test that OpenAI SDK call captures usage information."""
        # Mock OpenAI client and response
        mock_client = Mock()

        # Create mock response with usage
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Test response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        # Mock usage object
        mock_usage = Mock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150
        mock_response.usage = mock_usage

        mock_client.chat.completions.create.return_value = mock_response

        # Create a mock OpenAI class
        mock_openai_class = Mock(return_value=mock_client)

        # Patch OpenAI import
        import sys
        mock_openai_module = Mock()
        mock_openai_module.OpenAI = mock_openai_class
        sys.modules['openai'] = mock_openai_module

        # Mock ModelConfig
        monkeypatch.setattr(
            agent_backend_module.ModelConfig,
            'get_provider_from_model',
            lambda x: 'openai'
        )
        monkeypatch.setattr(
            agent_backend_module.ModelConfig,
            'requires_responses_api',
            lambda x: False
        )

        try:
            # Create backend and make call
            backend = OmicVerseLLMBackend(
                system_prompt="Test prompt",
                model="gpt-4o",
                api_key="test-key"
            )

            result = backend._chat_via_openai_compatible("Test user prompt")

            # Verify usage was captured
            assert backend.last_usage is not None
            assert backend.last_usage.input_tokens == 100
            assert backend.last_usage.output_tokens == 50
            assert backend.last_usage.total_tokens == 150
            assert backend.last_usage.model == "gpt-4o"
            assert backend.last_usage.provider == "openai"
        finally:
            # Clean up sys.modules
            if 'openai' in sys.modules:
                del sys.modules['openai']

    def test_openai_http_usage_tracking(self, monkeypatch):
        """Test that OpenAI HTTP fallback captures usage information."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.read.return_value = b'''{
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 100,
                "total_tokens": 300
            }
        }'''
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        # Mock urlopen
        mock_urlopen = Mock(return_value=mock_response)
        monkeypatch.setattr(agent_backend_module.urllib_request, 'urlopen', mock_urlopen)

        # Mock ModelConfig
        monkeypatch.setattr(
            agent_backend_module.ModelConfig,
            'get_provider_from_model',
            lambda x: 'openai'
        )
        monkeypatch.setattr(
            agent_backend_module.ModelConfig,
            'requires_responses_api',
            lambda x: False
        )

        # Create backend and make call
        backend = OmicVerseLLMBackend(
            system_prompt="Test prompt",
            model="gpt-4o",
            api_key="test-key"
        )

        result = backend._chat_via_openai_http(
            "https://api.openai.com/v1",
            "test-key",
            "Test user prompt"
        )

        # Verify usage was captured
        assert backend.last_usage is not None
        assert backend.last_usage.input_tokens == 200
        assert backend.last_usage.output_tokens == 100
        assert backend.last_usage.total_tokens == 300
        assert backend.last_usage.model == "gpt-4o"
        assert backend.last_usage.provider == "openai"


class TestResponsesAPIUsageTracking:
    """Test usage tracking for OpenAI Responses API (gpt-5 series)."""

    def test_responses_api_sdk_usage_tracking(self, monkeypatch):
        """Test that Responses API SDK call captures usage information."""
        # Create mock OpenAI client and response
        mock_client = Mock()

        # Create mock response with usage
        mock_response = Mock()
        mock_response.output_text = "Test response"

        # Mock usage object for Responses API
        mock_usage = Mock()
        mock_usage.input_tokens = 150
        mock_usage.output_tokens = 75
        mock_usage.total_tokens = 225
        mock_response.usage = mock_usage

        mock_client.responses.create.return_value = mock_response

        # Mock OpenAI class
        mock_openai_class = Mock(return_value=mock_client)

        # Inject mock via sys.modules to handle lazy import
        mock_openai_module = Mock()
        mock_openai_module.OpenAI = mock_openai_class

        with patch.dict('sys.modules', {'openai': mock_openai_module}):
            # Mock ModelConfig
            monkeypatch.setattr(
                agent_backend_module.ModelConfig,
                'get_provider_from_model',
                lambda x: 'openai'
            )

            # Create backend and make call
            backend = OmicVerseLLMBackend(
                system_prompt="Test prompt",
                model="gpt-5",
                api_key="test-key"
            )

            result = backend._chat_via_openai_responses(
                "https://api.openai.com/v1",
                "test-key",
                "Test user prompt"
            )

            # Verify usage was captured
            assert backend.last_usage is not None
            assert backend.last_usage.input_tokens == 150
            assert backend.last_usage.output_tokens == 75
            assert backend.last_usage.total_tokens == 225
            assert backend.last_usage.model == "gpt-5"
            assert backend.last_usage.provider == "openai"

    def test_responses_api_http_usage_tracking(self, monkeypatch):
        """Test that Responses API HTTP fallback captures usage information."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.read.return_value = b'''{
            "output_text": "Test response",
            "usage": {
                "input_tokens": 180,
                "output_tokens": 90,
                "total_tokens": 270
            }
        }'''
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        # Mock urlopen
        mock_urlopen = Mock(return_value=mock_response)
        monkeypatch.setattr(agent_backend_module.urllib_request, 'urlopen', mock_urlopen)

        # Mock ModelConfig
        monkeypatch.setattr(
            agent_backend_module.ModelConfig,
            'get_provider_from_model',
            lambda x: 'openai'
        )

        # Create backend and make call
        backend = OmicVerseLLMBackend(
            system_prompt="Test prompt",
            model="gpt-5",
            api_key="test-key"
        )

        result = backend._chat_via_openai_responses_http(
            "https://api.openai.com/v1",
            "test-key",
            "Test user prompt"
        )

        # Verify usage was captured
        assert backend.last_usage is not None
        assert backend.last_usage.input_tokens == 180
        assert backend.last_usage.output_tokens == 90
        assert backend.last_usage.total_tokens == 270
        assert backend.last_usage.model == "gpt-5"
        assert backend.last_usage.provider == "openai"


class TestAnthropicUsageTracking:
    """Test usage tracking for Anthropic SDK."""

    def test_anthropic_usage_tracking(self, monkeypatch):
        """Test that Anthropic SDK call captures usage information."""
        # Mock Anthropic client and response
        mock_client = Mock()

        # Create mock response with usage
        mock_response = Mock()
        mock_block = Mock()
        mock_block.type = "text"
        mock_block.text = "Test response"
        mock_response.content = [mock_block]

        # Mock usage object
        mock_usage = Mock()
        mock_usage.input_tokens = 120
        mock_usage.output_tokens = 60
        mock_response.usage = mock_usage

        mock_client.messages.create.return_value = mock_response

        # Mock Anthropic class
        mock_anthropic_class = Mock(return_value=mock_client)

        # Inject mock via sys.modules to handle lazy import
        mock_anthropic_module = Mock()
        mock_anthropic_module.Anthropic = mock_anthropic_class

        with patch.dict('sys.modules', {'anthropic': mock_anthropic_module}):
            # Mock ModelConfig
            monkeypatch.setattr(
                agent_backend_module.ModelConfig,
                'get_provider_from_model',
                lambda x: 'anthropic'
            )

            # Create backend and make call
            backend = OmicVerseLLMBackend(
                system_prompt="Test prompt",
                model="claude-3-sonnet",
                api_key="test-key"
            )

            result = backend._chat_via_anthropic("Test user prompt")

            # Verify usage was captured
            assert backend.last_usage is not None
            assert backend.last_usage.input_tokens == 120
            assert backend.last_usage.output_tokens == 60
            assert backend.last_usage.total_tokens == 180
            assert backend.last_usage.model == "claude-3-sonnet"
            assert backend.last_usage.provider == "anthropic"


class TestGeminiUsageTracking:
    """Test usage tracking for Google Gemini SDK."""

    def test_gemini_usage_tracking(self, monkeypatch):
        """Test that Gemini SDK call captures usage information."""
        # Create real classes with __init__ to set instance attributes
        class MockUsage:
            def __init__(self):
                self.prompt_token_count = 140
                self.candidates_token_count = 70
                self.total_token_count = 210

        class MockResponse:
            def __init__(self):
                self.text = "Test response"
                self.usage_metadata = MockUsage()

        # Build realistic module stubs using ModuleType to avoid Mock getattr pitfalls
        import types as _types

        # Create a minimal google.generativeai module
        genai_mod = _types.ModuleType('google.generativeai')

        # Mock Gemini model with generate_content returning our MockResponse
        class _GenModel:
            def __init__(self, *args, **kwargs):
                pass

            def generate_content(self, *args, **kwargs):
                return MockResponse()

        genai_mod.GenerativeModel = _GenModel
        genai_mod.configure = lambda **kwargs: None
        genai_mod.types = SimpleNamespace(GenerationConfig=lambda **kwargs: {})

        # Create the parent google module and attach generativeai
        google_mod = _types.ModuleType('google')
        google_mod.generativeai = genai_mod

        # Inject module stubs via sys.modules to handle lazy import
        with patch.dict('sys.modules', {
            'google': google_mod,
            'google.generativeai': genai_mod,
        }):
            # Mock ModelConfig
            monkeypatch.setattr(
                agent_backend_module.ModelConfig,
                'get_provider_from_model',
                lambda x: 'google'
            )

            # Create backend and make call
            backend = OmicVerseLLMBackend(
                system_prompt="Test prompt",
                model="google/gemini-pro",
                api_key="test-key"
            )

            result = backend._chat_via_gemini("Test user prompt")

            # Verify usage was captured
            assert backend.last_usage is not None
            assert backend.last_usage.input_tokens == 140
            assert backend.last_usage.output_tokens == 70
            assert backend.last_usage.total_tokens == 210
            assert backend.last_usage.model == "google/gemini-pro"
            assert backend.last_usage.provider == "google"


class TestDashScopeUsageTracking:
    """Test usage tracking for DashScope SDK."""

    def test_dashscope_usage_tracking(self, monkeypatch):
        """Test that DashScope SDK call captures usage information."""
        # Create mock response with usage
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.output = {
            "choices": [{
                "message": {"content": "Test response"}
            }]
        }

        # Mock usage object
        mock_usage = Mock()
        mock_usage.input_tokens = 160
        mock_usage.output_tokens = 80
        mock_usage.total_tokens = 240
        mock_response.usage = mock_usage

        # Mock Generation class
        mock_generation = Mock()
        mock_generation.call.return_value = mock_response

        # Mock HTTPStatus
        mock_httpstatus = SimpleNamespace(OK=200)

        # Inject mocks via sys.modules to handle lazy import
        mock_dashscope = Mock()
        mock_dashscope.Generation = mock_generation

        mock_http = Mock()
        mock_http.HTTPStatus = mock_httpstatus

        with patch.dict('sys.modules', {
            'dashscope': mock_dashscope,
            'http': mock_http
        }):
            # Mock ModelConfig
            monkeypatch.setattr(
                agent_backend_module.ModelConfig,
                'get_provider_from_model',
                lambda x: 'dashscope'
            )

            # Create backend and make call
            backend = OmicVerseLLMBackend(
                system_prompt="Test prompt",
                model="qwen-turbo",
                api_key="test-key"
            )

            result = backend._chat_via_dashscope("Test user prompt")

            # Verify usage was captured
            assert backend.last_usage is not None
            assert backend.last_usage.input_tokens == 160
            assert backend.last_usage.output_tokens == 80
            assert backend.last_usage.total_tokens == 240
            assert backend.last_usage.model == "qwen-turbo"
            assert backend.last_usage.provider == "dashscope"


class TestStreamingUsageTracking:
    """Test usage tracking for streaming responses."""

    @pytest.mark.asyncio
    async def test_openai_streaming_usage_tracking(self, monkeypatch):
        """Test that OpenAI streaming captures usage information."""
        # Mock OpenAI client
        mock_client = Mock()

        # Create mock streaming response with usage in final chunk
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="Test "))]),
            Mock(choices=[Mock(delta=Mock(content="response"))]),
            Mock(choices=[Mock(delta=Mock(content=None))], usage=Mock(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150
            ))
        ]

        mock_client.chat.completions.create.return_value = iter(mock_chunks)

        # Mock OpenAI class
        mock_openai_class = Mock(return_value=mock_client)

        # Inject mock via sys.modules to handle lazy import
        mock_openai_module = Mock()
        mock_openai_module.OpenAI = mock_openai_class

        with patch.dict('sys.modules', {'openai': mock_openai_module}):
            # Mock ModelConfig
            monkeypatch.setattr(
                agent_backend_module.ModelConfig,
                'get_provider_from_model',
                lambda x: 'openai'
            )
            monkeypatch.setattr(
                agent_backend_module.ModelConfig,
                'requires_responses_api',
                lambda x: False
            )

            # Create backend and stream
            backend = OmicVerseLLMBackend(
                system_prompt="Test prompt",
                model="gpt-4o",
                api_key="test-key"
            )

            chunks = []
            async for chunk in backend._stream_openai_compatible("Test user prompt"):
                chunks.append(chunk)

            # Verify usage was captured after streaming
            assert backend.last_usage is not None
            assert backend.last_usage.input_tokens == 100
            assert backend.last_usage.output_tokens == 50
            assert backend.last_usage.total_tokens == 150


class TestUsageTrackingIntegration:
    """Integration tests for usage tracking across different scenarios."""

    def test_last_usage_initially_none(self, monkeypatch):
        """Test that last_usage is None when backend is first created."""
        monkeypatch.setattr(
            agent_backend_module.ModelConfig,
            'get_provider_from_model',
            lambda x: 'openai'
        )

        backend = OmicVerseLLMBackend(
            system_prompt="Test prompt",
            model="gpt-4o",
            api_key="test-key"
        )

        assert backend.last_usage is None

    def test_last_usage_updates_on_new_call(self, monkeypatch):
        """Test that last_usage updates with each new call."""
        # Mock OpenAI client
        mock_client = Mock()

        # First response
        mock_response1 = Mock()
        mock_choice1 = Mock()
        mock_message1 = Mock()
        mock_message1.content = "First response"
        mock_choice1.message = mock_message1
        mock_response1.choices = [mock_choice1]
        mock_usage1 = Mock()
        mock_usage1.prompt_tokens = 100
        mock_usage1.completion_tokens = 50
        mock_usage1.total_tokens = 150
        mock_response1.usage = mock_usage1

        # Second response
        mock_response2 = Mock()
        mock_choice2 = Mock()
        mock_message2 = Mock()
        mock_message2.content = "Second response"
        mock_choice2.message = mock_message2
        mock_response2.choices = [mock_choice2]
        mock_usage2 = Mock()
        mock_usage2.prompt_tokens = 200
        mock_usage2.completion_tokens = 100
        mock_usage2.total_tokens = 300
        mock_response2.usage = mock_usage2

        mock_client.chat.completions.create.side_effect = [mock_response1, mock_response2]

        # Mock OpenAI class
        mock_openai_class = Mock(return_value=mock_client)

        # Inject mock via sys.modules to handle lazy import
        mock_openai_module = Mock()
        mock_openai_module.OpenAI = mock_openai_class

        with patch.dict('sys.modules', {'openai': mock_openai_module}):
            # Mock ModelConfig
            monkeypatch.setattr(
                agent_backend_module.ModelConfig,
                'get_provider_from_model',
                lambda x: 'openai'
            )
            monkeypatch.setattr(
                agent_backend_module.ModelConfig,
                'requires_responses_api',
                lambda x: False
            )

            # Create backend and make calls
            backend = OmicVerseLLMBackend(
                system_prompt="Test prompt",
                model="gpt-4o",
                api_key="test-key"
            )

            # First call
            backend._chat_via_openai_compatible("First prompt")
            assert backend.last_usage.total_tokens == 150

            # Second call should update usage
            backend._chat_via_openai_compatible("Second prompt")
            assert backend.last_usage.total_tokens == 300

    def test_usage_without_usage_field(self, monkeypatch):
        """Test graceful handling when response doesn't include usage."""
        # Mock OpenAI client
        mock_client = Mock()

        # Response without usage field
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Test response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = None  # No usage

        mock_client.chat.completions.create.return_value = mock_response

        # Mock OpenAI class
        mock_openai_class = Mock(return_value=mock_client)

        # Inject mock via sys.modules to handle lazy import
        mock_openai_module = Mock()
        mock_openai_module.OpenAI = mock_openai_class

        with patch.dict('sys.modules', {'openai': mock_openai_module}):
            # Mock ModelConfig
            monkeypatch.setattr(
                agent_backend_module.ModelConfig,
                'get_provider_from_model',
                lambda x: 'openai'
            )
            monkeypatch.setattr(
                agent_backend_module.ModelConfig,
                'requires_responses_api',
                lambda x: False
            )

            # Create backend and make call
            backend = OmicVerseLLMBackend(
                system_prompt="Test prompt",
                model="gpt-4o",
                api_key="test-key"
            )

            result = backend._chat_via_openai_compatible("Test prompt")

            # Verify last_usage remains None (not updated)
            assert backend.last_usage is None

    @pytest.mark.asyncio
    async def test_stale_usage_reset_on_new_call(self, monkeypatch):
        """Test that last_usage is reset to None when 2nd call returns no usage.

        Scenario:
        1st call sets usage, 2nd call returns no usage -> last_usage should be None
        This prevents stale data from appearing to belong to the 2nd call.
        """
        # Mock ModelConfig
        monkeypatch.setattr(
            agent_backend_module.ModelConfig,
            'get_provider_from_model',
            lambda x: 'openai'
        )
        monkeypatch.setattr(
            agent_backend_module.ModelConfig,
            'requires_responses_api',
            lambda x: False
        )

        # Create backend
        backend = OmicVerseLLMBackend(
            system_prompt="Test prompt",
            model="gpt-4o",
            api_key="test-key"
        )

        # Manually set usage to simulate a previous call with usage data
        backend.last_usage = Usage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            model="gpt-4o",
            provider="openai"
        )
        assert backend.last_usage is not None
        assert backend.last_usage.total_tokens == 150

        # Mock _run_sync to return a response without setting usage
        # This simulates a call where the API doesn't return usage data
        def mock_run_sync_no_usage(prompt):
            # Reset happens in run() before this is called
            # This mock doesn't set last_usage, simulating missing usage data
            return "Mock response"

        monkeypatch.setattr(backend, '_run_sync', mock_run_sync_no_usage)

        # Second call - should reset last_usage to None at the start of run()
        # Since _run_sync doesn't set usage, it should remain None
        await backend.run("Second prompt")
        assert backend.last_usage is None, "last_usage should be None after call with no usage data"

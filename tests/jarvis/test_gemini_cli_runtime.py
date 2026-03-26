from __future__ import annotations

import unittest
from unittest import mock

from omicverse.utils.ovagent.auth import collect_api_key_env
from omicverse.utils.agent_backend import OmicVerseLLMBackend
from omicverse.utils.smart_agent import _resolve_agent_llm_credentials
from omicverse.jarvis import gemini_cli_oauth


class GeminiCliRuntimeTests(unittest.TestCase):
    def test_collect_api_key_env_maps_gemini_model_to_google_api_key(self) -> None:
        env_mapping = collect_api_key_env(
            model="gemini-2.5-flash",
            endpoint=None,
            api_key='{"token":"oauth-token","projectId":"demo"}',
        )
        self.assertEqual(
            env_mapping["GOOGLE_API_KEY"],
            '{"token":"oauth-token","projectId":"demo"}',
        )

    def test_resolve_agent_llm_credentials_rewrites_stale_openai_endpoint_for_gemini_oauth(self) -> None:
        with mock.patch(
            "omicverse.utils.smart_agent.GeminiCliOAuthManager.build_api_key_payload",
            return_value='{"token":"oauth-token","projectId":"demo"}',
        ):
            model, api_key, endpoint, auth_mode = _resolve_agent_llm_credentials(
                model="gemini-2.5-flash",
                api_key=None,
                endpoint="https://chatgpt.com/backend-api",
                auth_mode="gemini_cli_oauth",
                auth_provider="gemini_cli",
                auth_file=None,
            )

        self.assertEqual(model, "gemini-2.5-flash")
        self.assertEqual(api_key, '{"token":"oauth-token","projectId":"demo"}')
        self.assertEqual(endpoint, "https://cloudcode-pa.googleapis.com")
        self.assertEqual(auth_mode, "gemini_cli_oauth")

    def test_resolve_google_oauth_identity_tolerates_project_discovery_failure(self) -> None:
        with mock.patch.object(
            gemini_cli_oauth,
            "_fetch_user_email",
            return_value="user@example.com",
        ), mock.patch.object(
            gemini_cli_oauth,
            "discover_google_project",
            side_effect=gemini_cli_oauth.GeminiCliOAuthError("loadCodeAssist failed: 400 Bad Request"),
        ):
            identity = gemini_cli_oauth.resolve_google_oauth_identity("oauth-token")

        self.assertEqual(identity["email"], "user@example.com")
        self.assertNotIn("project_id", identity)

    def test_gemini_auth_headers_support_oauth_payload(self) -> None:
        headers = OmicVerseLLMBackend._gemini_auth_headers('{"token":"oauth-token","projectId":"demo"}')
        self.assertEqual(headers["Authorization"], "Bearer oauth-token")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertNotIn("x-goog-api-key", headers)

    def test_chat_via_gemini_uses_rest_for_oauth_payload(self) -> None:
        backend = OmicVerseLLMBackend(
            system_prompt="system",
            model="gemini-2.5-flash",
            api_key='{"token":"oauth-token","projectId":"demo"}',
        )

        with mock.patch(
            "omicverse.utils.agent_backend_gemini._gemini_cli_request",
            return_value={
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "hello from gemini oauth"}],
                            }
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 3,
                        "candidatesTokenCount": 5,
                        "totalTokenCount": 8,
                    },
                },
            },
        ) as rest_request:
            result = backend._chat_via_gemini("hi")

        self.assertEqual(result, "hello from gemini oauth")
        self.assertEqual(rest_request.call_count, 1)
        self.assertIsNotNone(backend.last_usage)
        self.assertEqual(backend.last_usage.total_tokens, 8)

    def test_chat_tools_gemini_rest_parses_function_calls(self) -> None:
        backend = OmicVerseLLMBackend(
            system_prompt="system",
            model="gemini-2.5-flash",
            api_key='{"token":"oauth-token","projectId":"demo"}',
        )

        with mock.patch(
            "omicverse.utils.agent_backend_gemini._gemini_cli_request",
            return_value={
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "functionCall": {
                                            "name": "python_exec",
                                            "args": {"code": "print('hi')"},
                                        }
                                    }
                                ]
                            },
                            "finishReason": "STOP",
                        }
                    ]
                }
            },
        ):
            response = backend._chat_tools_gemini(
                messages=[{"role": "user", "content": "run python"}],
                tools=[
                    {
                        "name": "python_exec",
                        "description": "Run Python code",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string"},
                            },
                            "required": ["code"],
                        },
                    }
                ],
                tool_choice="auto",
            )

        self.assertEqual(response.stop_reason, "tool_use")
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].name, "python_exec")
        self.assertEqual(response.tool_calls[0].arguments["code"], "print('hi')")
        self.assertEqual(response.raw_message["role"], "assistant")
        self.assertEqual(response.raw_message["tool_calls"][0]["function"]["name"], "python_exec")

    def test_chat_via_gemini_cli_oauth_uses_cloudcode_wrapper(self) -> None:
        backend = OmicVerseLLMBackend(
            system_prompt="system",
            model="gemini-2.5-flash",
            api_key='{"token":"oauth-token","projectId":"demo-project"}',
            endpoint="https://cloudcode-pa.googleapis.com",
        )

        with mock.patch(
            "omicverse.utils.agent_backend_gemini._gemini_cli_request",
            return_value={
                "response": {
                    "candidates": [
                        {"content": {"parts": [{"text": "cloudcode ok"}]}}
                    ]
                }
            },
        ) as cli_request:
            result = backend._chat_via_gemini("hello")

        self.assertEqual(result, "cloudcode ok")
        # Module-level fn signature: _gemini_cli_request(backend, body, api_key)
        payload = cli_request.call_args.args[1]
        self.assertEqual(payload["model"], "gemini-2.5-flash")
        self.assertEqual(payload["project"], "demo-project")
        self.assertEqual(payload["request"]["systemInstruction"]["role"], "system")

    def test_chat_tools_gemini_cli_omits_tool_config_and_uppercases_schema_types(self) -> None:
        backend = OmicVerseLLMBackend(
            system_prompt="system",
            model="gemini-2.5-flash",
            api_key='{"token":"oauth-token","projectId":"demo-project"}',
            endpoint="https://cloudcode-pa.googleapis.com",
        )

        with mock.patch(
            "omicverse.utils.agent_backend_gemini._gemini_cli_request",
            return_value={"response": {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}},
        ) as cli_request:
            backend._chat_tools_gemini(
                messages=[{"role": "user", "content": "run tool"}],
                tools=[
                    {
                        "name": "python_exec",
                        "description": "Run Python",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string", "format": "python", "default": "print('hi')"},
                            },
                            "required": ["code"],
                        },
                    }
                ],
                tool_choice="required",
            )

        # Module-level fn signature: _gemini_cli_request(backend, body, api_key)
        payload = cli_request.call_args.args[1]
        self.assertNotIn("toolConfig", payload["request"])
        schema = payload["request"]["tools"][0]["functionDeclarations"][0]["parameters"]
        self.assertEqual(schema["type"], "OBJECT")
        self.assertEqual(schema["properties"]["code"]["type"], "STRING")
        self.assertNotIn("format", schema["properties"]["code"])
        self.assertNotIn("default", schema["properties"]["code"])


if __name__ == "__main__":
    unittest.main()

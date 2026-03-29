from __future__ import annotations

import unittest
from unittest import mock

from omicverse.utils.ovagent.auth import collect_api_key_env
from omicverse.utils.agent_backend import OmicVerseLLMBackend
from omicverse.utils.ovagent.auth import resolve_credentials as _resolve_agent_llm_credentials
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
            "omicverse.utils.ovagent.auth.GeminiCliOAuthManager.build_api_key_payload",
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


    # ------------------------------------------------------------------
    # P1 bug-fix coverage: collision-safe IDs and empty-candidate handling
    # ------------------------------------------------------------------

    def test_gemini_rest_tool_call_ids_are_collision_safe(self) -> None:
        """Tool-call IDs from _extract_gemini_text_and_tool_calls use UUIDs,
        not sequential counters, and must not collide across calls."""
        from omicverse.utils.agent_backend_gemini import _extract_gemini_text_and_tool_calls

        payload = {
            "candidates": [{
                "content": {
                    "parts": [
                        {"functionCall": {"name": "tool_a", "args": {}}},
                        {"functionCall": {"name": "tool_a", "args": {}}},
                    ]
                },
                "finishReason": "STOP",
            }]
        }
        _, tc1, _, _ = _extract_gemini_text_and_tool_calls(payload)
        _, tc2, _, _ = _extract_gemini_text_and_tool_calls(payload)

        # Within a single call, two tool calls with the same name get distinct IDs
        self.assertNotEqual(tc1[0].id, tc1[1].id)
        # Across calls, IDs must differ (no sequential-counter reuse)
        all_ids = {tc.id for tc in tc1} | {tc.id for tc in tc2}
        self.assertEqual(len(all_ids), 4)
        # IDs must not use sequential suffixes like _1, _2
        for tc in tc1 + tc2:
            suffix = tc.id.rsplit("_", 1)[-1]
            self.assertFalse(suffix.isdigit(), f"ID {tc.id} uses sequential counter")

    def test_gemini_sdk_tool_call_ids_are_collision_safe(self) -> None:
        """Tool-call IDs from _chat_tools_gemini (SDK path) use UUIDs."""
        backend = OmicVerseLLMBackend(
            system_prompt="system",
            model="gemini-2.5-flash",
            api_key="test-key-non-oauth",
        )

        mock_fc1 = mock.MagicMock()
        mock_fc1.name = "run_code"
        mock_fc1.args = {"code": "1+1"}

        mock_fc2 = mock.MagicMock()
        mock_fc2.name = "run_code"
        mock_fc2.args = {"code": "2+2"}

        mock_part1 = mock.MagicMock()
        mock_part1.text = ""
        mock_part1.function_call = mock_fc1

        mock_part2 = mock.MagicMock()
        mock_part2.text = ""
        mock_part2.function_call = mock_fc2

        mock_content = mock.MagicMock()
        mock_content.parts = [mock_part1, mock_part2]

        mock_candidate = mock.MagicMock()
        mock_candidate.content = mock_content

        mock_resp = mock.MagicMock()
        mock_resp.candidates = [mock_candidate]
        mock_resp.usage_metadata = None

        mock_genai = mock.MagicMock()
        mock_model_instance = mock.MagicMock()
        mock_model_instance.generate_content.return_value = mock_resp
        mock_genai.GenerativeModel.return_value = mock_model_instance

        # Link mock_google.generativeai to mock_genai so import resolves correctly
        mock_google = mock.MagicMock()
        mock_google.generativeai = mock_genai

        with mock.patch.dict("sys.modules", {"google.generativeai": mock_genai, "google": mock_google}):
            response = backend._chat_tools_gemini(
                messages=[{"role": "user", "content": "run"}],
                tools=[{"name": "run_code", "description": "run", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}],
                tool_choice="auto",
            )

        self.assertEqual(len(response.tool_calls), 2)
        self.assertNotEqual(response.tool_calls[0].id, response.tool_calls[1].id)
        for tc in response.tool_calls:
            suffix = tc.id.rsplit("_", 1)[-1]
            self.assertFalse(suffix.isdigit(), f"ID {tc.id} uses sequential counter")

    def test_gemini_rest_empty_candidates_returns_gracefully(self) -> None:
        """Empty Gemini candidate list must not crash — returns empty content."""
        from omicverse.utils.agent_backend_gemini import _extract_gemini_text_and_tool_calls

        # Empty candidates list
        content, tcs, raw, stop = _extract_gemini_text_and_tool_calls({"candidates": []})
        self.assertIsNone(content)
        self.assertEqual(tcs, [])
        self.assertIsNone(raw)
        self.assertEqual(stop, "end_turn")

        # Missing candidates key
        content, tcs, raw, stop = _extract_gemini_text_and_tool_calls({})
        self.assertIsNone(content)
        self.assertEqual(tcs, [])

        # candidates is None
        content, tcs, raw, stop = _extract_gemini_text_and_tool_calls({"candidates": None})
        self.assertIsNone(content)
        self.assertEqual(tcs, [])

    def test_chat_tools_gemini_rest_empty_candidates_no_crash(self) -> None:
        """_chat_tools_gemini_rest returns empty ChatResponse on empty candidates."""
        backend = OmicVerseLLMBackend(
            system_prompt="system",
            model="gemini-2.5-flash",
            api_key='{"token":"oauth-token","projectId":"demo"}',
        )

        with mock.patch(
            "omicverse.utils.agent_backend_gemini._gemini_cli_request",
            return_value={"response": {"candidates": []}},
        ):
            response = backend._chat_tools_gemini(
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"name": "t", "description": "d", "parameters": {}}],
                tool_choice="auto",
            )

        self.assertIsNone(response.content)
        self.assertEqual(response.tool_calls, [])
        self.assertEqual(response.stop_reason, "end_turn")


if __name__ == "__main__":
    unittest.main()

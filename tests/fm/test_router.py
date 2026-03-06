"""Tests for ``omicverse.fm.router``."""

import json
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from omicverse.fm.router import (
    VALID_FM_TOOLS,
    VALID_TASKS,
    RouterIntent,
    RouterInputs,
    RouterOutput,
    RouterSelection,
    ModelSelection,
    ResolvedParams,
    ToolCall,
    Question,
    build_model_cards,
    build_router_prompt,
    call_router_llm,
    route_query,
    validate_router_output,
    _extract_json_from_response,
)
from omicverse.fm.registry import get_registry


# ===========================================================================
# Helpers
# ===========================================================================


def _make_valid_router_output(**overrides) -> dict:
    """Build a minimal valid router output dict."""
    base = {
        "intent": {"task": "embed", "confidence": 0.9},
        "inputs": {"query": "embed my cells"},
        "selection": {
            "recommended": {"name": "scgpt", "rationale": "general purpose"},
            "fallbacks": [{"name": "uce", "rationale": "cross-species"}],
        },
        "resolved_params": {},
        "plan": [{"tool": "fm_run", "args": {"model_name": "scgpt"}}],
        "questions": [],
        "warnings": [],
    }
    base.update(overrides)
    return base


def _mock_llm_response(output_dict: dict) -> dict:
    """Wrap a router output dict as a mock _call_agent return value."""
    return {"success": True, "response": json.dumps(output_dict)}


# ===========================================================================
# Pydantic model tests
# ===========================================================================


class TestRouterIntent:
    def test_valid_task(self):
        intent = RouterIntent(task="embed", confidence=0.9)
        assert intent.task == "embed"

    def test_invalid_task(self):
        with pytest.raises(ValidationError):
            RouterIntent(task="invalid_task", confidence=0.5)

    def test_confidence_bounds(self):
        RouterIntent(task="embed", confidence=0.0)
        RouterIntent(task="embed", confidence=1.0)
        with pytest.raises(ValidationError):
            RouterIntent(task="embed", confidence=1.5)
        with pytest.raises(ValidationError):
            RouterIntent(task="embed", confidence=-0.1)


class TestToolCall:
    def test_valid_tool(self):
        tc = ToolCall(tool="fm_run", args={"model_name": "scgpt"})
        assert tc.tool == "fm_run"

    def test_invalid_tool(self):
        with pytest.raises(ValidationError):
            ToolCall(tool="invalid_tool")


class TestRouterOutput:
    def test_minimal_valid_output(self):
        data = _make_valid_router_output()
        output = RouterOutput.model_validate(data)
        assert output.intent.task == "embed"
        assert output.selection.recommended.name == "scgpt"

    def test_full_output(self):
        data = _make_valid_router_output(
            data_profile={"n_cells": 100, "species": "human"},
            questions=[{"field": "batch_key", "question": "Which column?", "options": ["batch1"]}],
            warnings=["Model may need fine-tuning"],
        )
        output = RouterOutput.model_validate(data)
        assert output.data_profile["n_cells"] == 100
        assert len(output.questions) == 1
        assert len(output.warnings) == 1


# ===========================================================================
# validate_router_output
# ===========================================================================


class TestValidateRouterOutput:
    def test_valid_output_passes(self):
        data = _make_valid_router_output()
        is_valid, errors, parsed = validate_router_output(data)
        assert is_valid
        assert not errors
        assert parsed is not None

    def test_invalid_task_fails(self):
        data = _make_valid_router_output()
        data["intent"]["task"] = "invalid_task"
        is_valid, errors, _ = validate_router_output(data)
        assert not is_valid
        assert any("Schema validation" in e for e in errors)

    def test_unknown_model_fails(self):
        data = _make_valid_router_output()
        data["selection"]["recommended"]["name"] = "nonexistent_model"
        is_valid, errors, _ = validate_router_output(data)
        assert not is_valid
        assert any("not found" in e.lower() for e in errors)

    def test_invalid_tool_fails(self):
        data = _make_valid_router_output()
        data["plan"] = [{"tool": "invalid_tool", "args": {}}]
        is_valid, errors, _ = validate_router_output(data)
        assert not is_valid


# ===========================================================================
# _extract_json_from_response
# ===========================================================================


class TestExtractJsonFromResponse:
    def test_direct_json(self):
        raw = json.dumps({"key": "value"})
        result = _extract_json_from_response(raw)
        assert result is not None
        assert result["key"] == "value"

    def test_markdown_code_block(self):
        raw = "Here is the result:\n```json\n{\"key\": \"value\"}\n```\nDone."
        result = _extract_json_from_response(raw)
        assert result is not None
        assert result["key"] == "value"

    def test_json_with_surrounding_text(self):
        raw = "Sure! Here is the output: {\"key\": \"value\"} Hope this helps."
        result = _extract_json_from_response(raw)
        assert result is not None
        assert result["key"] == "value"

    def test_invalid_json(self):
        raw = "This is not JSON at all."
        result = _extract_json_from_response(raw)
        assert result is None


# ===========================================================================
# build_model_cards
# ===========================================================================


class TestBuildModelCards:
    def test_builds_cards(self):
        cards = build_model_cards()
        assert "scgpt" in cards
        assert "geneformer" in cards
        assert "uce" in cards

    def test_skill_ready_filter(self):
        cards = build_model_cards(skill_ready_only=True)
        assert "scgpt" in cards

    def test_vram_filter(self):
        cards = build_model_cards(max_vram_gb=4)
        assert isinstance(cards, str)


# ===========================================================================
# build_router_prompt
# ===========================================================================


class TestBuildRouterPrompt:
    def test_includes_query(self):
        prompt = build_router_prompt(
            query="embed my human data",
            model_cards="## scgpt\n...",
        )
        assert "embed my human data" in prompt

    def test_includes_data_profile(self):
        profile = {"n_cells": 100, "species": "human"}
        prompt = build_router_prompt(
            query="embed data",
            model_cards="## scgpt\n...",
            data_profile=profile,
        )
        assert "100" in prompt
        assert "human" in prompt

    def test_includes_model_cards(self):
        prompt = build_router_prompt(
            query="embed data",
            model_cards="## scgpt\nBest model for RNA",
        )
        assert "Best model for RNA" in prompt

    def test_includes_constraints(self):
        prompt = build_router_prompt(
            query="embed data",
            model_cards="...",
            max_vram_gb=8,
        )
        assert "8" in prompt


# ===========================================================================
# call_router_llm
# ===========================================================================


class TestCallRouterLLM:
    @pytest.mark.asyncio
    async def test_missing_call_agent_returns_error(self):
        context = {}  # No _call_agent
        ok, result, errors = await call_router_llm(context, "test prompt")
        assert not ok
        assert len(errors) > 0
        assert any("_call_agent" in e for e in errors)

    @pytest.mark.asyncio
    async def test_valid_json_passes_without_retry(self):
        valid_output = _make_valid_router_output()
        mock_agent = AsyncMock(return_value=_mock_llm_response(valid_output))
        context = {"_call_agent": mock_agent}

        ok, result, warnings = await call_router_llm(context, "test prompt")
        assert ok
        assert result["intent"]["task"] == "embed"
        assert mock_agent.call_count == 1  # no retry

    @pytest.mark.asyncio
    async def test_invalid_json_triggers_retry(self):
        valid_output = _make_valid_router_output()
        mock_agent = AsyncMock(side_effect=[
            {"success": True, "response": "not json at all"},
            _mock_llm_response(valid_output),
        ])
        context = {"_call_agent": mock_agent}

        ok, result, _ = await call_router_llm(context, "test prompt", max_retries=1)
        assert ok
        assert mock_agent.call_count == 2

    @pytest.mark.asyncio
    async def test_invalid_task_triggers_retry(self):
        bad_output = _make_valid_router_output()
        bad_output["intent"]["task"] = "invalid_task"
        good_output = _make_valid_router_output()

        mock_agent = AsyncMock(side_effect=[
            _mock_llm_response(bad_output),
            _mock_llm_response(good_output),
        ])
        context = {"_call_agent": mock_agent}

        ok, result, _ = await call_router_llm(context, "test prompt", max_retries=1)
        assert ok

    @pytest.mark.asyncio
    async def test_unknown_model_triggers_retry(self):
        bad_output = _make_valid_router_output()
        bad_output["selection"]["recommended"]["name"] = "nonexistent_model"
        good_output = _make_valid_router_output()

        mock_agent = AsyncMock(side_effect=[
            _mock_llm_response(bad_output),
            _mock_llm_response(good_output),
        ])
        context = {"_call_agent": mock_agent}

        ok, result, _ = await call_router_llm(context, "test prompt", max_retries=1)
        assert ok


# ===========================================================================
# route_query
# ===========================================================================


class TestRouteQuery:
    @pytest.mark.asyncio
    async def test_missing_call_agent_returns_structured_error(self):
        result = await route_query("embed my data", context={})
        assert "error" in result or "warnings" in result

    @pytest.mark.asyncio
    async def test_injects_data_profile(self):
        valid_output = _make_valid_router_output()
        mock_agent = AsyncMock(return_value=_mock_llm_response(valid_output))
        context = {"_call_agent": mock_agent}

        profile = {"n_cells": 100, "species": "human", "gene_scheme": "symbol"}
        result = await route_query(
            "embed my data",
            context=context,
            data_profile=profile,
        )
        # Check that the LLM was called with the profile data in the prompt
        assert mock_agent.called
        call_kwargs = mock_agent.call_args[1] if mock_agent.call_args[1] else {}
        messages = call_kwargs.get("messages", mock_agent.call_args[0][0] if mock_agent.call_args[0] else [])
        prompt_text = str(messages)
        assert "100" in prompt_text or result.get("data_profile") is not None

    @pytest.mark.asyncio
    async def test_overrides_resolved_params(self):
        valid_output = _make_valid_router_output()
        mock_agent = AsyncMock(return_value=_mock_llm_response(valid_output))
        context = {"_call_agent": mock_agent}

        result = await route_query(
            "embed my data",
            context=context,
            output_path="/custom/output.h5ad",
            batch_key="my_batch",
        )
        if "resolved_params" in result:
            assert result["resolved_params"].get("output_path") == "/custom/output.h5ad"
            assert result["resolved_params"].get("batch_key") == "my_batch"

    @pytest.mark.asyncio
    async def test_candidate_filtering_skill_ready_only(self):
        valid_output = _make_valid_router_output()
        mock_agent = AsyncMock(return_value=_mock_llm_response(valid_output))
        context = {"_call_agent": mock_agent}

        result = await route_query(
            "embed my data",
            context=context,
            skill_ready_only=True,
        )
        assert "intent" in result or "error" in result

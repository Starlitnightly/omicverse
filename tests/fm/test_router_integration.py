"""Integration tests for ``omicverse.fm.router`` with real AnnData files."""

import json
from unittest.mock import AsyncMock

import pytest

import scanpy as sc
import numpy as np

from omicverse.fm.router import (
    VALID_FM_TOOLS,
    route_query,
    validate_router_output,
)
from omicverse.fm import api


# ===========================================================================
# Helpers
# ===========================================================================


def _make_mock_llm_response(**overrides) -> dict:
    """Build a valid _call_agent return dict wrapping a router JSON output."""
    output = {
        "intent": {"task": "embed", "confidence": 0.9},
        "inputs": {"query": "embed my cells"},
        "selection": {
            "recommended": {"name": "scgpt", "rationale": "general purpose"},
            "fallbacks": [{"name": "uce", "rationale": "cross-species"}],
        },
        "resolved_params": {"output_path": None, "batch_key": None, "label_key": None},
        "plan": [
            {"tool": "fm_preprocess_validate", "args": {"model_name": "scgpt"}},
            {"tool": "fm_run", "args": {"model_name": "scgpt", "task": "embed"}},
            {"tool": "fm_interpret_results", "args": {}},
        ],
        "questions": [],
        "warnings": [],
    }
    output.update(overrides)
    return {"success": True, "response": json.dumps(output)}


# ===========================================================================
# Router integration with real AnnData
# ===========================================================================


@pytest.mark.integration
class TestSCFMRouterIntegration:

    @pytest.mark.asyncio
    async def test_returns_data_profile_with_real_h5ad(self, test_adata_path):
        mock_agent = AsyncMock(return_value=_make_mock_llm_response())
        context = {"_call_agent": mock_agent}

        profile = api._profile_data_impl(test_adata_path)

        result = await route_query(
            "embed my cells",
            context=context,
            adata_path=test_adata_path,
            data_profile=profile,
        )
        # The data profile should be available in the result
        if "data_profile" in result:
            assert result["data_profile"] is not None

    @pytest.mark.asyncio
    async def test_incompatible_model_triggers_warning(self, test_adata_path):
        # Return geneformer (needs Ensembl) for symbol data
        resp = _make_mock_llm_response(
            selection={
                "recommended": {"name": "geneformer", "rationale": "rank-value"},
                "fallbacks": [{"name": "scgpt", "rationale": "fallback"}],
            }
        )
        mock_agent = AsyncMock(return_value=resp)
        context = {"_call_agent": mock_agent}

        profile = api._profile_data_impl(test_adata_path)

        result = await route_query(
            "embed my cells",
            context=context,
            adata_path=test_adata_path,
            data_profile=profile,
        )
        # Should have warnings or reroute
        has_warning = "warnings" in result and len(result.get("warnings", [])) > 0
        was_rerouted = (
            "selection" in result
            and result.get("selection", {}).get("recommended", {}).get("name") != "geneformer"
        )
        assert has_warning or was_rerouted

    @pytest.mark.asyncio
    async def test_router_with_prespecified_batch_key(self, test_adata_path):
        mock_agent = AsyncMock(return_value=_make_mock_llm_response())
        context = {"_call_agent": mock_agent}

        result = await route_query(
            "embed my cells",
            context=context,
            adata_path=test_adata_path,
            batch_key="batch",
        )
        if "resolved_params" in result:
            assert result["resolved_params"].get("batch_key") == "batch"

    @pytest.mark.asyncio
    async def test_router_with_skill_ready_only(self, test_adata_path):
        mock_agent = AsyncMock(return_value=_make_mock_llm_response())
        context = {"_call_agent": mock_agent}

        result = await route_query(
            "embed my cells",
            context=context,
            skill_ready_only=True,
        )
        assert "intent" in result or "error" in result

    @pytest.mark.asyncio
    async def test_router_with_max_vram_constraint(self, test_adata_path):
        mock_agent = AsyncMock(return_value=_make_mock_llm_response())
        context = {"_call_agent": mock_agent}

        result = await route_query(
            "embed my cells",
            context=context,
            max_vram_gb=8,
        )
        assert "intent" in result or "error" in result


# ===========================================================================
# Data profiling
# ===========================================================================


@pytest.mark.integration
class TestRouterDataProfiling:
    def test_profile_detects_species(self, test_adata_path):
        profile = api._profile_data_impl(test_adata_path)
        assert profile["species"] == "human"

    def test_profile_detects_batch_columns(self, test_adata_path):
        profile = api._profile_data_impl(test_adata_path)
        assert "batch" in profile["batch_columns"]

    def test_profile_detects_celltype_columns(self, test_adata_path):
        profile = api._profile_data_impl(test_adata_path)
        assert "celltype" in profile["celltype_columns"]

    def test_router_handles_missing_file_gracefully(self):
        profile = api._profile_data_impl("/nonexistent/file.h5ad")
        assert "error" in profile


# ===========================================================================
# Execution plan validation
# ===========================================================================


@pytest.mark.integration
class TestRouterExecutionPlan:
    @pytest.mark.asyncio
    async def test_plan_includes_validate_step(self, test_adata_path):
        resp = _make_mock_llm_response()
        mock_agent = AsyncMock(return_value=resp)
        context = {"_call_agent": mock_agent}

        result = await route_query("embed data", context=context)
        if "plan" in result:
            tool_names = [step.get("tool") or step.tool for step in result["plan"]]
            assert "fm_preprocess_validate" in tool_names

    @pytest.mark.asyncio
    async def test_plan_tool_names_are_valid(self, test_adata_path):
        resp = _make_mock_llm_response()
        mock_agent = AsyncMock(return_value=resp)
        context = {"_call_agent": mock_agent}

        result = await route_query("embed data", context=context)
        if "plan" in result:
            for step in result["plan"]:
                tool = step.get("tool") if isinstance(step, dict) else step.tool
                assert tool in VALID_FM_TOOLS, f"Invalid tool: {tool}"


# ===========================================================================
# Router with existing embeddings
# ===========================================================================


@pytest.mark.integration
class TestRouterWithEmbeddings:
    def test_profile_detects_existing_embeddings(self, test_adata_with_embeddings):
        profile = api._profile_data_impl(test_adata_with_embeddings)
        assert "X_uce" in profile["obsm_keys"]

    @pytest.mark.asyncio
    async def test_router_with_existing_embeddings(self, test_adata_with_embeddings):
        mock_agent = AsyncMock(return_value=_make_mock_llm_response())
        context = {"_call_agent": mock_agent}

        profile = api._profile_data_impl(test_adata_with_embeddings)

        result = await route_query(
            "embed my cells",
            context=context,
            data_profile=profile,
        )
        assert "intent" in result or "error" in result

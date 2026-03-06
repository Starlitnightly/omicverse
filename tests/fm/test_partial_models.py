"""Tests for partial/blocked FM model integrations."""

from pathlib import Path
from unittest.mock import patch

import pytest

from omicverse.fm import api
from omicverse.fm.registry import get_registry
from omicverse.llm.model_factory import ModelFactory


PARTIAL_MODEL_NAMES = [
    "scbert",
    "genecompass",
    "nicheformer",
    "scmulan",
    "tgpt",
    "cellfm",
    "sccello",
    "scprint",
    "aidocell",
    "pulsar",
    "atacformer",
    "scplantllm",
    "langcell",
    "cell2sentence",
    "genept",
    "chatcell",
    "tabula",
]


class TestPartialModelRegistration:
    def test_model_factory_exposes_partial_models(self):
        available = set(ModelFactory.available_models())
        assert set(PARTIAL_MODEL_NAMES) <= available

    def test_registry_resolves_partial_model_adapters(self):
        registry = get_registry()
        for model_name in PARTIAL_MODEL_NAMES:
            assert registry.get(model_name) is not None
            assert registry.get_adapter_class(model_name) is not None


@pytest.mark.integration
class TestBlockedModelExecution:
    @pytest.mark.parametrize("model_name", ["scbert", "genept", "scmulan"])
    def test_api_run_returns_blocked_status(self, model_name, test_adata_path, tmp_path):
        output_path = str(Path(tmp_path) / f"{model_name}.h5ad")
        with patch.object(api, "_maybe_run_in_conda_env", return_value=None):
            result = api.run(
                task="embed",
                model_name=model_name,
                adata_path=test_adata_path,
                output_path=output_path,
            )

        assert result["status"] == "blocked"
        assert result["model"] == model_name
        assert "error" in result

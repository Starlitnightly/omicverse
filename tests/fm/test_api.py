"""Tests for ``omicverse.fm.api``."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from omicverse.fm.registry import (
    GeneIDScheme,
    HardwareRequirements,
    ModelSpec,
    Modality,
    OutputKeys,
    SkillReadyStatus,
    TaskType,
    get_registry,
)
from omicverse.fm import api

import scanpy as sc
import anndata as ad
from sklearn.metrics import silhouette_score  # noqa: F401


# ===========================================================================
# Model Discovery
# ===========================================================================


class TestListModels:
    def test_list_models(self):
        result = api.list_models()
        assert "count" in result
        assert "models" in result
        assert result["count"] >= 22

    def test_list_models_with_task_filter(self):
        result = api.list_models(task="embed")
        assert result["count"] >= 3

        result_ann = api.list_models(task="annotate")
        model_names = [m["name"] for m in result_ann["models"]]
        # UCE doesn't support annotate
        assert "uce" not in model_names

    def test_list_models_skill_ready_only(self):
        result = api.list_models(skill_ready_only=True)
        assert result["count"] == 5
        model_names = {m["name"] for m in result["models"]}
        assert {"scgpt", "geneformer", "uce", "scfoundation", "cellplm"} <= model_names
        for m in result["models"]:
            assert m["status"] == "ready"


class TestDescribeModel:
    def test_describe_model(self):
        result = api.describe_model("uce")
        assert "model" in result
        assert "input_contract" in result
        assert "output_contract" in result
        assert "resources" in result
        assert result["model"]["name"] == "uce"

    def test_describe_model_not_found(self):
        result = api.describe_model("nonexistent_model_xyz")
        assert "error" in result
        assert "available_models" in result


# ===========================================================================
# Data Profiling
# ===========================================================================


class TestProfileData:
    def test_profile_data_file_not_found(self):
        result = api.profile_data("/nonexistent/path.h5ad")
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_profile_data_wrong_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".csv") as f:
            result = api.profile_data(f.name)
            assert "error" in result
            assert ".h5ad" in result["error"]

    @pytest.mark.integration
    def test_profile_real_data(self, test_adata_path):
        result = api.profile_data(test_adata_path)
        assert "error" not in result
        assert result["n_cells"] == 100
        assert result["n_genes"] == 200
        assert result["species"] == "human"
        assert result["gene_scheme"] == "symbol"
        assert "celltype" in result["celltype_columns"]
        assert "batch" in result["batch_columns"]

    @pytest.mark.integration
    def test_profile_detects_batch_columns(self, test_adata_path):
        result = api.profile_data(test_adata_path)
        assert "batch" in result["batch_columns"]

    @pytest.mark.integration
    def test_profile_detects_celltype_columns(self, test_adata_path):
        result = api.profile_data(test_adata_path)
        assert "celltype" in result["celltype_columns"]

    @pytest.mark.integration
    def test_profile_model_compatibility(self, test_adata_path):
        result = api.profile_data(test_adata_path)
        assert "model_compatibility" in result
        compat = result["model_compatibility"]
        assert isinstance(compat, dict)
        # scgpt should be compatible with human + symbol data
        assert "scgpt" in compat

    @pytest.mark.integration
    def test_profile_detects_existing_embeddings(self, test_adata_with_embeddings):
        result = api.profile_data(test_adata_with_embeddings)
        assert "X_uce" in result["obsm_keys"]


# ===========================================================================
# Model Selection
# ===========================================================================


class TestSelectModel:
    def test_select_model_file_not_found(self):
        result = api.select_model("/nonexistent/path.h5ad", "embed")
        assert "error" in result

    @pytest.mark.integration
    def test_select_model_for_embed(self, test_adata_path):
        result = api.select_model(test_adata_path, "embed")
        assert "error" not in result
        assert "recommended" in result
        assert "name" in result["recommended"]
        assert "rationale" in result["recommended"]
        assert "data_profile" in result

    @pytest.mark.integration
    def test_select_model_fallbacks(self, test_adata_path):
        result = api.select_model(test_adata_path, "embed")
        assert "fallbacks" in result
        assert len(result["fallbacks"]) >= 1


# ===========================================================================
# Preprocess & Validate
# ===========================================================================


class TestPreprocessValidate:
    def test_unknown_model(self):
        result = api.preprocess_validate("/some/path.h5ad", "nonexistent_model", "embed")
        assert "error" in result

    @pytest.mark.integration
    def test_validate_compatible(self, test_adata_path):
        result = api.preprocess_validate(test_adata_path, "uce", "embed")
        assert "error" not in result
        assert result["status"] in ["ready", "needs_preprocessing"]
        assert "diagnostics" in result

    @pytest.mark.integration
    def test_validate_incompatible_task(self, test_adata_path):
        result = api.preprocess_validate(test_adata_path, "uce", "annotate")
        assert result["status"] == "incompatible"


# ===========================================================================
# Run
# ===========================================================================


class TestRun:
    def test_run_unknown_model(self):
        result = api.run(
            task="embed", model_name="nonexistent_model_xyz",
            adata_path="/some/path.h5ad",
        )
        assert "error" in result

    @pytest.mark.integration
    def test_run_success_with_mock_adapter(self, test_adata_path, tmp_path):
        """Run with a mock adapter to verify the full pipeline."""
        from omicverse.fm.adapters.base import BaseAdapter

        class MockAdapter(BaseAdapter):
            def run(self, task, adata_path, output_path, batch_key=None,
                    label_key=None, device="auto", batch_size=64):
                import anndata as ad
                adata = ad.read_h5ad(adata_path)
                embeddings = np.random.randn(adata.n_obs, 512).astype(np.float32)
                output_keys = self._postprocess(adata, embeddings, task)
                self._add_provenance(adata, task, output_keys)
                adata.write_h5ad(output_path)
                return {
                    "status": "success",
                    "output_path": output_path,
                    "output_keys": output_keys,
                    "n_cells": adata.n_obs,
                }

            def _load_model(self, device):
                pass

            def _preprocess(self, adata, task):
                return adata

            def _postprocess(self, adata, embeddings, task):
                adata.obsm["X_mock"] = embeddings
                return ["X_mock"]

        output_path = str(tmp_path / "output.h5ad")
        spec = get_registry().get("scgpt")
        mock_adapter = MockAdapter(spec)

        with patch.object(api, "_get_model_adapter", return_value=mock_adapter):
            with patch.object(api, "_maybe_run_in_conda_env", return_value=None):
                result = api.run(
                    task="embed", model_name="scgpt",
                    adata_path=test_adata_path, output_path=output_path,
                )

        assert result["status"] == "success"
        assert Path(result["output_path"]).exists()


# ===========================================================================
# Interpret Results
# ===========================================================================


@pytest.mark.integration
class TestInterpretResults:
    def test_interpret_with_embeddings(self, test_adata_with_embeddings, tmp_path):
        output_dir = str(tmp_path / "output")
        result = api.interpret_results(
            test_adata_with_embeddings,
            task="embed",
            output_dir=output_dir,
            generate_umap=False,
        )
        assert "error" not in result
        assert "metrics" in result
        assert "embeddings" in result["metrics"]
        assert "X_uce" in result["metrics"]["embeddings"]
        assert result["metrics"]["embeddings"]["X_uce"]["dim"] == 1280
        assert result["metrics"]["n_cells"] == 100

    def test_interpret_computes_silhouette(self, test_adata_with_embeddings, tmp_path):
        output_dir = str(tmp_path / "output")
        result = api.interpret_results(
            test_adata_with_embeddings,
            task="embed",
            output_dir=output_dir,
            generate_umap=False,
        )
        assert "silhouette" in result["metrics"]["embeddings"]["X_uce"]
        assert result["metrics"]["embeddings"]["X_uce"]["silhouette"] > 0

    def test_interpret_no_embeddings(self, test_adata_path, tmp_path):
        output_dir = str(tmp_path / "output")
        result = api.interpret_results(
            test_adata_path,
            task="embed",
            output_dir=output_dir,
            generate_umap=False,
        )
        assert "error" not in result
        assert "warnings" in result
        assert any("no foundation model embeddings" in w.lower() for w in result["warnings"])

    def test_interpret_generates_umap(self, test_adata_with_embeddings, tmp_path):
        output_dir = str(tmp_path / "output")
        result = api.interpret_results(
            test_adata_with_embeddings,
            task="embed",
            output_dir=output_dir,
            generate_umap=True,
            color_by=["celltype"],
        )
        assert "error" not in result
        if "visualizations" in result:
            for viz in result["visualizations"]:
                assert "path" in viz

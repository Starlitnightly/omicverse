"""Real-data inference tests for ``ov.fm`` adapters.

Run on Taiwan server with GPU::

    OV_FM_DISABLE_CONDA_SUBPROCESS=1 \
    pytest tests/fm/test_real_inference.py -v -s --tb=long
"""

import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

pytestmark = [pytest.mark.fm_real, pytest.mark.fm_gpu]


# ---------------------------------------------------------------------------
# scGPT Real Inference
# ---------------------------------------------------------------------------


class TestScGPTRealInference:
    """Validate scGPT adapter with real data and checkpoint."""

    def test_embed_produces_valid_output(
        self, real_data_path, scgpt_checkpoint_dir, tmp_path
    ):
        from omicverse.fm.adapters._scgpt import ScGPTAdapter

        adapter = ScGPTAdapter(checkpoint_dir=scgpt_checkpoint_dir)
        output_path = str(tmp_path / "scgpt_output.h5ad")

        result = adapter.run(
            task="embed",
            adata_path=real_data_path,
            output_path=output_path,
        )

        assert result["status"] == "success"
        assert Path(result["output_path"]).exists()

        adata = ad.read_h5ad(output_path)
        emb_key = "X_scGPT"
        assert emb_key in adata.obsm, f"Missing {emb_key} in obsm"
        embeddings = adata.obsm[emb_key]
        assert embeddings.ndim == 2
        assert embeddings.shape[0] == adata.n_obs
        assert embeddings.dtype == np.float32
        assert not np.any(np.isnan(embeddings)), "Embeddings contain NaN"
        assert not np.all(embeddings == 0), "Embeddings are all zeros"

    def test_provenance_recorded(
        self, real_data_path, scgpt_checkpoint_dir, tmp_path
    ):
        from omicverse.fm.adapters._scgpt import ScGPTAdapter

        adapter = ScGPTAdapter(checkpoint_dir=scgpt_checkpoint_dir)
        output_path = str(tmp_path / "scgpt_prov.h5ad")

        adapter.run(
            task="embed",
            adata_path=real_data_path,
            output_path=output_path,
        )

        adata = ad.read_h5ad(output_path)
        assert "fm" in adata.uns
        latest = json.loads(adata.uns["fm"]["latest_json"])
        assert latest["model_name"] == "scgpt"
        assert latest["task"] == "embed"

    def test_embed_via_api(
        self, real_data_path, scgpt_checkpoint_dir, tmp_path
    ):
        """Test through the public api.run() interface."""
        os.environ["OV_FM_CHECKPOINT_DIR_SCGPT"] = scgpt_checkpoint_dir
        from omicverse.fm import api

        output_path = str(tmp_path / "scgpt_api.h5ad")

        result = api.run(
            task="embed",
            model_name="scgpt",
            adata_path=real_data_path,
            output_path=output_path,
        )

        assert result.get("status") == "success" or "error" not in result


# ---------------------------------------------------------------------------
# Geneformer Real Inference
# ---------------------------------------------------------------------------


class TestGeneformerRealInference:
    """Validate Geneformer adapter with real data and checkpoint."""

    def test_embed_produces_valid_output(
        self, real_data_path, geneformer_checkpoint_dir, tmp_path
    ):
        from omicverse.fm.adapters._geneformer import GeneformerAdapter

        adapter = GeneformerAdapter(checkpoint_dir=geneformer_checkpoint_dir)
        output_path = str(tmp_path / "geneformer_output.h5ad")

        result = adapter.run(
            task="embed",
            adata_path=real_data_path,
            output_path=output_path,
        )

        assert result["status"] == "success"
        assert Path(result["output_path"]).exists()

        adata = ad.read_h5ad(output_path)
        emb_key = "X_geneformer"
        assert emb_key in adata.obsm, f"Missing {emb_key} in obsm"
        embeddings = adata.obsm[emb_key]
        assert embeddings.ndim == 2
        assert embeddings.shape[0] == adata.n_obs
        assert not np.any(np.isnan(embeddings)), "Embeddings contain NaN"

    def test_cpu_fallback(
        self, real_data_path, geneformer_checkpoint_dir, tmp_path
    ):
        """Geneformer spec has cpu_fallback=True, verify it works on CPU."""
        from omicverse.fm.adapters._geneformer import GeneformerAdapter

        adapter = GeneformerAdapter(checkpoint_dir=geneformer_checkpoint_dir)
        output_path = str(tmp_path / "geneformer_cpu.h5ad")

        result = adapter.run(
            task="embed",
            adata_path=real_data_path,
            output_path=output_path,
            device="cpu",
        )

        assert result["status"] == "success"
        assert result["device"] == "cpu"

    def test_provenance_recorded(
        self, real_data_path, geneformer_checkpoint_dir, tmp_path
    ):
        from omicverse.fm.adapters._geneformer import GeneformerAdapter

        adapter = GeneformerAdapter(checkpoint_dir=geneformer_checkpoint_dir)
        output_path = str(tmp_path / "geneformer_prov.h5ad")

        adapter.run(
            task="embed",
            adata_path=real_data_path,
            output_path=output_path,
        )

        adata = ad.read_h5ad(output_path)
        assert "fm" in adata.uns
        latest = json.loads(adata.uns["fm"]["latest_json"])
        assert latest["model_name"] == "geneformer"


# ---------------------------------------------------------------------------
# UCE Real Inference
# ---------------------------------------------------------------------------


class TestUCERealInference:
    """Validate UCE adapter with real data and checkpoint."""

    def test_embed_produces_valid_output(
        self, real_data_path, uce_checkpoint_dir, tmp_path
    ):
        from omicverse.fm.adapters._uce import UCEAdapter

        adapter = UCEAdapter(checkpoint_dir=uce_checkpoint_dir)
        output_path = str(tmp_path / "uce_output.h5ad")

        result = adapter.run(
            task="embed",
            adata_path=real_data_path,
            output_path=output_path,
        )

        assert result["status"] == "success"
        assert Path(result["output_path"]).exists()

        adata = ad.read_h5ad(output_path)
        emb_key = "X_uce"
        assert emb_key in adata.obsm, f"Missing {emb_key} in obsm"
        embeddings = adata.obsm[emb_key]
        assert embeddings.ndim == 2
        assert embeddings.shape[0] == adata.n_obs
        assert embeddings.shape[1] == 1280
        assert embeddings.dtype == np.float32
        assert not np.any(np.isnan(embeddings)), "Embeddings contain NaN"

    def test_provenance_recorded(
        self, real_data_path, uce_checkpoint_dir, tmp_path
    ):
        from omicverse.fm.adapters._uce import UCEAdapter

        adapter = UCEAdapter(checkpoint_dir=uce_checkpoint_dir)
        output_path = str(tmp_path / "uce_prov.h5ad")

        adapter.run(
            task="embed",
            adata_path=real_data_path,
            output_path=output_path,
        )

        adata = ad.read_h5ad(output_path)
        assert "fm" in adata.uns
        latest = json.loads(adata.uns["fm"]["latest_json"])
        assert latest["model_name"] == "uce"
        assert "X_uce" in latest["output_keys"]


# ---------------------------------------------------------------------------
# scFoundation Real Inference
# ---------------------------------------------------------------------------


class TestScFoundationRealInference:
    """Validate scFoundation adapter with real data."""

    def test_embed_produces_valid_output(
        self, real_data_path, scfoundation_checkpoint_dir, tmp_path
    ):
        from omicverse.fm.adapters._scfoundation import ScFoundationAdapter

        adapter = ScFoundationAdapter(checkpoint_dir=scfoundation_checkpoint_dir)
        output_path = str(tmp_path / "scfoundation_output.h5ad")

        result = adapter.run(
            task="embed",
            adata_path=real_data_path,
            output_path=output_path,
        )

        assert result["status"] == "success"
        assert Path(result["output_path"]).exists()

        adata = ad.read_h5ad(output_path)
        emb_key = "X_scfoundation"
        assert emb_key in adata.obsm


# ---------------------------------------------------------------------------
# CellPLM Real Inference
# ---------------------------------------------------------------------------


class TestCellPLMRealInference:
    """Validate CellPLM adapter with real data."""

    def test_embed_produces_valid_output(
        self, real_data_path, cellplm_checkpoint_dir, tmp_path
    ):
        from omicverse.fm.adapters._cellplm import CellPLMAdapter

        adapter = CellPLMAdapter(checkpoint_dir=cellplm_checkpoint_dir)
        output_path = str(tmp_path / "cellplm_output.h5ad")

        result = adapter.run(
            task="embed",
            adata_path=real_data_path,
            output_path=output_path,
        )

        assert result["status"] == "success"
        assert Path(result["output_path"]).exists()

        adata = ad.read_h5ad(output_path)
        emb_key = "X_cellplm"
        assert emb_key in adata.obsm


# ---------------------------------------------------------------------------
# End-to-end API Pipeline
# ---------------------------------------------------------------------------


class TestFMAPIEndToEnd:
    """Full pipeline test through the public API."""

    def test_full_pipeline_scgpt(
        self, real_data_path, scgpt_checkpoint_dir, tmp_path
    ):
        os.environ["OV_FM_CHECKPOINT_DIR_SCGPT"] = scgpt_checkpoint_dir
        from omicverse.fm import api

        # Step 1: Profile data
        profile = api.profile_data(real_data_path)
        assert "error" not in profile
        assert any(s in profile["species"] for s in ("human", "mouse", "unknown"))

        # Step 2: Preprocess & validate
        validate = api.preprocess_validate(real_data_path, "scgpt", "embed")
        assert validate["status"] in ("ready", "needs_preprocessing")

        # Step 3: Run
        output_path = str(tmp_path / "e2e_scgpt.h5ad")
        result = api.run(
            task="embed",
            model_name="scgpt",
            adata_path=real_data_path,
            output_path=output_path,
        )
        assert result.get("status") == "success"
        assert Path(output_path).exists()

        # Step 4: Interpret results
        interpret = api.interpret_results(
            output_path,
            task="embed",
            output_dir=str(tmp_path / "interpret"),
            generate_umap=False,
        )
        assert "error" not in interpret
        assert "metrics" in interpret

    def test_full_pipeline_uce(
        self, real_data_path, uce_checkpoint_dir, tmp_path
    ):
        os.environ["OV_FM_CHECKPOINT_DIR_UCE"] = uce_checkpoint_dir
        from omicverse.fm import api

        profile = api.profile_data(real_data_path)
        assert "error" not in profile

        output_path = str(tmp_path / "e2e_uce.h5ad")
        result = api.run(
            task="embed",
            model_name="uce",
            adata_path=real_data_path,
            output_path=output_path,
        )
        assert result.get("status") == "success"

        interpret = api.interpret_results(
            output_path,
            task="embed",
            output_dir=str(tmp_path / "interpret_uce"),
            generate_umap=False,
        )
        assert "error" not in interpret

    def test_full_pipeline_geneformer(
        self, real_data_path, geneformer_checkpoint_dir, tmp_path
    ):
        os.environ["OV_FM_CHECKPOINT_DIR_GENEFORMER"] = geneformer_checkpoint_dir
        from omicverse.fm import api

        output_path = str(tmp_path / "e2e_geneformer.h5ad")
        result = api.run(
            task="embed",
            model_name="geneformer",
            adata_path=real_data_path,
            output_path=output_path,
        )
        assert result.get("status") == "success"

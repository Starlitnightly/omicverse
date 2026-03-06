"""
Geneformer Adapter — bridges ``omicverse.llm`` Geneformer to ``ov.fm``.
"""

from pathlib import Path
from typing import Any, Optional

from .base import BaseAdapter
from ..registry import ModelSpec, TaskType, get_registry


class GeneformerAdapter(BaseAdapter):
    """Adapter for Geneformer foundation model."""

    # Default dictionary file names within checkpoint directory
    _DICT_NAMES = {
        "gene_median_file": "gene_median_dictionary_gc104M.pkl",
        "token_dictionary_file": "token_dictionary_gc104M.pkl",
        "gene_mapping_file": "ensembl_mapping_dict_gc104M.pkl",
    }

    def __init__(self, checkpoint_dir: Optional[str] = None):
        spec = get_registry().get("geneformer")
        if spec is None:
            raise RuntimeError("geneformer not found in registry")
        super().__init__(spec, checkpoint_dir)

    def run(
        self,
        task: TaskType,
        adata_path: str,
        output_path: str,
        batch_key: Optional[str] = None,
        label_key: Optional[str] = None,
        device: str = "auto",
        batch_size: int = 64,
    ) -> dict[str, Any]:
        import anndata as ad

        device = self._resolve_device(device)
        adata = ad.read_h5ad(adata_path)

        # Resolve checkpoint directory
        checkpoint_dir = self._resolve_checkpoint_dir(require=True)
        dict_kwargs = self._resolve_dict_files(Path(checkpoint_dir))

        from omicverse.llm import ModelFactory

        # Detect model version from dictionary file names
        # gc104M → V2, gc30M → V1
        model_version = "V2"  # default to V2 for 104M dicts
        if any("30M" in v for v in dict_kwargs.values()):
            model_version = "V1"

        model = ModelFactory.create_model("geneformer")
        model.load_model(
            model_path=str(checkpoint_dir),
            device=device,
            model_version=model_version,
            **dict_kwargs,
        )

        embeddings = model.get_embeddings(
            adata, batch_size=batch_size, max_ncells=None,
        )

        output_keys = self._postprocess(adata, embeddings, task)
        self._add_provenance(adata, task, output_keys)
        adata.write_h5ad(output_path)

        return {
            "status": "success",
            "output_path": output_path,
            "output_keys": output_keys,
            "n_cells": adata.n_obs,
            "device": device,
        }

    def _resolve_dict_files(self, checkpoint_dir: Path) -> dict[str, str]:
        """Resolve Geneformer dictionary file paths from checkpoint directory."""
        import os

        result: dict[str, str] = {}
        # Search in checkpoint_dir itself and common subdirectories
        search_dirs = [
            checkpoint_dir,
            checkpoint_dir / "geneformer",
        ]
        for key, default_name in self._DICT_NAMES.items():
            # Check env var override first
            env_var = f"OV_FM_GENEFORMER_{key.upper()}"
            env_val = os.environ.get(env_var)
            if env_val and Path(env_val).exists():
                result[key] = env_val
                continue
            # Search in known directories
            for d in search_dirs:
                p = d / default_name
                if p.exists():
                    result[key] = str(p)
                    break
        return result

    def _load_model(self, device: str):
        pass  # Handled in run() via ModelFactory

    def _preprocess(self, adata, task: TaskType):
        return adata

    def _postprocess(self, adata, embeddings, task: TaskType) -> list[str]:
        key = self.spec.output_keys.embedding_key or "X_geneformer"
        adata.obsm[key] = embeddings
        return [key]

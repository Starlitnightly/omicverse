"""
UCE Adapter — bridges ``omicverse.llm`` UCE to ``ov.fm``.
"""

from pathlib import Path
from typing import Any, Optional

from .base import BaseAdapter
from ..registry import ModelSpec, TaskType, get_registry


class UCEAdapter(BaseAdapter):
    """Adapter for UCE (Universal Cell Embedding) foundation model."""

    # Default asset file names within checkpoint directory
    _ASSET_NAMES = {
        "token_file": "token_to_pos.torch",
        "protein_embeddings_dir": "protein_embeddings",
        "spec_chrom_csv_path": "species_chrom.csv",
        "offset_pkl_path": "species_offsets.pkl",
    }
    _ASSET_CANDIDATES = {
        "token_file": ["token_to_pos.torch", "all_tokens.torch"],
        "protein_embeddings_dir": ["protein_embeddings"],
        "spec_chrom_csv_path": ["species_chrom.csv"],
        "offset_pkl_path": ["species_offsets.pkl"],
    }

    def __init__(self, checkpoint_dir: Optional[str] = None):
        spec = get_registry().get("uce")
        if spec is None:
            raise RuntimeError("uce not found in registry")
        super().__init__(spec, checkpoint_dir)

    def run(
        self,
        task: TaskType,
        adata_path: str,
        output_path: str,
        batch_key: Optional[str] = None,
        label_key: Optional[str] = None,
        device: str = "auto",
        batch_size: int = 25,
    ) -> dict[str, Any]:
        import anndata as ad

        device = self._resolve_device(device)
        adata = ad.read_h5ad(adata_path)

        # Resolve checkpoint directory and UCE-specific asset paths
        checkpoint_dir = self._resolve_checkpoint_dir(require=True)
        assets = self._resolve_uce_assets(checkpoint_dir)
        model_path = assets.pop("model_path")

        from omicverse.llm import ModelFactory

        # Pass model_path + asset paths to create_model, which handles
        # both construction and load_model internally
        model = ModelFactory.create_model(
            "uce",
            model_path=str(model_path),
            device=device,
            token_file=str(assets["token_file"]),
            protein_embeddings_dir=str(assets["protein_embeddings_dir"]),
            spec_chrom_csv_path=str(assets["spec_chrom_csv_path"]),
            offset_pkl_path=str(assets["offset_pkl_path"]),
        )

        embeddings = model.get_embeddings(adata, batch_size=batch_size)

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

    def _resolve_uce_assets(self, checkpoint_dir: Path) -> dict[str, Any]:
        """Resolve UCE-specific asset file paths from checkpoint directory."""
        import os

        assets: dict[str, Any] = {}

        # Try to find model checkpoint
        model_candidates = ["model.torch", "model.pt", "best_model.pt", "4layer_model.torch"]
        for candidate in model_candidates:
            p = checkpoint_dir / candidate
            if p.exists():
                assets["model_path"] = p
                break
        if "model_path" not in assets:
            assets["model_path"] = self._find_checkpoint(checkpoint_dir, [".torch", ".pt"])

        # Resolve each asset — check env var override first, then default location
        for key, default_name in self._ASSET_NAMES.items():
            env_var = f"OV_FM_UCE_{key.upper()}"
            env_val = os.environ.get(env_var)
            if env_val and Path(env_val).exists():
                assets[key] = Path(env_val)
            else:
                asset_candidates = self._ASSET_CANDIDATES.get(key, [default_name])
                resolved = next(
                    (checkpoint_dir / candidate for candidate in asset_candidates if (checkpoint_dir / candidate).exists()),
                    checkpoint_dir / default_name,
                )
                assets[key] = resolved

        return assets

    def _load_model(self, device: str):
        pass  # Handled in run() via ModelFactory

    def _preprocess(self, adata, task: TaskType):
        return adata

    def _postprocess(self, adata, embeddings, task: TaskType) -> list[str]:
        key = self.spec.output_keys.embedding_key or "X_uce"
        adata.obsm[key] = embeddings
        return [key]

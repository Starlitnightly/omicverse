"""
scGPT Adapter — bridges ``omicverse.llm`` scGPT to the ``ov.fm`` adapter interface.
"""

from typing import Any, Optional

from .base import BaseAdapter
from ..registry import ModelSpec, TaskType, get_registry


class ScGPTAdapter(BaseAdapter):
    """Adapter for scGPT foundation model."""

    def __init__(self, checkpoint_dir: Optional[str] = None):
        spec = get_registry().get("scgpt")
        if spec is None:
            raise RuntimeError("scgpt not found in registry")
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

        from omicverse.llm import ModelFactory

        model = ModelFactory.create_model("scgpt")
        model.load_model(model_path=str(checkpoint_dir), device=device)

        # Preprocess: scGPT needs binning
        adata = model.preprocess(adata)

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

    def _load_model(self, device: str):
        pass  # Handled in run() via ModelFactory

    def _preprocess(self, adata, task: TaskType):
        return adata  # Handled in run() via model.preprocess()

    def _postprocess(self, adata, embeddings, task: TaskType) -> list[str]:
        key = self.spec.output_keys.embedding_key or "X_scgpt"
        adata.obsm[key] = embeddings
        return [key]

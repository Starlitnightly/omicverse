"""
Factory-backed adapters for experimental or blocked foundation models.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import BaseAdapter
from ..registry import TaskType, get_registry
from ...llm.base import ModelExecutionBlockedError


class FactoryBackedAdapter(BaseAdapter):
    """Adapter that delegates execution to ``omicverse.llm.ModelFactory``."""

    model_type: str = ""

    def __init__(self, checkpoint_dir: Optional[str] = None):
        spec = get_registry().get(self.model_type)
        if spec is None:
            raise RuntimeError(f"{self.model_type} not found in registry")
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

        from omicverse.llm import ModelFactory

        device = self._resolve_device(device)
        adata = ad.read_h5ad(adata_path)
        model = ModelFactory.create_model(self.model_type, device=device)
        checkpoint_dir = self._resolve_checkpoint_dir(require=False)

        try:
            self._load_factory_model(model, checkpoint_dir, task)
            output_keys = self._execute_task(
                model=model,
                adata=adata,
                task=task,
                batch_key=batch_key,
                label_key=label_key,
                batch_size=batch_size,
            )
        except ModelExecutionBlockedError as exc:
            return exc.to_result(task=self._task_name(task), output_path=output_path)

        self._add_provenance(adata, task, output_keys)
        adata.write_h5ad(output_path)
        return {
            "status": "success",
            "output_path": output_path,
            "output_keys": output_keys,
            "n_cells": adata.n_obs,
            "device": device,
        }

    def _load_factory_model(self, model, checkpoint_dir, task: TaskType) -> None:
        task_name = self._task_name(task)
        try:
            model.load_model(str(checkpoint_dir) if checkpoint_dir is not None else None, task=task_name)
        except TypeError:
            if checkpoint_dir is not None:
                model.load_model(str(checkpoint_dir))
            else:
                model.load_model(None)

    def _execute_task(
        self,
        *,
        model,
        adata,
        task: TaskType,
        batch_key: Optional[str],
        label_key: Optional[str],
        batch_size: int,
    ) -> list[str]:
        if task == TaskType.EMBED or task == TaskType.SPATIAL:
            embeddings = model.get_embeddings(adata, batch_size=batch_size)
            return self._postprocess(adata, embeddings, task)

        if task == TaskType.ANNOTATE:
            result = model.predict(
                adata,
                task="annotation",
                batch_size=batch_size,
                batch_key=batch_key,
                label_key=label_key,
            )
            return self._write_annotation_outputs(adata, result)

        if task == TaskType.INTEGRATE:
            if hasattr(model, "integrate"):
                result = model.integrate(adata, batch_key=batch_key or "batch", batch_size=batch_size)
                embeddings = result.get("embeddings", result.get("integrated_embeddings"))
            else:
                embeddings = model.get_embeddings(adata, batch_size=batch_size)
            if embeddings is None:
                raise RuntimeError(f"{self.model_type} did not return embeddings for integrate")
            key = self.spec.output_keys.integration_key or self.spec.output_keys.embedding_key or f"X_{self.model_type}"
            adata.obsm[key] = np.asarray(embeddings, dtype=np.float32)
            return [key]

        raise ModelExecutionBlockedError(
            self.model_type,
            f"{self.model_type} task '{self._task_name(task)}' is not implemented.",
            checkpoint_url=self.spec.checkpoint_url,
        )

    def _write_annotation_outputs(self, adata, result: Any) -> list[str]:
        if not isinstance(result, dict):
            raise RuntimeError(f"{self.model_type} annotation result must be a dict, got: {type(result)!r}")

        output_keys: list[str] = []
        labels = result.get("predicted_celltypes")
        if labels is None:
            labels = result.get("predictions")
        if labels is None:
            labels = result.get("annotations")
        if labels is None:
            raise RuntimeError(f"{self.model_type} annotation result did not include labels")

        ann_key = self.spec.output_keys.annotation_key or f"{self.model_type}_pred"
        adata.obs[ann_key] = labels
        output_keys.append(ann_key)

        embeddings = result.get("embeddings")
        if embeddings is not None and self.spec.output_keys.embedding_key:
            emb_key = self.spec.output_keys.embedding_key
            adata.obsm[emb_key] = np.asarray(embeddings, dtype=np.float32)
            output_keys.append(emb_key)

        confidence = result.get("confidence")
        if confidence is None:
            confidence = result.get("confidence_scores")
        if confidence is not None and self.spec.output_keys.confidence_key:
            conf_key = self.spec.output_keys.confidence_key
            adata.obs[conf_key] = confidence
            output_keys.append(conf_key)

        return output_keys

    def _load_model(self, device: str):
        pass

    def _preprocess(self, adata, task: TaskType):
        return adata

    def _postprocess(self, adata, embeddings, task: TaskType) -> list[str]:
        key = self.spec.output_keys.embedding_key or f"X_{self.model_type}"
        adata.obsm[key] = np.asarray(embeddings, dtype=np.float32)
        return [key]

    @staticmethod
    def _task_name(task: TaskType) -> str:
        return task.value if hasattr(task, "value") else str(task)


def _build_adapter(class_name: str, model_type: str):
    return type(class_name, (FactoryBackedAdapter,), {"model_type": model_type})


ScBERTAdapter = _build_adapter("ScBERTAdapter", "scbert")
GeneCompassAdapter = _build_adapter("GeneCompassAdapter", "genecompass")
NicheformerAdapter = _build_adapter("NicheformerAdapter", "nicheformer")
ScMulanAdapter = _build_adapter("ScMulanAdapter", "scmulan")
TGPTAdapter = _build_adapter("TGPTAdapter", "tgpt")
CellFMAdapter = _build_adapter("CellFMAdapter", "cellfm")
ScCelloAdapter = _build_adapter("ScCelloAdapter", "sccello")
ScPrintAdapter = _build_adapter("ScPrintAdapter", "scprint")
AiDocellAdapter = _build_adapter("AiDocellAdapter", "aidocell")
PulsarAdapter = _build_adapter("PulsarAdapter", "pulsar")
AtacformerAdapter = _build_adapter("AtacformerAdapter", "atacformer")
ScPlantLLMAdapter = _build_adapter("ScPlantLLMAdapter", "scplantllm")
LangCellAdapter = _build_adapter("LangCellAdapter", "langcell")
Cell2SentenceAdapter = _build_adapter("Cell2SentenceAdapter", "cell2sentence")
GenePTAdapter = _build_adapter("GenePTAdapter", "genept")
ChatCellAdapter = _build_adapter("ChatCellAdapter", "chatcell")
TabulaAdapter = _build_adapter("TabulaAdapter", "tabula")


__all__ = [
    "FactoryBackedAdapter",
    "ScBERTAdapter",
    "GeneCompassAdapter",
    "NicheformerAdapter",
    "ScMulanAdapter",
    "TGPTAdapter",
    "CellFMAdapter",
    "ScCelloAdapter",
    "ScPrintAdapter",
    "AiDocellAdapter",
    "PulsarAdapter",
    "AtacformerAdapter",
    "ScPlantLLMAdapter",
    "LangCellAdapter",
    "Cell2SentenceAdapter",
    "GenePTAdapter",
    "ChatCellAdapter",
    "TabulaAdapter",
]

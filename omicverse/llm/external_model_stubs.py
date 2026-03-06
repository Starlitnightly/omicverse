"""
Stub model wrappers for registered foundation models without local runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from anndata import AnnData

from .base import ModelExecutionBlockedError, SCLLMBase


@dataclass(frozen=True)
class StubModelConfig:
    model_name: str
    checkpoint_url: str
    reason: str
    requires_checkpoint: bool = True
    required_env_vars: tuple[str, ...] = ()


STUB_MODEL_CONFIGS: dict[str, StubModelConfig] = {
    "scbert": StubModelConfig(
        model_name="scbert",
        checkpoint_url="https://github.com/TencentAILabHealthcare/scBERT",
        reason="scBERT is registered but the upstream inference runtime is not yet integrated into omicverse.llm.",
    ),
    "genecompass": StubModelConfig(
        model_name="genecompass",
        checkpoint_url="https://github.com/xCompass-AI/GeneCompass",
        reason="GeneCompass is registered but requires upstream graph-prior assets and runtime wiring before local inference can run.",
    ),
    "nicheformer": StubModelConfig(
        model_name="nicheformer",
        checkpoint_url="https://github.com/theislab/nicheformer",
        reason="Nicheformer needs an upstream spatial runtime and checkpoint layout that are not bundled in this repository.",
    ),
    "tgpt": StubModelConfig(
        model_name="tgpt",
        checkpoint_url="https://github.com/deeplearningplus/tGPT",
        reason="tGPT is registered but its autoregressive runtime is not wired into omicverse.llm yet.",
    ),
    "cellfm": StubModelConfig(
        model_name="cellfm",
        checkpoint_url="https://github.com/cellverse/CellFM",
        reason="CellFM is registered but its upstream MLP checkpoint/runtime is not integrated into omicverse.llm yet.",
    ),
    "sccello": StubModelConfig(
        model_name="sccello",
        checkpoint_url="https://github.com/cellarium-ai/scCello",
        reason="scCello download metadata exists, but the model runtime and zero-shot annotation stack are not integrated yet.",
    ),
    "scprint": StubModelConfig(
        model_name="scprint",
        checkpoint_url="https://github.com/scprint/scPRINT",
        reason="scPRINT is registered but its upstream denoising runtime is not integrated into omicverse.llm yet.",
    ),
    "aidocell": StubModelConfig(
        model_name="aidocell",
        checkpoint_url="https://github.com/genbio-ai/AIDO",
        reason="AiDocell is registered but its upstream AIDO runtime is not exposed through omicverse.llm yet.",
    ),
    "pulsar": StubModelConfig(
        model_name="pulsar",
        checkpoint_url="https://github.com/pulsar-ai/PULSAR",
        reason="PULSAR is registered but its multicellular runtime is not integrated into omicverse.llm yet.",
    ),
    "atacformer": StubModelConfig(
        model_name="atacformer",
        checkpoint_url="https://github.com/Atacformer/Atacformer",
        reason="Atacformer requires an ATAC-specific peak tokenizer/runtime that is not bundled in this repository.",
    ),
    "scplantllm": StubModelConfig(
        model_name="scplantllm",
        checkpoint_url="https://github.com/scPlantLLM/scPlantLLM",
        reason="scPlantLLM is registered but its plant-specific runtime is not integrated into omicverse.llm yet.",
    ),
    "langcell": StubModelConfig(
        model_name="langcell",
        checkpoint_url="https://github.com/langcell/LangCell",
        reason="LangCell is registered but its text-cell dual-tower runtime is not integrated into omicverse.llm yet.",
    ),
    "cell2sentence": StubModelConfig(
        model_name="cell2sentence",
        checkpoint_url="https://github.com/vandijklab/cell2sentence",
        reason="Cell2Sentence is registered but the cells-to-text conversion and downstream embedding stack are not integrated yet.",
    ),
    "genept": StubModelConfig(
        model_name="genept",
        checkpoint_url="https://github.com/yiqunchen/GenePT",
        reason="GenePT requires external API-backed gene embeddings and is not runnable through the local omicverse.llm stack yet.",
        requires_checkpoint=False,
        required_env_vars=("OPENAI_API_KEY",),
    ),
    "chatcell": StubModelConfig(
        model_name="chatcell",
        checkpoint_url="https://github.com/chatcell/CHATCELL",
        reason="ChatCell requires the upstream conversational analysis stack and credentials, which are not integrated into omicverse.llm yet.",
        requires_checkpoint=False,
    ),
    "tabula": StubModelConfig(
        model_name="tabula",
        checkpoint_url="https://github.com/aristoteleo/tabula",
        reason="Tabula is registered but its federated perturbation runtime is not integrated into omicverse.llm yet.",
    ),
}


class ExternalModelStub(SCLLMBase):
    """Common blocked-model wrapper used until an upstream runtime is integrated."""

    config: StubModelConfig

    def __init__(self, device: Optional[str] = None, **kwargs):
        super().__init__(self.config.model_name, device)
        self.checkpoint_path = kwargs.get("model_path")
        self.requires_checkpoint = self.config.requires_checkpoint

    def _blocked(self, task: str, extra_reason: Optional[str] = None) -> ModelExecutionBlockedError:
        reason = extra_reason or self.config.reason
        raise ModelExecutionBlockedError(
            self.config.model_name,
            f"{self.config.model_name} cannot run task '{task}': {reason}",
            checkpoint_url=self.config.checkpoint_url,
            required_env_vars=list(self.config.required_env_vars),
            required_checkpoint=self.config.requires_checkpoint,
        )

    def load_model(self, model_path: Optional[str | Path], **kwargs) -> None:
        self.checkpoint_path = None if model_path in (None, "") else Path(model_path)
        if self.requires_checkpoint and self.checkpoint_path is None:
            self._blocked("load_model", "checkpoint not configured")
        if self.requires_checkpoint and self.checkpoint_path is not None and not self.checkpoint_path.exists():
            self._blocked("load_model", f"checkpoint path does not exist: {self.checkpoint_path}")
        self.is_loaded = True

    def preprocess(self, adata: AnnData, **kwargs) -> AnnData:
        return adata

    def predict(self, adata: AnnData, task: str = "annotation", **kwargs) -> dict[str, Any]:
        self._blocked(task)

    def fine_tune(
        self,
        train_adata: AnnData,
        valid_adata: Optional[AnnData] = None,
        **kwargs,
    ) -> dict[str, Any]:
        self._blocked("fine_tune")

    def get_embeddings(self, adata: AnnData, **kwargs) -> np.ndarray:
        self._blocked("embed")

    def _save_model_specific(self, save_path: Path, **kwargs) -> None:
        self._blocked("save_model")


def _build_stub_class(class_name: str, config_key: str):
    return type(class_name, (ExternalModelStub,), {"config": STUB_MODEL_CONFIGS[config_key]})


ScBERTModel = _build_stub_class("ScBERTModel", "scbert")
GeneCompassModel = _build_stub_class("GeneCompassModel", "genecompass")
NicheformerModel = _build_stub_class("NicheformerModel", "nicheformer")
TGPTModel = _build_stub_class("TGPTModel", "tgpt")
CellFMModel = _build_stub_class("CellFMModel", "cellfm")
ScCelloModel = _build_stub_class("ScCelloModel", "sccello")
ScPrintModel = _build_stub_class("ScPrintModel", "scprint")
AiDocellModel = _build_stub_class("AiDocellModel", "aidocell")
PulsarModel = _build_stub_class("PulsarModel", "pulsar")
AtacformerModel = _build_stub_class("AtacformerModel", "atacformer")
ScPlantLLMModel = _build_stub_class("ScPlantLLMModel", "scplantllm")
LangCellModel = _build_stub_class("LangCellModel", "langcell")
Cell2SentenceModel = _build_stub_class("Cell2SentenceModel", "cell2sentence")
GenePTModel = _build_stub_class("GenePTModel", "genept")
ChatCellModel = _build_stub_class("ChatCellModel", "chatcell")
TabulaModel = _build_stub_class("TabulaModel", "tabula")


__all__ = [
    "ModelExecutionBlockedError",
    "ExternalModelStub",
    "STUB_MODEL_CONFIGS",
    "ScBERTModel",
    "GeneCompassModel",
    "NicheformerModel",
    "TGPTModel",
    "CellFMModel",
    "ScCelloModel",
    "ScPrintModel",
    "AiDocellModel",
    "PulsarModel",
    "AtacformerModel",
    "ScPlantLLMModel",
    "LangCellModel",
    "Cell2SentenceModel",
    "GenePTModel",
    "ChatCellModel",
    "TabulaModel",
]

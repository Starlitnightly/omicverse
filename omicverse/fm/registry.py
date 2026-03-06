"""
Foundation Model Registry for Single-Cell Analysis
===================================================

Defines model capabilities, I/O contracts, and hardware requirements
for all supported single-cell foundation models.

This is the core data layer of ``ov.fm`` — every model known to the
framework is described by a :class:`ModelSpec` dataclass and stored in
a global :class:`ModelRegistry` singleton.
"""

import importlib
import importlib.metadata
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    """Supported foundation model tasks."""
    EMBED = "embed"
    ANNOTATE = "annotate"
    INTEGRATE = "integrate"
    PERTURB = "perturb"
    SPATIAL = "spatial"
    DRUG_RESPONSE = "drug_response"


class Modality(str, Enum):
    """Data modalities."""
    RNA = "RNA"
    ATAC = "ATAC"
    SPATIAL = "Spatial"
    PROTEIN = "Protein"
    MULTIOMICS = "Multi-omics"


class GeneIDScheme(str, Enum):
    """Gene identifier schemes."""
    SYMBOL = "symbol"    # HGNC gene symbols (e.g., TP53)
    ENSEMBL = "ensembl"  # Ensembl IDs (e.g., ENSG00000141510)
    CUSTOM = "custom"    # Model-specific gene set


class SkillReadyStatus(str, Enum):
    """Adapter implementation readiness."""
    READY = "ready"        # Full adapter implemented
    PARTIAL = "partial"    # Partial spec, needs validation
    REFERENCE = "reference"  # Reference docs only, no adapter


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class HardwareRequirements:
    """Hardware requirements for model inference."""
    gpu_required: bool = True
    min_vram_gb: int = 8
    recommended_vram_gb: int = 16
    cpu_fallback: bool = False
    default_batch_size: int = 64


@dataclass
class OutputKeys:
    """Standard AnnData output keys for each task."""
    embedding_key: str = ""       # obsm key for embeddings
    annotation_key: str = ""      # obs key for predictions
    confidence_key: str = ""      # obs key for confidence scores
    integration_key: str = ""     # obsm key for integrated embeddings


@dataclass
class ModelSpec:
    """Complete specification for a foundation model.

    Parameters
    ----------
    name : str
        Lowercase model identifier (e.g. ``"scgpt"``).
    version : str
        Model version string.
    skill_ready : SkillReadyStatus
        Adapter readiness level.
    tasks : list[TaskType]
        Supported tasks.
    modalities : list[Modality]
        Supported data modalities.
    species : list[str]
        Supported species (lowercase).
    gene_id_scheme : GeneIDScheme
        Expected gene identifier format.
    requires_finetuning : bool
        Whether annotation/some tasks need fine-tuning.
    zero_shot_embedding : bool
        Whether zero-shot embedding is supported.
    zero_shot_annotation : bool
        Whether zero-shot annotation is supported.
    output_keys : OutputKeys
        Standard AnnData keys written by this model.
    embedding_dim : int
        Dimension of cell embeddings.
    hardware : HardwareRequirements
        GPU/CPU requirements.
    differentiator : str
        Unique feature that distinguishes this model.
    prefer_when : str
        When to specifically choose this model.
    checkpoint_url : str
        URL for downloading model weights.
    documentation_url : str
        URL for model documentation.
    paper_url : str
        URL for the model paper.
    license_notes : str
        License information.
    """
    # Identity
    name: str
    version: str
    skill_ready: SkillReadyStatus = SkillReadyStatus.REFERENCE

    # Capabilities
    tasks: list = field(default_factory=list)
    modalities: list = field(default_factory=list)
    species: list = field(default_factory=list)

    # Input requirements
    gene_id_scheme: GeneIDScheme = GeneIDScheme.SYMBOL
    requires_finetuning: bool = False
    zero_shot_embedding: bool = True
    zero_shot_annotation: bool = False

    # Output contract
    output_keys: OutputKeys = field(default_factory=OutputKeys)
    embedding_dim: int = 512

    # Hardware
    hardware: HardwareRequirements = field(default_factory=HardwareRequirements)

    # Routing hints
    differentiator: str = ""
    prefer_when: str = ""

    # Resources
    checkpoint_url: str = ""
    documentation_url: str = ""
    paper_url: str = ""
    license_notes: str = ""

    def supports_task(self, task: TaskType) -> bool:
        """Check if model supports a given task."""
        return task in self.tasks

    def supports_modality(self, modality: Modality) -> bool:
        """Check if model supports a given modality."""
        return modality in self.modalities

    def supports_species(self, species: str) -> bool:
        """Check if model supports a given species."""
        return species.lower() in [s.lower() for s in self.species]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "skill_ready": self.skill_ready.value,
            "tasks": [t.value for t in self.tasks],
            "modalities": [m.value for m in self.modalities],
            "species": self.species,
            "gene_id_scheme": self.gene_id_scheme.value,
            "zero_shot_embedding": self.zero_shot_embedding,
            "zero_shot_annotation": self.zero_shot_annotation,
            "requires_finetuning": self.requires_finetuning,
            "embedding_dim": self.embedding_dim,
            "output_keys": {
                "embedding": self.output_keys.embedding_key,
                "annotation": self.output_keys.annotation_key,
                "confidence": self.output_keys.confidence_key,
            },
            "hardware": {
                "gpu_required": self.hardware.gpu_required,
                "min_vram_gb": self.hardware.min_vram_gb,
                "cpu_fallback": self.hardware.cpu_fallback,
            },
            "differentiator": self.differentiator,
            "prefer_when": self.prefer_when,
            "documentation_url": self.documentation_url,
        }


# ===========================================================================
# Model Specifications — Skill-Ready (✅)
# ===========================================================================

SCGPT_SPEC = ModelSpec(
    name="scgpt",
    version="whole-human-2024",
    skill_ready=SkillReadyStatus.READY,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE],
    modalities=[Modality.RNA, Modality.ATAC, Modality.SPATIAL],
    species=["human", "mouse"],
    gene_id_scheme=GeneIDScheme.SYMBOL,
    requires_finetuning=True,
    zero_shot_embedding=True,
    zero_shot_annotation=False,
    output_keys=OutputKeys(
        embedding_key="X_scGPT",
        annotation_key="scgpt_pred",
        confidence_key="scgpt_pred_score",
        integration_key="X_scGPT_integrated",
    ),
    embedding_dim=512,
    hardware=HardwareRequirements(
        gpu_required=True, min_vram_gb=8, recommended_vram_gb=16,
        cpu_fallback=True, default_batch_size=64,
    ),
    differentiator="Multi-modal transformer (RNA+ATAC+Spatial), attention-based gene interaction modeling",
    prefer_when="User needs multi-modal analysis (RNA+ATAC or spatial), or explicit attention-based gene interaction maps",
    checkpoint_url="https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo",
    documentation_url="https://scgpt.readthedocs.io/",
    paper_url="https://www.nature.com/articles/s41592-024-02201-0",
    license_notes="Check upstream LICENSE; treat as restricted until verified",
)

GENEFORMER_SPEC = ModelSpec(
    name="geneformer",
    version="v2-106M",
    skill_ready=SkillReadyStatus.READY,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE],
    modalities=[Modality.RNA],
    species=["human"],
    gene_id_scheme=GeneIDScheme.ENSEMBL,
    requires_finetuning=True,
    zero_shot_embedding=True,
    zero_shot_annotation=False,
    output_keys=OutputKeys(
        embedding_key="X_geneformer",
        annotation_key="geneformer_pred",
        confidence_key="",
    ),
    embedding_dim=512,
    hardware=HardwareRequirements(
        gpu_required=False, min_vram_gb=4, recommended_vram_gb=16,
        cpu_fallback=True, default_batch_size=32,
    ),
    differentiator="Rank-value encoded transformer, Ensembl gene IDs, CPU-capable, network biology pretraining",
    prefer_when="User has Ensembl gene IDs, needs CPU-only inference, or wants gene-network-aware embeddings",
    checkpoint_url="https://huggingface.co/ctheodoris/Geneformer",
    documentation_url="https://geneformer.readthedocs.io/",
    paper_url="https://www.nature.com/articles/s41586-023-06139-9",
    license_notes="Apache 2.0 (code); check model weights terms",
)

UCE_SPEC = ModelSpec(
    name="uce",
    version="4-layer",
    skill_ready=SkillReadyStatus.READY,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE],
    modalities=[Modality.RNA],
    species=["human", "mouse", "zebrafish", "mouse_lemur", "macaque", "frog", "pig"],
    gene_id_scheme=GeneIDScheme.SYMBOL,
    requires_finetuning=False,
    zero_shot_embedding=True,
    zero_shot_annotation=False,
    output_keys=OutputKeys(
        embedding_key="X_uce",
        annotation_key="",
        confidence_key="",
    ),
    embedding_dim=1280,
    hardware=HardwareRequirements(
        gpu_required=True, min_vram_gb=16, recommended_vram_gb=16,
        cpu_fallback=False, default_batch_size=100,
    ),
    differentiator="Broadest species support (7 species), 1280-dim embeddings, universal cell embedding via protein structure",
    prefer_when="User has non-human/non-mouse species (zebrafish, frog, pig, macaque, lemur), or needs cross-species comparison",
    checkpoint_url="https://github.com/snap-stanford/UCE",
    documentation_url="https://github.com/snap-stanford/UCE",
    paper_url="https://www.nature.com/articles/s41592-024-02201-0",
    license_notes="MIT License",
)

# ===========================================================================
# Model Specifications — Partial Specs (⚠️)
# ===========================================================================

SCFOUNDATION_SPEC = ModelSpec(
    name="scfoundation",
    version="xTrimoGene",
    skill_ready=SkillReadyStatus.READY,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE],
    modalities=[Modality.RNA],
    species=["human"],
    gene_id_scheme=GeneIDScheme.CUSTOM,
    requires_finetuning=True,
    zero_shot_embedding=True,
    zero_shot_annotation=False,
    output_keys=OutputKeys(embedding_key="X_scfoundation", annotation_key="scfoundation_pred"),
    embedding_dim=512,
    hardware=HardwareRequirements(gpu_required=True, min_vram_gb=16, recommended_vram_gb=32, cpu_fallback=False),
    differentiator="Large-scale asymmetric transformer (xTrimoGene), custom 19264 gene vocabulary, pre-trained for perturbation/drug response",
    prefer_when="User needs perturbation prediction, drug response modeling, or works with the xTrimoGene gene vocabulary",
    checkpoint_url="https://github.com/biomap-research/scFoundation",
    paper_url="https://www.nature.com/articles/s41592-024-02305-7",
)

SCBERT_SPEC = ModelSpec(
    name="scbert",
    version="v1.0",
    skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE],
    modalities=[Modality.RNA],
    species=["human"],
    gene_id_scheme=GeneIDScheme.SYMBOL,
    requires_finetuning=True,
    zero_shot_embedding=True,
    output_keys=OutputKeys(embedding_key="X_scBERT", annotation_key="scbert_pred"),
    embedding_dim=200,
    hardware=HardwareRequirements(gpu_required=True, min_vram_gb=8, cpu_fallback=True),
    differentiator="Compact 200-dim embeddings, BERT-style masked gene pretraining, lightweight model",
    prefer_when="User needs compact 200-dim embeddings, BERT-style pretraining, or a lightweight model for constrained hardware",
    checkpoint_url="https://github.com/TencentAILabHealthcare/scBERT",
    paper_url="https://www.nature.com/articles/s42256-022-00534-z",
)

GENECOMPASS_SPEC = ModelSpec(
    name="genecompass",
    version="120M-cells",
    skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE],
    modalities=[Modality.RNA],
    species=["human", "mouse"],
    gene_id_scheme=GeneIDScheme.SYMBOL,
    requires_finetuning=True,
    zero_shot_embedding=True,
    output_keys=OutputKeys(embedding_key="X_genecompass", annotation_key="genecompass_pred"),
    embedding_dim=512,
    hardware=HardwareRequirements(gpu_required=True, min_vram_gb=16, recommended_vram_gb=32, cpu_fallback=False, default_batch_size=32),
    differentiator="Prior-knowledge-enhanced pretraining (gene regulatory networks + pathway info), 120M cell training corpus",
    prefer_when="User mentions prior knowledge, gene regulatory networks, pathway-informed embeddings, or mouse+human cross-species",
    checkpoint_url="https://github.com/xCompass-AI/GeneCompass",
)

CELLPLM_SPEC = ModelSpec(
    name="cellplm",
    version="v1.0",
    skill_ready=SkillReadyStatus.READY,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE],
    modalities=[Modality.RNA],
    species=["human"],
    gene_id_scheme=GeneIDScheme.SYMBOL,
    requires_finetuning=False,
    zero_shot_embedding=True,
    output_keys=OutputKeys(embedding_key="X_cellplm"),
    embedding_dim=512,
    hardware=HardwareRequirements(gpu_required=True, min_vram_gb=8, cpu_fallback=True, default_batch_size=128),
    differentiator="Cell-centric (not gene-centric) architecture, highest batch throughput (batch_size=128), fast inference",
    prefer_when="User needs fast inference, high throughput, million-cell scale processing, or cell-level (not gene-level) modeling",
    checkpoint_url="https://github.com/OmicsML/CellPLM",
)

NICHEFORMER_SPEC = ModelSpec(
    name="nicheformer",
    version="v1.0",
    skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE, TaskType.SPATIAL],
    modalities=[Modality.SPATIAL, Modality.RNA],
    species=["human", "mouse"],
    gene_id_scheme=GeneIDScheme.SYMBOL,
    zero_shot_embedding=True,
    output_keys=OutputKeys(embedding_key="X_nicheformer"),
    embedding_dim=512,
    hardware=HardwareRequirements(gpu_required=True, min_vram_gb=16, recommended_vram_gb=32, cpu_fallback=False, default_batch_size=32),
    differentiator="Niche-aware spatial transformer, jointly models spatial coordinates and gene expression",
    prefer_when="User has spatial transcriptomics data (Visium, MERFISH, Slide-seq) and wants niche-aware or spatial-context embeddings",
    checkpoint_url="https://github.com/theislab/nicheformer",
)

SCMULAN_SPEC = ModelSpec(
    name="scmulan",
    version="v1.0",
    skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE],
    modalities=[Modality.RNA, Modality.ATAC, Modality.PROTEIN, Modality.MULTIOMICS],
    species=["human"],
    gene_id_scheme=GeneIDScheme.SYMBOL,
    zero_shot_embedding=True,
    output_keys=OutputKeys(embedding_key="X_scmulan"),
    embedding_dim=512,
    hardware=HardwareRequirements(gpu_required=True, min_vram_gb=16, recommended_vram_gb=32, cpu_fallback=False, default_batch_size=32),
    differentiator="Native multi-omics joint modeling (RNA+ATAC+Protein simultaneously), designed for CITE-seq/10x Multiome",
    prefer_when="User has multi-omics data (CITE-seq, 10x Multiome, RNA+ATAC+Protein), or wants joint multi-modal embedding",
    checkpoint_url="https://github.com/SuperBianC/scMulan",
)

# ===========================================================================
# Model Specifications — Specialized & Emerging (2024-2025)
# ===========================================================================

TGPT_SPEC = ModelSpec(
    name="tgpt", version="v1.0", skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE], modalities=[Modality.RNA], species=["human"],
    zero_shot_embedding=True, output_keys=OutputKeys(embedding_key="X_tgpt"),
    hardware=HardwareRequirements(gpu_required=True, min_vram_gb=16, recommended_vram_gb=32, cpu_fallback=False, default_batch_size=32),
    differentiator="Autoregressive next-token prediction of gene expression values (GPT-style, not masked)",
    prefer_when="User wants autoregressive/generative modeling, next-token prediction of gene expression, or GPT-style generation",
    checkpoint_url="https://github.com/deeplearningplus/tGPT",
)

CELLFM_SPEC = ModelSpec(
    name="cellfm", version="v1.0", skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE], modalities=[Modality.RNA], species=["human"],
    zero_shot_embedding=True, output_keys=OutputKeys(embedding_key="X_cellfm"),
    hardware=HardwareRequirements(gpu_required=True, min_vram_gb=16, recommended_vram_gb=32, cpu_fallback=False, default_batch_size=32),
    differentiator="MLP architecture (not transformer), trained on ~126M cells (largest training corpus)",
    prefer_when="User explicitly wants MLP-based (not transformer) model, or wants the largest pretraining scale (~126M cells)",
    checkpoint_url="https://github.com/cellverse/CellFM",
)

SCCELLO_SPEC = ModelSpec(
    name="sccello", version="v1.0", skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE, TaskType.ANNOTATE], modalities=[Modality.RNA], species=["human"],
    zero_shot_embedding=True, zero_shot_annotation=True,
    output_keys=OutputKeys(embedding_key="X_sccello", annotation_key="sccello_pred"),
    hardware=HardwareRequirements(gpu_required=True, min_vram_gb=16, recommended_vram_gb=32, cpu_fallback=False, default_batch_size=32),
    differentiator="Cell ontology-aligned embeddings, zero-shot cell type annotation with hierarchical coherence",
    prefer_when="User wants zero-shot cell type annotation, ontology-consistent predictions, or hierarchical cell-type labeling",
    checkpoint_url="https://github.com/cellarium-ai/scCello",
)

SCPRINT_SPEC = ModelSpec(
    name="scprint", version="v1.0", skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE], modalities=[Modality.RNA], species=["human"],
    zero_shot_embedding=True, output_keys=OutputKeys(embedding_key="X_scprint"),
    hardware=HardwareRequirements(gpu_required=True, min_vram_gb=16, recommended_vram_gb=32, cpu_fallback=False, default_batch_size=32),
    differentiator="Protein-coding gene focus with built-in denoising, robust batch integration",
    prefer_when="User mentions denoising, protein-coding genes, ambient RNA removal, or wants built-in noise reduction",
    checkpoint_url="https://github.com/scprint/scPRINT",
)

AIDOCELL_SPEC = ModelSpec(
    name="aidocell", version="v1.0", skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE], modalities=[Modality.RNA], species=["human"],
    zero_shot_embedding=True, output_keys=OutputKeys(embedding_key="X_aidocell"),
    hardware=HardwareRequirements(gpu_required=True, min_vram_gb=16, recommended_vram_gb=32, cpu_fallback=False, default_batch_size=32),
    differentiator="Dense transformer optimized for unsupervised cell clustering without predefined labels",
    prefer_when="User wants unsupervised clustering, label-free cell grouping, or dense transformer embeddings for discovery",
    checkpoint_url="https://github.com/genbio-ai/AIDO",
)

PULSAR_SPEC = ModelSpec(
    name="pulsar", version="v1.0", skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE], modalities=[Modality.RNA], species=["human"],
    zero_shot_embedding=True, output_keys=OutputKeys(embedding_key="X_pulsar"),
    hardware=HardwareRequirements(gpu_required=True, min_vram_gb=16, recommended_vram_gb=32, cpu_fallback=False, default_batch_size=32),
    differentiator="Multi-scale multicellular biology modeling, captures cell-cell communication and tissue-level organization",
    prefer_when="User wants cell-cell communication analysis, tissue-level modeling, multicellular programs, or intercellular signaling",
    checkpoint_url="https://github.com/pulsar-ai/PULSAR",
)

ATACFORMER_SPEC = ModelSpec(
    name="atacformer", version="v1.0", skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE], modalities=[Modality.ATAC], species=["human"],
    gene_id_scheme=GeneIDScheme.CUSTOM, zero_shot_embedding=True,
    output_keys=OutputKeys(embedding_key="X_atacformer"),
    hardware=HardwareRequirements(gpu_required=True, min_vram_gb=16, recommended_vram_gb=32, cpu_fallback=False, default_batch_size=32),
    differentiator="ATAC-seq-native transformer, peak-based (not gene-based) input, chromatin accessibility specialist",
    prefer_when="User has ATAC-seq data, chromatin accessibility profiles, or peak-based (not gene expression) inputs",
    checkpoint_url="https://github.com/Atacformer/Atacformer",
)

SCPLANTLLM_SPEC = ModelSpec(
    name="scplantllm", version="v1.0", skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE], modalities=[Modality.RNA], species=["plant"],
    zero_shot_embedding=True, output_keys=OutputKeys(embedding_key="X_scplantllm"),
    hardware=HardwareRequirements(gpu_required=True, min_vram_gb=16, recommended_vram_gb=32, cpu_fallback=False, default_batch_size=32),
    differentiator="Plant-specific single-cell model, handles polyploidy and plant gene nomenclature",
    prefer_when="User has plant single-cell data (Arabidopsis, rice, maize, etc.) or mentions polyploidy",
    checkpoint_url="https://github.com/scPlantLLM/scPlantLLM",
)

LANGCELL_SPEC = ModelSpec(
    name="langcell", version="v1.0", skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE], modalities=[Modality.RNA], species=["human"],
    zero_shot_embedding=True, output_keys=OutputKeys(embedding_key="X_langcell"),
    hardware=HardwareRequirements(gpu_required=True, min_vram_gb=16, recommended_vram_gb=32, cpu_fallback=False, default_batch_size=32),
    differentiator="Two-tower (text + cell) architecture, aligns natural language descriptions with cell embeddings",
    prefer_when="User wants text-guided cell retrieval, natural language cell queries, or text-cell alignment",
    checkpoint_url="https://github.com/langcell/LangCell",
)

CELL2SENTENCE_SPEC = ModelSpec(
    name="cell2sentence", version="v1.0", skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED], modalities=[Modality.RNA], species=["human"],
    requires_finetuning=True, zero_shot_embedding=False,
    output_keys=OutputKeys(embedding_key="X_cell2sentence"),
    embedding_dim=768,
    hardware=HardwareRequirements(gpu_required=True, min_vram_gb=16, recommended_vram_gb=32, cpu_fallback=False, default_batch_size=16),
    differentiator="Converts cells to text sentences for LLM fine-tuning, 768-dim LLM embeddings",
    prefer_when="User wants to leverage general-purpose LLMs, convert cells to text, or use LLM fine-tuning workflows",
    checkpoint_url="https://github.com/vandijklab/cell2sentence",
)

GENEPT_SPEC = ModelSpec(
    name="genept", version="v1.0", skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED], modalities=[Modality.RNA], species=["human"],
    zero_shot_embedding=True,
    output_keys=OutputKeys(embedding_key="X_genept"),
    embedding_dim=1536,
    hardware=HardwareRequirements(gpu_required=False, min_vram_gb=0, recommended_vram_gb=0, cpu_fallback=True, default_batch_size=32),
    differentiator="API-based GPT-3.5 gene embeddings (1536-dim), no local GPU required, gene-level (not cell-level)",
    prefer_when="User wants gene-level embeddings (not cell-level), has no local GPU, or wants API-based OpenAI embeddings",
    checkpoint_url="https://github.com/yiqunchen/GenePT",
)

CHATCELL_SPEC = ModelSpec(
    name="chatcell", version="v1.0", skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.ANNOTATE], modalities=[Modality.RNA], species=["human"],
    zero_shot_embedding=True, zero_shot_annotation=True,
    output_keys=OutputKeys(embedding_key="X_chatcell", annotation_key="chatcell_pred"),
    hardware=HardwareRequirements(gpu_required=True, min_vram_gb=16, recommended_vram_gb=32, cpu_fallback=False, default_batch_size=16),
    differentiator="Conversational chat interface for single-cell analysis, zero-shot annotation via dialogue",
    prefer_when="User wants interactive chat-based cell analysis, conversational annotation, or dialogue-driven exploration",
    checkpoint_url="https://github.com/chatcell/CHATCELL",
)

TABULA_SPEC = ModelSpec(
    name="tabula", version="federated-v1", skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.ANNOTATE, TaskType.INTEGRATE, TaskType.PERTURB],
    modalities=[Modality.RNA], species=["human"],
    gene_id_scheme=GeneIDScheme.CUSTOM,
    requires_finetuning=True, zero_shot_embedding=True,
    output_keys=OutputKeys(
        embedding_key="X_tabula", annotation_key="tabula_pred",
        confidence_key="tabula_pred_score", integration_key="X_tabula_integrated",
    ),
    embedding_dim=192,
    hardware=HardwareRequirements(gpu_required=True, min_vram_gb=8, cpu_fallback=False),
    differentiator="Privacy-preserving federated learning + tabular transformer, 60697 gene vocabulary, quantile-binned expression, FlashAttention",
    prefer_when="User needs privacy-preserving analysis, federated-trained embeddings, or perturbation prediction with tabular modeling approach",
    checkpoint_url="https://github.com/aristoteleo/tabula",
)


# ===========================================================================
# Model Registry
# ===========================================================================

class ModelRegistry:
    """Registry of available single-cell foundation models.

    Provides capability-driven model discovery and selection.

    Models can be registered from three sources:

    - **Built-in**: Hardcoded default models shipped with omicverse.
    - **Entry points**: Third-party pip packages using the ``omicverse.fm``
      entry-point group.
    - **Local plugins**: Python files in ``~/.omicverse/plugins/fm/``.
    """

    def __init__(self):
        self._models: dict = {}
        self._adapters: dict = {}
        self._model_sources: dict = {}
        self._builtin_adapter_imports: dict = {}
        self._register_default_models()
        self._register_builtin_adapters()
        self._discover_plugins()

    # -----------------------------------------------------------------------
    # Built-in registration
    # -----------------------------------------------------------------------

    def _register_default_models(self):
        """Register all built-in model specs."""
        # Skill-ready (✅)
        self.register(SCGPT_SPEC)
        self.register(GENEFORMER_SPEC)
        self.register(UCE_SPEC)
        # Partial (⚠️) — Core
        self.register(SCFOUNDATION_SPEC)
        self.register(SCBERT_SPEC)
        self.register(GENECOMPASS_SPEC)
        self.register(CELLPLM_SPEC)
        self.register(NICHEFORMER_SPEC)
        self.register(SCMULAN_SPEC)
        # Partial (⚠️) — Specialized & Emerging
        self.register(TGPT_SPEC)
        self.register(CELLFM_SPEC)
        self.register(SCCELLO_SPEC)
        self.register(SCPRINT_SPEC)
        self.register(AIDOCELL_SPEC)
        self.register(PULSAR_SPEC)
        self.register(ATACFORMER_SPEC)
        self.register(SCPLANTLLM_SPEC)
        self.register(LANGCELL_SPEC)
        self.register(CELL2SENTENCE_SPEC)
        self.register(GENEPT_SPEC)
        self.register(CHATCELL_SPEC)
        self.register(TABULA_SPEC)

    def _register_builtin_adapters(self):
        """Register lazy-import paths for built-in adapters."""
        self._builtin_adapter_imports = {
            "scgpt": (".adapters._scgpt", "ScGPTAdapter"),
            "geneformer": (".adapters._geneformer", "GeneformerAdapter"),
            "uce": (".adapters._uce", "UCEAdapter"),
            "scfoundation": (".adapters._scfoundation", "ScFoundationAdapter"),
            "cellplm": (".adapters._cellplm", "CellPLMAdapter"),
            "scbert": (".adapters._experimental", "ScBERTAdapter"),
            "genecompass": (".adapters._experimental", "GeneCompassAdapter"),
            "nicheformer": (".adapters._experimental", "NicheformerAdapter"),
            "scmulan": (".adapters._experimental", "ScMulanAdapter"),
            "tgpt": (".adapters._experimental", "TGPTAdapter"),
            "cellfm": (".adapters._experimental", "CellFMAdapter"),
            "sccello": (".adapters._experimental", "ScCelloAdapter"),
            "scprint": (".adapters._experimental", "ScPrintAdapter"),
            "aidocell": (".adapters._experimental", "AiDocellAdapter"),
            "pulsar": (".adapters._experimental", "PulsarAdapter"),
            "atacformer": (".adapters._experimental", "AtacformerAdapter"),
            "scplantllm": (".adapters._experimental", "ScPlantLLMAdapter"),
            "langcell": (".adapters._experimental", "LangCellAdapter"),
            "cell2sentence": (".adapters._experimental", "Cell2SentenceAdapter"),
            "genept": (".adapters._experimental", "GenePTAdapter"),
            "chatcell": (".adapters._experimental", "ChatCellAdapter"),
            "tabula": (".adapters._experimental", "TabulaAdapter"),
        }

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def register(self, spec: ModelSpec, adapter_class=None, *, source: str = "builtin"):
        """Register a model specification and optionally its adapter class.

        Parameters
        ----------
        spec : ModelSpec
            Model specification.
        adapter_class : type, optional
            Adapter class (subclass of BaseAdapter).
        source : str
            Where this registration came from.
        """
        name = spec.name.lower()
        if name in self._models and source != "builtin":
            existing_source = self._model_sources.get(name, "builtin")
            if existing_source == "builtin":
                logger.warning(
                    "Plugin '%s' (source=%s) conflicts with built-in model '%s'; skipping.",
                    name, source, name,
                )
                return
            else:
                logger.warning(
                    "Plugin '%s' (source=%s) overrides previous plugin (source=%s).",
                    name, source, existing_source,
                )
        self._models[name] = spec
        if adapter_class is not None:
            self._adapters[name] = adapter_class
        self._model_sources[name] = source

    def get(self, name: str) -> Optional[ModelSpec]:
        """Get model spec by name."""
        return self._models.get(name.lower())

    def get_adapter_class(self, name: str):
        """Get the adapter class for a model (lazy-loaded)."""
        name = name.lower()
        if name in self._adapters:
            return self._adapters[name]

        if name in self._builtin_adapter_imports:
            rel_module, class_name = self._builtin_adapter_imports[name]
            try:
                module = importlib.import_module(rel_module, package="omicverse.fm")
                cls = getattr(module, class_name)
                self._adapters[name] = cls
                return cls
            except (ImportError, AttributeError) as exc:
                logger.warning("Failed to import built-in adapter for '%s': %s", name, exc)
                return None

        return None

    def list_models(self, skill_ready_only: bool = False) -> list:
        """List all registered models.

        Parameters
        ----------
        skill_ready_only : bool
            If True, only return models with ``SkillReadyStatus.READY``.

        Returns
        -------
        list[ModelSpec]
        """
        models = list(self._models.values())
        if skill_ready_only:
            models = [m for m in models if m.skill_ready == SkillReadyStatus.READY]
        return models

    def find_models(
        self,
        task: Optional[TaskType] = None,
        modality: Optional[Modality] = None,
        species: Optional[str] = None,
        gene_scheme: Optional[GeneIDScheme] = None,
        zero_shot: bool = False,
        max_vram_gb: Optional[int] = None,
    ) -> list:
        """Find models matching criteria.

        Parameters
        ----------
        task : TaskType, optional
            Required task.
        modality : Modality, optional
            Required modality.
        species : str, optional
            Required species support.
        gene_scheme : GeneIDScheme, optional
            Required gene ID scheme.
        zero_shot : bool
            If True, only return zero-shot capable models.
        max_vram_gb : int, optional
            Maximum VRAM constraint.

        Returns
        -------
        list[ModelSpec]
            Matching specs sorted by skill-ready status.
        """
        matches = []

        for spec in self._models.values():
            if task and not spec.supports_task(task):
                continue
            if modality and not spec.supports_modality(modality):
                continue
            if species and not spec.supports_species(species):
                continue
            if gene_scheme and spec.gene_id_scheme != gene_scheme:
                continue
            if zero_shot and task == TaskType.ANNOTATE and not spec.zero_shot_annotation:
                continue
            if zero_shot and task == TaskType.EMBED and not spec.zero_shot_embedding:
                continue
            if max_vram_gb and spec.hardware.min_vram_gb > max_vram_gb:
                continue
            matches.append(spec)

        def _sort_key(s):
            if s.skill_ready == SkillReadyStatus.READY:
                return 0
            elif s.skill_ready == SkillReadyStatus.PARTIAL:
                return 1
            return 2

        return sorted(matches, key=_sort_key)

    # -----------------------------------------------------------------------
    # Plugin discovery
    # -----------------------------------------------------------------------

    def _discover_plugins(self):
        """Discover and load plugins from entry points and local directory."""
        self._discover_entry_point_plugins()
        self._discover_local_plugins()

    def _discover_entry_point_plugins(self):
        """Load plugins registered via the ``omicverse.fm`` entry-point group."""
        try:
            all_eps = importlib.metadata.entry_points()
            if isinstance(all_eps, dict):
                eps = all_eps.get("omicverse.fm", [])
            else:
                eps = all_eps.select(group="omicverse.fm")
        except Exception as exc:
            logger.debug("Entry point discovery failed: %s", exc)
            return

        for ep in eps:
            try:
                register_fn = ep.load()
                result = register_fn()
                self._process_plugin_result(result, source=f"entrypoint:{ep.name}")
            except Exception as exc:
                logger.warning("Failed to load FM plugin entry point '%s': %s", ep.name, exc)

    def _discover_local_plugins(self):
        """Load plugins from ``~/.omicverse/plugins/fm/`` directory."""
        plugin_dir = Path.home() / ".omicverse" / "plugins" / "fm"
        if not plugin_dir.is_dir():
            return

        for py_file in sorted(plugin_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                spec_obj = importlib.util.spec_from_file_location(
                    f"omicverse_fm_local_plugin_{py_file.stem}", py_file,
                )
                if spec_obj is None or spec_obj.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec_obj)
                spec_obj.loader.exec_module(module)

                register_fn = getattr(module, "register", None)
                if register_fn is None:
                    logger.warning("Local plugin '%s' has no register() function; skipping.", py_file.name)
                    continue

                result = register_fn()
                self._process_plugin_result(result, source=f"local:{py_file.name}")
            except Exception as exc:
                logger.warning("Failed to load local FM plugin '%s': %s", py_file.name, exc)

    def _process_plugin_result(self, result, source: str):
        """Normalize and validate plugin register() output."""
        if isinstance(result, tuple) and len(result) == 2:
            registrations = [result]
        elif isinstance(result, list):
            registrations = result
        else:
            logger.warning("Plugin (source=%s) returned unexpected type %s; skipping.", source, type(result).__name__)
            return

        for spec, adapter_cls in registrations:
            self._validate_and_register(spec, adapter_cls, source=source)

    def _validate_and_register(self, spec, adapter_cls, source: str):
        """Validate a plugin's spec and adapter class, then register."""
        if not isinstance(spec, ModelSpec):
            logger.warning("Plugin (source=%s) provided non-ModelSpec object: %s; skipping.", source, type(spec).__name__)
            return

        from .adapters.base import BaseAdapter
        if not (isinstance(adapter_cls, type) and issubclass(adapter_cls, BaseAdapter)):
            logger.warning(
                "Plugin '%s' (source=%s) adapter class %s does not subclass BaseAdapter; skipping.",
                spec.name, source, adapter_cls,
            )
            return

        self.register(spec, adapter_cls, source=source)
        logger.info("Registered FM plugin '%s' from %s", spec.name, source)


# ===========================================================================
# Global singleton
# ===========================================================================

_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get the global model registry singleton."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry

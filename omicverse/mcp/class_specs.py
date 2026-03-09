"""
Wrapper specifications for class-backed MCP tools.

Each wrapped class has a ``ClassWrapperSpec`` that defines its fixed action
set, parameter schemas, and lifecycle rules.  No generic method dispatch is
ever allowed — every callable action is explicitly enumerated here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ActionSpec:
    """Describes one allowed action on a wrapped class."""

    name: str                                # "create", "annotate", etc.
    method: str                              # Python method name (e.g. "cell_anno")
    params: Dict[str, dict] = field(default_factory=dict)  # JSON Schema properties
    required_params: List[str] = field(default_factory=list)
    needs_instance: bool = True              # False only for "create"
    needs_adata: bool = False                # True if action uses adata_id
    returns: str = "json"                    # "instance_id", "json", "object_ref", "void"
    description: str = ""
    handler: Optional[Callable] = field(default=None, repr=False)  # custom handler


@dataclass
class ClassWrapperSpec:
    """Complete wrapper definition for a class-backed tool."""

    full_name: str                           # "omicverse.single._anno.pySCSA"
    tool_name: str                           # "ov.single.pyscsa"
    actions: Dict[str, ActionSpec] = field(default_factory=dict)
    description: str = ""
    rollout_phase: str = "P2"
    available: bool = True                   # False = spec defined but not usable
    runtime_requirements: Dict[str, List[str]] = field(default_factory=dict)
    # {"packages": ["SEACells"], "modules": ["omicverse.single._metacell"]}


# ---------------------------------------------------------------------------
# Spec registry
# ---------------------------------------------------------------------------

_CLASS_SPECS: Dict[str, ClassWrapperSpec] = {}


def register_spec(spec: ClassWrapperSpec) -> None:
    """Register a wrapper spec by full_name."""
    _CLASS_SPECS[spec.full_name] = spec


def get_spec(full_name: str) -> Optional[ClassWrapperSpec]:
    """Retrieve spec or None."""
    return _CLASS_SPECS.get(full_name)


def all_specs() -> Dict[str, ClassWrapperSpec]:
    """Return all registered specs."""
    return dict(_CLASS_SPECS)


def get_spec_by_tool_name(tool_name: str) -> Optional[ClassWrapperSpec]:
    """Look up a spec by its MCP tool name (e.g. ``"ov.bulk.pydeg"``)."""
    for spec in _CLASS_SPECS.values():
        if spec.tool_name == tool_name:
            return spec
    return None


# ---------------------------------------------------------------------------
# Helper: build combined JSON Schema for a spec
# ---------------------------------------------------------------------------


def build_class_tool_schema(spec: ClassWrapperSpec) -> dict:
    """Build a flat JSON Schema for a class tool with ``action`` enum.

    All action-specific params are merged as optional properties.
    The adapter validates per-action requirements at invocation time.
    """
    action_names = sorted(spec.actions.keys())

    properties: Dict[str, dict] = {
        "action": {
            "type": "string",
            "enum": action_names,
            "description": "Action to perform: " + ", ".join(action_names),
        },
        "instance_id": {
            "type": "string",
            "description": "Instance reference (required for all actions except create)",
        },
        "adata_id": {
            "type": "string",
            "description": "Dataset reference (required when action needs AnnData)",
        },
    }

    # Merge all action-specific params (union of all actions)
    for action in spec.actions.values():
        for param_name, param_schema in action.params.items():
            if param_name not in properties:
                properties[param_name] = param_schema

    return {
        "type": "object",
        "properties": properties,
        "required": ["action"],
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# Spec definitions for first-batch classes
# ---------------------------------------------------------------------------


def _register_all_specs():
    """Register wrapper specs for all Phase 2 classes."""
    _register_pydeg()
    _register_pyscsa()
    _register_metacell()
    _register_dct()
    _register_lda_topic()


def _register_pydeg():
    spec = ClassWrapperSpec(
        full_name="omicverse.bulk._Deseq2.pyDEG",
        tool_name="ov.bulk.pydeg",
        description="Differential expression analysis for bulk RNA-seq",
        runtime_requirements={"packages": [], "modules": ["omicverse.bulk._Deseq2"]},
        actions={
            "create": ActionSpec(
                name="create",
                method="__init__",
                needs_instance=False,
                needs_adata=True,
                returns="instance_id",
                description="Create pyDEG instance from AnnData count matrix",
                params={
                    "layer": {
                        "type": "string",
                        "description": "Layer to extract as count matrix (default: X)",
                        "default": "X",
                    },
                },
            ),
            "run": ActionSpec(
                name="run",
                method="deg_analysis",
                needs_instance=True,
                returns="json",
                description="Run DEG analysis: normalize, test, and classify genes",
                params={
                    "treatment_groups": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Treatment sample names",
                    },
                    "control_groups": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Control sample names",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["DEseq2", "ttest", "wilcox"],
                        "default": "DEseq2",
                        "description": "Statistical method",
                    },
                    "fc_threshold": {
                        "type": "number",
                        "default": 2.0,
                        "description": "Fold change threshold",
                    },
                    "pval_threshold": {
                        "type": "number",
                        "default": 0.05,
                        "description": "P-value threshold",
                    },
                },
                required_params=["treatment_groups", "control_groups"],
            ),
            "results": ActionSpec(
                name="results",
                method="results",
                needs_instance=True,
                returns="json",
                description="Get DEG results as table",
                params={
                    "n_genes": {
                        "type": "integer",
                        "description": "Max genes to return (0=all)",
                        "default": 0,
                    },
                },
            ),
            "destroy": ActionSpec(
                name="destroy",
                method="destroy",
                needs_instance=True,
                returns="void",
                description="Delete instance and free memory",
            ),
        },
    )
    register_spec(spec)


def _register_pyscsa():
    spec = ClassWrapperSpec(
        full_name="omicverse.single._anno.pySCSA",
        tool_name="ov.single.pyscsa",
        description="Automated cell type annotation using SCSA",
        runtime_requirements={"packages": [], "modules": ["omicverse.single._anno"]},
        actions={
            "create": ActionSpec(
                name="create",
                method="__init__",
                needs_instance=False,
                needs_adata=True,
                returns="instance_id",
                description="Create pySCSA annotator from AnnData",
                params={
                    "foldchange": {
                        "type": "number",
                        "default": 1.5,
                        "description": "Fold change threshold for markers",
                    },
                    "pvalue": {
                        "type": "number",
                        "default": 0.05,
                        "description": "P-value threshold",
                    },
                    "species": {
                        "type": "string",
                        "enum": ["Human", "Mouse"],
                        "default": "Human",
                        "description": "Species",
                    },
                    "tissue": {
                        "type": "string",
                        "default": "All",
                        "description": "Tissue type for annotation DB",
                    },
                    "target": {
                        "type": "string",
                        "enum": ["cellmarker", "cancersea", "bindea"],
                        "default": "cellmarker",
                        "description": "Annotation database",
                    },
                },
            ),
            "annotate": ActionSpec(
                name="annotate",
                method="cell_anno",
                needs_instance=True,
                needs_adata=True,
                returns="json",
                description="Run cell type annotation and write to adata.obs",
                params={
                    "clustertype": {
                        "type": "string",
                        "default": "leiden",
                        "description": "Clustering column in adata.obs",
                    },
                    "cluster": {
                        "type": "string",
                        "default": "all",
                        "description": "Cluster to annotate ('all' or specific)",
                    },
                    "key_added": {
                        "type": "string",
                        "default": "scsa_celltype",
                        "description": "Key to add in adata.obs",
                    },
                },
            ),
            "destroy": ActionSpec(
                name="destroy",
                method="destroy",
                needs_instance=True,
                returns="void",
                description="Delete instance and free memory",
            ),
        },
    )
    register_spec(spec)


def _register_metacell():
    spec = ClassWrapperSpec(
        full_name="omicverse.single._metacell.MetaCell",
        tool_name="ov.single.metacell",
        description="Construct metacells using SEACells algorithm",
        runtime_requirements={"packages": ["SEACells"], "modules": ["omicverse.single._metacell"]},
        actions={
            "create": ActionSpec(
                name="create",
                method="__init__",
                needs_instance=False,
                needs_adata=True,
                returns="instance_id",
                description="Create MetaCell model from AnnData",
                params={
                    "use_rep": {
                        "type": "string",
                        "default": "X_pca",
                        "description": "Representation to use (e.g. X_pca)",
                    },
                    "n_metacells": {
                        "type": "integer",
                        "description": "Number of metacells (default: n_cells/75)",
                    },
                    "use_gpu": {
                        "type": "boolean",
                        "default": False,
                        "description": "Use GPU acceleration",
                    },
                    "n_neighbors": {
                        "type": "integer",
                        "default": 15,
                        "description": "Number of neighbors for kernel",
                    },
                },
            ),
            "train": ActionSpec(
                name="train",
                method="train",
                needs_instance=True,
                returns="json",
                description="Initialize archetypes and train SEACells model",
                params={
                    "min_iter": {
                        "type": "integer",
                        "default": 10,
                        "description": "Minimum training iterations",
                    },
                    "max_iter": {
                        "type": "integer",
                        "default": 50,
                        "description": "Maximum training iterations",
                    },
                },
            ),
            "predict": ActionSpec(
                name="predict",
                method="predicted",
                needs_instance=True,
                returns="object_ref",
                description="Generate metacell AnnData (returns new adata_id)",
                params={
                    "method": {
                        "type": "string",
                        "enum": ["soft", "hard"],
                        "default": "soft",
                        "description": "Aggregation method",
                    },
                    "celltype_label": {
                        "type": "string",
                        "default": "celltype",
                        "description": "Column name for cell type labels",
                    },
                },
            ),
            "destroy": ActionSpec(
                name="destroy",
                method="destroy",
                needs_instance=True,
                returns="void",
                description="Delete instance and free memory",
            ),
        },
    )
    register_spec(spec)


def _register_dct():
    """DCT — deferred (requires pertpy)."""
    spec = ClassWrapperSpec(
        full_name="omicverse.single._deg_ct.DCT",
        tool_name="ov.single.dct",
        description="Differential cell type composition analysis",
        available=False,
        runtime_requirements={"packages": ["pertpy"], "modules": ["omicverse.single._deg_ct"]},
        actions={
            "create": ActionSpec(
                name="create", method="__init__",
                needs_instance=False, needs_adata=True,
                returns="instance_id",
                description="Create DCT model (requires pertpy)",
            ),
            "run": ActionSpec(
                name="run", method="run",
                needs_instance=True, returns="json",
                description="Run differential abundance analysis",
            ),
            "results": ActionSpec(
                name="results", method="get_results",
                needs_instance=True, returns="json",
                description="Get differential abundance results",
            ),
            "destroy": ActionSpec(
                name="destroy", method="destroy",
                needs_instance=True, returns="void",
            ),
        },
    )
    register_spec(spec)


def _register_lda_topic():
    """LDA_topic — deferred (requires mira + PyTorch)."""
    spec = ClassWrapperSpec(
        full_name="omicverse.utils._cluster.LDA_topic",
        tool_name="ov.utils.lda_topic",
        description="LDA topic modeling for single-cell data",
        available=False,
        runtime_requirements={"packages": ["mira"], "modules": ["omicverse.utils._cluster"]},
        actions={
            "create": ActionSpec(
                name="create", method="__init__",
                needs_instance=False, needs_adata=True,
                returns="instance_id",
                description="Create LDA topic model (requires mira + PyTorch)",
            ),
            "fit": ActionSpec(
                name="fit", method="predicted",
                needs_instance=True, returns="json",
                description="Fit topic model and predict clusters",
            ),
            "destroy": ActionSpec(
                name="destroy", method="destroy",
                needs_instance=True, returns="void",
            ),
        },
    )
    register_spec(spec)


# Auto-register on import
_register_all_specs()

"""
Public API for ``ov.fm`` — Foundation Model operations.
========================================================

Provides user-facing functions for model discovery, data profiling,
model selection, validation, execution, and result interpretation.

Examples
--------
>>> import omicverse as ov
>>> ov.fm.list_models(task="embed")
>>> ov.fm.profile_data("my_data.h5ad")
>>> ov.fm.select_model("my_data.h5ad", task="embed")
>>> ov.fm.run(task="embed", model_name="scgpt", adata_path="my_data.h5ad")
"""

import json
import os
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

from .registry import (
    GeneIDScheme,
    ModelRegistry,
    ModelSpec,
    Modality,
    SkillReadyStatus,
    TaskType,
    get_registry,
)


# ===========================================================================
# Model Discovery
# ===========================================================================

def list_models(
    task: Optional[str] = None,
    skill_ready_only: bool = False,
) -> dict:
    """List available single-cell foundation models.

    Parameters
    ----------
    task : str, optional
        Filter by task type (``'embed'``, ``'annotate'``, ``'integrate'``,
        ``'perturb'``, ``'spatial'``, ``'drug_response'``).
    skill_ready_only : bool
        If True, only show models with complete adapter specs.

    Returns
    -------
    dict
        ``{"count": int, "models": [...]}``.
    """
    registry = get_registry()
    task_filter = TaskType(task) if task else None
    models = registry.list_models(skill_ready_only=skill_ready_only)

    if task_filter:
        models = [m for m in models if m.supports_task(task_filter)]

    result = {"count": len(models), "models": []}
    for spec in models:
        result["models"].append({
            "name": spec.name,
            "version": spec.version,
            "status": spec.skill_ready.value,
            "tasks": [t.value for t in spec.tasks],
            "modalities": [m.value for m in spec.modalities],
            "species": spec.species,
            "gene_id_scheme": spec.gene_id_scheme.value,
            "zero_shot": spec.zero_shot_embedding,
            "gpu_required": spec.hardware.gpu_required,
            "min_vram_gb": spec.hardware.min_vram_gb,
        })
    return result


def describe_model(model_name: str) -> dict:
    """Get detailed specification for a foundation model.

    Parameters
    ----------
    model_name : str
        Model identifier (e.g. ``'scgpt'``, ``'geneformer'``, ``'uce'``).

    Returns
    -------
    dict
        Full model spec including I/O contract, requirements, and resources.
    """
    registry = get_registry()
    spec = registry.get(model_name)
    if not spec:
        available = [m.name for m in registry.list_models()]
        return {"error": f"Model '{model_name}' not found", "available_models": available}

    return {
        "model": spec.to_dict(),
        "input_contract": {
            "gene_id_scheme": spec.gene_id_scheme.value,
            "gene_id_notes": _get_gene_id_notes(spec),
            "required_obs": _get_required_obs(spec),
            "preprocessing": _get_preprocessing_notes(spec),
        },
        "output_contract": {
            "embedding_key": f"obsm['{spec.output_keys.embedding_key}']" if spec.output_keys.embedding_key else None,
            "annotation_key": f"obs['{spec.output_keys.annotation_key}']" if spec.output_keys.annotation_key else None,
            "confidence_key": f"obs['{spec.output_keys.confidence_key}']" if spec.output_keys.confidence_key else None,
            "embedding_dim": spec.embedding_dim,
        },
        "resources": {
            "checkpoint": spec.checkpoint_url,
            "documentation": spec.documentation_url,
            "paper": spec.paper_url,
            "license": spec.license_notes,
        },
    }


# ===========================================================================
# Data Profiling
# ===========================================================================

def profile_data(adata_path: str) -> dict:
    """Profile an AnnData file to detect species, gene scheme, and modality.

    Parameters
    ----------
    adata_path : str
        Path to ``.h5ad`` file.

    Returns
    -------
    dict
        Data profile including species, gene_scheme, modality, n_cells, n_genes,
        and model compatibility notes.
    """
    return _profile_data_impl(adata_path)


def _profile_data_impl(adata_path: str) -> dict:
    path = Path(adata_path)
    if not path.exists():
        return {"error": f"File not found: {adata_path}"}
    if path.suffix != ".h5ad":
        return {"error": f"Expected .h5ad file, got: {path.suffix}"}

    try:
        import scanpy as sc
        adata = sc.read_h5ad(adata_path, backed="r")
    except ImportError:
        return {"error": "scanpy not installed. Install with: pip install scanpy"}
    except Exception as exc:
        return {"error": f"Failed to read AnnData: {exc}"}

    profile = {
        "file": str(path.name),
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "species": _detect_species(adata),
        "gene_scheme": _detect_gene_scheme(adata),
        "modality": _detect_modality(adata),
        "has_raw": adata.raw is not None,
        "layers": list(adata.layers.keys()) if adata.layers else [],
        "obs_columns": list(adata.obs.columns)[:20],
        "obsm_keys": list(adata.obsm.keys()),
    }

    batch_cols = [c for c in adata.obs.columns if "batch" in c.lower()]
    profile["batch_columns"] = batch_cols

    celltype_cols = [
        c for c in adata.obs.columns
        if any(x in c.lower() for x in ["celltype", "cell_type", "annotation"])
    ]
    profile["celltype_columns"] = celltype_cols

    profile["model_compatibility"] = _check_model_compatibility(profile)

    adata.file.close()
    return profile


# ===========================================================================
# Model Selection
# ===========================================================================

def select_model(
    adata_path: str,
    task: str,
    prefer_zero_shot: bool = True,
    max_vram_gb: Optional[int] = None,
) -> dict:
    """Select the best foundation model for a task and dataset.

    Parameters
    ----------
    adata_path : str
        Path to ``.h5ad`` file.
    task : str
        Task type (``'embed'``, ``'annotate'``, ``'integrate'``).
    prefer_zero_shot : bool
        Prefer models that don't require fine-tuning.
    max_vram_gb : int, optional
        Maximum VRAM constraint.

    Returns
    -------
    dict
        Recommended model with rationale and fallback options.
    """
    registry = get_registry()
    profile = _profile_data_impl(adata_path)
    if "error" in profile:
        return profile

    task_type = TaskType(task)
    species = profile["species"].replace(" (inferred)", "")
    if species == "unknown":
        species = None

    gene_scheme = profile["gene_scheme"]
    gene_scheme_enum = None
    if gene_scheme == "ensembl":
        gene_scheme_enum = GeneIDScheme.ENSEMBL
    elif gene_scheme == "symbol":
        gene_scheme_enum = GeneIDScheme.SYMBOL

    models = registry.find_models(
        task=task_type, species=species, gene_scheme=gene_scheme_enum,
        max_vram_gb=max_vram_gb,
    )

    if not models:
        models = registry.find_models(task=task_type, species=species, max_vram_gb=max_vram_gb)

    if not models:
        return {
            "error": "No compatible models found",
            "data_profile": profile,
            "suggestion": "Try relaxing constraints or check data format",
        }

    scored = [(spec, _score_model(spec, profile, task_type, prefer_zero_shot)) for spec in models]
    scored.sort(key=lambda x: x[1], reverse=True)

    recommended = scored[0][0]
    fallbacks = [m[0] for m in scored[1:3]]

    return {
        "recommended": {
            "name": recommended.name,
            "version": recommended.version,
            "rationale": _generate_rationale(recommended, profile, task_type),
        },
        "fallbacks": [
            {"name": f.name, "rationale": _generate_rationale(f, profile, task_type)}
            for f in fallbacks
        ],
        "preprocessing_notes": _get_preprocessing_notes(recommended),
        "data_profile": {
            "species": profile["species"],
            "gene_scheme": profile["gene_scheme"],
            "n_cells": profile["n_cells"],
            "n_genes": profile["n_genes"],
        },
    }


# ===========================================================================
# Validation
# ===========================================================================

def preprocess_validate(
    adata_path: str,
    model_name: str,
    task: str,
) -> dict:
    """Validate data compatibility with a model and suggest preprocessing.

    Parameters
    ----------
    adata_path : str
        Path to ``.h5ad`` file.
    model_name : str
        Target model name.
    task : str
        Task type.

    Returns
    -------
    dict
        Validation result with status, diagnostics, and auto-fix suggestions.
    """
    registry = get_registry()
    spec = registry.get(model_name)
    if not spec:
        return {"error": f"Model '{model_name}' not found"}

    profile = _profile_data_impl(adata_path)
    if "error" in profile:
        return profile

    task_type = TaskType(task)
    diagnostics = []
    auto_fixes = []
    status = "ready"

    if not spec.supports_task(task_type):
        diagnostics.append({"severity": "error", "message": f"Model '{model_name}' does not support task '{task}'"})
        status = "incompatible"

    gene_scheme = profile["gene_scheme"]
    if gene_scheme == "ensembl" and spec.gene_id_scheme == GeneIDScheme.SYMBOL:
        diagnostics.append({"severity": "warning", "message": "Data has Ensembl IDs but model requires gene symbols"})
        auto_fixes.append({"action": "convert_gene_ids", "from": "ensembl", "to": "symbol"})
        status = "needs_preprocessing"
    elif gene_scheme == "symbol" and spec.gene_id_scheme == GeneIDScheme.ENSEMBL:
        diagnostics.append({"severity": "warning", "message": "Data has gene symbols but model requires Ensembl IDs"})
        auto_fixes.append({"action": "convert_gene_ids", "from": "symbol", "to": "ensembl"})
        status = "needs_preprocessing"

    species = profile["species"].replace(" (inferred)", "")
    if species != "unknown" and not spec.supports_species(species):
        diagnostics.append({"severity": "error", "message": f"Species '{species}' not supported by {model_name}"})
        status = "incompatible"

    if task_type == TaskType.INTEGRATE and not profile.get("batch_columns"):
        diagnostics.append({"severity": "warning", "message": "No batch column found for integration task"})
        auto_fixes.append({"action": "add_batch_column", "code": "adata.obs['batch_id'] = 'batch_1'"})

    if task_type == TaskType.ANNOTATE and spec.requires_finetuning and not profile.get("celltype_columns"):
        diagnostics.append({"severity": "info", "message": "No celltype column found. Fine-tuning requires labeled data."})

    if not profile.get("has_raw") and "counts" not in profile.get("layers", []):
        diagnostics.append({
            "severity": "info",
            "message": "No raw counts found. Some models require unnormalized counts in .raw or layers['counts'].",
        })

    return {
        "status": status,
        "model": model_name,
        "task": task,
        "diagnostics": diagnostics,
        "auto_fixes": auto_fixes,
        "data_summary": {
            "n_cells": profile["n_cells"],
            "n_genes": profile["n_genes"],
            "species": profile["species"],
            "gene_scheme": profile["gene_scheme"],
        },
    }


# ===========================================================================
# Execution
# ===========================================================================

def run(
    task: str,
    model_name: str,
    adata_path: str,
    output_path: Optional[str] = None,
    batch_key: Optional[str] = None,
    label_key: Optional[str] = None,
    device: str = "auto",
    batch_size: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
) -> dict:
    """Execute a foundation model task.

    Parameters
    ----------
    task : str
        Task type (``'embed'``, ``'annotate'``, ``'integrate'``).
    model_name : str
        Model to use.
    adata_path : str
        Path to input ``.h5ad`` file.
    output_path : str, optional
        Path for output ``.h5ad`` (default: overwrites input).
    batch_key : str, optional
        Column in ``.obs`` for batch information.
    label_key : str, optional
        Column in ``.obs`` for cell type labels.
    device : str
        Device to use (``'auto'``, ``'cuda'``, ``'cpu'``).
    batch_size : int, optional
        Batch size for inference.
    checkpoint_dir : str, optional
        Path to model checkpoints.

    Returns
    -------
    dict
        Execution result with output path, keys, and statistics.
    """
    registry = get_registry()
    spec = registry.get(model_name)
    if not spec:
        return {"error": f"Model '{model_name}' not found"}

    validation = preprocess_validate(adata_path, model_name, task)
    if validation.get("status") == "incompatible":
        return {"error": "Data incompatible with model", "validation": validation}

    task_type = TaskType(task)
    output_path = output_path or adata_path

    # Try conda subprocess first
    conda_result = _maybe_run_in_conda_env(
        model_name=model_name, task=task, adata_path=adata_path,
        output_path=output_path, batch_key=batch_key, label_key=label_key,
        device=device, batch_size=batch_size or spec.hardware.default_batch_size,
        checkpoint_dir=checkpoint_dir,
    )
    if conda_result is not None:
        return conda_result

    # Fall back to in-process adapter
    adapter = _get_model_adapter(model_name, checkpoint_dir)
    if adapter is None:
        return {
            "error": f"No adapter implemented for model '{model_name}'",
            "status": "not_implemented",
            "suggestion": "Install model dependencies or use a supported model",
            "model_spec": spec.to_dict(),
        }

    try:
        result = adapter.run(
            task=task_type,
            adata_path=adata_path,
            output_path=output_path,
            batch_key=batch_key,
            label_key=label_key,
            device=device,
            batch_size=batch_size or spec.hardware.default_batch_size,
        )
        return result
    except Exception as exc:
        return {"error": f"Execution failed: {exc}", "model": model_name, "task": task}


# ===========================================================================
# Interpretation
# ===========================================================================

def interpret_results(
    adata_path: str,
    task: str,
    output_dir: Optional[str] = None,
    generate_umap: bool = True,
    color_by: Optional[list] = None,
) -> dict:
    """Generate QA metrics and visualizations for model results.

    Parameters
    ----------
    adata_path : str
        Path to ``.h5ad`` file with model outputs.
    task : str
        Task that was executed.
    output_dir : str, optional
        Directory for visualization outputs.
    generate_umap : bool
        Whether to generate UMAP visualizations.
    color_by : list[str], optional
        List of obs columns to color UMAP by.

    Returns
    -------
    dict
        QA metrics, visualization paths, and warnings.
    """
    try:
        import scanpy as sc
        adata = sc.read_h5ad(adata_path)
    except ImportError:
        return {"error": "scanpy not installed"}
    except Exception as exc:
        return {"error": f"Failed to read AnnData: {exc}"}

    registry = get_registry()
    output_dir = output_dir or os.path.dirname(adata_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    metrics: dict = {}
    visualizations: list = []
    warnings_list: list = []

    embedding_keys = [k for k in adata.obsm.keys() if k.startswith("X_")]
    model_names = [spec.name for spec in registry.list_models()]
    fm_keys = [k for k in embedding_keys if any(m in k.lower() for m in model_names)]

    if not fm_keys:
        warnings_list.append("No foundation model embeddings found in obsm")

    if fm_keys:
        metrics["embeddings"] = {}
        for key in fm_keys:
            emb = adata.obsm[key]
            metrics["embeddings"][key] = {"dim": emb.shape[1], "n_cells": emb.shape[0]}

            sil_score = _compute_silhouette(adata, key)
            if sil_score is not None:
                metrics["embeddings"][key]["silhouette"] = round(sil_score, 4)

    annotation_cols = [
        c for c in adata.obs.columns
        if any(m in c.lower() for m in ["pred", "annotation"])
    ]
    if annotation_cols:
        metrics["annotations"] = {"columns": annotation_cols}

    if "scfm" in adata.uns:
        metrics["provenance"] = adata.uns["scfm"]

    metrics["n_cells"] = adata.n_obs
    metrics["n_genes"] = adata.n_vars

    return {
        "metrics": metrics,
        "visualizations": visualizations,
        "warnings": warnings_list,
        "embedding_keys": fm_keys,
        "annotation_columns": annotation_cols,
    }


# ===========================================================================
# Internal Helpers
# ===========================================================================

def _detect_species(adata) -> str:
    if "species" in adata.uns:
        return adata.uns["species"]

    gene_names = adata.var_names[:100].tolist()
    human_markers = {"ACTB", "GAPDH", "CD4", "CD8A", "MS4A1", "CD14"}
    mouse_markers = {"Actb", "Gapdh", "Cd4", "Cd8a", "Ms4a1", "Cd14"}
    gene_set = set(gene_names)
    human_hits = len(human_markers & gene_set)
    mouse_hits = len(mouse_markers & gene_set)

    if human_hits > mouse_hits:
        return "human"
    elif mouse_hits > human_hits:
        return "mouse"

    uppercase_count = sum(1 for g in gene_names if g.isupper() or (len(g) > 1 and g[1:].isupper()))
    mixed_count = sum(1 for g in gene_names if len(g) > 0 and g[0].isupper() and g[1:].islower())

    if uppercase_count > mixed_count:
        return "human (inferred)"
    elif mixed_count > uppercase_count:
        return "mouse (inferred)"
    return "unknown"


def _detect_gene_scheme(adata) -> str:
    gene_names = adata.var_names[:50].tolist()
    ensembl_count = sum(1 for g in gene_names if g.startswith(("ENSG", "ENSMUSG", "ENS")))
    if ensembl_count > len(gene_names) * 0.5:
        return "ensembl"

    symbol_count = sum(1 for g in gene_names if g.isalnum() or "-" in g)
    if symbol_count > len(gene_names) * 0.8:
        return "symbol"

    return "unknown"


def _detect_modality(adata) -> str:
    if "modality" in adata.uns:
        return adata.uns["modality"]
    if any("peak" in k.lower() or "atac" in k.lower() for k in adata.var.columns):
        return "ATAC"
    if "spatial" in adata.obsm or "X_spatial" in adata.obsm:
        return "Spatial"
    return "RNA"


def _check_model_compatibility(profile: dict) -> dict:
    registry = get_registry()
    compatibility = {}

    for spec in registry.list_models():
        issues = []
        recommendations = []

        species = profile["species"].replace(" (inferred)", "")
        if species != "unknown" and not spec.supports_species(species):
            issues.append(f"Species '{species}' not supported")

        gene_scheme = profile["gene_scheme"]
        if gene_scheme != "unknown":
            if spec.gene_id_scheme == GeneIDScheme.ENSEMBL and gene_scheme != "ensembl":
                issues.append("Model requires Ensembl IDs")
                recommendations.append("Convert gene symbols to Ensembl IDs")
            elif spec.gene_id_scheme == GeneIDScheme.SYMBOL and gene_scheme == "ensembl":
                issues.append("Model requires gene symbols")
                recommendations.append("Convert Ensembl IDs to gene symbols")

        modality = profile["modality"]
        if modality and not spec.supports_modality(Modality(modality)):
            issues.append(f"Modality '{modality}' not supported")

        compatibility[spec.name] = {
            "compatible": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations,
        }

    return compatibility


def _score_model(spec: ModelSpec, profile: dict, task: TaskType, prefer_zero_shot: bool) -> float:
    score = 0.0
    if spec.skill_ready == SkillReadyStatus.READY:
        score += 100
    elif spec.skill_ready == SkillReadyStatus.PARTIAL:
        score += 50

    if prefer_zero_shot:
        if task == TaskType.EMBED and spec.zero_shot_embedding:
            score += 30
        elif task == TaskType.ANNOTATE and spec.zero_shot_annotation:
            score += 30

    gene_scheme = profile["gene_scheme"]
    if gene_scheme == "ensembl" and spec.gene_id_scheme == GeneIDScheme.ENSEMBL:
        score += 20
    elif gene_scheme == "symbol" and spec.gene_id_scheme == GeneIDScheme.SYMBOL:
        score += 20

    if spec.hardware.cpu_fallback:
        score += 10
    if spec.hardware.min_vram_gb <= 8:
        score += 5

    return score


def _generate_rationale(spec: ModelSpec, profile: dict, task: TaskType) -> str:
    reasons = []
    if spec.skill_ready == SkillReadyStatus.READY:
        reasons.append("fully implemented adapter")
    gene_scheme = profile["gene_scheme"]
    if gene_scheme == "ensembl" and spec.gene_id_scheme == GeneIDScheme.ENSEMBL:
        reasons.append("matches Ensembl gene IDs")
    elif gene_scheme == "symbol" and spec.gene_id_scheme == GeneIDScheme.SYMBOL:
        reasons.append("matches gene symbols")
    species = profile["species"].replace(" (inferred)", "")
    if species in spec.species:
        reasons.append(f"supports {species}")
    if task == TaskType.EMBED and spec.zero_shot_embedding:
        reasons.append("zero-shot embedding (no fine-tuning needed)")
    if spec.hardware.cpu_fallback:
        reasons.append("CPU fallback available")
    return "; ".join(reasons) if reasons else "general purpose model"


def _get_gene_id_notes(spec: ModelSpec) -> str:
    notes = {
        "scgpt": "Uses HGNC gene symbols. Convert Ensembl IDs to symbols if needed.",
        "geneformer": "Requires Ensembl IDs (ENSG...). Strip version suffix (.15) if present.",
        "uce": "Uses gene symbols. Not compatible with Ensembl IDs directly.",
        "scfoundation": "Uses custom 19,264 gene set. Map genes to model vocabulary.",
    }
    return notes.get(spec.name, "Check model documentation for gene ID requirements.")


def _get_required_obs(spec: ModelSpec) -> list:
    required = []
    if TaskType.INTEGRATE in spec.tasks:
        required.append("batch_id (for integration)")
    if TaskType.ANNOTATE in spec.tasks and spec.requires_finetuning:
        required.append("celltype (for annotation training)")
    return required


def _get_preprocessing_notes(spec: ModelSpec) -> str:
    notes = {
        "scgpt": "Normalize to 1e4 via sc.pp.normalize_total, then bin into 51 expression bins.",
        "geneformer": "Rank-value encoding. Use geneformer.preprocess() for proper tokenization.",
        "uce": "Standard log-normalization. Model handles tokenization internally.",
        "scfoundation": "Match genes to model vocabulary. Follow xTrimoGene preprocessing.",
    }
    return notes.get(spec.name, "See model documentation for preprocessing requirements.")


def _compute_silhouette(adata, embedding_key: str):
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        return None

    label_cols = [
        c for c in adata.obs.columns
        if any(x in c.lower() for x in ["celltype", "cell_type", "cluster", "leiden", "louvain"])
    ]
    if not label_cols:
        return None

    labels = adata.obs[label_cols[0]]
    if labels.nunique() < 2:
        return None

    try:
        import numpy as np
        max_cells = 10000
        if adata.n_obs > max_cells:
            idx = np.random.choice(adata.n_obs, max_cells, replace=False)
            embeddings = adata.obsm[embedding_key][idx]
            sample_labels = labels.iloc[idx]
        else:
            embeddings = adata.obsm[embedding_key]
            sample_labels = labels
        return silhouette_score(embeddings, sample_labels, metric="euclidean")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Conda subprocess execution
# ---------------------------------------------------------------------------

_conda_env_cache: dict = {}


def _conda_env_name(model_name: str) -> str:
    return f"scfm-{model_name.lower()}"


def _conda_env_available(model_name: str) -> bool:
    model_name = model_name.lower()
    if os.environ.get("OV_FM_DISABLE_CONDA_SUBPROCESS"):
        return False
    if model_name in _conda_env_cache:
        return _conda_env_cache[model_name]
    if shutil.which("conda") is None:
        _conda_env_cache[model_name] = False
        return False

    env_name = _conda_env_name(model_name)
    try:
        probe = subprocess.run(
            ["conda", "run", "-n", env_name, "python", "-c", "print('ok')"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        ok = probe.returncode == 0
    except Exception:
        ok = False

    _conda_env_cache[model_name] = ok
    return ok


def _maybe_run_in_conda_env(
    model_name: str,
    task: str,
    adata_path: str,
    output_path: str,
    batch_key: Optional[str],
    label_key: Optional[str],
    device: str,
    batch_size: int,
    checkpoint_dir: Optional[str] = None,
    timeout: int = 7200,
) -> Optional[dict]:
    model_name = model_name.lower()
    if not _conda_env_available(model_name):
        return None

    runner_path = Path(__file__).resolve().with_name("_conda_runner.py")
    if not runner_path.exists():
        return None

    payload = {
        "model_name": model_name,
        "task": task,
        "adata_path": adata_path,
        "output_path": output_path,
        "batch_key": batch_key,
        "label_key": label_key,
        "device": device,
        "batch_size": batch_size,
        "checkpoint_dir": checkpoint_dir,
    }

    env_name = _conda_env_name(model_name)
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("OV_FM_BACKEND", "conda-subprocess")

    with tempfile.TemporaryDirectory(prefix="ov_fm_conda_") as tmpdir:
        payload_path = Path(tmpdir) / "payload.json"
        payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        cmd = ["conda", "run", "-n", env_name, "python", str(runner_path), "--payload", str(payload_path)]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False, env=env)
        except subprocess.TimeoutExpired:
            return {"error": "Subprocess timed out", "model": model_name, "task": task, "timeout": timeout}
        except FileNotFoundError as exc:
            return {"error": f"Subprocess failed: {exc}", "model": model_name, "task": task}

    stdout_lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
    parsed = None
    for line in reversed(stdout_lines):
        try:
            parsed = json.loads(line)
            break
        except Exception:
            continue

    if parsed is None:
        return {
            "error": f"Subprocess failed (exit {proc.returncode}); no JSON result found",
            "model": model_name, "task": task,
            "stderr_tail": (proc.stderr or "")[-4000:],
            "stdout_tail": (proc.stdout or "")[-4000:],
        }

    if proc.returncode != 0:
        parsed.setdefault("error", f"Subprocess failed (exit {proc.returncode})")
        parsed.setdefault("stderr_tail", (proc.stderr or "")[-4000:])

    return parsed


def _get_model_adapter(model_name: str, checkpoint_dir: Optional[str] = None):
    registry = get_registry()
    adapter_cls = registry.get_adapter_class(model_name)
    if adapter_cls is None:
        return None
    try:
        return adapter_cls(checkpoint_dir)
    except Exception:
        return None

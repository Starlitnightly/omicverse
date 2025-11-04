# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OmicVerse** is a comprehensive Python framework for multi-omics data analysis, supporting bulk RNA-seq, single-cell RNA-seq, and spatial transcriptomics. The framework integrates 60+ published bioinformatics tools and features a Smart Agent system for natural language-driven analysis.

**Key Publication**: [Nature Communications 2024](https://doi.org/10.1038/s41467-024-50194-3)

## Common Commands

### Installation & Setup
```bash
# Install from PyPI
pip install omicverse

# Install with all optional dependencies
pip install omicverse[full]

# Install from source (development)
python setup.py install
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/single/test_single.py
pytest tests/bulk/test_bulk.py

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest --cov=omicverse tests/
```

### Documentation
```bash
# Build and serve documentation locally
cd omicverse_guide
mkdocs serve  # Preview at http://localhost:8000
mkdocs build  # Build static site
```

## Architecture & Code Organization

### Core Module Structure

The codebase follows a modular architecture with clear separation of concerns:

```
omicverse/
├── pp/           # Preprocessing (QC, normalization, HVG selection, PCA, UMAP)
├── single/       # Single-cell analysis (clustering, annotation, trajectory, DEG)
├── bulk/         # Bulk RNA-seq (DESeq2, WGCNA, GSEA, PPI networks)
├── space/        # Spatial transcriptomics analysis
├── bulk2single/  # Deconvolution and bulk-to-single integration
├── utils/        # Core utilities (I/O, plotting, agent system, registry)
├── pl/           # Visualization and plotting functions
├── llm/          # LLM integration (CellPLM, Geneformer, scGPT, etc.)
├── external/     # 45+ integrated external packages (avoid installation conflicts)
└── agent/        # Smart agent system for natural language processing
```

### Data Flow Pattern

All analysis follows the AnnData-centric (single-cell) or DataFrame-centric (bulk) pattern:

```python
# Typical single-cell workflow
adata = ov.read('data.h5ad')                          # Load data
adata = ov.pp.qc(adata, tresh={...})                  # Quality control
adata = ov.pp.preprocess(adata, mode='shiftlog|pearson', n_HVGs=2000)
ov.pp.scale(adata)                                    # Scale data
ov.pp.pca(adata, n_pcs=50)                           # PCA
ov.pp.umap(adata)                                     # UMAP
ov.single.leiden(adata, resolution=1.0)              # Clustering
ov.pl.embedding(adata, basis='X_umap', color=['leiden'])  # Visualization
```

### Smart Agent System (Pantheon Framework)

The framework includes a natural language processing system for function discovery and execution:

**Key Components**:
- `omicverse/utils/registry.py` - Function registry with semantic search
- `omicverse/utils/smart_agent.py` - Agent implementation
- `omicverse/__init__.py` - Exports `Agent`, `list_supported_models`

**Usage**:
```python
import omicverse as ov
agent = ov.Agent("gpt-4o-mini", api_key="your-key")
adata = agent.run("quality control with nUMI>500, mito<0.2", adata)
adata = agent.run("leiden clustering resolution=1.0", adata)
```

## Development Conventions

### Adding New Functions

**Location**: Functions belong in module-specific files with underscore prefix (e.g., `single/_clustering.py`)

**Import Pattern**: Always import in module's `__init__.py`:
```python
# In omicverse/single/_clustering.py
def new_clustering_method(adata):
    pass

# In omicverse/single/__init__.py
from ._clustering import new_clustering_method
```

### External Package Integration

For non-core dependencies, **always use lazy imports** to avoid installation conflicts:

**❌ WRONG** (causes import errors):
```python
import external_package

def my_function():
    external_package.run()
```

**✅ CORRECT** (lazy import):
```python
def my_function():
    try:
        import external_package
    except ImportError:
        raise ImportError('Please install external_package from https://...')
    external_package.run()
```

### Function Documentation Standard

All public functions require comprehensive docstrings with type hints:

```python
def preprocess(adata: anndata.AnnData, mode: str='scanpy',
               target_sum: int=50*1e4, n_HVGs: int=2000,
               organism: str='human', no_cc: bool=False) -> anndata.AnnData:
    """
    Preprocesses AnnData using specified normalization and HVG selection.

    Arguments:
        adata: The data matrix.
        mode: Normalization mode. Can be 'scanpy' or 'pearson'.
        target_sum: Target total count after normalization.
        n_HVGs: Number of highly variable genes to select.
        organism: The organism ('human' or 'mouse').
        no_cc: Whether to remove cell-cycle correlated genes.

    Returns:
        adata: The preprocessed data matrix.
    """
```

### Registering Functions for Smart Agent

Functions discoverable by the Smart Agent must use the `@register_function` decorator:

```python
from omicverse.utils.registry import register_function

@register_function(
    aliases=["质控", "qc", "quality_control"],
    category="preprocessing",
    description="Perform standard single-cell quality control filtering",
    examples=["ov.pp.qc(adata, tresh={'mito_perc': 0.15, 'nUMIs': 500})"],
)
def qc(adata, tresh=None):
    """Run OmicVerse QC on the provided AnnData."""
    ...
```

**Required fields**:
- `aliases` - List of alternative names (include English + Chinese if applicable)
- `category` - Function category (preprocessing, clustering, visualization, etc.)
- `description` - Clear, concise description
- `examples` - Representative usage examples

## Key Files for Development

**Core Entry Points**:
- `omicverse/__init__.py` - Package initialization, main imports, agent system
- `omicverse/utils/registry.py` - Function registry system
- `omicverse/utils/smart_agent.py` - Smart agent implementation

**Configuration**:
- `pyproject.toml` - Build configuration (Flit Core backend), dependencies, entry points
- `setup.py` - Compatibility shim for legacy installations
- `setup.cfg` - pytest configuration (excludes `OvStudent/Converted_Scripts_Annotated`)

**Documentation**:
- `omicverse_guide/docs/Developer_guild.md` - Developer guide and contribution workflow
- `omicverse_guide/mkdocs.yml` - MkDocs configuration
- `.readthedocs.yaml` - Read the Docs build config (Ubuntu 20.04, Python 3.9)

**Testing**:
- `tests/single/` - Single-cell tests
- `tests/bulk/` - Bulk RNA-seq tests
- `tests/utils/` - Utility function tests
- `tests/llm/` - LLM integration tests

## Build System

**Backend**: Flit Core (modern, standards-based)
**Python Support**: 3.8 - 3.12
**License**: GPL 3.0

**Core Dependencies**:
- Scientific stack: numpy>=1.23, pandas>=1.5, scipy>=1.8,<1.12, scikit-learn>=1.2
- Single-cell: scanpy>=1.9, anndata
- Visualization: matplotlib>3.6, seaborn>=0.11, plotly
- Network analysis: networkx>=2.8, igraph>=0.10, leidenalg>=0.9
- Statistical: statsmodels>=0.13, lifelines>=0.27, pydeseq2>=0.4.1

**Optional Extras**:
- `[full]` - Includes scVI, DynAmo, squidpy, tangram, pertpy
- `[skillseeker]` - Includes beautifulsoup4, PyGithub, PyMuPDF

## Testing Framework

**Framework**: pytest
**Configuration**: `setup.cfg` (excludes converted script directories)

**Test Pattern** (from single-cell tests):
```python
import pytest
import omicverse as ov

def test_pp():
    adata = ov.utils.pancreas()  # Built-in dataset
    adata = ov.pp.qc(adata, tresh={'mito_perc': 0.05, 'nUMIs': 500, 'detected_genes': 250})
    ov.utils.store_layers(adata, layers='counts')
    adata = ov.pp.preprocess(adata, mode='shiftlog|pearson', n_HVGs=2000)
    ov.pp.scale(adata)
    assert adata.layers['scaled'] is not None
```

## CI/CD

**GitHub Actions** (`.github/workflows/`):
- `python-package.yml` - Automated testing on Python 3.10, 3.11
- `claude.yml` - Claude AI code analysis
- `claude-code-review.yml` - Automated code review

## Design Patterns

1. **Lazy Imports**: Heavy optional dependencies imported only when needed
2. **Decorator-Based Registry**: Functions discoverable via metadata (`@register_function`)
3. **AnnData-Centric**: All single-cell operations use scanpy's AnnData objects
4. **DataFrame-Centric**: Bulk operations use pandas DataFrames
5. **Modular Architecture**: Clear separation between preprocessing, analysis, visualization

## External Packages Directory

**Location**: `omicverse/external/` (45+ integrated packages)

**Purpose**: Include external tools without causing dependency conflicts

**Key Integrations**:
- Graph methods: VIA, PROST, GraphST, STAGATE, SEACells
- Batch correction: Harmony, CellANOVA
- Spatial analysis: STAligner, PROST
- Trajectory: latentvelo, graphvelo
- Enrichment: gseapy, BINARY
- Communication: commot, flowsig

## Claude Skills System

**Location**: `.claude/skills/` (25+ specialized skills)

**Skill Categories**:
- Bulk analysis: `bulk-deg-analysis`, `bulk-deseq2-analysis`, `bulk-wgcna-analysis`
- Single-cell: `single-clustering`, `single-annotation`, `single-trajectory-inference`
- Integration: `bulk-to-single-deconvolution`, `single-to-spatial-mapping`
- Visualization: `data-viz-plots`, `plotting-visualization`
- Data utilities: `data-transform`, `data-stats-analysis`

Each skill contains `SKILL.md` (instructions/examples) and `reference.md` (API references).

## Pull Request Workflow

1. Fork the repository: https://github.com/Starlitnightly/omicverse
2. Clone your fork
3. Create feature branch
4. Implement function in appropriate module (single, bulk, utils, etc.)
5. Add function import to module's `__init__.py`
6. Decorate with `@register_function` for agent discovery
7. Create test in `tests/module/test_feature.py`
8. Update documentation if needed
9. Commit with clear messages
10. Push and create PR
11. Wait for review and CI/CD checks

## Important Notes

- **Always use lazy imports** for non-core dependencies to avoid installation conflicts
- **Register new functions** with `@register_function` decorator for Smart Agent discovery
- **Follow type hint conventions** - use `anndata.AnnData` for single-cell, `pd.DataFrame` for bulk
- **Test coverage** - add tests for all new functions in appropriate `tests/` subdirectory
- **Documentation** - include comprehensive docstrings with Arguments/Returns sections
- **License compliance** - ensure external packages are GPL-compatible

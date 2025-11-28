# OmicVerse Dependencies Documentation

## Overview

This document provides a comprehensive overview of OmicVerse dependencies, including recent additions, version constraints, rationale, and dependency management best practices.

## Table of Contents

1. [Dependency Categories](#dependency-categories)
2. [Recent Additions](#recent-additions)
3. [Version Constraints](#version-constraints)
4. [Required vs Optional Dependencies](#required-vs-optional-dependencies)
5. [Dependency Installation](#dependency-installation)
6. [Troubleshooting](#troubleshooting)
7. [Dependency Management](#dependency-management)

---

## Dependency Categories

### Core Scientific Computing

**Purpose:** Foundation for numerical computation and data manipulation

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.23 | Array operations, numerical computing |
| scipy | >=1.8, <1.12 | Scientific algorithms, statistics |
| pandas | >=1.5 | Data frames, tabular data manipulation |
| scikit-learn | >=1.2 | Machine learning algorithms |

**Why these versions:**
- `numpy>=1.23`: Required for Python 3.11+ compatibility
- `scipy<1.12`: Upper bound to avoid breaking API changes
- `pandas>=1.5`: Performance improvements and new features
- `scikit-learn>=1.2`: Modern API compatibility

---

### Single-Cell Analysis

**Purpose:** Core single-cell RNA-seq analysis functionality

| Package | Version | Purpose |
|---------|---------|---------|
| scanpy | >=1.9 | Single-cell analysis framework |
| anndata | (via scanpy) | Annotated data matrices |
| igraph | >=0.10 | Graph algorithms for clustering |
| leidenalg | >=0.9 | Leiden clustering algorithm |

**Why these versions:**
- `scanpy>=1.9`: Modern API, better performance
- `igraph>=0.10`: Python 3.10+ compatibility
- `leidenalg>=0.9`: Latest clustering improvements

---

### Visualization

**Purpose:** Data visualization and plotting

| Package | Version | Purpose |
|---------|---------|---------|
| matplotlib | >3.6 | Core plotting library |
| seaborn | >=0.11 | Statistical visualizations |
| plotly | (latest) | Interactive plots |

**Why these versions:**
- `matplotlib>3.6`: Python 3.11 compatibility, improved layouts
- `seaborn>=0.11`: Modern API, better defaults
- `plotly`: Interactive visualizations for web interfaces

---

### Network and Graph Analysis

**Purpose:** Network analysis and graph algorithms

| Package | Version | Purpose |
|---------|---------|---------|
| networkx | >=2.8 | General graph algorithms |
| graphtools | >=1.5 | Graph-based analysis tools |
| igraph | >=0.10 | Fast graph algorithms |

**Why these versions:**
- `networkx>=2.8`: Performance improvements
- `graphtools>=1.5`: Compatibility with modern numpy

---

### Statistical Analysis

**Purpose:** Statistical testing and modeling

| Package | Version | Purpose |
|---------|---------|---------|
| statsmodels | >=0.13 | Statistical models, tests |
| lifelines | >=0.27 | Survival analysis |
| pydeseq2 | >=0.4.1 | Differential expression (DESeq2) |
| pygam | >=0.8.0 | Generalized additive models |

**Why these versions:**
- `statsmodels>=0.13`: Modern statistical APIs
- `lifelines>=0.27`: Improved survival models
- `pydeseq2>=0.4.1`: Bug fixes and performance
- `pygam>=0.8.0`: Latest GAM implementations

---

### Machine Learning & Deep Learning

**Purpose:** Advanced ML and neural network capabilities

| Package | Version | Purpose |
|---------|---------|---------|
| transformers | >=0.30 | Transformer models, embeddings |
| tensorboard | >=2.6 | Training visualization |
| numba | >=0.56 | JIT compilation for speed |
| einops | >=0.6 | Tensor operations |

**Why these versions:**
- `transformers>=0.30`: Modern transformer architectures
- `tensorboard>=2.6`: TensorFlow 2.x compatibility
- `numba>=0.56`: Python 3.11 support
- `einops>=0.6`: Clean tensor manipulation API

---

### Data Processing & I/O

**Purpose:** Data loading, processing, and storage

| Package | Version | Purpose |
|---------|---------|---------|
| tqdm | >=4.64 | Progress bars |
| gdown | >=4.6 | Google Drive downloads |
| boltons | >=23.0 | Utility functions |
| multiprocess | >=0.70 | Parallel processing |

**Why these versions:**
- `tqdm>=4.64`: Modern progress bar features
- `gdown>=4.6`: Google Drive API compatibility
- `boltons>=23.0`: Latest utility functions
- `multiprocess>=0.70`: Better pickling support

---

### Bioinformatics-Specific

**Purpose:** Specialized bioinformatics tools

| Package | Version | Purpose |
|---------|---------|---------|
| metatime | >=1.3.0 | Cell type annotation |
| ktplotspy | >=0.1 | Cell-cell communication visualization |
| python-dotplot | >=0.0.1 | Dotplot visualizations |
| mofax | >=0.3 | Multi-omics factor analysis |
| scikit-misc | >=0.1 | Loess smoothing and more |

**Why these versions:**
- `metatime>=1.3.0`: Latest annotation algorithms
- Domain-specific tools for specialized analyses

---

### Utility & Infrastructure

**Purpose:** General utilities and infrastructure

| Package | Version | Purpose |
|---------|---------|---------|
| requests | >=2.0 | HTTP requests |
| tomli | (latest) | TOML parsing |
| datetime | >=4.5 | Date/time operations |
| termcolor | >=2.1 | Colored terminal output |
| pillow | >=9.0 | Image processing |
| ctxcore | >=0.2 | Context management |
| ipywidgets | >=8.0 | Jupyter widgets |

**Why these versions:**
- `requests>=2.0`: Stable HTTP API
- `ipywidgets>=8.0`: Jupyter Lab compatibility
- `pillow>=9.0`: Security updates

---

### Visualization & Layout

**Purpose:** Advanced visualization layouts

| Package | Version | Purpose |
|---------|---------|---------|
| marsilea | (latest) | Complex layout visualization |
| adjustText | >=0.8 | Text label adjustment in plots |

**Why these versions:**
- Latest versions for best features and compatibility

---

### LLM & Agent Dependencies

**Purpose:** Agent system and LLM integration

| Package | Version | Purpose |
|---------|---------|---------|
| transformers | >=0.30 | Transformer models |
| wandb | (latest) | Experiment tracking |
| requests | >=2.0 | API calls |

**Why these versions:**
- `transformers>=0.30`: Modern LLM support
- `wandb`: Latest features for experiment tracking

---

## Recent Additions

### 2025-01-08: Testing Infrastructure

**Added:**
```toml
[project.optional-dependencies]
tests = [
  "pytest>=7.0",
  "pytest-asyncio>=0.23",
]
```

**Rationale:**

1. **pytest>=7.0**
   - Modern pytest features
   - Better error reporting
   - Async test support
   - Parametrization improvements

2. **pytest-asyncio>=0.23**
   - **Critical for agent system testing**
   - Enables `async def` test functions
   - Required for testing streaming APIs
   - Supports `@pytest.mark.asyncio` decorator

**Version Constraints:**
- `pytest>=7.0`: Minimum for Python 3.11 support
- `pytest-asyncio>=0.23`: Latest version with auto mode support

**Impact:**
- Enables comprehensive async testing
- Required for agent backend streaming tests
- No impact on production dependencies (optional)

**Usage:**
```bash
# Install with test dependencies
pip install -e ".[tests]"

# Run async tests
pytest tests/utils/test_agent_backend_streaming.py
```

---

### 2024-12: Agent System Foundation

**Added (implicitly via existing dependencies):**
- `transformers>=0.30` - Already present for embeddings
- `requests>=2.0` - Already present for data downloads
- `wandb` - Already present for experiment tracking

**No new production dependencies added for agent system!**

The agent system was designed to work with existing dependencies plus external LLM provider SDKs (installed by users as needed).

---

### 2024-11: Skill Seeker System

**Added:**
```toml
[project.optional-dependencies]
skillseeker = [
  "beautifulsoup4>=4.12",
  "PyGithub>=2.2",
  "PyMuPDF>=1.24",
]
```

**Rationale:**

1. **beautifulsoup4>=4.12**
   - HTML/XML parsing for documentation scraping
   - Extracting skill content from web pages
   - Used in `ov-skill-seeker` CLI tool

2. **PyGithub>=2.2**
   - GitHub API integration
   - Fetching documentation from repositories
   - Issue tracking and repository management

3. **PyMuPDF>=1.24**
   - PDF parsing for documentation
   - Extracting text from PDF tutorials
   - Converting PDF docs to skills

**Version Constraints:**
- `beautifulsoup4>=4.12`: Latest HTML5 parsing
- `PyGithub>=2.2`: Modern GitHub API v3/v4
- `PyMuPDF>=1.24`: Latest PDF parsing features

**Impact:**
- Optional dependency group
- Only needed for skill development
- Not required for end users
- Enables automated skill creation

**Usage:**
```bash
# Install skill seeker dependencies
pip install -e ".[skillseeker]"

# Use the CLI tool
ov-skill-seeker --help
```

---

### 2024-10: Enhanced Visualization

**Added:**
```python
dependencies = [
    'marsilea',  # Complex layout visualization
]
```

**Rationale:**
- Advanced multi-panel layouts
- Heatmap enhancements
- Better figure composition

**Impact:**
- Enables advanced visualization features
- Required for complex plotting functions

---

## Version Constraints

### Constraint Types

#### 1. Lower Bounds (`>=`)

**Purpose:** Ensure minimum feature availability

**Examples:**
```python
'numpy>=1.23'     # Need Python 3.11 support
'scanpy>=1.9'     # Need modern API
'pytest>=7.0'     # Need async support
```

**Guidelines:**
- Set when specific features are required
- Update when code uses new API features
- Document why minimum version is needed

#### 2. Upper Bounds (`<`)

**Purpose:** Prevent breaking changes

**Examples:**
```python
'scipy>=1.8, <1.12'  # Avoid scipy 1.12+ API changes
```

**Guidelines:**
- Use sparingly (pins can cause conflicts)
- Only when known breaking changes exist
- Test and update regularly

#### 3. Exact Versions (`==`)

**Purpose:** Reproducible builds (avoid in libraries)

**Examples:**
```python
# Avoid in library code
'numpy==1.23.5'  # Too restrictive!

# OK in requirements.txt for deployment
numpy==1.23.5
```

**Guidelines:**
- **Never use in `pyproject.toml` dependencies**
- OK in `requirements.txt` for reproducibility
- Use ranges instead: `'numpy>=1.23,<2.0'`

#### 4. Compatible Release (`~=`)

**Purpose:** Patch updates only

**Examples:**
```python
'requests~=2.31.0'  # Allows 2.31.x, not 2.32.0
```

**Guidelines:**
- Rarely needed
- Prefer `>=` with optional upper bound

---

### Version Constraint Best Practices

#### For Libraries (like OmicVerse)

```python
# Good: Flexible ranges
dependencies = [
    'numpy>=1.23',           # Lower bound only
    'scipy>=1.8, <1.12',    # Range when needed
]

# Bad: Too restrictive
dependencies = [
    'numpy==1.23.5',        # Exact pin - avoid!
    'scipy>=1.11.0',        # Too recent - excludes users
]
```

#### For Applications (deployment)

```python
# requirements.txt (exact pins OK)
numpy==1.23.5
scipy==1.11.2
pandas==2.1.0

# Generated from:
# pip freeze > requirements.txt
```

#### Testing Constraints

```toml
[project.optional-dependencies]
tests = [
    "pytest>=7.0",         # Modern features
    "pytest-cov>=4.0",     # Coverage reporting
    "pytest-asyncio>=0.23" # Async support
]
```

---

### Version Update Policy

#### When to Update Constraints

1. **Lower Bound:**
   - New feature used in code
   - Security vulnerability in old version
   - Python version support changes

2. **Upper Bound:**
   - Known breaking change in new version
   - Major API rewrite upstream
   - Test failures with new version

#### Testing New Versions

```bash
# Test with latest versions
pip install --upgrade numpy scipy pandas
pytest

# Test with minimum versions
pip install numpy==1.23 scipy==1.8 pandas==1.5
pytest

# Test in fresh environment
python -m venv test_env
source test_env/bin/activate
pip install -e ".[tests]"
pytest
```

---

## Required vs Optional Dependencies

### Required Dependencies

**Definition:** Needed for core OmicVerse functionality

**Listed in:** `pyproject.toml` → `dependencies`

**Examples:**
```python
dependencies = [
    'numpy>=1.23',
    'scanpy>=1.9',
    'matplotlib>3.6',
    # ... all core packages
]
```

**Installation:**
```bash
# Automatically installed
pip install omicverse
```

**Rationale:**
- Needed by 90%+ of users
- Core analysis functions depend on them
- Breaking if missing

---

### Optional Dependencies

**Definition:** Needed for specific features or development

**Listed in:** `pyproject.toml` → `optional-dependencies`

#### 1. Full Feature Set

```toml
[project.optional-dependencies]
full = [
  "dynamo-release",    # RNA velocity
  "squidpy",           # Spatial analysis
  "tangram-sc",        # Spatial mapping
  "pertpy",            # Perturbation analysis
  "toytree",           # Phylogenetics
  "arviz",             # Bayesian analysis
  "ete3",              # Tree visualization
  "scvi-tools",        # Deep learning models
  "pymde",             # Embeddings
  "torchdr",           # Dimensionality reduction
  "memento-de"         # Differential expression
]
```

**Rationale:**
- Heavy dependencies (PyTorch, etc.)
- Specialized use cases
- Not needed by all users

**Installation:**
```bash
pip install omicverse[full]
```

---

#### 2. Skill Seeker Tools

```toml
skillseeker = [
  "beautifulsoup4>=4.12",  # Web scraping
  "PyGithub>=2.2",         # GitHub API
  "PyMuPDF>=1.24",         # PDF parsing
]
```

**Rationale:**
- Only for skill development
- Not needed by end users
- Developers/contributors only

**Installation:**
```bash
pip install omicverse[skillseeker]
```

---

#### 3. Testing Tools

```toml
tests = [
  "pytest>=7.0",           # Test framework
  "pytest-asyncio>=0.23",  # Async testing
]
```

**Rationale:**
- Development only
- Not needed in production
- CI/CD environments

**Installation:**
```bash
pip install omicverse[tests]
```

---

### Dependency Installation Matrix

| Use Case | Command | Installs |
|----------|---------|----------|
| Basic user | `pip install omicverse` | Required only |
| Power user | `pip install omicverse[full]` | Required + Full |
| Developer | `pip install -e ".[tests,skillseeker]"` | All optional |
| Testing | `pip install -e ".[tests]"` | Required + Tests |
| Documentation | `pip install -e ".[skillseeker]"` | Required + Skill tools |

---

## Dependency Installation

### Standard Installation

```bash
# Latest stable release
pip install omicverse

# Specific version
pip install omicverse==1.7.9

# From source (development)
git clone https://github.com/Starlitnightly/omicverse.git
cd omicverse
pip install -e .
```

---

### Installation with Optional Dependencies

```bash
# Install with all features
pip install omicverse[full]

# Install with specific feature sets
pip install omicverse[tests]
pip install omicverse[skillseeker]

# Install multiple feature sets
pip install omicverse[full,tests,skillseeker]

# Development installation with all features
pip install -e ".[full,tests,skillseeker]"
```

---

### LLM Provider SDKs (User Installed)

The agent system requires LLM provider SDKs, installed separately by users:

```bash
# OpenAI
pip install openai

# Anthropic (Claude)
pip install anthropic

# Google (Gemini)
pip install google-generativeai

# Alibaba DashScope
pip install dashscope
```

**Why not included:**
- User choice of provider
- Avoid unnecessary dependencies
- Version flexibility
- Licensing considerations

---

### Requirements Files

#### requirements.txt (Full Development)

```bash
# Install exact versions from requirements.txt
pip install -r requirements.txt
```

**Purpose:**
- Reproducible development environment
- Documentation building
- Specific version pins

#### requirements-latest.txt (Latest Versions)

```bash
# Install latest compatible versions
pip install -r requirements-latest.txt
```

**Purpose:**
- Testing with latest dependencies
- Finding compatibility issues early
- Preparing for updates

---

## Troubleshooting

### Common Issues

#### 1. Scipy Version Conflict

**Error:**
```
ERROR: scipy 1.12.0 has requirement numpy>=1.26.0, but you have numpy 1.23.5
```

**Solution:**
```bash
# Update numpy
pip install --upgrade numpy

# Or use compatible scipy
pip install 'scipy>=1.8,<1.12'
```

---

#### 2. Missing Async Test Support

**Error:**
```
PytestConfigWarning: Unknown config option: asyncio_mode
```

**Solution:**
```bash
# Install pytest-asyncio
pip install pytest-asyncio>=0.23

# Verify installation
pytest --version
python -c "import pytest_asyncio; print(pytest_asyncio.__version__)"
```

---

#### 3. Matplotlib Import Error

**Error:**
```
ModuleNotFoundError: No module named 'matplotlib'
```

**Solution:**
```bash
# Install omicverse properly
pip install omicverse

# Or install matplotlib separately
pip install 'matplotlib>3.6'
```

---

#### 4. Conflicting Dependencies

**Error:**
```
ERROR: Cannot install omicverse and package-x because these package versions have conflicting dependencies.
```

**Solution:**
```bash
# Create fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate
pip install omicverse

# Or use conda
conda create -n omicverse python=3.11
conda activate omicverse
pip install omicverse
```

---

#### 5. Python Version Incompatibility

**Error:**
```
ERROR: omicverse requires Python '>=3.8'
```

**Solution:**
```bash
# Check Python version
python --version

# Use compatible Python version
conda create -n omicverse python=3.11
conda activate omicverse
pip install omicverse
```

---

### Dependency Resolution

#### Check Installed Versions

```bash
# List all installed packages
pip list

# Check specific package
pip show numpy

# Check for conflicts
pip check
```

#### Dependency Tree

```bash
# Install pipdeptree
pip install pipdeptree

# Show dependency tree
pipdeptree

# Show reverse dependencies
pipdeptree -r -p numpy

# Show conflicts
pipdeptree --warn fail
```

---

## Dependency Management

### For Contributors

#### Adding a New Dependency

1. **Evaluate Necessity:**
   - Is it really needed?
   - Can we use existing dependencies?
   - What's the maintenance status?

2. **Choose Version Constraint:**
   ```python
   # Determine minimum version
   'package>=X.Y'

   # Add upper bound if needed
   'package>=X.Y,<Z.0'
   ```

3. **Update pyproject.toml:**
   ```toml
   [project]
   dependencies = [
       # ... existing
       'new-package>=1.0',
   ]
   ```

4. **Test Compatibility:**
   ```bash
   pip install -e ".[tests]"
   pytest
   ```

5. **Document:**
   - Add to this file (DEPENDENCIES.md)
   - Explain rationale
   - Note version constraint reasoning

---

#### Updating a Dependency

1. **Test with New Version:**
   ```bash
   pip install --upgrade package-name
   pytest
   ```

2. **Update Constraint if Needed:**
   ```python
   # Old
   'package>=1.0'

   # New (if using new features)
   'package>=2.0'
   ```

3. **Update Documentation:**
   - Note changes in CHANGELOG
   - Update DEPENDENCIES.md
   - Update migration guides if breaking

---

#### Removing a Dependency

1. **Check Usage:**
   ```bash
   # Search for imports
   grep -r "import package" omicverse/

   # Check reverse dependencies
   pipdeptree -r -p package-name
   ```

2. **Remove from pyproject.toml:**
   ```toml
   # Remove line
   'package>=X.Y',
   ```

3. **Update Code:**
   - Remove imports
   - Refactor code
   - Update tests

4. **Test:**
   ```bash
   pytest
   ```

5. **Document:**
   - Note in CHANGELOG
   - Update migration guide

---

### For Maintainers

#### Regular Dependency Audits

**Monthly:**
```bash
# Check for security vulnerabilities
pip install safety
safety check

# Check for outdated packages
pip list --outdated
```

**Quarterly:**
```bash
# Test with latest versions
pip install --upgrade-strategy eager --upgrade omicverse[full,tests]
pytest

# Update version constraints if compatible
# Document any breaking changes
```

**Annually:**
```bash
# Review all dependencies
# Remove unused dependencies
# Consolidate similar dependencies
# Update major versions
```

---

#### CI/CD Dependency Testing

```yaml
# .github/workflows/test-dependencies.yml

name: Test Dependencies

on: [push, pull_request]

jobs:
  test-minimum:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install minimum versions
        run: |
          pip install numpy==1.23 scipy==1.8 pandas==1.5
          pip install -e ".[tests]"
      - name: Run tests
        run: pytest

  test-latest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install latest versions
        run: |
          pip install --upgrade numpy scipy pandas
          pip install -e ".[tests]"
      - name: Run tests
        run: pytest
```

---

## Dependency Security

### Security Best Practices

1. **Pin Minimum Versions:**
   ```python
   # Include security fixes
   'requests>=2.31.0'  # CVE-2023-xxxxx fixed in 2.31.0
   ```

2. **Monitor Vulnerabilities:**
   ```bash
   # Use safety
   pip install safety
   safety check

   # Use dependabot (GitHub)
   # Enable in repository settings
   ```

3. **Regular Updates:**
   ```bash
   # Update dependencies regularly
   pip install --upgrade-strategy eager omicverse[full]
   pytest
   ```

---

## Summary

### Key Points

1. **Required Dependencies (48 packages):**
   - Core scientific stack (numpy, scipy, pandas, scikit-learn)
   - Single-cell analysis (scanpy, anndata)
   - Visualization (matplotlib, seaborn, plotly)
   - Bioinformatics tools

2. **Optional Dependencies (3 groups):**
   - `full`: Advanced features (13 packages)
   - `skillseeker`: Documentation tools (3 packages)
   - `tests`: Testing infrastructure (2 packages)

3. **Recent Additions:**
   - `pytest-asyncio>=0.23` for async testing
   - No new production dependencies for agent system

4. **Version Constraints:**
   - Lower bounds for minimum features
   - Upper bounds only when necessary
   - Tested regularly for compatibility

5. **Dependency Philosophy:**
   - Minimize required dependencies
   - Use optional dependencies for specialized features
   - Keep constraints flexible
   - Test with minimum and latest versions

---

## Resources

### Documentation

- [Python Packaging Guide](https://packaging.python.org/)
- [PEP 508 - Dependency specification](https://peps.python.org/pep-0508/)
- [PEP 631 - pyproject.toml dependencies](https://peps.python.org/pep-0631/)

### Tools

- [pipdeptree](https://github.com/tox-dev/pipdeptree) - Dependency tree visualization
- [safety](https://github.com/pyupio/safety) - Security vulnerability scanning
- [pip-tools](https://github.com/jazzband/pip-tools) - Requirements management

### Monitoring

- [Dependabot](https://github.com/dependabot) - Automated dependency updates
- [PyUp](https://pyup.io/) - Python dependency monitoring
- [Snyk](https://snyk.io/) - Security monitoring

---

**Document Version:** 1.0
**Last Updated:** 2025-01-08
**Author:** OmicVerse Development Team

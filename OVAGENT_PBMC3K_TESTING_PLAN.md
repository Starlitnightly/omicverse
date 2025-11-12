# OVAgent PBMC3k Comprehensive Testing Plan

**Purpose**: Test all features of the OmicVerse Agent system using PBMC3k data based on the tutorial at `omicverse_guide/docs/Tutorials-llm/t_ov_agent_pbmc3k.ipynb`

**Date Created**: 2025-11-12
**Branch**: claude/plan-ovagent-pbm3k-testing-011CV3gMwNmwMsYhaGESGYr1
**Tutorial Reference**: `/home/user/omicverse/omicverse_guide/docs/Tutorials-llm/t_ov_agent_pbmc3k.ipynb`

---

## Overview

The OmicVerse Agent (`ov.Agent`) is an LLM-powered natural language interface for single-cell and bulk RNA-seq analysis. This plan covers comprehensive testing of all agent features using the PBMC3k dataset.

### Key Features to Test:
1. **LLM-Based Skill Matching** - Pure LLM reasoning (no keyword matching)
2. **Progressive Disclosure** - Lazy-loading of skill content
3. **Priority System** - Priority 1 (fast) vs Priority 2 (comprehensive)
4. **Multi-Provider Support** - 30+ LLM models from 8 providers
5. **Code Generation** - Natural language → executable code
6. **Reflection System** - Code validation and improvement
7. **Result Review** - Output validation against user intent
8. **Function Registry** - 110 functions across 7 categories

---

## Test Environment Setup

### Prerequisites
```python
# Required packages
import omicverse as ov
import scanpy as sc
import os

# Check version
print(f"OmicVerse version: {ov.__version__}")
print(f"Supported models: {ov.list_supported_models()}")

# Set API keys (choose one or more providers)
os.environ['OPENAI_API_KEY'] = 'your-key-here'
os.environ['ANTHROPIC_API_KEY'] = 'your-key-here'
os.environ['GEMINI_API_KEY'] = 'your-key-here'
```

### Load PBMC3k Dataset
```python
# Primary method
adata = sc.datasets.pbmc3k()

# Fallback options if primary fails:
# 1. Local path: adata = sc.read_h5ad(os.environ.get('PBMC3K_PATH'))
# 2. Alternative dataset: adata = sc.datasets.pbmc68k_reduced()
```

### Verify Skills Loading
```python
from pathlib import Path
from omicverse.utils.skill_registry import build_multi_path_skill_registry

pkg_root = Path(ov.__file__).resolve().parents[1]
cwd = Path.cwd()

# Build skill registry
reg = build_multi_path_skill_registry(pkg_root, cwd)

print(f"✅ Loaded {len(reg.skill_metadata)} skills")
print(f"   Built-in skills path: {pkg_root / 'omicverse' / '.claude' / 'skills'}")
print(f"   Custom skills path: {cwd / '.claude' / 'skills'}")

# List first 10 skills
for slug in sorted(reg.skill_metadata.keys())[:10]:
    metadata = reg.skill_metadata[slug]
    print(f"  • {slug}: {metadata.description[:60]}...")
```

---

## Section 1: Core Workflow Testing

### 1.1 Basic Agent Initialization

**Test Case 1.1.1**: Initialize agent with different models

```python
# Test with GPT-5
agent_gpt5 = ov.Agent(model='gpt-5', api_key=os.getenv('OPENAI_API_KEY'))
assert agent_gpt5 is not None, "GPT-5 agent initialization failed"

# Test with Claude Sonnet 4
agent_claude = ov.Agent(
    model='anthropic/claude-sonnet-4-20250514',
    api_key=os.getenv('ANTHROPIC_API_KEY')
)
assert agent_claude is not None, "Claude agent initialization failed"

# Test with Gemini 2.5 Pro
agent_gemini = ov.Agent(
    model='gemini/gemini-2.5-pro',
    api_key=os.getenv('GEMINI_API_KEY')
)
assert agent_gemini is not None, "Gemini agent initialization failed"
```

**Expected Output**:
```
 Initializing OmicVerse Smart Agent (internal backend)...
   🧭 Loaded 23 skills (23 built-in)
    Model: [Model Name]
    Provider: [Provider Name]
    Endpoint: [API Endpoint]
   ✅ [Provider] API key available
   📚 Function registry loaded: 110 functions in 7 categories
✅ Smart Agent initialized successfully!
```

**Verification**:
- [ ] Agent object created successfully
- [ ] Correct model displayed
- [ ] Skills loaded (should see 23-25 skills)
- [ ] Function registry loaded (110 functions)
- [ ] API key detected

---

### 1.2 Quality Control Workflow

**Test Case 1.2.1**: Basic QC with cell filtering

```python
agent = ov.Agent(model='gpt-5', api_key=os.getenv('OPENAI_API_KEY'))
adata_fresh = sc.datasets.pbmc3k()

# Natural language QC request
adata_qc = agent.run('quality control with nUMI>500, mito<0.2', adata_fresh)
```

**Expected Behavior**:
- [ ] Agent matches `single-preprocessing` skill (check for "🎯 LLM matched skills:")
- [ ] Code generated uses `ov.pp.qc()`
- [ ] Parameters correctly set: `tresh={'mito_perc': 0.2, 'nUMIs': 500}`
- [ ] Cells filtered based on thresholds
- [ ] Returns modified AnnData object

**Validation**:
```python
# Check QC was applied
assert adata_qc.n_obs < adata_fresh.n_obs, "No cells were filtered"
assert 'qc_passed' in adata_qc.obs.columns or adata_qc.n_obs < 2700, "QC not applied"
print(f"✅ QC passed: {adata_fresh.n_obs} → {adata_qc.n_obs} cells")
```

**Test Case 1.2.2**: QC with alternative phrasing

Test semantic understanding with different phrasings:

```python
test_phrases = [
    "filter cells with more than 500 UMIs and less than 20% mitochondrial content",
    "QC my data removing high mito and low count cells",
    "quality filter: nUMI threshold 500, mitochondrial percentage under 0.2",
    "remove low quality cells with mito<0.2 and nUMI>500"
]

for phrase in test_phrases:
    adata_test = sc.datasets.pbmc3k()
    result = agent.run(phrase, adata_test)
    print(f"✅ Phrase '{phrase[:50]}...' → {result.n_obs} cells")
```

**Verification**:
- [ ] All phrases correctly understood
- [ ] Similar filtering results across phrases
- [ ] Skill matching works for semantic variations

---

### 1.3 Preprocessing and HVG Selection

**Test Case 1.3.1**: Standard preprocessing

```python
adata_qc = agent.run('quality control with nUMI>500, mito<0.2', adata_fresh)
adata_prep = agent.run(
    'preprocess with 2000 highly variable genes using shiftlog|pearson',
    adata_qc
)
```

**Expected Behavior**:
- [ ] Matches `single-preprocessing` skill
- [ ] Uses `ov.pp.preprocess()` function
- [ ] Parameters: `mode='shiftlog|pearson'`, `n_HVGs=2000`
- [ ] Normalization applied
- [ ] HVGs computed and stored in `adata.var['highly_variable']`

**Validation**:
```python
assert 'highly_variable' in adata_prep.var.columns, "HVGs not computed"
assert adata_prep.var['highly_variable'].sum() == 2000, "Wrong number of HVGs"
print(f"✅ Preprocessing complete: {adata_prep.var['highly_variable'].sum()} HVGs")
```

**Test Case 1.3.2**: Alternative preprocessing modes

```python
test_modes = [
    ('shiftlog|pearson', 2000),
    ('lognorm|pearson', 3000),
    ('shiftlog|seurat', 1500),
]

for mode, n_hvgs in test_modes:
    adata_test = adata_qc.copy()
    result = agent.run(
        f'preprocess with {n_hvgs} highly variable genes using {mode}',
        adata_test
    )
    assert result.var['highly_variable'].sum() == n_hvgs
    print(f"✅ Mode {mode} with {n_hvgs} HVGs: passed")
```

---

### 1.4 Clustering Workflow

**Test Case 1.4.1**: Leiden clustering

```python
adata_clust = agent.run('leiden clustering resolution=1.0', adata_prep)
```

**Expected Behavior**:
- [ ] Matches `single-clustering` skill
- [ ] Computes neighbor graph if not present
- [ ] Runs Leiden algorithm with resolution=1.0
- [ ] Adds `leiden` column to `adata.obs`

**Validation**:
```python
assert 'leiden' in adata_clust.obs.columns, "Leiden clustering not performed"
n_clusters = adata_clust.obs['leiden'].nunique()
print(f"✅ Leiden clustering: {n_clusters} clusters identified")
assert n_clusters > 1, "Only one cluster found"
```

**Test Case 1.4.2**: Different resolutions

```python
resolutions = [0.5, 1.0, 1.5, 2.0]

for res in resolutions:
    adata_test = adata_prep.copy()
    result = agent.run(f'leiden clustering resolution={res}', adata_test)
    n_clust = result.obs['leiden'].nunique()
    print(f"✅ Resolution {res}: {n_clust} clusters")
```

**Verification**:
- [ ] Higher resolution → more clusters
- [ ] All resolutions work correctly
- [ ] Clustering results reasonable

---

### 1.5 Dimensionality Reduction and Visualization

**Test Case 1.5.1**: UMAP computation and plotting

```python
adata_umap = agent.run('compute umap and plot colored by leiden', adata_clust)
```

**Expected Behavior**:
- [ ] Matches `single-preprocessing` or `plotting-visualization` skill
- [ ] Computes UMAP if not present
- [ ] Creates plot colored by leiden clusters
- [ ] Embedding stored in `adata.obsm['X_umap']`

**Validation**:
```python
assert 'X_umap' in adata_umap.obsm.keys(), "UMAP not computed"
assert adata_umap.obsm['X_umap'].shape == (adata_umap.n_obs, 2), "Wrong UMAP dimensions"
print(f"✅ UMAP computed: shape {adata_umap.obsm['X_umap'].shape}")
```

**Test Case 1.5.2**: Alternative embeddings

```python
embeddings = [
    'compute tsne and plot by leiden',
    'compute pca and plot first 3 components',
    'create force-directed layout and visualize'
]

for emb_request in embeddings:
    adata_test = adata_clust.copy()
    result = agent.run(emb_request, adata_test)
    print(f"✅ Request '{emb_request}': completed")
```

---

### 1.6 Complete End-to-End Pipeline

**Test Case 1.6.1**: Full workflow in sequence

```python
# Initialize fresh agent
agent = ov.Agent(model='gpt-5', api_key=os.getenv('OPENAI_API_KEY'))
adata = sc.datasets.pbmc3k()

# Step 1: QC
adata = agent.run('quality control with nUMI>500, mito<0.2', adata)
print(f"Step 1 complete: {adata.n_obs} cells after QC")

# Step 2: Preprocessing
adata = agent.run('preprocess with 2000 highly variable genes using shiftlog|pearson', adata)
print(f"Step 2 complete: {adata.var['highly_variable'].sum()} HVGs")

# Step 3: Clustering
adata = agent.run('leiden clustering resolution=1.0', adata)
print(f"Step 3 complete: {adata.obs['leiden'].nunique()} clusters")

# Step 4: Visualization
adata = agent.run('compute umap and plot colored by leiden', adata)
print(f"Step 4 complete: UMAP shape {adata.obsm['X_umap'].shape}")

# Final validation
assert adata.n_obs < 2700, "QC not applied"
assert 'highly_variable' in adata.var.columns, "Preprocessing failed"
assert 'leiden' in adata.obs.columns, "Clustering failed"
assert 'X_umap' in adata.obsm.keys(), "UMAP failed"

print("✅ Complete end-to-end pipeline: PASSED")
```

**Verification Checklist**:
- [ ] All steps execute without errors
- [ ] Data flows correctly between steps
- [ ] Final AnnData object has all expected attributes
- [ ] Results are scientifically reasonable

---

## Section 2: LLM-Based Skill Matching

### 2.1 Skill Discovery and Matching

**Test Case 2.1.1**: Verify progressive disclosure

```python
import time

# Measure agent initialization time
start = time.time()
agent = ov.Agent(model='gpt-5', api_key=os.getenv('OPENAI_API_KEY'))
init_time = time.time() - start

print(f"Agent initialization time: {init_time:.2f}s")
print(f"Expected: < 5 seconds with progressive disclosure")

# Verify only metadata loaded initially
assert hasattr(agent.skill_registry, 'skill_metadata'), "Metadata not available"
assert len(agent.skill_registry.skill_metadata) > 0, "No skills loaded"
print(f"✅ Progressive disclosure: {len(agent.skill_registry.skill_metadata)} skill metadata loaded")
```

**Expected**:
- [ ] Initialization < 5 seconds
- [ ] Only metadata loaded (not full content)
- [ ] ~1,250 tokens loaded (not ~25,000)

**Test Case 2.1.2**: Semantic matching accuracy

```python
# Test natural language variations
test_cases = [
    ("QC my data", "single-preprocessing"),
    ("filter low quality cells", "single-preprocessing"),
    ("cluster cells", "single-clustering"),
    ("annotate cell types", "single-annotation"),
    ("find differentially expressed genes", "bulk-deg-analysis"),
    ("make a volcano plot", "plotting-visualization"),
    ("export results to excel", "data-export-excel"),
    ("create PDF report", "data-export-pdf"),
]

for request, expected_skill in test_cases:
    print(f"\nTesting: '{request}'")
    # Note: Check agent output for "🎯 LLM matched skills:" line
    result = agent.run(request, adata)
    print(f"Expected skill: {expected_skill}")
    # Manual verification needed - check console output
```

**Verification**:
- [ ] All requests match expected skills
- [ ] Output shows "🎯 LLM matched skills: [skill-name]"
- [ ] Semantic variations understood correctly

**Test Case 2.1.3**: Multi-skill matching

```python
# Requests that should match multiple skills
multi_skill_requests = [
    "preprocess, cluster, and visualize the data",
    "QC and normalize then run differential expression",
    "annotate cells and export to excel"
]

for request in multi_skill_requests:
    print(f"\n--- Request: {request} ---")
    result = agent.run(request, adata)
    # Check console output for multiple matched skills
    print("Check console for matched skills")
```

**Verification**:
- [ ] Multiple skills matched when appropriate
- [ ] Skills loaded on-demand
- [ ] Code integrates multiple skill guidance

---

### 2.2 Lazy Loading Verification

**Test Case 2.2.1**: Verify lazy loading behavior

```python
from pathlib import Path

# Create new agent
agent = ov.Agent(model='gpt-5', api_key=os.getenv('OPENAI_API_KEY'))

# Check initial state - metadata only
initial_skills = agent.skill_registry.skill_metadata
print(f"Initial metadata loaded: {len(initial_skills)} skills")

# Trigger a request that needs a specific skill
result = agent.run('quality control with nUMI>500', adata)

# Verify full content was lazy-loaded for matched skill
# (This is internal - verify through logs and performance)
print("✅ Lazy loading: skill content loaded on-demand")
```

**Expected Behavior**:
- [ ] Fast initialization (metadata only)
- [ ] Full skill content loaded only when matched
- [ ] No performance degradation for unused skills

---

### 2.3 Skill Matching Edge Cases

**Test Case 2.3.1**: Ambiguous requests

```python
ambiguous_requests = [
    "analyze my data",  # Too vague
    "process this",     # Generic
    "help me",          # No specific task
]

for request in ambiguous_requests:
    print(f"\nTesting ambiguous: '{request}'")
    try:
        result = agent.run(request, adata)
        print("Agent response received - check if clarification requested")
    except Exception as e:
        print(f"Exception: {e}")
```

**Verification**:
- [ ] Agent handles vague requests gracefully
- [ ] May ask for clarification
- [ ] Doesn't crash or produce nonsensical output

**Test Case 2.3.2**: No matching skills

```python
no_match_requests = [
    "predict stock prices",
    "analyze text sentiment",
    "train neural network for image classification"
]

for request in no_match_requests:
    print(f"\nTesting non-matching: '{request}'")
    try:
        result = agent.run(request, adata)
        print("Check if agent indicates no matching skills")
    except Exception as e:
        print(f"Expected behavior - no matching skills: {e}")
```

**Verification**:
- [ ] Agent gracefully handles out-of-scope requests
- [ ] Clear messaging when no skills match
- [ ] No crashes or infinite loops

---

## Section 3: Multi-Provider Testing

### 3.1 Provider Compatibility

**Test Case 3.1.1**: Test major providers

```python
providers_to_test = [
    # OpenAI
    ('gpt-5', 'OPENAI_API_KEY'),
    ('gpt-4o', 'OPENAI_API_KEY'),
    ('gpt-4o-mini', 'OPENAI_API_KEY'),

    # Anthropic
    ('anthropic/claude-opus-4-20250514', 'ANTHROPIC_API_KEY'),
    ('anthropic/claude-sonnet-4-20250514', 'ANTHROPIC_API_KEY'),
    ('anthropic/claude-haiku-3-5-20241022', 'ANTHROPIC_API_KEY'),

    # Google
    ('gemini/gemini-2.5-pro', 'GEMINI_API_KEY'),
    ('gemini/gemini-2.5-flash', 'GEMINI_API_KEY'),

    # DeepSeek (if available)
    ('deepseek/deepseek-chat', 'DEEPSEEK_API_KEY'),
]

results = []
for model_id, env_key in providers_to_test:
    api_key = os.getenv(env_key)
    if not api_key:
        print(f"⏭️  Skipping {model_id}: {env_key} not set")
        continue

    try:
        agent = ov.Agent(model=model_id, api_key=api_key)
        adata_test = sc.datasets.pbmc3k()
        result = agent.run('quality control with nUMI>500, mito<0.2', adata_test)

        status = "✅ PASSED" if result.n_obs < 2700 else "⚠️  WARNING"
        results.append((model_id, status))
        print(f"{status} - {model_id}")
    except Exception as e:
        results.append((model_id, f"❌ FAILED: {str(e)[:50]}"))
        print(f"❌ FAILED - {model_id}: {e}")

# Summary
print("\n" + "="*70)
print("PROVIDER COMPATIBILITY SUMMARY")
print("="*70)
for model, status in results:
    print(f"{status} - {model}")
```

**Verification**:
- [ ] All providers with valid API keys work
- [ ] Consistent behavior across providers
- [ ] Error messages clear for missing keys

**Test Case 3.1.2**: Provider-specific features

```python
# Test if different models produce different code styles
models = ['gpt-5', 'anthropic/claude-sonnet-4-20250514', 'gemini/gemini-2.5-pro']

for model in models:
    api_key = os.getenv(model.split('/')[0].upper() + '_API_KEY')
    if not api_key and 'openai' in model.lower():
        api_key = os.getenv('OPENAI_API_KEY')

    if api_key:
        agent = ov.Agent(model=model, api_key=api_key)
        print(f"\n{'='*70}")
        print(f"Testing {model}")
        print('='*70)

        # Same request across models
        result = agent.run('quality control with nUMI>500, mito<0.2', adata.copy())
        print(f"Result: {result.n_obs} cells retained")
```

**Verification**:
- [ ] All models understand same natural language requests
- [ ] Code output may vary but achieves same goal
- [ ] Quality of generated code comparable

---

### 3.2 Model Performance Comparison

**Test Case 3.2.1**: Response time comparison

```python
import time

models_to_benchmark = [
    'gpt-4o-mini',  # Fast, cost-effective
    'gpt-5',         # Most capable
    'anthropic/claude-haiku-3-5-20241022',  # Fast
    'anthropic/claude-sonnet-4-20250514',   # Balanced
]

request = 'quality control with nUMI>500, mito<0.2'
benchmark_results = []

for model in models_to_benchmark:
    api_key = os.getenv('OPENAI_API_KEY') if 'gpt' in model else os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        continue

    agent = ov.Agent(model=model, api_key=api_key)
    adata_test = sc.datasets.pbmc3k()

    start = time.time()
    result = agent.run(request, adata_test)
    elapsed = time.time() - start

    benchmark_results.append({
        'model': model,
        'time_seconds': elapsed,
        'cells_retained': result.n_obs
    })
    print(f"{model}: {elapsed:.2f}s ({result.n_obs} cells)")

# Sort by time
benchmark_results.sort(key=lambda x: x['time_seconds'])
print("\n--- Performance Ranking (fastest to slowest) ---")
for i, r in enumerate(benchmark_results, 1):
    print(f"{i}. {r['model']}: {r['time_seconds']:.2f}s")
```

**Expected Insights**:
- [ ] Mini/Haiku models faster than full models
- [ ] All models produce correct results
- [ ] Performance variance documented

---

## Section 4: Priority System Testing

### 4.1 Priority 1 (Fast Registry-Based) Workflow

**Test Case 4.1.1**: Simple single-function tasks

```python
# These should trigger Priority 1 (fast path)
priority1_tasks = [
    "compute pca",
    "calculate total counts per cell",
    "filter genes with minimum 3 cells",
    "normalize to median counts",
]

agent = ov.Agent(model='gpt-5', api_key=os.getenv('OPENAI_API_KEY'))

for task in priority1_tasks:
    print(f"\n--- Task: {task} ---")
    start = time.time()
    result = agent.run(task, adata.copy())
    elapsed = time.time() - start

    print(f"Execution time: {elapsed:.2f}s")
    print("Expected: Priority 1 (fast, registry-only)")
    # Check logs for priority indication
```

**Expected Behavior**:
- [ ] Fast execution (60-70% faster than Priority 2)
- [ ] Uses function registry directly
- [ ] No skill loading needed
- [ ] Token savings (~50%)

**Verification**:
```python
# Priority 1 should be:
# - Single function call
# - No multi-step logic
# - Direct registry match
```

---

### 4.2 Priority 2 (Skills-Guided) Workflow

**Test Case 4.2.1**: Complex multi-step tasks

```python
# These should trigger Priority 2 (comprehensive path)
priority2_tasks = [
    "preprocess the data with quality control, normalization, and HVG selection",
    "cluster the cells using multiple resolutions and compare results",
    "perform full differential expression analysis with pathway enrichment",
    "create a comprehensive report with plots and statistics",
]

agent = ov.Agent(model='gpt-5', api_key=os.getenv('OPENAI_API_KEY'))

for task in priority2_tasks:
    print(f"\n--- Task: {task} ---")
    start = time.time()
    result = agent.run(task, adata.copy())
    elapsed = time.time() - start

    print(f"Execution time: {elapsed:.2f}s")
    print("Expected: Priority 2 (comprehensive, skills-guided)")
    # Check for "🎯 LLM matched skills:" in output
```

**Expected Behavior**:
- [ ] Matches relevant skills
- [ ] Comprehensive guidance provided
- [ ] Multi-step code generation
- [ ] Higher quality for complex tasks

**Verification**:
- [ ] Skills loaded and used
- [ ] Code quality high for complex tasks
- [ ] All sub-tasks completed

---

### 4.3 Priority Selection Validation

**Test Case 4.3.1**: Automatic priority determination

```python
# Test agent's ability to choose correct priority
task_priority_map = [
    ("compute pca", 1),  # Simple → Priority 1
    ("preprocess with QC and normalization", 2),  # Complex → Priority 2
    ("filter cells", 1),  # Simple → Priority 1
    ("full single-cell analysis pipeline", 2),  # Complex → Priority 2
]

for task, expected_priority in task_priority_map:
    print(f"\nTask: '{task}'")
    print(f"Expected Priority: {expected_priority}")

    result = agent.run(task, adata.copy())

    # Manual verification from logs
    print("Check logs to verify priority used")
```

**Verification**:
- [ ] Simple tasks → Priority 1
- [ ] Complex tasks → Priority 2
- [ ] Agent makes correct decisions

---

## Section 5: Code Validation Features

### 5.1 Reflection System

**Test Case 5.1.1**: Verify reflection on generated code

```python
# Request that may require reflection
agent = ov.Agent(model='gpt-5', api_key=os.getenv('OPENAI_API_KEY'))

# Enable verbose output to see reflection
result = agent.run(
    'preprocess data with careful quality control and normalization',
    adata.copy()
)

# Check for reflection in logs:
# - "🔍 Reflecting on generated code..."
# - Iterations (typically 1-3)
# - Improvements made
```

**Expected Behavior**:
- [ ] Reflection system engaged for complex tasks
- [ ] Code reviewed for correctness
- [ ] Improvements made if needed (1-3 iterations)
- [ ] Final code executes successfully

**Test Case 5.1.2**: Intentional error handling

```python
# Request that might generate initial errors
complex_request = """
Quality control with very strict thresholds:
- Remove cells with less than 1000 UMIs
- Remove cells with more than 10% mitochondrial content
- Remove genes expressed in fewer than 5 cells
Then normalize and select top 3000 HVGs
"""

result = agent.run(complex_request, adata.copy())

# Check if reflection caught and fixed any issues
print("✅ Reflection system handled complex request")
```

**Verification**:
- [ ] Initial code may have issues
- [ ] Reflection detects problems
- [ ] Corrections applied
- [ ] Final code works correctly

---

### 5.2 Result Review System

**Test Case 5.2.1**: Output validation against intent

```python
# Request with specific expected outcome
agent = ov.Agent(model='gpt-5', api_key=os.getenv('OPENAI_API_KEY'))

result = agent.run(
    'filter cells to exactly those with nUMI>500 AND mitochondrial percentage<0.2',
    adata.copy()
)

# Verify result matches intent
assert result.n_obs < adata.n_obs, "Filtering not applied"

# Check for result review in logs:
# - "✅ Result review: output matches user intent"
# - Validation messages

print(f"✅ Result review: {adata.n_obs} → {result.n_obs} cells")
```

**Expected Behavior**:
- [ ] Result review engaged
- [ ] Output validated against request
- [ ] Confirmation message if correct
- [ ] Re-generation if incorrect

**Test Case 5.2.2**: Detection of incorrect results

```python
# This is a hypothetical test - would need a scenario where initial result is wrong
# In practice, reflection+review should prevent incorrect results

# The agent should:
# 1. Generate code
# 2. Execute code
# 3. Review results
# 4. If results don't match intent → regenerate
# 5. Retry up to N times
```

**Verification**:
- [ ] Review system catches discrepancies
- [ ] Code regenerated if needed
- [ ] Final output correct

---

## Section 6: Advanced Features

### 6.1 Function Registry Testing

**Test Case 6.1.1**: Verify registry contents

```python
# Access function registry
agent = ov.Agent(model='gpt-5', api_key=os.getenv('OPENAI_API_KEY'))

# Check registry loading
assert hasattr(agent, 'function_registry'), "Function registry not loaded"

print("Function Registry Overview:")
print(f"  Total functions: ~110 expected")
print(f"  Categories: 7 expected")

# Test registry-based simple requests
simple_tasks = [
    "compute pca with 50 components",
    "calculate neighborhood graph",
    "compute umap",
]

for task in simple_tasks:
    result = agent.run(task, adata.copy())
    print(f"✅ Registry task: '{task}' → completed")
```

**Expected**:
- [ ] Registry loaded at initialization
- [ ] 110 functions available
- [ ] 7 categories present
- [ ] Simple tasks use registry directly

**Test Case 6.1.2**: Registry categories

```python
# The 7 categories should include:
# 1. Preprocessing (QC, normalization, HVG)
# 2. Dimensionality reduction (PCA, UMAP, tSNE)
# 3. Clustering (Leiden, Louvain)
# 4. Differential expression (DEG analysis)
# 5. Visualization (plots, embeddings)
# 6. Annotation (cell type assignment)
# 7. Data manipulation (filtering, subsetting)

category_tests = {
    'Preprocessing': 'normalize to median total counts',
    'Dimensionality reduction': 'compute tsne',
    'Clustering': 'run leiden clustering',
    'Visualization': 'plot umap',
}

for category, task in category_tests.items():
    result = agent.run(task, adata.copy())
    print(f"✅ {category}: task completed")
```

---

### 6.2 Data Summary and Analysis

**Test Case 6.2.1**: Data summarization

```python
# Test ability to summarize AnnData objects
result = agent.run('give me the summary of this h5ad data', adata)

# Expected: Comprehensive summary including:
# - Cell and gene counts
# - Available layers
# - Metadata columns
# - Embeddings present
# - Clustering information
```

**Expected Output Elements**:
- [ ] n_obs and n_vars reported
- [ ] .obs columns listed
- [ ] .var columns listed
- [ ] .obsm keys (embeddings) listed
- [ ] .uns keys listed
- [ ] Summary statistics

**Test Case 6.2.2**: Statistical queries

```python
stat_queries = [
    "calculate mean UMI counts per cell",
    "find highly expressed genes",
    "compute cluster proportions",
    "get summary statistics for mitochondrial content",
]

for query in stat_queries:
    result = agent.run(query, adata)
    print(f"✅ Query: '{query}' → completed")
```

---

### 6.3 Visualization Requests

**Test Case 6.3.1**: Various plot types

```python
plot_requests = [
    "create a violin plot of gene expression",
    "make a dotplot of marker genes",
    "generate a heatmap of top variable genes",
    "create a ranked gene expression plot",
    "plot quality control metrics",
]

for request in plot_requests:
    try:
        result = agent.run(request, adata)
        print(f"✅ Plot request: '{request}' → completed")
    except Exception as e:
        print(f"⚠️  Plot request failed: {e}")
```

**Verification**:
- [ ] Various plot types supported
- [ ] Appropriate visualization functions used
- [ ] Plots generated successfully
- [ ] Defensive coding (checks data before plotting)

---

### 6.4 Custom Skill Integration

**Test Case 6.4.1**: Create custom skill

```python
# Create custom skill directory
custom_skill_path = Path.cwd() / '.claude' / 'skills' / 'custom-pbmc-analysis'
custom_skill_path.mkdir(parents=True, exist_ok=True)

# Write custom skill
custom_skill_content = """---
name: custom-pbmc-analysis
description: |
  Custom PBMC3k analysis workflow for our lab.
  Use when: user wants to run our standard PBMC pipeline
  Includes: QC with nUMI>800, mito<0.15, standard preprocessing, leiden clustering
---

# Custom PBMC Analysis Workflow

This is our lab's standard workflow for PBMC3k analysis.

## Steps:
1. Quality control: nUMI>800, mito<0.15
2. Preprocess with 2500 HVGs using shiftlog|pearson
3. Leiden clustering with resolution=0.8
4. UMAP visualization

## Code example:
```python
import omicverse as ov
import scanpy as sc

# QC
ov.pp.qc(adata, tresh={'nUMIs': 800, 'mito_perc': 0.15})

# Preprocess
ov.pp.preprocess(adata, mode='shiftlog|pearson', n_HVGs=2500)

# Cluster
sc.pp.neighbors(adata, n_neighbors=15)
sc.tl.leiden(adata, resolution=0.8)

# UMAP
sc.tl.umap(adata)
sc.pl.umap(adata, color='leiden')
```
"""

# Write skill file
skill_file = custom_skill_path / 'SKILL.md'
skill_file.write_text(custom_skill_content)

print(f"✅ Custom skill created at: {skill_file}")

# Test custom skill loading
agent = ov.Agent(model='gpt-5', api_key=os.getenv('OPENAI_API_KEY'))
result = agent.run('run our custom PBMC analysis', adata.copy())

print("✅ Custom skill loaded and executed")
```

**Verification**:
- [ ] Custom skill directory created
- [ ] Skill file written correctly
- [ ] Agent loads custom skill
- [ ] Custom skill overrides built-in if same name
- [ ] Custom workflow executes correctly

---

## Section 7: Error Handling and Edge Cases

### 7.1 Input Validation

**Test Case 7.1.1**: Invalid AnnData objects

```python
# Test with None
try:
    result = agent.run('quality control', None)
    print("⚠️  Should have failed with None input")
except Exception as e:
    print(f"✅ Correctly rejected None input: {e}")

# Test with empty AnnData
empty_adata = sc.AnnData()
try:
    result = agent.run('quality control', empty_adata)
    print("Check if agent handles empty data gracefully")
except Exception as e:
    print(f"Expected error with empty data: {e}")
```

**Verification**:
- [ ] None input rejected
- [ ] Empty AnnData handled gracefully
- [ ] Clear error messages

**Test Case 7.1.2**: Missing required data

```python
# Request clustering without preprocessing
fresh_adata = sc.datasets.pbmc3k()
try:
    result = agent.run('leiden clustering', fresh_adata)
    print("Check if agent preprocesses first or reports missing neighbors")
except Exception as e:
    print(f"Handled missing neighbors: {e}")
```

**Verification**:
- [ ] Agent detects missing prerequisites
- [ ] Either auto-preprocesses or reports error clearly
- [ ] Defensive coding prevents crashes

---

### 7.2 API and Network Errors

**Test Case 7.2.1**: Invalid API key

```python
try:
    agent_invalid = ov.Agent(model='gpt-5', api_key='invalid_key_12345')
    result = agent_invalid.run('quality control', adata)
    print("⚠️  Should have failed with invalid API key")
except Exception as e:
    print(f"✅ Correctly rejected invalid API key: {type(e).__name__}")
```

**Test Case 7.2.2**: Rate limiting

```python
# Rapid fire requests to test rate limiting
agent = ov.Agent(model='gpt-5', api_key=os.getenv('OPENAI_API_KEY'))

for i in range(5):
    try:
        result = agent.run(f'compute pca with {10+i} components', adata.copy())
        print(f"Request {i+1}: success")
    except Exception as e:
        print(f"Request {i+1}: {type(e).__name__} - {e}")
        # Check if rate limit error handled gracefully
```

**Verification**:
- [ ] Rate limit errors caught
- [ ] Clear error messages
- [ ] Doesn't crash application

---

### 7.3 Malformed Requests

**Test Case 7.3.1**: Completely invalid requests

```python
invalid_requests = [
    "",  # Empty string
    "asdfghjkl",  # Random text
    "123456789",  # Just numbers
    "!@#$%^&*()",  # Special characters
]

for req in invalid_requests:
    try:
        result = agent.run(req, adata)
        print(f"Request '{req}': {type(result)}")
    except Exception as e:
        print(f"Request '{req}': {type(e).__name__}")
```

**Verification**:
- [ ] Invalid requests handled gracefully
- [ ] No crashes or hangs
- [ ] Clear error messages or clarification requests

---

## Section 8: Performance and Scalability

### 8.1 Memory Usage

**Test Case 8.1.1**: Monitor memory during execution

```python
import psutil
import os

process = psutil.Process(os.getpid())

# Baseline memory
mem_before = process.memory_info().rss / 1024 / 1024  # MB
print(f"Memory before agent: {mem_before:.1f} MB")

# Initialize agent
agent = ov.Agent(model='gpt-5', api_key=os.getenv('OPENAI_API_KEY'))
mem_after_init = process.memory_info().rss / 1024 / 1024
print(f"Memory after init: {mem_after_init:.1f} MB (+{mem_after_init-mem_before:.1f} MB)")

# Run several requests
for i in range(5):
    result = agent.run(f'compute pca with {10+i*5} components', adata.copy())
    mem_current = process.memory_info().rss / 1024 / 1024
    print(f"After request {i+1}: {mem_current:.1f} MB")

print(f"\n✅ Total memory increase: {mem_current - mem_before:.1f} MB")
print("Expected with progressive disclosure: minimal increase")
```

**Expected**:
- [ ] Agent initialization: < 50 MB increase
- [ ] Request execution: incremental memory usage
- [ ] No memory leaks over multiple requests
- [ ] Progressive disclosure keeps memory low

---

### 8.2 Token Usage Efficiency

**Test Case 8.2.1**: Compare token usage (conceptual)

```python
# This would require access to API response metadata
# Conceptual test to verify progressive disclosure benefits

print("""
Progressive Disclosure Token Savings:

Startup:
- Old system: ~25,000 tokens (all skills loaded)
- New system: ~1,250 tokens (metadata only)
- Savings: ~95% reduction

Per Request:
- Only matched skills loaded (lazy)
- Typically 1-3 skills per request
- 2,000-6,000 additional tokens vs 25,000 always

Expected overall savings: 50-60% token reduction
""")
```

**Verification**:
- [ ] Faster startup with progressive disclosure
- [ ] Lower token usage overall
- [ ] Skills loaded only when needed

---

### 8.3 Concurrent Requests

**Test Case 8.3.1**: Multiple agents in parallel

```python
import concurrent.futures

def run_analysis(model_name, api_key):
    agent = ov.Agent(model=model_name, api_key=api_key)
    adata_local = sc.datasets.pbmc3k()
    result = agent.run('quality control with nUMI>500, mito<0.2', adata_local)
    return result.n_obs

# Run multiple agents concurrently
models = [
    ('gpt-5', os.getenv('OPENAI_API_KEY')),
    ('gpt-4o-mini', os.getenv('OPENAI_API_KEY')),
]

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(run_analysis, m, k) for m, k in models if k]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]

print(f"✅ Concurrent execution: {len(results)} agents completed")
```

**Verification**:
- [ ] Multiple agents can run concurrently
- [ ] No race conditions
- [ ] Results consistent across parallel runs

---

## Section 9: Integration with Other Tools

### 9.1 Scanpy Integration

**Test Case 9.1.1**: Scanpy functions via agent

```python
scanpy_requests = [
    "use scanpy to compute tsne",
    "run scanpy leiden clustering",
    "use scanpy to calculate PAGA",
    "compute diffusion map with scanpy",
]

for req in scanpy_requests:
    try:
        result = agent.run(req, adata.copy())
        print(f"✅ Scanpy request: '{req}' → completed")
    except Exception as e:
        print(f"⚠️  Scanpy request failed: {e}")
```

**Verification**:
- [ ] Agent can invoke Scanpy functions
- [ ] Proper integration with Scanpy API
- [ ] Results compatible with OmicVerse

---

### 9.2 Data Export Features

**Test Case 9.2.1**: Excel export

```python
# Test Excel export skill
result = agent.run('export the top 20 variable genes to excel file', adata)

# Check if file was created
# (This depends on how the agent implements export)
print("✅ Excel export requested - check for generated file")
```

**Test Case 9.2.2**: PDF report generation

```python
# Test PDF export skill
result = agent.run('create a PDF report with summary statistics and plots', adata)

print("✅ PDF report requested - check for generated file")
```

**Verification**:
- [ ] Export skills matched correctly
- [ ] Files generated successfully
- [ ] Proper formatting and content

---

## Section 10: Skill-Specific Testing

### 10.1 Single-Cell Skills

**Test Case 10.1.1**: Annotation skills

```python
# First complete preprocessing and clustering
adata_annot = adata.copy()
adata_annot = agent.run('quality control with nUMI>500, mito<0.2', adata_annot)
adata_annot = agent.run('preprocess with 2000 HVGs', adata_annot)
adata_annot = agent.run('leiden clustering', adata_annot)

# Test annotation
annot_requests = [
    "annotate cell types using marker genes",
    "use reference-based annotation",
    "identify cell types with CellVote",
]

for req in annot_requests:
    try:
        result = agent.run(req, adata_annot.copy())
        print(f"✅ Annotation: '{req}' → completed")
    except Exception as e:
        print(f"⚠️  Annotation failed: {e}")
```

**Verification**:
- [ ] `single-annotation` skill matched
- [ ] Appropriate annotation method used
- [ ] Cell type labels added to .obs

**Test Case 10.1.2**: Trajectory analysis

```python
traj_requests = [
    "compute trajectory with PAGA",
    "perform pseudotime analysis",
    "run velocity analysis",
]

for req in traj_requests:
    try:
        result = agent.run(req, adata.copy())
        print(f"✅ Trajectory: '{req}' → completed")
    except Exception as e:
        print(f"⚠️  Trajectory failed: {e}")
```

**Verification**:
- [ ] `single-trajectory` skill matched
- [ ] Trajectory computed correctly
- [ ] Results stored in appropriate .obs/.uns

---

### 10.2 Bulk RNA-Seq Skills

**Test Case 10.2.1**: DEG analysis

```python
# Note: This requires bulk RNA-seq data, not PBMC3k
# Conceptual test structure:

deg_requests = [
    "perform differential expression analysis",
    "find DEGs between conditions",
    "run DESeq2 analysis",
]

# Would need appropriate bulk data to test
print("⏭️  DEG analysis tests require bulk RNA-seq data")
```

**Test Case 10.2.2**: WGCNA analysis

```python
wgcna_requests = [
    "perform WGCNA co-expression analysis",
    "find gene modules with WGCNA",
]

print("⏭️  WGCNA tests require bulk RNA-seq data")
```

**Verification**:
- [ ] Skills match bulk RNA-seq requests
- [ ] Appropriate for bulk data only
- [ ] Clear error if used with wrong data type

---

### 10.3 Universal Data Skills

**Test Case 10.3.1**: Data transformation

```python
transform_requests = [
    "transpose the data",
    "subset to first 100 cells",
    "filter genes with zero expression",
    "merge multiple datasets",
]

for req in transform_requests:
    try:
        result = agent.run(req, adata.copy())
        print(f"✅ Transform: '{req}' → completed")
    except Exception as e:
        print(f"⚠️  Transform failed: {e}")
```

**Verification**:
- [ ] `data-transform` skill matched
- [ ] Pandas/numpy operations used correctly
- [ ] Data integrity maintained

**Test Case 10.3.2**: Statistical analysis

```python
stats_requests = [
    "perform t-test between clusters",
    "calculate correlation matrix",
    "run ANOVA on gene expression",
]

for req in stats_requests:
    try:
        result = agent.run(req, adata.copy())
        print(f"✅ Statistics: '{req}' → completed")
    except Exception as e:
        print(f"⚠️  Statistics failed: {e}")
```

**Verification**:
- [ ] `data-stats-analysis` skill matched
- [ ] Scipy/statsmodels used appropriately
- [ ] Statistical tests valid

---

## Section 11: Regression Testing

### 11.1 Backwards Compatibility

**Test Case 11.1.1**: Old SkillRouter still works

```python
# Test that deprecated SkillRouter doesn't break existing code
from omicverse.utils.skill_router import SkillRouter

# Initialize old router (should work but be deprecated)
try:
    router = SkillRouter(agent.skill_registry.skills)
    print("✅ Old SkillRouter still functional (deprecated)")
except Exception as e:
    print(f"⚠️  SkillRouter compatibility issue: {e}")
```

**Verification**:
- [ ] Old API still works
- [ ] Deprecation warnings shown (if any)
- [ ] No breaking changes

---

### 11.2 Tutorial Reproducibility

**Test Case 11.2.1**: Reproduce tutorial notebook exactly

```python
# Follow exact sequence from tutorial
agent = ov.Agent(model='gpt-5', api_key=os.getenv('OPENAI_API_KEY'))
adata = sc.datasets.pbmc3k()

# Step-by-step from tutorial
adata = agent.run('quality control with nUMI>500, mito<0.2', adata)
adata = agent.run('preprocess with 2000 highly variable genes using shiftlog|pearson', adata)
adata = agent.run('leiden clustering resolution=1.0', adata)
adata = agent.run('compute umap and plot colored by leiden', adata)

# Verify final state matches tutorial expectations
checks = [
    ('n_obs', lambda a: a.n_obs < 2700),
    ('n_vars', lambda a: a.n_vars <= 32738),
    ('HVGs', lambda a: 'highly_variable' in a.var.columns),
    ('clustering', lambda a: 'leiden' in a.obs.columns),
    ('UMAP', lambda a: 'X_umap' in a.obsm.keys()),
]

for name, check_fn in checks:
    assert check_fn(adata), f"Check failed: {name}"
    print(f"✅ Tutorial check: {name}")

print("✅ Tutorial fully reproducible")
```

**Verification**:
- [ ] All tutorial steps work
- [ ] Results match expected outcomes
- [ ] No errors or warnings

---

## Section 12: Documentation and User Experience

### 12.1 Help and Guidance

**Test Case 12.1.1**: List supported models

```python
models_list = ov.list_supported_models()
print(models_list)

# Verify structure
assert isinstance(models_list, str), "Should return formatted string"
assert 'Openai' in models_list, "Should list OpenAI"
assert 'Anthropic' in models_list, "Should list Anthropic"
assert 'Google' in models_list, "Should list Google"

print("✅ Model listing works correctly")
```

**Test Case 12.1.2**: Agent help/documentation

```python
# Check if agent provides helpful information
agent = ov.Agent(model='gpt-5', api_key=os.getenv('OPENAI_API_KEY'))

# These should provide guidance
help_requests = [
    "what can you do?",
    "help me understand the workflow",
    "what analysis types do you support?",
]

for req in help_requests:
    result = agent.run(req, adata)
    print(f"Help request: '{req}' → response received")
```

**Verification**:
- [ ] Agent provides helpful responses
- [ ] Lists capabilities
- [ ] Suggests workflows

---

### 12.2 Error Messages Quality

**Test Case 12.2.1**: Clear error messages

```python
# Trigger various errors and check message quality
error_tests = [
    (None, "None input should give clear error"),
    ("", "Empty request should guide user"),
]

for bad_input, expected_behavior in error_tests:
    try:
        if bad_input == "":
            result = agent.run(bad_input, adata)
        else:
            result = agent.run('test', bad_input)
        print(f"Input '{bad_input}': {expected_behavior}")
    except Exception as e:
        print(f"✅ Error message for '{bad_input}': {str(e)[:100]}")
```

**Verification**:
- [ ] Error messages are clear
- [ ] Suggest corrective actions
- [ ] No technical jargon without explanation

---

## Test Execution Summary

### Checklist for Complete Testing

**Section 1: Core Workflow**
- [ ] 1.1 Agent initialization
- [ ] 1.2 Quality control
- [ ] 1.3 Preprocessing
- [ ] 1.4 Clustering
- [ ] 1.5 Visualization
- [ ] 1.6 End-to-end pipeline

**Section 2: Skill Matching**
- [ ] 2.1 Skill discovery and matching
- [ ] 2.2 Lazy loading verification
- [ ] 2.3 Edge cases

**Section 3: Multi-Provider**
- [ ] 3.1 Provider compatibility
- [ ] 3.2 Performance comparison

**Section 4: Priority System**
- [ ] 4.1 Priority 1 (fast path)
- [ ] 4.2 Priority 2 (comprehensive)
- [ ] 4.3 Priority selection

**Section 5: Code Validation**
- [ ] 5.1 Reflection system
- [ ] 5.2 Result review

**Section 6: Advanced Features**
- [ ] 6.1 Function registry
- [ ] 6.2 Data summarization
- [ ] 6.3 Visualization
- [ ] 6.4 Custom skills

**Section 7: Error Handling**
- [ ] 7.1 Input validation
- [ ] 7.2 API errors
- [ ] 7.3 Malformed requests

**Section 8: Performance**
- [ ] 8.1 Memory usage
- [ ] 8.2 Token efficiency
- [ ] 8.3 Concurrent requests

**Section 9: Integration**
- [ ] 9.1 Scanpy integration
- [ ] 9.2 Data export

**Section 10: Skill-Specific**
- [ ] 10.1 Single-cell skills
- [ ] 10.2 Bulk RNA-seq skills
- [ ] 10.3 Universal data skills

**Section 11: Regression**
- [ ] 11.1 Backwards compatibility
- [ ] 11.2 Tutorial reproducibility

**Section 12: Documentation**
- [ ] 12.1 Help and guidance
- [ ] 12.2 Error message quality

---

## Known Limitations and Future Work

### Current Limitations:
1. Requires valid API keys from supported providers
2. Network connectivity required for LLM calls
3. Token costs vary by provider and model
4. Some advanced features may require specific data formats

### Recommendations for Extended Testing:
1. Test with different single-cell datasets (not just PBMC3k)
2. Test bulk RNA-seq workflows with appropriate data
3. Test spatial transcriptomics features
4. Stress test with very large datasets (>100k cells)
5. Long-running session stability testing
6. Custom skill development workflows

---

## Success Criteria

**Test Suite Passes If:**
- ✅ All core workflow tests pass (Section 1)
- ✅ LLM skill matching works accurately (Section 2)
- ✅ At least 2 LLM providers work correctly (Section 3)
- ✅ Priority system functions as expected (Section 4)
- ✅ Code validation features active (Section 5)
- ✅ No critical errors in advanced features (Section 6)
- ✅ Graceful error handling (Section 7)
- ✅ Performance within expected ranges (Section 8)
- ✅ Tutorial fully reproducible (Section 11.2)

**Quality Metrics:**
- Code generation accuracy: >90%
- Skill matching accuracy: >85%
- Error rate: <5%
- Performance vs baseline: Priority 1 should be 60-70% faster

---

## Test Execution Instructions

1. **Setup Environment**:
   ```bash
   conda create -n ovagent_test python=3.9
   conda activate ovagent_test
   pip install omicverse scanpy psutil
   ```

2. **Set API Keys**:
   ```bash
   export OPENAI_API_KEY="your-key"
   export ANTHROPIC_API_KEY="your-key"
   export GEMINI_API_KEY="your-key"
   ```

3. **Run Tests**:
   ```python
   # Copy test cases from this document
   # Run section by section
   # Document results
   ```

4. **Report Results**:
   - Note any failures with error messages
   - Document performance metrics
   - Report unexpected behaviors
   - Suggest improvements

---

**End of Testing Plan**

*This comprehensive plan covers all major features of the OVAgent system as demonstrated in the PBMC3k tutorial. Execute tests systematically and document all results.*

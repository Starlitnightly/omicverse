# Prerequisite Tracking System Design
## ov.agent Registry Enhancement

**Status:** Approved Plan
**Date:** 2025-11-11
**Branch:** `claude/plan-registry-prerequisite-fix-011CV26QQwK9Lf98uFiM7Vyo`

---

## Problem Statement

The `ov.agent` registry function currently forgets to track and enforce prerequisites when executing user requests. For example:

**Current Behavior (BROKEN):**
```python
User: "Run PCA on my data"
Agent generates: ov.pp.pca(adata, n_pcs=50)
Result: KeyError: 'scaled' ❌
```

The agent generates code that fails because:
- PCA requires `adata.layers['scaled']` which is created by `ov.pp.scale()`
- The registry doesn't track that `pca` requires `scale` to run first
- The agent has no way to know the current state of the data
- No prerequisite chain is enforced or suggested

---

## Root Cause Analysis

### 1. Registry Metadata Gaps

**Location:** `/home/user/omicverse/omicverse/utils/registry.py:31-37`

The `@register_function()` decorator currently only supports:
- ✅ `aliases` - Alternative names for the function
- ✅ `category` - Function category
- ✅ `description` - What the function does
- ✅ `examples` - Usage examples
- ✅ `related` - Related functions (informational only, not enforced)

**Missing critical metadata:**
- ❌ `prerequisites` - What functions must run first
- ❌ `requires` - What data layers/structures must exist
- ❌ `produces` - What data layers/structures the function creates
- ❌ `auto_fix` - Whether missing prerequisites can be auto-inserted

### 2. Agent Instruction Gaps

**Location:** `/home/user/omicverse/omicverse/utils/smart_agent.py:306-369`

The system prompt sent to the LLM:
- ✅ Lists available functions with descriptions
- ❌ Doesn't instruct to check data state before function calls
- ❌ Doesn't provide prerequisite chains
- ❌ Doesn't tell LLM to validate required data structures exist

### 3. No Runtime Validation

- ❌ No mechanism to inspect `adata` state before generating code
- ❌ No validation that required layers/structures exist
- ❌ No automatic insertion of missing prerequisite steps

---

## Solution Architecture

### Three-Layer Prerequisite System

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: REGISTRY METADATA                                  │
│  - Declarative prerequisites in @register_function()         │
│  - Track: function deps, layer deps, structure deps          │
│  Files: omicverse/utils/registry.py                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2: DATA STATE INSPECTOR                               │
│  - Runtime inspection of adata object                        │
│  - Check: existing layers, obsm, varm, neighbors, etc.       │
│  Files: omicverse/utils/smart_agent.py (new class)           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: SMART CODE GENERATOR (Enhanced Agent Prompt)       │
│  - Receives: target function + prerequisites + data state    │
│  - Outputs: Complete code with missing steps auto-inserted   │
│  Files: omicverse/utils/smart_agent.py                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Registry Metadata Enhancement

### 1.1 Extended Decorator Signature

**File:** `/home/user/omicverse/omicverse/utils/registry.py`

**New Parameters for `@register_function()`:**

```python
@register_function(
    # Existing parameters
    aliases: List[str],
    category: str,
    description: str,
    examples: Optional[List[str]] = None,
    related: Optional[List[str]] = None,

    # NEW: Prerequisite tracking
    prerequisites: Optional[Dict[str, List[str]]] = None,
    # Format: {
    #     'functions': ['scale'],  # Must run these functions first
    #     'optional_functions': ['qc', 'preprocess']  # Recommended
    # }

    # NEW: Data structure requirements
    requires: Optional[Dict[str, List[str]]] = None,
    # Format: {
    #     'layers': ['scaled'],  # Required data layers
    #     'obsm': ['X_pca'],     # Required obsm keys
    #     'obsp': ['connectivities'],  # Required obsp keys
    #     'uns': ['neighbors'],  # Required uns keys
    #     'var': ['highly_variable_features'],  # Recommended var columns
    #     'obs': ['n_genes']     # Recommended obs columns
    # }

    # NEW: What this function produces
    produces: Optional[Dict[str, List[str]]] = None,
    # Format: {
    #     'layers': ['scaled'],
    #     'obsm': ['X_pca'],
    #     'varm': ['PCs'],
    #     'uns': ['pca']
    # }

    # NEW: Auto-fix configuration
    auto_fix: str = 'none'
    # Options:
    #   'auto' - Simple case, can auto-insert prerequisites (e.g., leiden needs neighbors)
    #   'escalate' - Complex case, suggest workflow instead (e.g., PCA needs full preprocessing)
    #   'none' - Just warn, don't auto-fix
)
```

### 1.2 Example: PCA Function Registration

```python
@register_function(
    aliases=["pca", "PCA", "主成分分析"],
    category="preprocessing",
    description="Perform Principal Component Analysis for dimensionality reduction",

    prerequisites={
        'functions': ['scale'],  # Must run scale() first
        'optional_functions': ['qc', 'preprocess'],  # Recommended but not required
    },

    requires={
        'layers': ['scaled'],  # Requires adata.layers['scaled']
        'var': ['highly_variable_features'],  # Recommended
    },

    produces={
        'obsm': ['X_pca'],  # Creates adata.obsm['X_pca']
        'varm': ['PCs'],    # Creates adata.varm['PCs']
        'uns': ['pca'],     # Creates adata.uns['pca']
    },

    auto_fix='escalate',  # Too complex, suggest ov.pp.preprocess() instead

    examples=["ov.pp.pca(adata, n_pcs=50)"],
    related=["umap", "tsne", "neighbors"]
)
def pca(adata, n_pcs=50, layer='scaled', **kwargs):
    """Perform PCA on scaled data."""
    ...
```

### 1.3 Example: Leiden Clustering Registration

```python
@register_function(
    aliases=["leiden", "clustering", "聚类"],
    category="clustering",
    description="Leiden community detection clustering algorithm",

    prerequisites={
        'functions': ['neighbors'],  # Must have neighbor graph
        'optional_functions': ['pca', 'umap'],  # Recommended dimensionality reduction
    },

    requires={
        'uns': ['neighbors'],  # Requires adata.uns['neighbors']
        'obsp': ['connectivities', 'distances'],  # Requires neighbor graphs
    },

    produces={
        'obs': ['leiden'],  # Creates adata.obs['leiden']
    },

    auto_fix='auto',  # Simple - just needs neighbors(), can auto-insert

    examples=["ov.single.leiden(adata, resolution=0.5)"],
    related=["louvain", "neighbors", "umap"]
)
def leiden(adata, resolution=1.0, **kwargs):
    """Run Leiden clustering."""
    ...
```

### 1.4 New FunctionRegistry Methods

**File:** `/home/user/omicverse/omicverse/utils/registry.py`

```python
class FunctionRegistry:
    """Extended with prerequisite tracking."""

    def get_prerequisites(self, func_name: str) -> Dict[str, Any]:
        """
        Get full prerequisite information for a function.

        Parameters
        ----------
        func_name : str
            Function name or alias

        Returns
        -------
        Dict with keys:
            - required_functions: List[str] - Must run these first
            - optional_functions: List[str] - Recommended to run first
            - requires: Dict - Required data structures
            - produces: Dict - What the function creates
            - auto_fix: str - Auto-fix strategy
        """

    def get_prerequisite_chain(self, func_name: str) -> List[str]:
        """
        Get ordered list of functions to run for prerequisites.

        Parameters
        ----------
        func_name : str
            Target function name

        Returns
        -------
        List[str]
            Ordered prerequisite chain, e.g., ['qc', 'scale', 'pca']

        Examples
        --------
        >>> registry.get_prerequisite_chain('pca')
        ['scale', 'pca']

        >>> registry.get_prerequisite_chain('leiden')
        ['neighbors', 'leiden']
        """

    def check_prerequisites(self, func_name: str, adata) -> Dict[str, Any]:
        """
        Validate if all prerequisites are satisfied for an AnnData object.

        Parameters
        ----------
        func_name : str
            Function to check
        adata : AnnData
            Data object to validate

        Returns
        -------
        Dict with keys:
            - satisfied: bool - All requirements met
            - missing_functions: List[str] - Functions not run yet
            - missing_structures: List[str] - Missing data layers/structures
            - recommendation: str - What to do next
        """

    def format_prerequisites_for_llm(self, func_name: str) -> str:
        """
        Format prerequisite info for LLM consumption in system prompt.

        Returns
        -------
        str
            Formatted text with prerequisite chain, requirements, and guidance

        Example output:
        '''
        Function: ov.pp.pca()
        Prerequisites:
          - Requires adata.layers['scaled'] (created by ov.pp.scale())
          - Recommended: Run QC and preprocessing first
        Prerequisite Chain: qc → preprocess → scale → pca
        Auto-fix: ESCALATE (complex, suggest using ov.pp.preprocess())
        '''
        """
```

### 1.5 Priority Functions for Metadata Annotation

**Phase 1 - Critical Functions (Week 1):**
1. `ov.pp.qc()` - Quality control
2. `ov.pp.preprocess()` - Complete preprocessing pipeline
3. `ov.pp.scale()` - Scaling
4. `ov.pp.pca()` - PCA
5. `ov.pp.neighbors()` - Neighbor graph computation
6. `ov.single.leiden()` - Leiden clustering
7. `ov.pp.umap()` - UMAP embedding

**Phase 2 - Common Functions (Week 2):**
8. `ov.pp.normalize()` - Normalization
9. `ov.single.tsne()` - t-SNE
10. `ov.single.louvain()` - Louvain clustering
11. `ov.pp.highly_variable_genes()` - HVG selection
12. `ov.single.rank_genes_groups()` - DEG analysis

---

## Layer 2: Data State Inspector

### 2.1 DataStateInspector Class

**File:** `/home/user/omicverse/omicverse/utils/smart_agent.py` (new class)

**Purpose:** Runtime inspection of AnnData objects to determine current preprocessing state.

```python
class DataStateInspector:
    """
    Inspects AnnData objects to determine what preprocessing steps
    have been completed and what data structures exist.
    """

    def inspect_adata(self, adata) -> Dict[str, Any]:
        """
        Comprehensive inspection of AnnData object state.

        Parameters
        ----------
        adata : AnnData
            Object to inspect

        Returns
        -------
        Dict with keys:
            shape: Tuple[int, int] - (n_cells, n_genes)
            layers: List[str] - Available data layers
            obsm: List[str] - Available embeddings
            obsp: List[str] - Available pairwise arrays (neighbor graphs)
            uns: List[str] - Available unstructured data
            var_columns: List[str] - Available gene metadata columns
            obs_columns: List[str] - Available cell metadata columns

            # High-level status inference
            preprocessing_complete: bool - Has normalized/scaled data
            dimensionality_reduction_complete: bool - Has PCA/UMAP/tSNE
            clustering_complete: bool - Has leiden/louvain clusters
            neighbors_computed: bool - Has neighbor graph
            qc_complete: bool - Has QC metrics computed

        Examples
        --------
        >>> inspector = DataStateInspector()
        >>> state = inspector.inspect_adata(adata)
        >>> state['preprocessing_complete']
        True
        >>> state['layers']
        ['counts', 'scaled']
        """

    def check_function_compatible(
        self,
        func_name: str,
        adata,
        registry: FunctionRegistry
    ) -> Dict[str, Any]:
        """
        Check if a function can be run on the current adata state.

        Parameters
        ----------
        func_name : str
            Function to check compatibility for
        adata : AnnData
            Current data object
        registry : FunctionRegistry
            Registry to look up requirements

        Returns
        -------
        Dict with keys:
            compatible: bool - Can run without errors
            missing_layers: List[str] - Required layers that don't exist
            missing_structures: List[str] - Required structures that don't exist
            recommendation: str - What to do next
            auto_fixable: bool - Can prerequisites be auto-inserted
            suggested_workflow: str - Workflow function to use instead

        Examples
        --------
        >>> result = inspector.check_function_compatible('pca', raw_adata, registry)
        >>> result
        {
            'compatible': False,
            'missing_layers': ['scaled'],
            'missing_structures': [],
            'recommendation': 'Run ov.pp.scale() first',
            'auto_fixable': False,
            'suggested_workflow': 'ov.pp.preprocess()'
        }
        """

    def generate_state_summary_for_llm(self, adata) -> str:
        """
        Generate human-readable summary for LLM system prompt.

        Parameters
        ----------
        adata : AnnData
            Data to summarize

        Returns
        -------
        str
            Formatted summary text

        Example output:
        '''
        Current AnnData State:
        - Shape: 3000 cells × 2000 genes
        - Raw data: YES (counts layer available)
        - Preprocessing: PARTIAL (has counts, missing scaled layer)
        - Dimensionality reduction: NO (no PCA, UMAP, or tSNE)
        - Clustering: NO (no leiden or louvain)
        - Neighbor graph: NO

        Recommended next steps: Run preprocessing (QC → normalize → scale)
        '''
        """

    def infer_preprocessing_status(self, adata) -> Dict[str, bool]:
        """
        Infer high-level preprocessing status from data structures.

        Logic:
        - QC complete: has 'n_genes', 'n_counts', 'pct_counts_mt' in obs
        - Preprocessing complete: has 'scaled' or 'lognorm' layer
        - Dimensionality reduction: has 'X_pca', 'X_umap', or 'X_tsne' in obsm
        - Neighbors computed: has 'neighbors' in uns and 'connectivities' in obsp
        - Clustering complete: has 'leiden' or 'louvain' in obs

        Returns
        -------
        Dict[str, bool]
            Status flags for each major stage
        """
```

### 2.2 Integration with Registry

```python
# In smart_agent.py

class SmartAgent:
    """Extended with data state awareness."""

    def __init__(self, ...):
        self.inspector = DataStateInspector()
        ...

    def _prepare_execution_context(self, user_request: str, adata) -> str:
        """
        Prepare enhanced context for LLM including:
        1. User request
        2. Current data state
        3. Available functions
        4. Prerequisite information for relevant functions

        This context is injected into the system prompt before code generation.
        """
        # Extract target function from user request
        target_func = self._extract_target_function(user_request)

        # Inspect current data state
        data_state = self.inspector.inspect_adata(adata)
        state_summary = self.inspector.generate_state_summary_for_llm(adata)

        # Check prerequisites
        if target_func:
            prereq_info = _global_registry.format_prerequisites_for_llm(target_func)
            compatibility = self.inspector.check_function_compatible(
                target_func, adata, _global_registry
            )

        # Build enhanced prompt
        context = f"""
        ## User Request
        {user_request}

        ## Current Data State
        {state_summary}

        ## Target Function Analysis
        {prereq_info if target_func else 'No specific function detected'}

        ## Compatibility Check
        {self._format_compatibility(compatibility) if target_func else 'N/A'}

        ## Instructions
        Generate code that:
        1. Checks if prerequisites are satisfied
        2. Auto-inserts missing prerequisites if auto_fix='auto'
        3. Suggests workflow if auto_fix='escalate'
        4. Executes the user's requested operation
        """

        return context
```

---

## Layer 3: Enhanced Agent Prompt System

### 3.1 Modified System Prompt

**File:** `/home/user/omicverse/omicverse/utils/smart_agent.py:306-369`

**New Instruction Sections:**

#### Section A: Data State Validation

```markdown
## CRITICAL: Data State Validation

Before generating code for ANY function, you MUST:

1. **Check data state** - Verify the adata object has required structures
2. **Validate prerequisites** - Ensure prerequisite functions have been run
3. **Auto-insert if needed** - Add missing prerequisites in correct order

### Examples

❌ WRONG - Will fail:
```python
# User: "Run PCA"
ov.pp.pca(adata, n_pcs=50)  # KeyError: 'scaled'
```

✅ CORRECT - Handles prerequisites:
```python
# User: "Run PCA"
# Check: does adata have scaled layer?
if 'scaled' not in adata.layers:
    print("Data needs scaling before PCA")
    adata = ov.pp.scale(adata)
ov.pp.pca(adata, n_pcs=50)
print(f"PCA complete: {adata.obsm['X_pca'].shape}")
```

✅ BETTER - Use workflow for complex cases:
```python
# User: "Run PCA" (on completely raw data)
# Detected: missing QC, normalization, scaling (3+ steps)
# Recommendation: Use integrated workflow instead
print("Running complete preprocessing pipeline before PCA...")
adata = ov.pp.preprocess(adata, mode='shiftlog|pearson', n_HVGs=2000)
ov.pp.pca(adata, n_pcs=50)
print(f"Complete! PCA shape: {adata.obsm['X_pca'].shape}")
```
```

#### Section B: Prerequisite Chain Reference

This section is **dynamically generated** from registry metadata:

```markdown
## Function Prerequisite Chains

When a function is requested, follow its prerequisite chain:

### ov.pp.pca()
- **Prerequisites:** scale → pca
- **Requires:** adata.layers['scaled']
- **Full recommended chain:** qc → preprocess → scale → pca
- **Auto-fix:** ESCALATE (suggest ov.pp.preprocess() instead)
- **Reason:** PCA needs 3+ missing steps on raw data

### ov.single.leiden()
- **Prerequisites:** neighbors → leiden
- **Requires:** adata.uns['neighbors'], adata.obsp['connectivities']
- **Full recommended chain:** pca → neighbors → leiden
- **Auto-fix:** AUTO (can insert ov.pp.neighbors() automatically)
- **Reason:** Only 1 simple step missing

### ov.pp.neighbors()
- **Prerequisites:** (dimensionality reduction: pca OR umap)
- **Requires:** adata.obsm['X_pca'] OR adata.obsm['X_umap']
- **Full recommended chain:** scale → pca → neighbors
- **Auto-fix:** AUTO (can insert ov.pp.pca() if missing)
- **Reason:** Only 1 simple step missing

### ov.pp.umap()
- **Prerequisites:** neighbors → umap
- **Requires:** adata.uns['neighbors']
- **Full recommended chain:** pca → neighbors → umap
- **Auto-fix:** AUTO
- **Reason:** Simple chain

[... for all registered functions ...]
```

#### Section C: Workflow Escalation Rules

```markdown
## Complex Prerequisite Handling

### Rule: 3+ Missing Prerequisites → Recommend Workflow

If a function needs 3 or more missing prerequisites, recommend an integrated workflow function instead of manual steps.

### Example: PCA on Raw Data

**Detected Missing:**
1. QC (quality control)
2. Normalization
3. Scaling

**Total:** 3 steps → ESCALATE to workflow

**Generated Code:**
```python
# Your data needs preprocessing before PCA.
# Instead of running each step manually, use the integrated workflow:
print("Running complete preprocessing pipeline...")
adata = ov.pp.preprocess(
    adata,
    mode='shiftlog|pearson',  # Log-normalize + Pearson residuals for HVGs
    n_HVGs=2000               # Select 2000 highly variable genes
)
print(f"Preprocessing complete: {adata.shape[0]} cells, {adata.shape[1]} genes")

# Now run PCA
ov.pp.pca(adata, n_pcs=50)
print(f"PCA complete: {adata.obsm['X_pca'].shape[1]} principal components")
print(f"Variance explained (first 5 PCs): {adata.uns['pca']['variance_ratio'][:5].sum():.1%}")
```

**Workflow:** This runs QC → normalize → HVG selection → scale → PCA in one call.
```

### Rule: 1-2 Missing Prerequisites → Auto-Insert

**Example: Leiden without Neighbors**

**Detected Missing:**
1. Neighbors (only 1 step)

**Total:** 1 step → AUTO-FIX

**Generated Code:**
```python
# Leiden clustering requires neighbor graph
if 'neighbors' not in adata.uns:
    print("Computing neighbor graph first...")
    ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50)

# Now run leiden
ov.single.leiden(adata, resolution=0.5)
print(f"Clustering complete: {adata.obs['leiden'].nunique()} clusters found")
```
```

---

## Implementation Workflow

### Phase 1: Registry Metadata (Week 1)

**Tasks:**
1. ✅ Extend `register_function()` signature with new parameters
2. ✅ Update `FunctionRegistry.register()` to store prerequisite metadata
3. ✅ Add validation for metadata format
4. ✅ Implement `get_prerequisites()`, `get_prerequisite_chain()`, `format_prerequisites_for_llm()`
5. ✅ Update 7 priority functions with full prerequisite metadata:
   - `qc`, `preprocess`, `scale`, `pca`, `neighbors`, `leiden`, `umap`

**Deliverables:**
- Updated `/home/user/omicverse/omicverse/utils/registry.py`
- Updated preprocessing functions in `/home/user/omicverse/omicverse/pp/_preprocess.py`
- Updated clustering functions in `/home/user/omicverse/omicverse/single/`
- Unit tests for prerequisite chain generation

### Phase 2: Data State Inspector (Week 2)

**Tasks:**
1. ✅ Create `DataStateInspector` class in `smart_agent.py`
2. ✅ Implement `inspect_adata()` - structure detection
3. ✅ Implement `check_function_compatible()` - compatibility checking
4. ✅ Implement `generate_state_summary_for_llm()` - LLM-friendly summaries
5. ✅ Implement `infer_preprocessing_status()` - high-level status detection
6. ✅ Test on PBMC3k dataset at various preprocessing stages

**Deliverables:**
- `DataStateInspector` class in `/home/user/omicverse/omicverse/utils/smart_agent.py`
- Unit tests for state detection accuracy
- Integration tests with real datasets

### Phase 3: Agent Integration (Week 3)

**Tasks:**
1. ✅ Modify `SmartAgent._setup_agent()` to inject prerequisite chains
2. ✅ Add data state validation to system prompt
3. ✅ Add workflow escalation guidance
4. ✅ Implement `_prepare_execution_context()` for enhanced prompting
5. ✅ Add helper functions accessible to agent:
   - `_check_state()` - Check current data state
   - `_get_prerequisites(func_name)` - Query prerequisites
6. ✅ Create execution flow:
   ```
   User request → Extract target function →
   Check prerequisites → Inspect adata state →
   Generate prerequisite chain OR escalate to workflow →
   Inject enhanced prompt → LLM generates complete code
   ```

**Deliverables:**
- Updated `SmartAgent` class with prerequisite awareness
- Enhanced system prompt generation
- Integration with `DataStateInspector` and `FunctionRegistry`

### Phase 4: Testing & Refinement (Week 4)

**Tasks:**
1. ✅ Unit tests:
   - Prerequisite chain generation for all functions
   - Data state detection accuracy (raw, QC'd, preprocessed, clustered)
   - Auto-fix vs escalate logic decision making
2. ✅ Integration tests:
   - "Run PCA" on raw data → suggests preprocess()
   - "Leiden clustering" with PCA but no neighbors → auto-inserts neighbors()
   - "UMAP visualization" on raw data → runs full chain
3. ✅ Edge case testing:
   - GPU mode vs CPU mode compatibility
   - Different preprocessing workflows (Seurat vs Pearson)
   - Custom layer names
   - Partial preprocessing (user ran some steps manually)
4. ✅ User acceptance testing with real workflows

**Deliverables:**
- Comprehensive test suite
- Documentation updates
- Example notebooks demonstrating prerequisite handling

---

## Example: Complete Flow

### User Input
```python
agent = ov.utils.SmartAgent(...)
adata = sc.read_h5ad('raw_pbmc3k.h5ad')  # Raw, unprocessed data

result = agent.execute("Perform PCA analysis", adata)
```

### Internal Flow

**Step 1: Request received**
- User wants: "Perform PCA analysis"
- Data object: raw PBMC3k (3000 cells × 32738 genes)

**Step 2: DataStateInspector checks adata**
```python
{
    'shape': (3000, 32738),
    'layers': [],  # Only has .X, no layers
    'obsm': [],
    'obsp': [],
    'uns': {},
    'preprocessing_complete': False,
    'dimensionality_reduction_complete': False,
    'qc_complete': False
}
```

**Step 3: Registry lookup for `pca`**
```python
{
    'prerequisites': {
        'functions': ['scale'],
        'optional_functions': ['qc', 'preprocess']
    },
    'requires': {
        'layers': ['scaled']
    },
    'auto_fix': 'escalate'  # Too complex
}
```

**Step 4: Compatibility check**
```python
{
    'compatible': False,
    'missing_layers': ['scaled'],
    'missing_structures': [],
    'recommendation': 'Run ov.pp.preprocess() workflow',
    'auto_fixable': False,  # 3+ steps missing
    'suggested_workflow': 'ov.pp.preprocess()'
}
```

**Step 5: Enhanced prompt sent to LLM**
```markdown
User Request: "Perform PCA analysis"

Current Data State:
- Shape: 3000 cells × 32738 genes
- Raw data: YES (X matrix present)
- Preprocessing: NO (no layers, no QC metrics)
- Dimensionality reduction: NO

Target Function: ov.pp.pca()
- Requires: adata.layers['scaled']
- Prerequisites: qc → preprocess → scale → pca (4 steps)
- Missing: All preprocessing steps

Recommendation: ESCALATE to workflow
- Reason: 3+ missing steps
- Suggested: Use ov.pp.preprocess() for integrated pipeline

Generate code that handles this intelligently.
```

**Step 6: LLM generates smart code**
```python
import omicverse as ov

# Data is completely raw - needs full preprocessing before PCA
print("Your data requires preprocessing before PCA can be performed.")
print("Running integrated preprocessing pipeline...")
print("Steps: QC → Normalization → HVG Selection → Scaling")

# Use integrated workflow (more efficient than manual steps)
adata = ov.pp.preprocess(
    adata,
    mode='shiftlog|pearson',  # Log-normalization + Pearson residuals for HVGs
    target_sum=50e4,          # Target total counts
    n_HVGs=2000               # Select 2000 highly variable genes
)
print(f"✓ Preprocessing complete: {adata.shape[0]} cells × {adata.shape[1]} genes")

# Now run PCA
print("Running PCA...")
ov.pp.pca(adata, n_pcs=50)

# Report results
print(f"✓ PCA complete!")
print(f"  - PCA matrix shape: {adata.obsm['X_pca'].shape}")
print(f"  - Variance explained (first 5 PCs): {adata.uns['pca']['variance_ratio'][:5].sum():.1%}")
print(f"  - Variance explained (first 10 PCs): {adata.uns['pca']['variance_ratio'][:10].sum():.1%}")
```

**Step 7: Code executes successfully**
```
Output:
Your data requires preprocessing before PCA can be performed.
Running integrated preprocessing pipeline...
Steps: QC → Normalization → HVG Selection → Scaling
✓ Preprocessing complete: 3000 cells × 2000 genes
Running PCA...
✓ PCA complete!
  - PCA matrix shape: (3000, 50)
  - Variance explained (first 5 PCs): 42.3%
  - Variance explained (first 10 PCs): 58.7%
```

**Result:** ✅ User gets working code with proper preprocessing, no errors!

---

## Benefits Summary

| **Problem** | **Solution** |
|-------------|--------------|
| ❌ Functions fail with cryptic errors (KeyError: 'scaled') | ✅ Prerequisites auto-detected, proper error messages |
| ❌ Users don't know function execution order | ✅ Prerequisite chains documented and enforced |
| ❌ Agent generates broken code | ✅ Agent receives data state + prerequisites → generates working code |
| ❌ Manual prerequisite hardcoding in agent prompts | ✅ Declarative metadata in decorators, self-documenting |
| ❌ Hard to maintain when functions change | ✅ Metadata lives with function definition, single source of truth |
| ❌ Poor user experience | ✅ Helpful messages: "Running preprocessing first..." |
| ❌ No runtime validation | ✅ Data state inspector validates before execution |
| ❌ Complex workflows require expert knowledge | ✅ Agent suggests workflows for complex cases |

---

## Alternative: Minimal Viable Solution

If the three-layer system is too complex for initial implementation, here's a simpler approach:

### Option 2: Static Prerequisite Lookup Table

**File:** `/home/user/omicverse/omicverse/utils/prerequisites.py` (new file)

```python
"""
Static prerequisite lookup table.
Simple alternative to full three-layer system.
"""

PREREQUISITE_CHAINS = {
    'pca': {
        'requires_layers': ['scaled'],
        'prerequisite_functions': ['scale'],
        'full_workflow': 'preprocess',
        'auto_insert': False,
        'message': 'PCA requires scaled data. Run ov.pp.scale() or ov.pp.preprocess() first.'
    },

    'leiden': {
        'requires_uns': ['neighbors'],
        'prerequisite_functions': ['neighbors'],
        'auto_insert': True,
        'message': 'Leiden requires neighbor graph. Running ov.pp.neighbors() first...'
    },

    'umap': {
        'requires_uns': ['neighbors'],
        'prerequisite_functions': ['neighbors'],
        'auto_insert': True,
        'message': 'UMAP requires neighbor graph. Running ov.pp.neighbors() first...'
    },

    'neighbors': {
        'requires_obsm': ['X_pca'],
        'prerequisite_functions': ['pca'],
        'auto_insert': True,
        'message': 'Neighbors requires dimensionality reduction. Running ov.pp.pca() first...'
    },

    # ... for all functions
}

def get_prerequisite_info(func_name: str) -> dict:
    """Get prerequisite info for a function."""
    return PREREQUISITE_CHAINS.get(func_name, {})

def format_for_agent_prompt() -> str:
    """Format all prerequisites for agent system prompt."""
    output = "## Function Prerequisites\n\n"
    for func, info in PREREQUISITE_CHAINS.items():
        output += f"**{func}**: {info['message']}\n"
    return output
```

**Integration:** Inject this table into agent system prompt as reference guide.

**Pros:**
- ✅ Simple to implement (1-2 days vs 4 weeks)
- ✅ No complex runtime inspection needed
- ✅ Easy to understand and maintain

**Cons:**
- ❌ Static (doesn't adapt to actual adata state)
- ❌ Less intelligent (always suggests full chain even if partially complete)
- ❌ Requires manual updates when functions change
- ❌ No automatic validation

---

## Recommendation

**Implement the full three-layer system** because:

1. **Scalability** - As OmicVerse grows, automatic detection >> manual tables
2. **User Experience** - Smart detection of actual data state > generic warnings
3. **Maintainability** - Metadata in decorators > separate lookup tables
4. **Future-proof** - Works for any new function, no updates needed
5. **Intelligence** - Adapts to partial preprocessing, custom workflows, edge cases

**Timeline:** 4 weeks for full implementation + testing

**Fallback:** If timeline is critical, implement Option 2 (lookup table) as v1.0, then migrate to full system in v2.0.

---

## Success Criteria

The implementation will be considered successful when:

1. ✅ **No more KeyError failures** - All prerequisite-related errors caught before execution
2. ✅ **Smart code generation** - Agent generates working code even for raw data
3. ✅ **Helpful messages** - Clear guidance on what's missing and what's being run
4. ✅ **Auto-fix simple cases** - Leiden without neighbors → auto-inserts neighbors()
5. ✅ **Escalate complex cases** - PCA on raw data → suggests preprocess() workflow
6. ✅ **Extensible** - New functions automatically get prerequisite support via decorators
7. ✅ **Tested** - 100% pass rate on integration tests with PBMC3k dataset
8. ✅ **Documented** - Users understand prerequisite system via examples and docs

---

## Next Steps

1. ✅ **Plan Approved** - This document represents the approved design
2. ⏸️ **Awaiting Implementation Signal** - Ready to begin coding when requested
3. 📋 **Implementation Order:**
   - Phase 1: Registry metadata enhancement
   - Phase 2: Data state inspector
   - Phase 3: Agent integration
   - Phase 4: Testing & refinement

---

## References

- **Registry Implementation:** `/home/user/omicverse/omicverse/utils/registry.py`
- **Agent Implementation:** `/home/user/omicverse/omicverse/utils/smart_agent.py`
- **Preprocessing Functions:** `/home/user/omicverse/omicverse/pp/_preprocess.py`
- **Test Data:** PBMC3k dataset (standard benchmark)

---

**Document Version:** 1.0
**Last Updated:** 2025-11-11
**Status:** ✅ APPROVED - Ready for Implementation

# Layer 2: DataStateInspector Implementation Plan

**Date**: 2025-11-11
**Status**: Planning Phase
**Prerequisite**: Layer 1 Complete (36/36 functions with metadata)
**Branch**: `claude/plan-registry-prerequisite-fix-011CV26QQwK9Lf98uFiM7Vyo`

---

## Executive Summary

Layer 2 implements **runtime validation** of prerequisite chains by inspecting actual AnnData object state. The DataStateInspector class will use Layer 1 metadata to verify that required data structures and prerequisite functions have been executed before allowing a function to run.

### Key Objectives

1. **Runtime Validation**: Check AnnData state against function requirements
2. **Prerequisite Verification**: Detect which prerequisite functions have been executed
3. **Intelligent Suggestions**: Provide actionable guidance when requirements are missing
4. **LLM-Ready Format**: Structure validation results for Layer 3 LLM integration
5. **User-Friendly Errors**: Generate clear, helpful error messages

---

## Architecture Overview

### Component Hierarchy

```
Layer 3: LLM Integration (Future)
    ↓
Layer 2: DataStateInspector (This Plan)
    ↓
Layer 1: Registry Metadata (Complete ✅)
```

### Core Components

```
DataStateInspector/
├── inspector.py          # Main inspector class
├── validators.py         # Data structure validators
├── prerequisite_checker.py   # Function execution detection
├── suggestion_engine.py  # Generate fix suggestions
└── formatters.py        # LLM-compatible output formatting
```

---

## Design Principles

### 1. Conservative Detection
- **Principle**: Only mark prerequisites as satisfied when clear evidence exists
- **Rationale**: False positives are more dangerous than false negatives
- **Example**: Require specific markers in `adata.uns['preprocessing']` rather than inferring from data shape

### 2. Evidence-Based Validation
- **Principle**: Use concrete markers rather than heuristics
- **Rationale**: Ensure accuracy and reproducibility
- **Example**: Check for `adata.uns['log1p']['base']` rather than analyzing X distribution

### 3. Graceful Degradation
- **Principle**: Provide useful information even with incomplete metadata
- **Rationale**: Support partially-tracked workflows
- **Example**: Suggest likely missing steps even if exact prerequisite chain is uncertain

### 4. LLM-First Design
- **Principle**: Structure all outputs for easy LLM consumption
- **Rationale**: Enable Layer 3 intelligent assistance
- **Example**: Use structured dicts with clear keys like `missing_prerequisites`, `suggested_fixes`

---

## Component 1: Core Inspector Class

### Class: `DataStateInspector`

**Purpose**: Central orchestrator for all validation operations

**Initialization**:
```python
class DataStateInspector:
    def __init__(self, adata: AnnData, registry: FunctionRegistry):
        """Initialize inspector with AnnData object and function registry.

        Args:
            adata: The AnnData object to inspect
            registry: Function registry with Layer 1 metadata
        """
        self.adata = adata
        self.registry = registry
        self.validators = DataValidators(adata)
        self.prereq_checker = PrerequisiteChecker(adata, registry)
        self.suggestion_engine = SuggestionEngine(registry)
```

**Core Methods**:

#### 1. `validate_prerequisites(function_name: str) -> ValidationResult`
```python
def validate_prerequisites(self, function_name: str) -> ValidationResult:
    """Validate all prerequisites for a given function.

    Checks:
    1. Required data structures exist
    2. Required prerequisite functions have been executed
    3. Optional prerequisites and their impact

    Returns:
        ValidationResult with status and detailed findings
    """
```

#### 2. `check_data_requirements(function_name: str) -> DataCheckResult`
```python
def check_data_requirements(self, function_name: str) -> DataCheckResult:
    """Check if required data structures exist in AnnData.

    Validates:
    - adata.obs columns
    - adata.obsm keys
    - adata.obsp keys
    - adata.uns keys
    - adata.layers keys
    - adata.var columns
    - adata.varm keys

    Returns:
        DataCheckResult with missing/present structures
    """
```

#### 3. `get_execution_state() -> ExecutionState`
```python
def get_execution_state(self) -> ExecutionState:
    """Get comprehensive state of executed functions.

    Analyzes AnnData to determine which registered functions
    have been executed based on their output signatures.

    Returns:
        ExecutionState mapping function names to confidence levels
    """
```

#### 4. `suggest_fixes(function_name: str) -> List[Suggestion]`
```python
def suggest_fixes(self, function_name: str) -> List[Suggestion]:
    """Generate actionable suggestions to fix missing prerequisites.

    Returns ordered list of suggestions with:
    - Code snippet to execute
    - Explanation of what it does
    - Estimated execution time
    - Priority level

    Returns:
        List of Suggestion objects in priority order
    """
```

### Usage Example

```python
# Initialize inspector
from omicverse.utils.registry import get_registry
from omicverse.utils.inspector import DataStateInspector

registry = get_registry()
inspector = DataStateInspector(adata, registry)

# Validate before calling a function
result = inspector.validate_prerequisites('leiden')

if result.is_valid:
    # Proceed with function
    ov.pp.leiden(adata)
else:
    # Show user what's missing
    print(f"Cannot run leiden: {result.message}")
    print("\nRequired prerequisites:")
    for prereq in result.missing_prerequisites:
        print(f"  - {prereq}")

    print("\nSuggested fixes:")
    for suggestion in inspector.suggest_fixes('leiden'):
        print(f"\n{suggestion.priority}: {suggestion.description}")
        print(f"Code: {suggestion.code}")
```

---

## Component 2: Data Structure Validators

### Class: `DataValidators`

**Purpose**: Validate presence and structure of data components

**Methods**:

#### 1. `check_obs(required_columns: List[str]) -> ObsCheckResult`
```python
def check_obs(self, required_columns: List[str]) -> ObsCheckResult:
    """Check if required columns exist in adata.obs.

    Validates:
    - Column existence
    - Data type appropriateness
    - Non-null values

    Returns:
        ObsCheckResult with missing columns and issues
    """
```

#### 2. `check_obsm(required_keys: List[str]) -> ObsmCheckResult`
```python
def check_obsm(self, required_keys: List[str]) -> ObsmCheckResult:
    """Check if required embeddings exist in adata.obsm.

    Validates:
    - Key existence
    - Array shape consistency
    - Dimensionality appropriateness

    Returns:
        ObsmCheckResult with missing keys and shape issues
    """
```

#### 3. `check_obsp(required_keys: List[str]) -> ObspCheckResult`
```python
def check_obsp(self, required_keys: List[str]) -> ObspCheckResult:
    """Check if required pairwise arrays exist in adata.obsp.

    Validates:
    - Key existence (connectivities, distances)
    - Sparse matrix structure
    - Symmetry properties

    Returns:
        ObspCheckResult with missing keys and structural issues
    """
```

#### 4. `check_uns(required_keys: List[str]) -> UnsCheckResult`
```python
def check_uns(self, required_keys: List[str]) -> UnsCheckResult:
    """Check if required unstructured data exists in adata.uns.

    Validates:
    - Key existence
    - Nested structure correctness
    - Metadata completeness

    Returns:
        UnsCheckResult with missing keys and structure
    """
```

#### 5. `check_layers(required_keys: List[str]) -> LayersCheckResult`
```python
def check_layers(self, required_keys: List[str]) -> LayersCheckResult:
    """Check if required layers exist in adata.layers.

    Validates:
    - Layer existence
    - Shape matching X
    - Data type appropriateness

    Returns:
        LayersCheckResult with missing layers
    """
```

### Validation Strategies by Data Type

#### adata.obs
- **Simple check**: Column name exists
- **Deep check**: Validate categorical vs numeric, check for NaN
- **Example**: `leiden` should be categorical with reasonable number of categories

#### adata.obsm
- **Simple check**: Key exists
- **Deep check**: Shape is (n_obs, n_features), reasonable dimensionality
- **Example**: `X_pca` typically (n_obs, 20-100), `X_umap` is (n_obs, 2)

#### adata.obsp
- **Simple check**: Key exists
- **Deep check**: Sparse matrix, shape is (n_obs, n_obs), symmetric for distances
- **Example**: `connectivities` and `distances` should be paired

#### adata.uns
- **Simple check**: Key exists
- **Deep check**: Validate nested structure, check for expected metadata
- **Example**: `neighbors` should contain `params`, `connectivities_key`, etc.

#### adata.layers
- **Simple check**: Key exists
- **Deep check**: Shape matches X, appropriate data type
- **Example**: `counts` should have integer-like values

---

## Component 3: Prerequisite Execution Checker

### Class: `PrerequisiteChecker`

**Purpose**: Detect which functions have been executed on the AnnData object

**Core Challenge**: Inferring execution history from current state

**Detection Strategies**:

#### Strategy 1: Metadata Markers (High Confidence)
Use explicit markers left by functions:
```python
# Many OmicVerse functions add metadata
adata.uns['preprocessing'] = {
    'qc': True,
    'normalize': True,
    'log1p': {'base': 2}
}
```

**Pros**: Definitive evidence
**Cons**: Requires functions to add markers (not all do)

#### Strategy 2: Output Signature Matching (Medium Confidence)
Match data structures to known function outputs:
```python
def detect_pca(self) -> float:
    """Detect if PCA has been run.

    Evidence:
    - adata.obsm['X_pca'] exists
    - adata.uns['pca'] exists with variance_ratio
    - Shape suggests PCA dimensionality

    Returns:
        Confidence score 0.0-1.0
    """
    confidence = 0.0

    if 'X_pca' in self.adata.obsm:
        confidence += 0.5

    if 'pca' in self.adata.uns:
        if 'variance_ratio' in self.adata.uns['pca']:
            confidence += 0.3
        if 'params' in self.adata.uns['pca']:
            confidence += 0.2

    return min(confidence, 1.0)
```

**Pros**: Works even without explicit markers
**Cons**: Can produce false positives

#### Strategy 3: Data Distribution Analysis (Low Confidence)
Analyze data properties to infer transformations:
```python
def detect_log_transformation(self) -> float:
    """Detect if log transformation has been applied.

    Evidence:
    - X values are mostly < 10
    - Distribution looks log-normal
    - No extremely large outliers

    Returns:
        Confidence score 0.0-1.0
    """
    # Analyze X distribution
    # This is less reliable and should be lowest priority
```

**Pros**: Can work with any data
**Cons**: Unreliable, many false positives/negatives

### Implementation Priority

1. **Phase 1**: Implement metadata marker detection (high confidence)
2. **Phase 2**: Implement output signature matching (medium confidence)
3. **Phase 3**: Consider distribution analysis (low priority, optional)

### Methods

#### 1. `check_function_executed(function_name: str) -> ExecutionCheckResult`
```python
def check_function_executed(self, function_name: str) -> ExecutionCheckResult:
    """Check if a specific function has been executed.

    Uses multiple detection strategies and returns
    confidence-weighted result.

    Returns:
        ExecutionCheckResult with confidence and evidence
    """
```

#### 2. `get_execution_chain() -> List[str]`
```python
def get_execution_chain(self) -> List[str]:
    """Get ordered list of detected function executions.

    Attempts to reconstruct the analysis workflow
    based on detected function signatures.

    Returns:
        List of function names in likely execution order
    """
```

#### 3. `find_missing_prerequisites(function_name: str) -> List[str]`
```python
def find_missing_prerequisites(self, function_name: str) -> List[str]:
    """Find which required prerequisites are missing.

    Uses function metadata to check all required
    prerequisites and returns those not detected.

    Returns:
        List of missing prerequisite function names
    """
```

---

## Component 4: Suggestion Engine

### Class: `SuggestionEngine`

**Purpose**: Generate actionable fix suggestions for missing prerequisites

**Core Features**:
1. Ordered suggestions (priority-based)
2. Code snippets ready to execute
3. Explanations for each step
4. Alternative approaches when applicable

### Methods

#### 1. `generate_fix_suggestions(function_name: str, missing_prereqs: List[str]) -> List[Suggestion]`
```python
def generate_fix_suggestions(
    self,
    function_name: str,
    missing_prereqs: List[str]
) -> List[Suggestion]:
    """Generate prioritized list of fix suggestions.

    For each missing prerequisite:
    1. Generate code to execute it
    2. Explain what it does
    3. Estimate execution time
    4. Assign priority level

    Returns:
        List of Suggestion objects ordered by priority
    """
```

#### 2. `generate_minimal_workflow(target_function: str) -> WorkflowPlan`
```python
def generate_minimal_workflow(self, target_function: str) -> WorkflowPlan:
    """Generate minimal workflow to reach target function.

    Creates shortest valid path from current state
    to target function, considering:
    - Required prerequisites
    - Current execution state
    - Function dependencies

    Returns:
        WorkflowPlan with ordered steps and code
    """
```

#### 3. `suggest_alternatives(function_name: str) -> List[Alternative]`
```python
def suggest_alternatives(self, function_name: str) -> List[Alternative]:
    """Suggest alternative functions when prerequisites missing.

    If prerequisites are complex or expensive, suggest
    alternative functions that don't require them.

    Returns:
        List of Alternative objects with trade-offs
    """
```

### Suggestion Types

#### Type 1: Direct Fix (auto_fix='auto')
```python
Suggestion(
    priority='HIGH',
    type='direct_fix',
    description='Run neighbors() to compute graph',
    code='sc.pp.neighbors(adata, n_neighbors=15)',
    explanation='Computes KNN graph required for leiden clustering',
    estimated_time='5-30 seconds',
    impact='Required for clustering'
)
```

#### Type 2: Workflow Guidance (auto_fix='escalate')
```python
Suggestion(
    priority='HIGH',
    type='workflow_guidance',
    description='Complete preprocessing pipeline required',
    code='''
# Step 1: Quality control
ov.pp.qc(adata)

# Step 2: Normalization
ov.pp.preprocess(adata, mode='shiftlog|pearson')

# Step 3: PCA
ov.pp.scale(adata)
ov.pp.pca(adata)

# Step 4: Neighbors
ov.pp.neighbors(adata)
''',
    explanation='Tangram requires fully preprocessed single-cell reference',
    estimated_time='1-5 minutes',
    impact='Required for spatial deconvolution'
)
```

#### Type 3: Alternative Approach
```python
Suggestion(
    priority='MEDIUM',
    type='alternative',
    description='Consider using pre-built reference',
    code='# Load pre-processed reference from tutorials',
    explanation='Avoid lengthy preprocessing by using validated reference',
    estimated_time='Immediate',
    impact='Alternative to preprocessing workflow'
)
```

---

## Component 5: Output Formatters

### Class: `LLMFormatter`

**Purpose**: Format validation results for LLM consumption (Layer 3)

**Key Requirements**:
1. Structured JSON/dict output
2. Clear action items
3. Context for decision making
4. Human-readable fallback

### Methods

#### 1. `format_validation_result(result: ValidationResult) -> dict`
```python
def format_validation_result(self, result: ValidationResult) -> dict:
    """Format validation result for LLM consumption.

    Returns structured dict with:
    - validation_status: bool
    - missing_prerequisites: List[str]
    - missing_data_structures: Dict[str, List[str]]
    - suggested_fixes: List[dict]
    - context: Dict[str, Any]
    """
```

#### 2. `format_for_agent(validation: ValidationResult, user_intent: str) -> str`
```python
def format_for_agent(
    self,
    validation: ValidationResult,
    user_intent: str
) -> str:
    """Format validation for LLM agent with natural language.

    Creates clear, actionable prompt for LLM that includes:
    - Current situation
    - Missing requirements
    - Available fixes
    - User's original intent

    Returns:
        Natural language prompt for LLM
    """
```

### Example LLM-Ready Output

```json
{
  "function": "leiden",
  "validation_status": false,
  "current_state": {
    "executed_functions": ["qc", "preprocess", "scale", "pca"],
    "missing_functions": ["neighbors"]
  },
  "missing_requirements": {
    "data_structures": {
      "obsp": ["connectivities", "distances"]
    },
    "prerequisites": {
      "required": ["neighbors"],
      "optional": []
    }
  },
  "suggested_fixes": [
    {
      "priority": "HIGH",
      "type": "direct_fix",
      "action": "run_neighbors",
      "code": "sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)",
      "explanation": "Compute KNN graph required for Leiden clustering",
      "estimated_time_seconds": 10,
      "auto_executable": true
    }
  ],
  "llm_prompt": "The user wants to run Leiden clustering, but the required neighbors graph is missing. The preprocessing steps (qc, normalize, scale, PCA) have been completed. To proceed, run: sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)"
}
```

---

## Data Structures

### ValidationResult

```python
@dataclass
class ValidationResult:
    """Result of prerequisite validation."""

    function_name: str
    is_valid: bool
    message: str

    # Missing components
    missing_prerequisites: List[str]
    missing_data_structures: Dict[str, List[str]]  # category -> keys

    # Current state
    executed_functions: List[str]
    confidence_scores: Dict[str, float]

    # Suggestions
    suggestions: List[Suggestion]
    minimal_workflow: Optional[WorkflowPlan]

    # Metadata
    validation_timestamp: datetime
    adata_hash: str  # For caching
```

### Suggestion

```python
@dataclass
class Suggestion:
    """A suggested fix for missing prerequisites."""

    priority: Literal['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
    type: Literal['direct_fix', 'workflow_guidance', 'alternative', 'optimization']

    description: str
    code: str
    explanation: str

    estimated_time: str  # Human-readable
    estimated_time_seconds: Optional[int]

    prerequisites: List[str]  # Prerequisites for this suggestion itself
    impact: str  # What this enables

    auto_executable: bool  # Can be run automatically?
```

### ExecutionState

```python
@dataclass
class ExecutionState:
    """State of detected function executions."""

    detected_functions: Dict[str, ExecutionEvidence]
    execution_order: List[str]  # Best-guess order

    confidence_summary: Dict[str, float]  # function -> confidence

    metadata: Dict[str, Any]  # Raw metadata found in adata.uns
```

### ExecutionEvidence

```python
@dataclass
class ExecutionEvidence:
    """Evidence that a function was executed."""

    function_name: str
    confidence: float  # 0.0 to 1.0

    evidence_type: Literal['metadata_marker', 'output_signature', 'distribution_analysis']
    evidence_details: Dict[str, Any]

    detected_outputs: List[str]  # Which outputs were found
    timestamp: Optional[datetime]  # If available in metadata
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)

**Goals**:
- Implement basic DataStateInspector class
- Implement DataValidators with simple checks
- Basic prerequisite detection using metadata markers

**Deliverables**:
```
omicverse/utils/inspector/
├── __init__.py
├── inspector.py          # DataStateInspector class
├── validators.py         # DataValidators class
├── data_structures.py    # Result classes
└── tests/
    ├── test_inspector.py
    └── test_validators.py
```

**Success Criteria**:
- Can validate simple data requirements (obs, obsm keys)
- Can detect functions with explicit metadata markers
- Basic validation passing/failing works

### Phase 2: Prerequisite Detection (Week 2)

**Goals**:
- Implement PrerequisiteChecker with multiple strategies
- Add output signature matching
- Build execution chain reconstruction

**Deliverables**:
```
omicverse/utils/inspector/
├── prerequisite_checker.py   # PrerequisiteChecker class
├── detection_strategies.py   # Individual detection methods
└── tests/
    └── test_prerequisite_checker.py
```

**Success Criteria**:
- Can detect 80%+ of preprocessing functions
- Confidence scores are calibrated and accurate
- Execution chain reconstruction works for standard workflows

### Phase 3: Suggestion Engine (Week 3)

**Goals**:
- Implement SuggestionEngine
- Generate code snippets for fixes
- Build workflow planning

**Deliverables**:
```
omicverse/utils/inspector/
├── suggestion_engine.py    # SuggestionEngine class
├── workflow_planner.py     # WorkflowPlan generation
└── tests/
    └── test_suggestion_engine.py
```

**Success Criteria**:
- Generates valid Python code for all suggestions
- Prioritization is sensible and helpful
- Workflow plans are minimal and correct

### Phase 4: LLM Integration Prep (Week 4)

**Goals**:
- Implement LLMFormatter
- Create structured output formats
- Build natural language generation

**Deliverables**:
```
omicverse/utils/inspector/
├── formatters.py          # LLMFormatter class
├── templates/            # Prompt templates
└── tests/
    └── test_formatters.py
```

**Success Criteria**:
- JSON output is valid and complete
- Natural language prompts are clear
- Ready for Layer 3 LLM integration

### Phase 5: Integration & Testing (Week 5)

**Goals**:
- Integrate with existing registry
- Comprehensive testing on real workflows
- Documentation and examples

**Deliverables**:
```
Documentation:
├── docs/inspector_guide.md
├── docs/api_reference.md
└── examples/
    ├── basic_validation.ipynb
    ├── workflow_debugging.ipynb
    └── llm_integration_demo.ipynb

Tests:
└── tests/integration/
    ├── test_full_workflows.py
    ├── test_spatial_workflows.py
    └── test_edge_cases.py
```

**Success Criteria**:
- All unit tests passing
- Integration tests cover 20+ real workflows
- Documentation is complete and clear
- Ready for production use

---

## Testing Strategy

### Unit Tests

Test each component in isolation:

```python
# test_validators.py
def test_check_obsm_missing_key():
    adata = create_test_adata()
    validator = DataValidators(adata)
    result = validator.check_obsm(['X_pca', 'X_umap'])
    assert 'X_umap' in result.missing_keys

def test_check_obsp_neighbors():
    adata = create_test_adata_with_neighbors()
    validator = DataValidators(adata)
    result = validator.check_obsp(['connectivities', 'distances'])
    assert result.is_valid
```

### Integration Tests

Test full validation workflows:

```python
# test_full_workflows.py
def test_standard_pipeline_validation():
    """Test validation through standard preprocessing pipeline."""
    adata = sc.datasets.pbmc3k()
    inspector = DataStateInspector(adata, registry)

    # Should fail before preprocessing
    result = inspector.validate_prerequisites('leiden')
    assert not result.is_valid

    # Run preprocessing
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)
    sc.pp.scale(adata)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)

    # Should pass after prerequisites
    result = inspector.validate_prerequisites('leiden')
    assert result.is_valid
```

### Workflow Tests

Test real-world analysis workflows:

```python
# test_spatial_workflows.py
def test_stagate_workflow():
    """Test STAGATE spatial clustering workflow validation."""
    adata = load_spatial_data()
    inspector = DataStateInspector(adata, registry)

    # Check what's needed for STAGATE
    result = inspector.validate_prerequisites('pySTAGATE')

    # Should only need spatial coordinates
    assert 'spatial' in result.missing_data_structures.get('obsm', [])
```

---

## Integration Points

### With Layer 1 (Registry)

```python
# Access Layer 1 metadata
function_meta = registry.get_function('leiden')
prerequisites = function_meta['prerequisites']
requires = function_meta['requires']
produces = function_meta['produces']
auto_fix = function_meta['auto_fix']
```

### With Layer 3 (LLM - Future)

```python
# Provide structured output for LLM
validation = inspector.validate_prerequisites('leiden')
llm_input = formatter.format_for_agent(validation, user_intent="cluster cells")

# LLM can then:
# 1. Understand what's missing
# 2. Decide on best fix
# 3. Execute suggested code
# 4. Verify success
```

### With OmicVerse Functions

```python
# Optional: Add validation hooks to functions
@register_function(...)
def leiden(adata, **kwargs):
    # Optional validation before execution
    if VALIDATION_ENABLED:
        inspector = get_inspector(adata)
        result = inspector.validate_prerequisites('leiden')
        if not result.is_valid:
            raise PrerequisiteError(result)

    # Actual function logic
    ...
```

---

## Configuration & Settings

### Global Configuration

```python
# omicverse/utils/inspector/config.py

class InspectorConfig:
    """Configuration for DataStateInspector."""

    # Validation behavior
    STRICT_MODE: bool = False  # Raise errors vs warnings
    CONFIDENCE_THRESHOLD: float = 0.7  # Min confidence for detection

    # Detection strategies
    ENABLE_METADATA_DETECTION: bool = True
    ENABLE_SIGNATURE_DETECTION: bool = True
    ENABLE_DISTRIBUTION_DETECTION: bool = False  # Experimental

    # Suggestion generation
    MAX_SUGGESTIONS: int = 5
    INCLUDE_ALTERNATIVES: bool = True
    GENERATE_WORKFLOWS: bool = True

    # Performance
    CACHE_VALIDATION_RESULTS: bool = True
    CACHE_TTL_SECONDS: int = 300
```

### User-Facing API

```python
# Enable validation globally
import omicverse as ov
ov.settings.enable_validation = True
ov.settings.validation_mode = 'strict'  # or 'warn'

# Validate specific function
ov.utils.validate('leiden', adata)

# Get current state
state = ov.utils.get_execution_state(adata)
print(f"Detected functions: {state.detected_functions}")
```

---

## Error Handling

### Custom Exceptions

```python
class PrerequisiteError(Exception):
    """Raised when prerequisites are not satisfied."""

    def __init__(self, validation_result: ValidationResult):
        self.result = validation_result
        super().__init__(validation_result.message)

class DataStructureError(Exception):
    """Raised when required data structures are missing."""

    def __init__(self, missing_structures: Dict[str, List[str]]):
        self.missing_structures = missing_structures
        message = self._format_message()
        super().__init__(message)
```

### Error Messages

Focus on actionable guidance:

❌ **Bad**: "Missing prerequisite: neighbors"

✅ **Good**:
```
Cannot run leiden clustering: neighbors graph not found.

Required prerequisite: sc.pp.neighbors()

To fix, run:
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)

This will compute the KNN graph needed for clustering.
Estimated time: 10-30 seconds
```

---

## Performance Considerations

### Caching Strategy

```python
class ValidationCache:
    """Cache validation results to avoid repeated checks."""

    def __init__(self, ttl_seconds: int = 300):
        self._cache = {}
        self._ttl = ttl_seconds

    def get(self, adata_hash: str, function_name: str) -> Optional[ValidationResult]:
        """Get cached validation result."""
        key = (adata_hash, function_name)
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                return result
        return None

    def set(self, adata_hash: str, function_name: str, result: ValidationResult):
        """Cache validation result."""
        key = (adata_hash, function_name)
        self._cache[key] = (result, time.time())
```

### Optimization Strategies

1. **Lazy Detection**: Only check prerequisites when needed
2. **Batch Validation**: Validate multiple functions at once
3. **Incremental Updates**: Track changes to avoid full re-validation
4. **Async Validation**: For expensive checks, run asynchronously

---

## Documentation Requirements

### API Documentation

Complete docstrings for all public methods:
```python
def validate_prerequisites(self, function_name: str) -> ValidationResult:
    """Validate all prerequisites for a given function.

    This method checks both data requirements and prerequisite
    function execution to determine if the target function can
    be safely executed.

    Args:
        function_name: Name of the function to validate

    Returns:
        ValidationResult containing:
            - is_valid: Whether all prerequisites are satisfied
            - missing_prerequisites: List of missing functions
            - missing_data_structures: Dict of missing data
            - suggestions: List of fix suggestions

    Example:
        >>> inspector = DataStateInspector(adata, registry)
        >>> result = inspector.validate_prerequisites('leiden')
        >>> if not result.is_valid:
        ...     print(result.suggestions[0].code)

    See Also:
        - check_data_requirements()
        - suggest_fixes()
    """
```

### User Guides

1. **Getting Started**: Basic validation usage
2. **Advanced Features**: Custom validators, caching
3. **Troubleshooting**: Common issues and solutions
4. **API Reference**: Complete function documentation

### Examples

Jupyter notebooks demonstrating:
- Basic validation workflow
- Debugging failed validations
- Custom validation rules
- Integration with LLM agents

---

## Success Metrics

### Quantitative Metrics

1. **Detection Accuracy**: ≥90% for common preprocessing functions
2. **False Positive Rate**: <5% for all detection strategies
3. **Performance**: Validation completes in <100ms for typical AnnData
4. **Coverage**: Supports all 36 Layer 1 functions

### Qualitative Metrics

1. **User Experience**: Error messages are clear and actionable
2. **Integration**: Works seamlessly with existing workflows
3. **Extensibility**: Easy to add new detection strategies
4. **Documentation**: Complete and easy to understand

---

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Inaccurate detection | Medium | High | Multiple detection strategies, conservative thresholds |
| Performance issues | Low | Medium | Caching, lazy evaluation, profiling |
| API instability | Low | High | Thorough testing, clear versioning |
| False positives | Medium | Medium | Conservative detection, confidence scores |

### Process Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Scope creep | Medium | Medium | Clear phase boundaries, MVP focus |
| Integration complexity | Low | High | Incremental integration, extensive testing |
| Documentation lag | Medium | Low | Write docs alongside code |

---

## Next Steps After Layer 2

### Layer 3: LLM Integration

Once Layer 2 is complete:

1. **LLM Agent Development**
   - Natural language workflow composition
   - Intelligent prerequisite resolution
   - Context-aware suggestions

2. **Automated Workflow Generation**
   - Generate complete workflows from high-level goals
   - Optimize execution order
   - Handle errors gracefully

3. **Interactive Assistance**
   - Real-time validation during analysis
   - Proactive suggestions
   - Learning from user patterns

---

## Appendix A: Detection Heuristics

### Common Function Signatures

```python
DETECTION_SIGNATURES = {
    'qc': {
        'obs_columns': ['n_genes_by_counts', 'pct_counts_mt'],
        'uns_keys': ['qc'],
        'confidence_weights': {'obs': 0.4, 'uns': 0.6}
    },
    'preprocess': {
        'uns_keys': ['log1p', 'hvg'],
        'layers': ['counts'],
        'confidence_weights': {'uns': 0.5, 'layers': 0.5}
    },
    'pca': {
        'obsm_keys': ['X_pca'],
        'varm_keys': ['PCs'],
        'uns_keys': ['pca'],
        'confidence_weights': {'obsm': 0.4, 'varm': 0.3, 'uns': 0.3}
    },
    'neighbors': {
        'obsp_keys': ['connectivities', 'distances'],
        'uns_keys': ['neighbors'],
        'confidence_weights': {'obsp': 0.6, 'uns': 0.4}
    },
    # ... add all 36 functions
}
```

### Detection Priority Order

1. **Explicit metadata** (`adata.uns['preprocessing']`)
2. **Output signatures** (obsm, obsp keys)
3. **Data transformations** (shape, distribution)

---

## Appendix B: Example Workflows

### Example 1: Standard Single-Cell Analysis

```python
import scanpy as sc
import omicverse as ov

# Load data
adata = sc.datasets.pbmc3k()

# Enable validation
ov.settings.enable_validation = True

# Create inspector
inspector = ov.utils.DataStateInspector(adata)

# Try to cluster (should fail)
try:
    sc.tl.leiden(adata)
except ov.utils.PrerequisiteError as e:
    print(e.result.message)
    print("\nSuggested fix:")
    print(e.result.suggestions[0].code)

# Follow suggestions
for suggestion in inspector.suggest_fixes('leiden'):
    print(f"\nExecuting: {suggestion.description}")
    exec(suggestion.code)

# Verify and proceed
result = inspector.validate_prerequisites('leiden')
if result.is_valid:
    sc.tl.leiden(adata)
```

### Example 2: Spatial Analysis

```python
import scanpy as sc
import omicverse as ov

# Load spatial data
adata = sc.datasets.visium_sge()

# Check STAGATE requirements
inspector = ov.utils.DataStateInspector(adata)
result = inspector.validate_prerequisites('pySTAGATE')

if not result.is_valid:
    print("Missing requirements:")
    for category, keys in result.missing_data_structures.items():
        print(f"  {category}: {keys}")

    # Get workflow to satisfy requirements
    workflow = inspector.generate_minimal_workflow('pySTAGATE')
    print(f"\nRequired steps: {len(workflow.steps)}")
    for step in workflow.steps:
        print(f"  {step.order}. {step.description}")
```

---

## Conclusion

Layer 2 (DataStateInspector) provides **runtime validation** that bridges the gap between Layer 1's declarative metadata and Layer 3's intelligent LLM assistance. By implementing conservative detection strategies, clear error messages, and actionable suggestions, we create a robust foundation for automated workflow guidance.

**Key Success Factors**:
1. ✅ Conservative, evidence-based detection
2. ✅ Clear, actionable error messages
3. ✅ Structured output for LLM integration
4. ✅ Comprehensive testing and validation
5. ✅ Seamless integration with existing code

**Timeline**: 5 weeks to production-ready Layer 2

**Prerequisites**: Layer 1 complete (✅ Done)

**Next**: Begin Phase 1 implementation or await approval

---

**Document Version**: 1.0
**Last Updated**: 2025-11-11
**Status**: Awaiting Review
**Author**: Claude (Anthropic)

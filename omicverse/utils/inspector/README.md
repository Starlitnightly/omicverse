# DataStateInspector - Layer 2 Runtime Validation

**Status**: Phase 1 Complete (Core Infrastructure)
**Version**: 0.1.0

---

## Overview

The DataStateInspector provides runtime validation of prerequisite chains by inspecting AnnData object state. It verifies that required data structures exist before allowing function execution.

### Phase 1 Deliverables (Complete ✅)

1. **Data Structures** (`data_structures.py`)
   - ValidationResult, DataCheckResult
   - ObsCheckResult, ObsmCheckResult, ObspCheckResult, UnsCheckResult, LayersCheckResult
   - Suggestion, ExecutionEvidence, ExecutionState

2. **Data Validators** (`validators.py`)
   - Check obs, obsm, obsp, uns, layers requirements
   - Comprehensive validation with issue detection
   - Shape and type validation

3. **Inspector Core** (`inspector.py`)
   - DataStateInspector main class
   - validate_prerequisites() method
   - Integration with Layer 1 registry

4. **Tests** (`tests/`)
   - Unit tests for all validators
   - Integration test examples

---

## Quick Start

### Basic Usage

```python
from omicverse.utils.inspector import DataStateInspector
from omicverse.utils.registry import get_registry

# Create inspector
inspector = DataStateInspector(adata, get_registry())

# Validate prerequisites
result = inspector.validate_prerequisites('leiden')

if result.is_valid:
    # Safe to proceed
    sc.tl.leiden(adata)
else:
    # Show what's missing
    print(result.message)
    print(f"Missing data structures: {result.missing_data_structures}")

    # Show suggestions
    for suggestion in result.suggestions:
        print(f"\n[{suggestion.priority}] {suggestion.description}")
        print(f"Code: {suggestion.code}")
```

### Check Specific Data Requirements

```python
from omicverse.utils.inspector import DataValidators

# Create validator
validator = DataValidators(adata)

# Check specific structures
obsm_result = validator.check_obsm(['X_pca', 'X_umap'])
print(f"Valid: {obsm_result.is_valid}")
print(f"Missing: {obsm_result.missing_keys}")

obsp_result = validator.check_obsp(['connectivities', 'distances'])
print(f"Valid: {obsp_result.is_valid}")
print(f"Missing: {obsp_result.missing_keys}")

# Check all requirements at once
requires = {
    'obsm': ['X_pca'],
    'obsp': ['connectivities', 'distances'],
    'obs': ['leiden']
}
result = validator.check_all_requirements(requires)
print(f"All valid: {result.is_valid}")
print(f"All missing: {result.all_missing_structures}")
```

---

## Components

### 1. DataStateInspector

Main orchestrator for validation.

**Methods**:
- `validate_prerequisites(function_name)` - Validate all requirements
- `check_data_requirements(function_name)` - Check only data structures
- `get_validation_summary(function_name)` - Get LLM-friendly dict

**Example**:
```python
inspector = DataStateInspector(adata, registry)
result = inspector.validate_prerequisites('leiden')
```

### 2. DataValidators

Validates data structure presence and correctness.

**Methods**:
- `check_obs(required_columns)` - Check obs columns
- `check_obsm(required_keys)` - Check obsm embeddings
- `check_obsp(required_keys)` - Check obsp pairwise arrays
- `check_uns(required_keys)` - Check uns unstructured data
- `check_layers(required_keys)` - Check layers
- `check_all_requirements(requires_dict)` - Check everything

**Example**:
```python
validator = DataValidators(adata)
result = validator.check_obsm(['X_pca', 'X_umap'])
if not result.is_valid:
    print(f"Missing: {result.missing_keys}")
```

### 3. Data Structures

Result classes for validation:

- **ValidationResult**: Complete validation result with suggestions
- **DataCheckResult**: Aggregate of all data checks
- **ObsCheckResult**: Result of obs validation
- **ObsmCheckResult**: Result of obsm validation
- **ObspCheckResult**: Result of obsp validation
- **UnsCheckResult**: Result of uns validation
- **LayersCheckResult**: Result of layers validation
- **Suggestion**: Actionable fix suggestion with code

---

## Examples

### Example 1: Validating Leiden Clustering

```python
from omicverse.utils.inspector import DataStateInspector

# Leiden requires neighbors graph
inspector = DataStateInspector(adata, registry)
result = inspector.validate_prerequisites('leiden')

if not result.is_valid:
    # Output:
    # Missing data structures: {'obsp': ['connectivities', 'distances']}
    # Suggestions:
    #   [HIGH] Compute neighbor graph
    #   Code: sc.pp.neighbors(adata, n_neighbors=15)

    # Execute suggested fix
    for suggestion in result.suggestions:
        if suggestion.auto_executable:
            exec(suggestion.code)
```

### Example 2: Checking Multiple Requirements

```python
from omicverse.utils.inspector import DataValidators

validator = DataValidators(adata)

# Check if ready for UMAP
umap_requires = {
    'obsm': ['X_pca'],
    'obsp': ['connectivities', 'distances']
}
result = validator.check_all_requirements(umap_requires)

if result.is_valid:
    print("✓ Ready for UMAP")
else:
    print(f"✗ Missing: {result.all_missing_structures}")
```

### Example 3: LLM-Ready Output

```python
inspector = DataStateInspector(adata, registry)
summary = inspector.get_validation_summary('leiden')

# Returns dict:
{
    'function': 'leiden',
    'valid': False,
    'message': 'Missing requirements for leiden',
    'missing_data_structures': {
        'obsp': ['connectivities', 'distances']
    },
    'suggestions': [
        {
            'priority': 'HIGH',
            'type': 'direct_fix',
            'description': 'Compute neighbor graph',
            'code': 'sc.pp.neighbors(adata, n_neighbors=15)',
            'auto_executable': False
        }
    ]
}
```

---

## Architecture

```
DataStateInspector
    ├── validate_prerequisites()
    │   ├── check_data_requirements()
    │   │   └── DataValidators.check_all_requirements()
    │   │       ├── check_obs()
    │   │       ├── check_obsm()
    │   │       ├── check_obsp()
    │   │       ├── check_uns()
    │   │       └── check_layers()
    │   └── _generate_data_suggestions()
    └── get_validation_summary()
```

---

## Phase 1 Status

### ✅ Completed

1. **Data Structures**
   - All result classes implemented
   - LLM-ready output format
   - String representations for debugging

2. **Data Validators**
   - All 5 validator methods (obs, obsm, obsp, uns, layers)
   - Shape and type validation
   - Issue detection

3. **Inspector Core**
   - Basic validation workflow
   - Suggestion generation for data requirements
   - Registry integration

4. **Tests**
   - 12 unit tests for validators
   - Integration test examples
   - Documentation

### 🔜 Next: Phase 2 (Week 2)

**PrerequisiteChecker** - Detect executed functions
- Metadata marker detection
- Output signature matching
- Execution chain reconstruction

---

## Integration with Layer 1

The inspector integrates with Layer 1 registry metadata:

```python
# Layer 1 provides metadata
function_meta = registry.get_function('leiden')
# Returns:
{
    'prerequisites': {'functions': ['neighbors'], 'optional_functions': []},
    'requires': {'obsp': ['connectivities', 'distances']},
    'produces': {'obs': ['leiden']},
    'auto_fix': 'none'
}

# Layer 2 uses this to validate
inspector = DataStateInspector(adata, registry)
result = inspector.validate_prerequisites('leiden')
# Checks if adata.obsp has 'connectivities' and 'distances'
```

---

## Testing

### Run Unit Tests

```bash
# Run all tests
python -m pytest omicverse/utils/inspector/tests/

# Run specific test file
python omicverse/utils/inspector/tests/test_validators.py
```

### Test Coverage

- ✅ obs validation (present, missing, with issues)
- ✅ obsm validation (present, missing, shape validation)
- ✅ obsp validation (present, missing, sparse detection)
- ✅ uns validation (present, missing, nested structure)
- ✅ layers validation (present, missing, shape validation)
- ✅ Comprehensive validation (all requirements)
- ✅ Empty requirements handling

---

## API Reference

### DataStateInspector

```python
class DataStateInspector:
    def __init__(self, adata: AnnData, registry: Any)

    def validate_prerequisites(self, function_name: str) -> ValidationResult
        """Main validation method."""

    def check_data_requirements(self, function_name: str) -> DataCheckResult
        """Check only data structures."""

    def get_validation_summary(self, function_name: str) -> Dict[str, Any]
        """Get LLM-ready dict summary."""
```

### DataValidators

```python
class DataValidators:
    def __init__(self, adata: AnnData)

    def check_obs(self, required_columns: List[str]) -> ObsCheckResult
    def check_obsm(self, required_keys: List[str]) -> ObsmCheckResult
    def check_obsp(self, required_keys: List[str]) -> ObspCheckResult
    def check_uns(self, required_keys: List[str]) -> UnsCheckResult
    def check_layers(self, required_keys: List[str]) -> LayersCheckResult

    def check_all_requirements(self, requires: dict) -> DataCheckResult
        """Check all requirements at once."""
```

---

## Future Phases

### Phase 2: PrerequisiteChecker (Week 2)
- Detect which functions have been executed
- Multiple detection strategies
- Confidence scoring

### Phase 3: SuggestionEngine (Week 3)
- Enhanced suggestion generation
- Workflow planning
- Alternative approaches

### Phase 4: LLMFormatter (Week 4)
- Natural language formatting
- Prompt templates
- Agent integration

### Phase 5: Integration & Testing (Week 5)
- Comprehensive testing
- Documentation
- Production deployment

---

## Contributing

When adding new validators or features:

1. Add to appropriate module (validators.py, inspector.py, etc.)
2. Create unit tests in tests/
3. Update this README
4. Follow existing code style and documentation patterns

---

## Version History

- **0.1.0** (2025-11-11): Phase 1 complete
  - Data structures implemented
  - Data validators complete
  - Basic inspector functionality
  - Unit tests

---

**Next Steps**: Begin Phase 2 (PrerequisiteChecker) implementation

**Status**: Phase 1 Complete ✅

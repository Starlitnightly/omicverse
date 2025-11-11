# Layer 2: DataStateInspector - Executive Summary

**Purpose**: Runtime validation of prerequisite chains by inspecting AnnData object state

**Status**: Planning Complete - Ready for Review

---

## What Layer 2 Does

Layer 2 adds **runtime intelligence** to the prerequisite tracking system:

```python
# User tries to run clustering
inspector = DataStateInspector(adata, registry)
result = inspector.validate_prerequisites('leiden')

if not result.is_valid:
    # Clear error message
    print("Cannot run leiden: neighbors graph is missing")

    # Actionable fix
    print("Run: sc.pp.neighbors(adata, n_neighbors=15)")

    # Auto-executable
    exec(result.suggestions[0].code)
```

**Before Layer 2**: Functions fail with cryptic errors
**After Layer 2**: Clear validation with actionable guidance

---

## Core Components (5)

### 1. DataStateInspector 🎯
**Purpose**: Main orchestrator
**What it does**: Validates prerequisites before function execution
**Key method**: `validate_prerequisites(function_name) -> ValidationResult`

### 2. DataValidators ✅
**Purpose**: Check data structure requirements
**What it does**: Verifies obs, obsm, obsp, uns, layers exist
**Key method**: `check_obsm(['X_pca', 'X_umap']) -> CheckResult`

### 3. PrerequisiteChecker 🔍
**Purpose**: Detect executed functions
**What it does**: Infers execution history from AnnData state
**Key method**: `check_function_executed('pca') -> ExecutionResult`

### 4. SuggestionEngine 💡
**Purpose**: Generate fix suggestions
**What it does**: Creates code snippets to resolve issues
**Key method**: `generate_fix_suggestions() -> List[Suggestion]`

### 5. LLMFormatter 🤖
**Purpose**: Format for Layer 3
**What it does**: Structures output for LLM consumption
**Key method**: `format_for_agent() -> dict`

---

## Detection Strategy (3 Levels)

### Level 1: Metadata Markers (High Confidence)
```python
# Functions leave explicit markers
adata.uns['preprocessing'] = {'log1p': True, 'scale': True}
```
**Confidence**: 0.9-1.0 | **Reliability**: High ✅

### Level 2: Output Signatures (Medium Confidence)
```python
# Match known outputs
if 'X_pca' in adata.obsm and 'pca' in adata.uns:
    confidence = 0.8  # PCA likely executed
```
**Confidence**: 0.6-0.9 | **Reliability**: Medium ⚠️

### Level 3: Distribution Analysis (Low Confidence)
```python
# Analyze data properties (optional, future)
if max(adata.X) < 10:
    confidence = 0.3  # Maybe log-transformed
```
**Confidence**: 0.0-0.6 | **Reliability**: Low (Optional)

---

## Implementation Timeline

**Total Duration**: 5 weeks

| Week | Phase | Deliverable | Status |
|------|-------|-------------|--------|
| 1 | Core Infrastructure | Inspector + Validators | Planned |
| 2 | Detection System | PrerequisiteChecker | Planned |
| 3 | Suggestion Engine | Code generation | Planned |
| 4 | LLM Integration | Formatters | Planned |
| 5 | Testing & Docs | Production ready | Planned |

---

## Key Design Decisions

### ✅ Conservative Detection
- **Decision**: Only mark prerequisites satisfied with clear evidence
- **Rationale**: False positives more dangerous than false negatives
- **Impact**: Users trust validation results

### ✅ Structured Output
- **Decision**: Format all results for LLM consumption
- **Rationale**: Enable Layer 3 intelligent assistance
- **Impact**: Seamless LLM integration

### ✅ Actionable Suggestions
- **Decision**: Generate executable code snippets
- **Rationale**: Users need immediate fixes, not explanations
- **Impact**: Faster problem resolution

### ✅ Confidence Scores
- **Decision**: Report detection confidence (0.0-1.0)
- **Rationale**: Allow users to evaluate reliability
- **Impact**: Transparent, trustworthy system

---

## Example Usage

### Basic Validation
```python
import omicverse as ov

# Load data
adata = sc.datasets.pbmc3k()

# Validate
inspector = ov.utils.DataStateInspector(adata)
result = inspector.validate_prerequisites('leiden')

# Result
print(result.is_valid)  # False
print(result.missing_prerequisites)  # ['neighbors']
print(result.suggestions[0].code)  # 'sc.pp.neighbors(adata, n_neighbors=15)'
```

### Automated Fixing
```python
# Check and auto-fix
result = inspector.validate_prerequisites('leiden')
if not result.is_valid:
    for suggestion in result.suggestions:
        if suggestion.auto_executable:
            print(f"Executing: {suggestion.description}")
            exec(suggestion.code)

# Verify and proceed
result = inspector.validate_prerequisites('leiden')
if result.is_valid:
    sc.tl.leiden(adata)
```

### LLM Integration (Layer 3 Preview)
```python
# Format for LLM
result = inspector.validate_prerequisites('leiden')
llm_input = formatter.format_for_agent(result, user_intent="cluster cells")

# LLM receives:
{
  "function": "leiden",
  "validation_status": false,
  "missing_prerequisites": ["neighbors"],
  "suggested_fixes": [{
    "code": "sc.pp.neighbors(adata, n_neighbors=15)",
    "explanation": "Compute KNN graph",
    "auto_executable": true
  }],
  "llm_prompt": "User wants to cluster cells but neighbors graph is missing. Run: sc.pp.neighbors(adata)"
}
```

---

## Success Metrics

### Quantitative
- ✅ **Detection Accuracy**: ≥90% for preprocessing functions
- ✅ **False Positive Rate**: <5%
- ✅ **Performance**: <100ms validation time
- ✅ **Coverage**: All 36 Layer 1 functions

### Qualitative
- ✅ **Clarity**: Error messages are actionable
- ✅ **Integration**: Seamless with existing workflows
- ✅ **Reliability**: Users trust validation results
- ✅ **Extensibility**: Easy to add new detectors

---

## Risk Assessment

### Low Risk ✅
- Performance issues (mitigated by caching)
- API instability (thorough testing)

### Medium Risk ⚠️
- Inaccurate detection (multiple strategies, confidence scores)
- False positives (conservative thresholds)
- Scope creep (clear phase boundaries)

### Mitigation
All risks have clear mitigation strategies in full plan

---

## Integration Points

### With Layer 1 (Registry) ✅
```python
# Access function metadata
meta = registry.get_function('leiden')
prerequisites = meta['prerequisites']
requires = meta['requires']
auto_fix = meta['auto_fix']
```

### With Layer 3 (LLM) 🔮
```python
# Provide structured output
validation = inspector.validate_prerequisites('leiden')
llm_input = formatter.format_for_agent(validation)
# LLM can understand and act on validation results
```

### With OmicVerse Functions 🔧
```python
# Optional validation hooks
@register_function(...)
def leiden(adata, **kwargs):
    if VALIDATION_ENABLED:
        validate_or_raise('leiden', adata)
    # ... function logic
```

---

## Files & Structure

```
omicverse/utils/inspector/
├── __init__.py
├── inspector.py              # DataStateInspector
├── validators.py             # DataValidators
├── prerequisite_checker.py   # PrerequisiteChecker
├── suggestion_engine.py      # SuggestionEngine
├── formatters.py             # LLMFormatter
├── data_structures.py        # Result classes
├── config.py                 # Configuration
└── tests/
    ├── test_inspector.py
    ├── test_validators.py
    ├── test_prerequisite_checker.py
    ├── test_suggestion_engine.py
    └── test_formatters.py

Documentation:
├── LAYER2_DATASTATEINSPECTOR_PLAN.md  (Complete spec)
├── LAYER2_EXECUTIVE_SUMMARY.md        (This file)
└── docs/
    ├── inspector_guide.md
    ├── api_reference.md
    └── examples/
```

---

## Key Takeaways

### What Layer 2 Provides
1. ✅ **Runtime Validation**: Check prerequisites before execution
2. ✅ **Clear Errors**: Actionable messages, not cryptic failures
3. ✅ **Auto-Fix**: Generate code to resolve issues
4. ✅ **LLM-Ready**: Structured output for Layer 3
5. ✅ **Reliable**: Conservative detection with confidence scores

### Why It Matters
- **Better UX**: Users understand what's wrong and how to fix it
- **Faster Development**: Automated prerequisite resolution
- **Fewer Errors**: Catch issues before they cause failures
- **LLM Integration**: Foundation for intelligent assistance
- **Production Quality**: Robust, tested, documented

### Next Steps
1. **Review this summary + full plan**
2. **Approve approach or suggest changes**
3. **Begin Phase 1 implementation**
4. **5-week timeline to production**

---

## Questions for Review

1. **Scope**: Is the 5-component architecture appropriate?
2. **Detection**: Are the 3 detection levels sufficient?
3. **Timeline**: Is 5 weeks realistic?
4. **Integration**: Any concerns about Layer 1/3 integration?
5. **Risks**: Any additional risks to consider?
6. **Features**: Any critical features missing?

---

## Approval Checklist

- [ ] Architecture approved
- [ ] Detection strategy approved
- [ ] Timeline acceptable
- [ ] Success metrics agreed
- [ ] Risk mitigation acceptable
- [ ] Ready to begin implementation

---

**Full Plan**: See `LAYER2_DATASTATEINSPECTOR_PLAN.md` (12,000+ words)

**Status**: Awaiting Review & Approval

**Next**: Begin Phase 1 implementation upon approval

**Contact**: Ready to answer questions and adjust plan as needed

---

**Version**: 1.0
**Date**: 2025-11-11
**Author**: Claude (Anthropic)

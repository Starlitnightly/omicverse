# OmicVerse Skills Verifier

A comprehensive verification system that tests skill selection and ordering using LLM reasoning, mimicking how Claude Code autonomously selects skills based on task descriptions extracted from tutorial notebooks.

## Overview

The OmicVerse Skills Verifier validates that:
1. ✅ **Skills work correctly** (execution validation)
2. ✅ **LLM can select the right skills** (selection verifier)
3. ✅ **LLM uses skills in correct order** (ordering verifier)
4. ✅ **Tasks can be extracted from notebooks** (task extractor)

## Architecture

### How Claude Code Selects Skills

Claude Code uses a **progressive disclosure** approach:
- **At startup**: Pre-loads only the name and description of each skill (~30-50 tokens each)
- **Matching**: Uses pure LLM reasoning - no algorithmic routing, embeddings, or classifiers
- **Key principle**: The `description` field in YAML frontmatter is the primary signal Claude uses
- **On-demand loading**: Full SKILL.md content is loaded only when needed

### Verifier Components

```
omicverse/utils/verifier/
├── __init__.py
├── data_structures.py              # Core data structures
├── skill_description_loader.py     # Phase 1: Progressive disclosure loader
├── llm_skill_selector.py           # Phase 2: LLM-based skill selection
├── skill_description_quality.py    # Phase 3: Description quality checker
├── notebook_task_extractor.py      # Phase 4: Task extraction from notebooks
├── skill_ordering_verifier.py      # Skill order verification
├── verification_report.py          # Report generation
└── README.md                        # This file
```

## Data Structures

### SkillDescription
Minimal skill info using progressive disclosure (mimics Claude Code):
```python
@dataclass
class SkillDescription:
    name: str          # Skill identifier
    description: str   # What the skill does & when to use it
```

### NotebookTask
Task extracted from tutorial notebook:
```python
@dataclass
class NotebookTask:
    task_id: str
    notebook_path: str
    task_description: str      # Natural language task
    expected_skills: List[str] # Ground truth
    expected_order: List[str]  # Expected execution order
    category: str              # bulk, single-cell, spatial, etc.
    difficulty: str            # single, workflow, complex, ambiguous
```

### LLMSelectionResult
Result of LLM skill selection:
```python
@dataclass
class LLMSelectionResult:
    task_id: str
    selected_skills: List[str]  # LLM's choices
    skill_order: List[str]      # LLM's ordering
    reasoning: str              # LLM's explanation
    confidence: Optional[float]
```

### VerificationResult
Comparison of LLM selection vs ground truth:
```python
@dataclass
class VerificationResult:
    task_id: str
    passed: bool
    precision: float           # Correct / selected
    recall: float             # Correct / expected
    f1_score: float
    ordering_accuracy: float  # Kendall tau correlation
```

## Phase 1: Skill Description Loader ✅ COMPLETED

### SkillDescriptionLoader

Loads skill descriptions using progressive disclosure approach.

**Features**:
- Loads only YAML frontmatter (name + description)
- Keeps memory footprint minimal (~30-50 tokens per skill)
- Formats skills for LLM consumption
- Validates description quality
- Provides statistics on skills

**Usage**:
```python
from omicverse.utils.verifier import SkillDescriptionLoader

# Load all skills
loader = SkillDescriptionLoader()
skills = loader.load_all_descriptions()  # List[SkillDescription]

# Format for LLM
formatted = loader.format_for_llm(skills)
print(formatted)
# Output:
# - bulk-deg-analysis: Perform differential expression...
# - single-preprocessing: Preprocess single-cell data...
# - ...

# Get statistics
stats = loader.get_statistics(skills)
print(f"Total skills: {stats['total_skills']}")
print(f"Avg tokens: {stats['avg_token_estimate']:.1f}")
print(f"Categories: {stats['categories']}")

# Validate descriptions
warnings = loader.validate_descriptions(skills)
for skill_name, skill_warnings in warnings.items():
    print(f"{skill_name}: {skill_warnings}")
```

**Description Quality Checks**:
- ✅ Has action verbs (what the skill does)
- ✅ States when to use it
- ✅ Concise (< 100 words recommended)
- ✅ Token efficient (< 80 tokens recommended)

**Good Description Example**:
```yaml
---
name: bulk-deg-analysis
description: Guide Claude through omicverse's bulk RNA-seq DEG pipeline, from gene ID mapping and DESeq2 normalization to statistical testing, visualization, and pathway enrichment. Use when a user has bulk count matrices and needs differential expression analysis in omicverse.
---
```

## Phase 2: LLM Skill Selector (PENDING)

Will implement LLM-based skill selection that mimics Claude Code's behavior.

**Planned Features**:
- Pure LLM reasoning (no algorithmic matching)
- Given task description → ask LLM which skills to use
- Returns selected skills with ordering and reasoning
- Supports multiple LLM backends (GPT, Claude, Gemini)

**Planned Usage**:
```python
selector = LLMSkillSelector(llm_backend, skill_descriptions)
result = selector.select_skills(task)
print(f"Selected: {result.selected_skills}")
print(f"Order: {result.skill_order}")
print(f"Reasoning: {result.reasoning}")
```

## Phase 3: Skill Description Quality Checker (PENDING)

Will verify skill descriptions are effective for LLM matching.

**Planned Checks**:
- Description completeness (what, when, examples)
- Description effectiveness (how well LLM selects correctly)
- A/B testing with modified descriptions
- Recommendations for improvement

## Phase 4: Notebook Task Extractor (PENDING)

Will extract task descriptions from tutorial notebooks.

**Planned Features**:
- Parse Jupyter notebooks
- Extract natural language tasks from markdown cells
- Infer tasks from code flow
- Build ground truth task → skills mapping

## Phase 5: End-to-End Verification (PENDING)

Will test complete workflow: notebook → task → LLM selection → verification.

**Success Criteria**:
- ≥90% F1-score for skill selection
- ≥85% ordering accuracy for workflows
- 100% notebook coverage (73 notebooks)
- 100% skill coverage (23 skills)

## Testing

### Current Status

**Implemented**:
- ✅ Data structures with comprehensive validation
- ✅ SkillDescriptionLoader with progressive disclosure
- ✅ Comprehensive test suite for loader
- ✅ Token estimation and statistics
- ✅ Description quality validation

**Test Files**:
- `tests/verifier/test_skill_description_loader.py` - Full test coverage

### Running Tests

```bash
# Install dependencies
pip install -e ".[tests]"

# Run verifier tests
pytest tests/verifier/ -v

# Run with coverage
pytest tests/verifier/ --cov=omicverse.utils.verifier --cov-report=html
```

## Verification Test Scenarios

### Level 1: Single-Skill Tasks
Task requires exactly one skill.
```
Task: "Preprocess PBMC3k dataset with QC filtering"
Expected: single-preprocessing
```

### Level 2: Multi-Skill Sequential Tasks
Task requires multiple skills in specific order.
```
Task: "Analyze bulk RNA-seq: normalize, find DEGs, run WGCNA"
Expected: bulk-deg-analysis → bulk-wgcna-analysis
```

### Level 3: Complex Workflow Tasks
Multi-step analysis with dependencies.
```
Task: "Single-cell workflow: QC → cluster → annotate → cell communication"
Expected: single-preprocessing → single-clustering → single-annotation → single-cellphone-db
```

### Level 4: Ambiguous Tasks
Task could match multiple skills.
```
Task: "Analyze differential expression"
Expected: bulk-deg-analysis (or bulk-deseq2-analysis)
Test: Verify LLM picks appropriate skill and explains reasoning
```

## Metrics

### Selection Accuracy
- **Precision**: Correct skills / selected skills
- **Recall**: Correct skills / expected skills
- **F1-score**: Harmonic mean of precision and recall

### Ordering Accuracy
- **Exact Match**: 1.0 if order matches exactly, 0.0 otherwise
- **Kendall Tau**: Correlation between expected and actual order (normalized to [0, 1])
- **Dependency Violations**: Count of out-of-order dependencies

### Coverage
- **Notebook Coverage**: % of notebooks with extracted tasks
- **Skill Coverage**: % of skills tested
- **Category Coverage**: Coverage per category (bulk, single, spatial)

## Ground Truth Dataset

Format for task dataset (planned):
```json
{
  "tasks": [
    {
      "task_id": "bulk-deg-001",
      "notebook_path": "Tutorials-bulk/t_deg.ipynb",
      "task_description": "Perform differential expression analysis...",
      "expected_skills": ["bulk-deg-analysis"],
      "expected_order": ["bulk-deg-analysis"],
      "category": "bulk",
      "difficulty": "single"
    }
  ]
}
```

## Implementation Timeline

- **Phase 1** (Skill Description Loader): ✅ **COMPLETED**
- **Phase 2** (LLM Skill Selector): 2 days
- **Phase 3** (Description Quality Checker): 2 days
- **Phase 4** (Notebook Task Extractor): 2-3 days
- **Phase 5** (End-to-End Verification): 2 days
- **Phase 6** (Documentation & CI): 1 day

**Total**: ~12-14 days

## Key Principles

1. **Test How Claude Code Actually Works**
   - Use LLM reasoning, not algorithmic matching
   - Progressive disclosure (load minimal info at startup)
   - Description field is critical for matching

2. **Validate with Real Tasks**
   - Extract tasks from actual tutorial notebooks
   - Test with realistic user requests
   - Measure against ground truth

3. **Comprehensive Coverage**
   - All 73 notebooks
   - All 23 skills
   - All categories (bulk, single, spatial, data, plotting)

4. **Actionable Results**
   - Identify weak descriptions
   - Recommend improvements
   - Track metrics over time

## Contributing

When adding new skills or modifying existing ones:

1. Ensure description clearly states:
   - **What** the skill does (use action verbs)
   - **When** to use it (triggers, use cases)
   - **Key features** (in brief)

2. Keep descriptions concise:
   - < 100 words
   - < 80 tokens (estimated)

3. Test with verifier:
   ```bash
   python -m omicverse.utils.verifier --validate-descriptions
   ```

## References

- Claude Code Skills Documentation: https://docs.claude.com/en/docs/claude-code/skills
- Progressive Disclosure Pattern: Loads minimal info (name + description) at startup
- LLM-based Matching: No algorithms, embeddings, or classifiers - pure language understanding

## License

Part of OmicVerse package. See main LICENSE file.

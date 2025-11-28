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

## Phase 2: LLM Skill Selector ✅ COMPLETED

Implements LLM-based skill selection that mimics Claude Code's behavior.

### LLMSkillSelector

Uses pure LLM reasoning to select skills - NO algorithmic routing!

**Features**:
- Pure language understanding (no embeddings, no classifiers)
- Asks LLM to select skills based on descriptions
- Returns selected skills with ordering and reasoning
- Supports async and sync execution
- Batch selection for multiple tasks in parallel
- Robust JSON parsing (handles markdown, malformed responses)
- Integrates with OmicVerse's LLM backend infrastructure

**Usage**:
```python
from omicverse.utils.verifier import LLMSkillSelector, create_skill_selector

# Option 1: Use convenience function
selector = create_skill_selector(
    skill_descriptions=skills,
    model="gpt-4o-mini",
    temperature=0.0
)

# Option 2: Create with custom backend
from omicverse.utils.agent_backend import OmicVerseLLMBackend

backend = OmicVerseLLMBackend(
    system_prompt="...",
    model="gpt-4o-mini",
    temperature=0.0
)
selector = LLMSkillSelector(
    llm_backend=backend,
    skill_descriptions=skills
)

# Select skills for a task
result = selector.select_skills(task)
print(f"Selected: {result.selected_skills}")
print(f"Order: {result.skill_order}")
print(f"Reasoning: {result.reasoning}")

# Async usage
result = await selector.select_skills_async(task)

# Batch selection (parallel)
results = selector.select_skills_batch([task1, task2, task3])
```

**How It Works**:
1. Formats skill descriptions into LLM prompt
2. Sends task description + skills to LLM
3. LLM responds with JSON: `{"skills": [...], "order": [...], "reasoning": "..."}`
4. Parses response (handles markdown, malformed JSON)
5. Returns `LLMSelectionResult` with selections

**Robust Parsing**:
- Handles clean JSON: `{"skills": ...}`
- Handles markdown: ` ```json ... ``` `
- Handles text with JSON: `"Here's my answer: {...}"`
- Graceful error handling for invalid responses

## Phase 3: Skill Description Quality Checker ✅ COMPLETED

Verifies that skill descriptions are effective for LLM matching.

### SkillDescriptionQualityChecker

Analyzes skill descriptions for quality, effectiveness, and provides recommendations.

**Features**:
- Completeness checking (has "what", "when", examples)
- Clarity metrics (conciseness, token efficiency)
- Effectiveness testing (LLM selection accuracy)
- A/B testing for comparing descriptions
- Recommendation generation
- Bulk operations for analyzing all skills
- Report generation

**Usage**:
```python
from omicverse.utils.verifier import (
    SkillDescriptionQualityChecker,
    create_quality_checker,
    SkillDescription
)

# Create quality checker
checker = create_quality_checker()

# Check a single skill
skill = SkillDescription(
    name="test-skill",
    description="Analyze data. Use when you need analysis."
)

metrics = checker.check_quality(skill)
print(f"Overall score: {metrics.overall_score:.2f}")
print(f"Completeness: {metrics.completeness_score:.2f}")
print(f"Warnings: {metrics.warnings}")
print(f"Recommendations: {metrics.recommendations}")

# Check all skills
results = checker.check_all_skills(skills)

# Get summary statistics
summary = checker.get_quality_summary(skills)
print(f"Average score: {summary['avg_overall_score']:.2f}")
print(f"Skills needing improvement: {summary['skills_needing_improvement']}")

# Generate report
report = checker.generate_report(skills, show_recommendations=True)
print(report)
```

**Effectiveness Testing** (requires LLM selector):
```python
from omicverse.utils.verifier import LLMSkillSelector

# Create checker with LLM selector
selector = create_skill_selector(skills, model="gpt-4o-mini")
checker = create_quality_checker(llm_selector=selector)

# Test how well LLM selects this skill
result = checker.test_effectiveness(
    skill=my_skill,
    positive_tasks=[
        "Tasks that SHOULD match this skill",
        "Another task that should match"
    ],
    negative_tasks=[
        "Tasks that should NOT match",
        "Another non-matching task"
    ],
    all_skills=all_skills
)

print(f"Precision: {result.precision:.2f}")
print(f"Recall: {result.recall:.2f}")
print(f"F1 Score: {result.f1_score:.2f}")
```

**A/B Testing**:
```python
# Compare two versions of a description
original = SkillDescription(name="skill", description="Original description")
modified = SkillDescription(name="skill", description="Improved description")

comparison = checker.compare_descriptions(
    original=original,
    modified=modified,
    test_tasks=["task1", "task2", "task3"],
    positive_task_indices=[0, 1],  # Tasks 0 and 1 should match
    all_skills=all_other_skills
)

print(f"Winner: {comparison.winner}")
print(f"Improvement: {comparison.improvement:.1f}%")
print(f"Recommendation: {comparison.recommendation}")
```

**Quality Metrics**:
- **Completeness Score**: 0.0-1.0 based on has_what + has_when + has_use_cases
- **Clarity Score**: 0.0-1.0 based on conciseness and token efficiency
- **Overall Score**: Weighted average (60% completeness, 40% clarity)

**Effectiveness Metrics** (with LLM testing):
- **Precision**: Correct selections / Total selections
- **Recall**: Correct selections / Should have selected
- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy**: (True positives + True negatives) / Total

## Phase 4: Notebook Task Extractor ✅ COMPLETED

Extracts task descriptions from Jupyter notebooks and maps them to skills.

### NotebookTaskExtractor

Parses .ipynb files to extract tasks and build ground truth dataset.

**Features**:
- Parse Jupyter notebooks using nbformat
- Extract task descriptions from markdown cells
- Detect task indicators ("In this tutorial", "The goal is", etc.)
- Map code patterns to skills (function call analysis)
- Ground truth mapping (notebook basename → expected skills)
- Coverage statistics calculation
- JSON serialization for task datasets

**Usage**:
```python
from omicverse.utils.verifier import NotebookTaskExtractor, create_task_extractor

# Create extractor
extractor = create_task_extractor()

# Extract from single notebook
tasks = extractor.extract_from_notebook("path/to/t_deg.ipynb")

for task in tasks:
    print(f"Task: {task.task_description}")
    print(f"Skills: {task.expected_skills}")
    print(f"Category: {task.category}")

# Extract from all notebooks in directory
all_tasks = extractor.extract_from_directory(
    "omicverse_guide/docs/Tutorials-bulk"
)

# Get coverage statistics
from omicverse.utils.verifier import SkillDescriptionLoader

loader = SkillDescriptionLoader()
skills = loader.load_all_descriptions()

stats = extractor.get_coverage_statistics(all_tasks, skills)
print(f"Coverage: {stats['coverage_percentage']:.1f}%")
print(f"Covered skills: {stats['covered_skills']}")
print(f"Not covered: {stats['not_covered_skills']}")

# Save to JSON
extractor.save_tasks_to_json(all_tasks, "tasks_dataset.json")

# Load from JSON
loaded_tasks = NotebookTaskExtractor.load_tasks_from_json("tasks_dataset.json")
```

**Task Extraction**:
- **Main task**: From notebook title + introduction
- **Sub-tasks**: From H2/H3 section headings
- **Task indicators detected**:
  - "in this tutorial"
  - "we will"
  - "the goal is"
  - "this tutorial demonstrates"
  - "here we"
  - "an important task"

**Skill Mapping**:
Maps code patterns to skills by detecting function calls:
- `ov.bulk.pyDEG` → `bulk-deg-analysis`
- `ov.pp.qc` + `ov.pp.preprocess` → `single-preprocessing`
- `ov.single.cluster` → `single-clustering`
- `ov.single.CellVote` → `single-annotation`
- And 20+ more patterns

**Ground Truth Mapping**:
Pre-built mapping for known notebooks:
```python
{
    't_deg.ipynb': ['bulk-deg-analysis'],
    't_preprocess.ipynb': ['single-preprocessing'],
    't_cluster.ipynb': ['single-clustering'],
    # ... 20+ more mappings
}
```

**Coverage Statistics**:
```python
{
    'total_tasks': 45,
    'total_notebooks': 15,
    'skills_covered': 18,
    'skills_not_covered': 5,
    'coverage_percentage': 78.3,
    'covered_skills': ['bulk-deg-analysis', 'single-preprocessing', ...],
    'not_covered_skills': ['data-export-pdf', ...]
}
```

## Phase 5: End-to-End Verification ✅ COMPLETED

Tests complete workflow: notebook → task → LLM selection → verification.

### EndToEndVerifier

Ties all components together for complete verification workflow.

**Features**:
- Single task verification (sync + async)
- Batch task verification with concurrency control
- Complete verification runs with configuration
- Comprehensive summary generation
- Detailed report generation
- Category and difficulty breakdown metrics
- Success criteria checking
- Failed task analysis

**Usage**:
```python
from omicverse.utils.verifier import (
    EndToEndVerifier,
    VerificationRunConfig,
    create_verifier,
)

# Create verifier
verifier = create_verifier()

# Configure verification run
config = VerificationRunConfig(
    notebooks_dir="omicverse_guide/docs/Tutorials-bulk",
    notebook_pattern="**/*.ipynb",
    model="gpt-4o-mini",
    temperature=0.0,
    max_concurrent_tasks=5,
    skip_notebooks=["old_notebook.ipynb"],
    only_categories=["bulk", "single-cell"],  # Optional filter
)

# Run verification
summary = verifier.run_verification(config)

# Check results
print(f"Tasks verified: {summary.tasks_verified}")
print(f"Tasks passed: {summary.tasks_passed}")
print(f"F1-Score: {summary.avg_f1_score:.3f}")
print(f"Ordering Accuracy: {summary.avg_ordering_accuracy:.3f}")

# Check success criteria
if summary.passed_criteria():
    print("✅ Verification PASSED!")
else:
    print("❌ Verification FAILED")

# Generate report
report = verifier.generate_report(summary, detailed=True)
print(report)

# Save report to file
verifier.save_report(summary, "verification_report.txt")
```

**Async Usage**:
```python
import asyncio

async def run_async_verification():
    verifier = create_verifier()
    config = VerificationRunConfig(notebooks_dir="...")
    summary = await verifier.run_verification_async(config)
    return summary

summary = asyncio.run(run_async_verification())
```

**Verification Summary**:
The `VerificationSummary` object contains:
- Overall metrics (precision, recall, F1, ordering accuracy)
- Coverage statistics (notebooks tested, skills tested)
- Category breakdown (metrics per category)
- Difficulty breakdown (metrics per difficulty level)
- Failed task details (for debugging)
- Success criteria check

**Success Criteria**:
- ≥90% F1-score for skill selection
- ≥85% ordering accuracy for workflows
- 100% notebook coverage (73 notebooks)
- 100% skill coverage (23 skills)

## Phase 6: CLI Tool ✅ COMPLETED

Command-line interface for running verification, validating descriptions, and extracting tasks.

### Command: verify

Run end-to-end verification on notebooks:

```bash
# Basic verification
python -m omicverse.utils.verifier verify ./notebooks

# With filters and options
python -m omicverse.utils.verifier verify ./notebooks \
    --pattern "**/*.ipynb" \
    --model gpt-4o-mini \
    --max-concurrent 5 \
    --categories bulk single-cell \
    --detailed \
    --output report.txt \
    --json-output summary.json

# Skip specific notebooks
python -m omicverse.utils.verifier verify ./notebooks \
    --skip old_notebook.ipynb deprecated.ipynb
```

**Options**:
- `--pattern`: Glob pattern for notebooks (default: `**/*.ipynb`)
- `--model`: LLM model (default: `gpt-4o-mini`)
- `--temperature`: LLM temperature (default: 0.0)
- `--max-concurrent`: Max parallel tasks (default: 5)
- `--skip`: Notebooks to skip
- `--categories`: Filter by categories (e.g., `bulk single-cell`)
- `--detailed`: Generate detailed report
- `--output`: Save report to file
- `--json-output`: Save JSON summary

**Exit Codes**:
- `0`: Verification passed criteria
- `1`: Verification failed criteria

### Command: validate

Validate skill descriptions:

```bash
# Basic validation
python -m omicverse.utils.verifier validate

# With quality checks
python -m omicverse.utils.verifier validate --check-quality

# Detailed quality metrics
python -m omicverse.utils.verifier validate --check-quality --detailed
```

**Options**:
- `--check-quality`: Run quality metrics analysis
- `--detailed`: Show detailed quality breakdown

**Checks**:
- Action verbs (what the skill does)
- "When to use" indicators
- Conciseness (< 100 words)
- Token efficiency (< 80 tokens)
- Completeness score
- Clarity score

### Command: extract

Extract tasks from notebooks:

```bash
# Extract from single notebook
python -m omicverse.utils.verifier extract --notebook ./t_deg.ipynb \
    --detailed \
    --show-coverage \
    --output tasks.json

# Extract from directory
python -m omicverse.utils.verifier extract --directory ./notebooks \
    --pattern "**/*.ipynb" \
    --show-coverage \
    --output tasks.json
```

**Options**:
- `--notebook`: Single notebook path
- `--directory`: Directory of notebooks
- `--pattern`: Glob pattern (default: `**/*.ipynb`)
- `--detailed`: Show detailed task info
- `--show-coverage`: Show skill coverage stats
- `--output`: Save tasks to JSON

### Command: test-selection

Test LLM skill selection interactively:

```bash
# Interactive mode
python -m omicverse.utils.verifier test-selection

# With task provided
python -m omicverse.utils.verifier test-selection \
    --task "Perform differential expression analysis on bulk RNA-seq" \
    --model gpt-4o-mini

# Custom model and temperature
python -m omicverse.utils.verifier test-selection \
    --task "Cluster single-cell data" \
    --model gpt-4o \
    --temperature 0.2
```

**Options**:
- `--task`: Task description (prompts if not provided)
- `--model`: LLM model (default: `gpt-4o-mini`)
- `--temperature`: LLM temperature (default: 0.0)

### Global Options

Available for all commands:

```bash
--skills-dir PATH    # Custom skills directory (default: .claude/skills/)
```

### Usage Examples

**Example 1: Quick validation**
```bash
python -m omicverse.utils.verifier validate
```

**Example 2: Full verification with report**
```bash
python -m omicverse.utils.verifier verify ./omicverse_guide/docs/Tutorials-bulk \
    --detailed \
    --output bulk_verification_report.txt \
    --json-output bulk_summary.json
```

**Example 3: Extract and analyze coverage**
```bash
python -m omicverse.utils.verifier extract \
    --directory ./omicverse_guide/docs \
    --show-coverage \
    --output all_tasks.json
```

**Example 4: Test skill selection**
```bash
python -m omicverse.utils.verifier test-selection \
    --task "Preprocess and cluster PBMC3k dataset"
```

**Example 5: CI/CD Integration**
```bash
# Validation check in CI
python -m omicverse.utils.verifier validate --check-quality || exit 1

# Verification test in CI
python -m omicverse.utils.verifier verify ./notebooks --categories bulk
# Exit code 0 if passed, 1 if failed
```

## Testing

### Current Status

**Implemented**:
- ✅ Data structures with comprehensive validation
- ✅ SkillDescriptionLoader with progressive disclosure
- ✅ LLMSkillSelector with pure LLM reasoning
- ✅ SkillDescriptionQualityChecker with effectiveness testing
- ✅ NotebookTaskExtractor with ground truth mapping
- ✅ EndToEndVerifier with complete workflow testing
- ✅ CLI tool with 4 commands (verify, validate, extract, test-selection)
- ✅ Comprehensive test suites (130+ tests total)
- ✅ Token estimation and statistics
- ✅ Description quality validation
- ✅ A/B testing framework
- ✅ Report generation with detailed breakdowns
- ✅ Notebook parsing and task extraction
- ✅ Coverage statistics
- ✅ Success criteria checking
- ✅ CI/CD integration support

**Test Files**:
- `tests/verifier/test_skill_description_loader.py` - SkillDescriptionLoader tests (20+ tests)
- `tests/verifier/test_llm_skill_selector.py` - LLMSkillSelector tests (25+ tests)
- `tests/verifier/test_skill_description_quality.py` - Quality checker tests (30+ tests)
- `tests/verifier/test_notebook_task_extractor.py` - NotebookTaskExtractor tests (35+ tests)
- `tests/verifier/test_end_to_end_verifier.py` - EndToEndVerifier tests (30+ tests)

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
- **Phase 2** (LLM Skill Selector): ✅ **COMPLETED**
- **Phase 3** (Description Quality Checker): ✅ **COMPLETED**
- **Phase 4** (Notebook Task Extractor): ✅ **COMPLETED**
- **Phase 5** (End-to-End Verification): ✅ **COMPLETED**
- **Phase 6** (CLI Tool & Documentation): ✅ **COMPLETED**

**Progress**: 6/6 phases complete (100%) 🎉

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

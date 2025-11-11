# Layer 3: Enhanced Agent Prompt System - DETAILED PLAN

**Status**: Planning Phase (No Code Changes)
**Dependencies**: Layer 2 Complete (v0.5.0)
**Goal**: Intelligent LLM agent integration with automatic prerequisite handling

---

## Executive Summary

Layer 3 transforms the DataStateInspector from a validation tool into an intelligent agent system that:
- **Injects prerequisite context** directly into LLM system prompts
- **Validates data state** before allowing agent execution
- **Escalates complex workflows** to suggest high-level functions
- **Auto-inserts simple prerequisites** to streamline workflows

**Key Innovation**: The LLM agent becomes prerequisite-aware and can automatically handle dependency chains without explicit user instructions.

---

## Problem Statement

### Current State (Layer 2)
With Layer 2 complete, we have:
- ✅ Prerequisite detection (PrerequisiteChecker)
- ✅ Intelligent suggestions (SuggestionEngine)
- ✅ LLM-friendly formatting (LLMFormatter)
- ✅ Production API (easy integration)

**But**: The LLM agent is **passive** - it only responds to validation errors. The user must explicitly check prerequisites and act on suggestions.

### Limitations
1. **Reactive, not proactive**: Agent waits for errors instead of preventing them
2. **Context-unaware**: Agent doesn't know what prerequisites are satisfied
3. **Manual workflow**: User must copy/paste suggestions and execute them
4. **Fragmented experience**: Validation and execution are separate steps

### Layer 3 Vision

**Proactive Agent**: The LLM agent knows the current data state and prerequisite chains, automatically handling simple dependencies and escalating complex ones.

**Example Flow**:
```
User: "Run leiden clustering"

Layer 2 (passive):
  → Check prerequisites
  → leiden needs neighbors
  → Return error with suggestion
  → User manually runs neighbors
  → User retries leiden

Layer 3 (proactive):
  → Agent checks prerequisites (automatic)
  → leiden needs neighbors
  → Agent knows: neighbors is "simple" (auto-insertable)
  → Agent generates code: ov.pp.neighbors(adata) + ov.pp.leiden(adata)
  → User executes complete workflow in one step
```

---

## Layer 3 Architecture

### Component Overview

```
Layer 3: Enhanced Agent Prompt System
├── Component 1: AgentContextInjector
│   └── Injects prerequisite chains into system prompt
├── Component 2: DataStateValidator
│   └── Pre-execution validation with auto-correction
├── Component 3: WorkflowEscalator
│   └── Smart escalation for complex workflows
└── Component 4: AutoPrerequisiteInserter
    └── Automatic insertion of simple prerequisites
```

### Integration with Layer 2

```
User Request → Agent (with Layer 3 context)
                ↓
          [AgentContextInjector]
          Adds prerequisite state to system prompt
                ↓
          Agent generates code
                ↓
          [DataStateValidator]
          Validates before execution
                ↓
      ┌─────────┴─────────┐
      │                   │
   Valid              Invalid
      │                   │
      ↓                   ↓
  Execute         [WorkflowEscalator]
                  Suggests escalation OR
                  [AutoPrerequisiteInserter]
                  Auto-adds simple prereqs
                        ↓
                  Execute complete workflow
```

---

## Component 1: AgentContextInjector

### Purpose
Inject current data state and prerequisite chains into the LLM's system prompt, making the agent aware of what's already been executed and what's missing.

### Key Features

**1. Dynamic System Prompt Enhancement**

Adds a section to the system prompt:
```
## Current AnnData State

You are working with an AnnData object that has:

Executed Functions:
  ✅ qc (confidence: 0.95)
  ✅ preprocess (confidence: 0.95)
  ✅ pca (confidence: 0.95)
  ❌ neighbors (not executed)
  ❌ leiden (not executed)

Available Data:
  ✅ adata.obsm['X_pca'] - PCA embedding (50 components)
  ✅ adata.uns['pca'] - PCA metadata
  ❌ adata.obsp['connectivities'] - neighbor graph (MISSING)
  ❌ adata.obs['leiden'] - cluster labels (MISSING)

Prerequisite Chains:
  neighbors → requires → pca ✅
  leiden → requires → neighbors ❌

IMPORTANT: When generating code, ensure prerequisites are satisfied.
For simple prerequisites (like neighbors, pca), you can auto-insert them.
For complex workflows, suggest high-level functions like ov.pp.preprocess().
```

**2. Function-Specific Context**

When user asks about a specific function (e.g., "run leiden"):
```
## Function: leiden

Prerequisites Status:
  Required:
    - neighbors: ❌ NOT EXECUTED (auto-insertable)
    - pca: ✅ EXECUTED

  Data Requirements:
    - adata.obsp['connectivities']: ❌ MISSING
    - adata.obsp['distances']: ❌ MISSING

Recommendation:
  Since neighbors is missing but is a simple prerequisite, you should
  generate code that includes BOTH neighbors and leiden:

  ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
  ov.pp.leiden(adata, resolution=1.0)
```

**3. Workflow State Tracking**

Maintains conversation-level state:
```python
class AgentContextInjector:
    def __init__(self, adata, registry):
        self.inspector = create_inspector(adata, registry)
        self.conversation_history = []
        self.executed_functions = set()

    def inject_context(self, system_prompt: str) -> str:
        """Enhance system prompt with current data state."""

        # Get current state
        executed = self.inspector.prerequisite_checker.get_execution_chain()

        # Build context
        context = self._build_context_section(executed)

        # Inject into system prompt
        return system_prompt + "\n\n" + context

    def update_after_execution(self, function_name: str):
        """Update state after function execution."""
        self.executed_functions.add(function_name)
        self.conversation_history.append({
            'function': function_name,
            'timestamp': datetime.now(),
        })
```

### Implementation Details

**Input**: System prompt + AnnData state
**Output**: Enhanced system prompt with prerequisite context
**Update Frequency**: After every code execution
**Cache Strategy**: Cache per AnnData object, invalidate on execution

### Benefits

1. **Context-Aware Agent**: Agent knows what's been done
2. **Prevents Errors**: Agent avoids generating invalid code
3. **Smarter Suggestions**: Agent can suggest complete workflows
4. **Better UX**: User doesn't need to manually check prerequisites

---

## Component 2: DataStateValidator

### Purpose
Validate generated code BEFORE execution, with automatic correction for simple cases.

### Key Features

**1. Pre-Execution Validation Hook**

Intercepts generated code before execution:
```python
class DataStateValidator:
    def validate_before_execution(
        self,
        code: str,
        adata: AnnData
    ) -> ValidationResult:
        """Validate code before execution."""

        # Parse code to extract function calls
        functions = self._extract_function_calls(code)

        # Validate each function
        results = []
        for func in functions:
            result = self.inspector.validate_prerequisites(func)
            results.append(result)

        # Aggregate results
        return self._aggregate_results(results)
```

**2. Auto-Correction for Simple Cases**

If validation fails but prerequisites are simple, auto-correct:
```python
def auto_correct(self, code: str, validation_result: ValidationResult) -> str:
    """Auto-correct code by inserting prerequisites."""

    if not validation_result.is_valid:
        missing = validation_result.missing_prerequisites

        # Check if all missing are "simple" (auto-insertable)
        if self._all_simple_prerequisites(missing):
            # Generate corrected code with prerequisites
            corrected = self._insert_prerequisites(code, missing)
            return corrected

    return code  # No correction needed or not auto-correctable
```

**3. Validation Feedback to Agent**

If validation fails and can't be auto-corrected, provide feedback:
```
⚠️ Code Validation Failed

Generated code:
  ov.pp.leiden(adata, resolution=1.0)

Issues:
  ✗ leiden requires neighbors (not executed)
  ✗ neighbors requires pca (not executed)

Suggested Fix:
  Run complete preprocessing workflow first:

  ov.pp.preprocess(adata, mode="shiftlog|pearson", n_HVGs=2000)
  ov.pp.scale(adata)
  ov.pp.pca(adata, n_pcs=50)
  ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
  ov.pp.leiden(adata, resolution=1.0)
```

### Implementation Details

**Validation Trigger**: Before executing any ov.pp.* or ov.single.* function
**Auto-Correction Policy**:
- Simple prerequisites (pca, neighbors, scale): Auto-insert
- Complex workflows (qc, preprocess): Escalate to user
**Feedback Format**: Structured JSON or natural language

### Benefits

1. **Prevents Invalid Execution**: Catches errors before they happen
2. **Automatic Fixes**: Corrects simple issues without user intervention
3. **Clear Feedback**: Shows exactly what's wrong and how to fix it
4. **Learning Agent**: Agent learns from validation feedback

---

## Component 3: WorkflowEscalator

### Purpose
Intelligently escalate complex workflows to high-level functions instead of suggesting many small steps.

### Key Features

**1. Complexity Analysis**

Determine if a workflow is "complex":
```python
class WorkflowEscalator:
    def analyze_complexity(
        self,
        missing_prerequisites: List[str]
    ) -> ComplexityLevel:
        """Analyze workflow complexity."""

        # Complexity indicators
        num_missing = len(missing_prerequisites)
        has_qc = 'qc' in missing_prerequisites
        has_preprocessing = 'preprocess' in missing_prerequisites
        dependency_depth = self._calculate_dependency_depth(missing_prerequisites)

        if num_missing >= 4 or has_qc or has_preprocessing:
            return ComplexityLevel.HIGH
        elif num_missing >= 2 or dependency_depth >= 3:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.LOW
```

**2. Escalation Rules**

Based on complexity, suggest appropriate approach:
```python
def escalate(self, target_function: str, complexity: ComplexityLevel) -> Suggestion:
    """Escalate to appropriate high-level function."""

    if complexity == ComplexityLevel.HIGH:
        # Suggest comprehensive preprocessing
        return Suggestion(
            priority='HIGH',
            suggestion_type='workflow_escalation',
            description='Run complete preprocessing pipeline',
            code='ov.pp.preprocess(adata, mode="shiftlog|pearson", n_HVGs=2000)',
            explanation=(
                f'{target_function} requires extensive preprocessing. '
                'Use ov.pp.preprocess() to handle qc, normalization, '
                'feature selection, scaling, and PCA in one step.'
            ),
            estimated_time='1-2 minutes',
        )

    elif complexity == ComplexityLevel.MEDIUM:
        # Suggest workflow chain
        return self._generate_workflow_chain(target_function)

    else:
        # Simple auto-insert
        return None  # Handled by AutoPrerequisiteInserter
```

**3. Escalation Examples**

**Example 1: High Complexity**
```
User: "Run leiden clustering on raw data"

Analysis:
  - Missing: qc, preprocess, scale, pca, neighbors
  - Complexity: HIGH (5 prerequisites, includes qc/preprocess)

Escalation:
  Instead of suggesting 5 separate functions, escalate to:

  # Complete preprocessing pipeline
  ov.pp.preprocess(adata, mode="shiftlog|pearson", n_HVGs=2000)
  ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
  ov.pp.leiden(adata, resolution=1.0)
```

**Example 2: Medium Complexity**
```
User: "Run UMAP visualization"

Analysis:
  - Missing: pca, neighbors
  - Complexity: MEDIUM (2 prerequisites)

Escalation:
  Generate ordered chain:

  ov.pp.pca(adata, n_pcs=50)
  ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
  ov.pp.umap(adata)
```

### Implementation Details

**Complexity Thresholds**:
- LOW: 0-1 missing prerequisites, depth ≤ 2
- MEDIUM: 2-3 missing prerequisites, depth ≤ 4
- HIGH: 4+ missing prerequisites OR includes qc/preprocess

**Escalation Targets**:
- `ov.pp.preprocess()`: Handles qc, normalization, HVG selection, scaling, PCA
- `ov.pp.batch_correct()`: Handles batch correction workflows
- Custom high-level wrappers as needed

### Benefits

1. **Simpler Workflows**: One high-level function instead of many steps
2. **Better Defaults**: High-level functions have tested default parameters
3. **Faster Execution**: Optimized workflows run faster
4. **Less Error-Prone**: Fewer chances for user mistakes

---

## Component 4: AutoPrerequisiteInserter

### Purpose
Automatically insert simple prerequisites into generated code, creating complete executable workflows.

### Key Features

**1. Simple Prerequisite Detection**

Define what counts as "simple" (auto-insertable):
```python
class AutoPrerequisiteInserter:
    # Simple prerequisites that can be auto-inserted
    SIMPLE_PREREQUISITES = {
        'scale': {
            'code_template': 'ov.pp.scale(adata)',
            'time_cost': 5,  # seconds
            'requires': [],
        },
        'pca': {
            'code_template': 'ov.pp.pca(adata, n_pcs=50)',
            'time_cost': 10,
            'requires': ['scale'],
        },
        'neighbors': {
            'code_template': 'ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50)',
            'time_cost': 15,
            'requires': ['pca'],
        },
    }

    # Complex prerequisites that should NOT be auto-inserted
    COMPLEX_PREREQUISITES = {
        'qc',          # Requires parameter tuning
        'preprocess',  # Multiple choices, complex configuration
        'batch_correct',  # Requires batch information
    }
```

**2. Code Insertion**

Insert prerequisites in correct order:
```python
def insert_prerequisites(
    self,
    code: str,
    missing_prerequisites: List[str]
) -> str:
    """Insert missing prerequisites into code."""

    # Filter to only simple prerequisites
    simple_missing = [
        p for p in missing_prerequisites
        if p in self.SIMPLE_PREREQUISITES
    ]

    # Check if all missing are simple
    if len(simple_missing) != len(missing_prerequisites):
        return code  # Has complex prerequisites, don't auto-insert

    # Resolve dependency order
    ordered = self._resolve_dependencies(simple_missing)

    # Generate prerequisite code
    prereq_code = '\n'.join(
        self.SIMPLE_PREREQUISITES[p]['code_template']
        for p in ordered
    )

    # Insert before original code
    return prereq_code + '\n\n' + code
```

**3. Smart Insertion Logic**

**Example 1: Simple Case**
```
User: "Run leiden clustering"

Generated Code (by agent):
  ov.pp.leiden(adata, resolution=1.0)

Validation:
  ✗ Missing: neighbors

Auto-Insertion:
  # Auto-inserted prerequisite
  ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50)

  # Original code
  ov.pp.leiden(adata, resolution=1.0)
```

**Example 2: Dependency Chain**
```
User: "Run UMAP"

Generated Code:
  ov.pp.umap(adata)

Validation:
  ✗ Missing: pca, neighbors

Auto-Insertion:
  # Auto-inserted prerequisites (ordered)
  ov.pp.pca(adata, n_pcs=50)
  ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50)

  # Original code
  ov.pp.umap(adata)
```

**Example 3: Complex Prerequisite (No Auto-Insert)**
```
User: "Run leiden clustering on raw data"

Generated Code:
  ov.pp.leiden(adata, resolution=1.0)

Validation:
  ✗ Missing: qc, preprocess, scale, pca, neighbors

Auto-Insertion:
  NOT PERFORMED (qc and preprocess are complex)

  Instead, escalate to WorkflowEscalator:
  "This requires extensive preprocessing. Consider using:
   ov.pp.preprocess(adata, mode='shiftlog|pearson', n_HVGs=2000)"
```

### Implementation Details

**Auto-Insert Policy**:
- ✅ Auto-insert: scale, pca, neighbors, umap (standard workflows)
- ❌ Never auto-insert: qc, preprocess, batch_correct (need user config)
- ⚠️ Conditional: leiden, louvain (only if neighbors exists)

**Code Generation**:
- Use default parameters from registry metadata
- Add comments explaining auto-insertion
- Preserve user's original code formatting

### Benefits

1. **Seamless UX**: User gets complete working code
2. **No Manual Steps**: Agent handles dependency management
3. **Correct Order**: Dependencies always in right order
4. **Transparent**: Comments show what was auto-inserted

---

## Integration Strategy

### Phase 1: AgentContextInjector Integration

**Week 1**: Implement dynamic system prompt enhancement
- Detect executed functions from AnnData state
- Format prerequisite chains for system prompt
- Inject context before each agent request

**Week 2**: Add conversation-level state tracking
- Track executed functions across conversation
- Update state after each code execution
- Cache context per AnnData object

### Phase 2: DataStateValidator Integration

**Week 3**: Implement pre-execution validation
- Hook into code execution pipeline
- Parse generated code to extract function calls
- Validate prerequisites before execution

**Week 4**: Add auto-correction logic
- Implement simple prerequisite detection
- Generate corrected code with auto-insertion
- Provide validation feedback to agent

### Phase 3: WorkflowEscalator Integration

**Week 5**: Implement complexity analysis
- Calculate workflow complexity scores
- Define complexity thresholds
- Classify workflows as LOW/MEDIUM/HIGH

**Week 6**: Add escalation logic
- Map complexity levels to escalation strategies
- Generate high-level function suggestions
- Integrate with SuggestionEngine

### Phase 4: AutoPrerequisiteInserter Integration

**Week 7**: Implement code insertion
- Define simple vs. complex prerequisites
- Implement dependency resolution for insertion
- Generate properly ordered prerequisite code

**Week 8**: Polish and testing
- End-to-end integration testing
- Performance optimization
- Documentation and examples

---

## Use Cases & Examples

### Use Case 1: Complete Beginner

**Scenario**: User with raw AnnData, no preprocessing

```
User: "Cluster my cells"

Without Layer 3:
  Agent: "Please run preprocessing first: qc, normalize, scale, pca, neighbors"
  User: *manually runs 5 functions*
  User: "Now cluster my cells"
  Agent: "Here's the clustering code"

With Layer 3:
  Agent (with context injection): "I see your data needs preprocessing.
  I'll generate a complete pipeline:"

  Code:
    # Complete preprocessing (escalated to high-level function)
    ov.pp.preprocess(adata, mode="shiftlog|pearson", n_HVGs=2000)
    ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
    ov.pp.leiden(adata, resolution=1.0)

  User: *runs once, done*
```

### Use Case 2: Intermediate User

**Scenario**: User has done QC and normalization

```
User: "Run UMAP visualization"

Without Layer 3:
  Agent: "Run PCA first, then neighbors, then UMAP"
  User: *manually runs 3 functions*

With Layer 3:
  Agent (context-aware): "I see you've completed QC and normalization.
  I'll add the remaining prerequisites:"

  Code:
    # Auto-inserted prerequisites
    ov.pp.pca(adata, n_pcs=50)
    ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50)

    # Requested visualization
    ov.pp.umap(adata)

  User: *runs once, gets UMAP*
```

### Use Case 3: Advanced User (Already Has Prerequisites)

**Scenario**: User has completed full preprocessing

```
User: "Run leiden with resolution=0.5"

Without Layer 3:
  Agent: "Here's the code: ov.pp.leiden(adata, resolution=0.5)"

With Layer 3:
  Agent (validates first): "All prerequisites satisfied ✓"

  Code:
    ov.pp.leiden(adata, resolution=0.5)

  (No unnecessary prerequisite insertion)
```

### Use Case 4: Error Prevention

**Scenario**: User requests invalid operation

```
User: "Run differential expression between clusters"

Without Layer 3:
  Agent: "Code: ov.single.pyDEG(adata, groupby='leiden')"
  User: *executes*
  Error: "Column 'leiden' not found in adata.obs"

With Layer 3:
  Agent (validates): "This requires leiden clustering first.
  I'll generate the complete workflow:"

  Code:
    # Prerequisite for DEG
    ov.pp.leiden(adata, resolution=1.0)

    # Requested analysis
    ov.single.pyDEG(adata, groupby='leiden')

  User: *executes successfully*
```

---

## Technical Challenges & Solutions

### Challenge 1: State Synchronization

**Problem**: AnnData state may change outside of agent (manual user edits)

**Solution**:
- Re-validate state at start of each request
- Use PrerequisiteChecker's detection (confidence-based, not exact)
- Cache with TTL (time-to-live) expiration

### Challenge 2: Parameter Inference

**Problem**: Auto-inserted functions need parameters (e.g., n_neighbors=?)

**Solution**:
- Use sensible defaults from registry metadata
- Allow user to override via system prompt configuration
- Learn from user's previous parameter choices

### Challenge 3: Context Length

**Problem**: Injecting full prerequisite state may exceed token limits

**Solution**:
- Summarize instead of listing all details
- Only inject relevant context (functions related to current request)
- Use token-efficient format (tables, bullet points)

### Challenge 4: Over-Insertion

**Problem**: Agent might insert prerequisites user already ran manually

**Solution**:
- Always validate current state first (PrerequisiteChecker)
- Only insert missing prerequisites with confidence < threshold
- Add comments explaining what was inserted and why

---

## Success Metrics

### Quantitative Metrics

1. **Workflow Completion Rate**: % of user requests that execute successfully on first try
   - Target: 85%+ (up from ~50% without Layer 3)

2. **Average Steps to Success**: Number of code executions needed to complete task
   - Target: 1.5 steps (down from 3-5 steps)

3. **Error Rate**: % of generated code that fails validation
   - Target: <10% (down from ~40%)

4. **Auto-Insertion Rate**: % of simple prerequisites auto-inserted correctly
   - Target: >90%

5. **Escalation Accuracy**: % of complex workflows correctly escalated
   - Target: >80%

### Qualitative Metrics

1. **User Satisfaction**: Measured via feedback
   - "Agent generated complete working code"
   - "Didn't need to manually check prerequisites"

2. **Code Quality**: Generated code is correct and efficient
   - Uses appropriate high-level functions
   - Includes proper parameter defaults
   - Follows best practices

3. **Transparency**: User understands what was auto-inserted
   - Clear comments
   - Obvious prerequisite sections
   - Explanation of escalation

---

## Risk Assessment

### Risk 1: Auto-Insertion Errors

**Likelihood**: Medium
**Impact**: High (generates incorrect code)
**Mitigation**:
- Extensive testing of insertion logic
- Conservative auto-insert policy (only proven-safe functions)
- Validation feedback loop catches errors

### Risk 2: Context Overhead

**Likelihood**: High
**Impact**: Low (increased latency, token usage)
**Mitigation**:
- Cache context per AnnData object
- Use efficient context format
- Only inject relevant subset of context

### Risk 3: Over-Automation

**Likelihood**: Medium
**Impact**: Medium (user loses control)
**Mitigation**:
- Always show what was auto-inserted (comments)
- Allow user to disable auto-insertion via config
- Provide explanation for escalation decisions

### Risk 4: Parameter Mismatch

**Likelihood**: Medium
**Impact**: Medium (suboptimal parameters)
**Mitigation**:
- Use well-tested defaults from registry
- Allow user to override in system prompt
- Learn from user's parameter history

---

## Implementation Phases (8 Weeks)

### Phase 1: AgentContextInjector (Weeks 1-2)
- Implement dynamic system prompt enhancement
- Add conversation-level state tracking
- Test with simple workflows

### Phase 2: DataStateValidator (Weeks 3-4)
- Implement pre-execution validation
- Add auto-correction for simple cases
- Integrate validation feedback

### Phase 3: WorkflowEscalator (Weeks 5-6)
- Implement complexity analysis
- Add escalation logic
- Define escalation targets

### Phase 4: AutoPrerequisiteInserter (Weeks 7-8)
- Implement code insertion logic
- Define simple vs. complex prerequisites
- Polish and end-to-end testing

---

## Success Criteria for Layer 3

Layer 3 will be considered complete when:

1. ✅ **AgentContextInjector** injects current data state into system prompt
2. ✅ **DataStateValidator** validates all code before execution
3. ✅ **WorkflowEscalator** correctly escalates complex workflows (>80% accuracy)
4. ✅ **AutoPrerequisiteInserter** auto-inserts simple prerequisites (>90% success)
5. ✅ **Integration tests** pass for all use cases (100% pass rate)
6. ✅ **Performance** overhead is acceptable (<100ms additional latency)
7. ✅ **Documentation** complete with examples

---

## Next Steps After Layer 3

Once Layer 3 is complete, potential Layer 4 enhancements:

### Layer 4: Learning Agent System
- Learn from user's parameter preferences
- Adapt auto-insertion policy to user's workflow style
- Predict what user wants based on history
- Suggest optimizations based on data characteristics

### Layer 5: Multi-Agent Collaboration
- Separate agents for different analysis types (QC, clustering, DEG, etc.)
- Agents share prerequisite state
- Collaborative workflow planning

---

## Conclusion

Layer 3 transforms the DataStateInspector from a **passive validation tool** into an **intelligent agent system** that:

1. **Knows the current state** (AgentContextInjector)
2. **Validates before execution** (DataStateValidator)
3. **Suggests smart shortcuts** (WorkflowEscalator)
4. **Automatically completes workflows** (AutoPrerequisiteInserter)

**Result**: Users get **complete, working code** on the first try, with minimal manual prerequisite management.

**Timeline**: 8 weeks to implement, test, and document
**Dependencies**: Layer 2 v0.5.0 (complete)
**Risk Level**: Medium (mitigated with testing and conservative policies)

---

**Status**: ✅ **PLAN COMPLETE - READY FOR REVIEW**

No code changes made. This document provides a comprehensive plan for implementing Layer 3: Enhanced Agent Prompt System.

---

**Generated**: 2025-11-11
**Author**: Claude (Anthropic)
**Version**: Layer 3 Planning v1.0

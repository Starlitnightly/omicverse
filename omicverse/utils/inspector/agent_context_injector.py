"""
AgentContextInjector - Inject prerequisite state into LLM system prompts.

This module provides the AgentContextInjector class which enhances LLM system
prompts with current data state, executed functions, and prerequisite information,
making the agent context-aware and proactive.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from anndata import AnnData

from .inspector import DataStateInspector


@dataclass
class ConversationState:
    """Track state across a conversation with the agent.

    Attributes:
        executed_functions: Set of function names executed in this conversation.
        execution_history: List of execution events with timestamps.
        data_snapshots: Snapshots of data state at different points.
        current_context: Current context string injected into system prompt.
    """
    executed_functions: Set[str] = field(default_factory=set)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    data_snapshots: List[Dict[str, Any]] = field(default_factory=list)
    current_context: str = ""

    def add_execution(self, function_name: str, timestamp: Optional[datetime] = None):
        """Record a function execution.

        Args:
            function_name: Name of the function that was executed.
            timestamp: When it was executed (defaults to now).
        """
        if timestamp is None:
            timestamp = datetime.now()

        self.executed_functions.add(function_name)
        self.execution_history.append({
            'function': function_name,
            'timestamp': timestamp,
            'order': len(self.execution_history) + 1,
        })

    def snapshot_data_state(self, state: Dict[str, Any]):
        """Save a snapshot of data state.

        Args:
            state: Dictionary describing current data state.
        """
        self.data_snapshots.append({
            'timestamp': datetime.now(),
            'state': state,
            'functions_executed': len(self.executed_functions),
        })


class AgentContextInjector:
    """Inject prerequisite context into LLM system prompts.

    This class enhances LLM system prompts with:
    - Current AnnData state (what data structures exist)
    - Executed functions (what's been run, with confidence scores)
    - Prerequisite chains (what's required, what's missing)
    - Function-specific context (for targeted operations)

    The injected context makes the LLM agent aware of the current data state
    and enables it to generate complete, working code that handles prerequisites
    automatically.

    Example:
        >>> injector = AgentContextInjector(adata, registry)
        >>> system_prompt = "You are a helpful bioinformatics assistant."
        >>> enhanced = injector.inject_context(system_prompt)
        >>> # Enhanced prompt now includes data state and prerequisite info
        >>> # Agent knows what's been executed and what's missing
    """

    def __init__(self, adata: AnnData, registry: Any):
        """Initialize the context injector.

        Args:
            adata: AnnData object to inspect and track.
            registry: Function registry with prerequisite metadata.
        """
        self.adata = adata
        self.registry = registry
        self.inspector = DataStateInspector(adata, registry)
        self.conversation_state = ConversationState()

        # Take initial snapshot
        initial_state = self._get_current_state()
        self.conversation_state.snapshot_data_state(initial_state)

    def inject_context(
        self,
        system_prompt: str,
        target_function: Optional[str] = None,
        include_general_state: bool = True,
        include_function_specific: bool = True,
        include_instructions: bool = True,
    ) -> str:
        """Inject prerequisite context into system prompt.

        Args:
            system_prompt: Original system prompt for the LLM.
            target_function: Optional specific function being targeted.
            include_general_state: Include general data state summary.
            include_function_specific: Include function-specific context.
            include_instructions: Include prerequisite handling instructions.

        Returns:
            Enhanced system prompt with injected context.

        Example:
            >>> prompt = "You are a helpful assistant."
            >>> enhanced = injector.inject_context(prompt, target_function='leiden')
            >>> # Enhanced prompt now includes:
            >>> # - Current data state (what's been executed)
            >>> # - leiden prerequisites and requirements
            >>> # - Instructions for handling missing prerequisites
        """
        sections = [system_prompt, "\n"]

        if include_general_state:
            sections.append(self._build_general_state_section())
            sections.append("\n")

        if include_function_specific and target_function:
            sections.append(self._build_function_specific_section(target_function))
            sections.append("\n")

        if include_instructions:
            sections.append(self._build_prerequisite_instructions())
            sections.append("\n")

        enhanced_prompt = "\n".join(sections)
        self.conversation_state.current_context = enhanced_prompt

        return enhanced_prompt

    def _build_general_state_section(self) -> str:
        """Build general data state section.

        Returns:
            Formatted section describing current data state.
        """
        state = self._get_current_state()
        executed_funcs = self._detect_executed_functions()

        lines = ["## Current AnnData State\n"]
        lines.append("You are working with an AnnData object that has:\n")

        # Executed functions with confidence
        lines.append("### Executed Functions:")
        if executed_funcs:
            for func_name, confidence in executed_funcs.items():
                status = "✅" if confidence >= 0.7 else "⚠️"
                lines.append(f"  {status} {func_name} (confidence: {confidence:.2f})")
        else:
            lines.append("  (No preprocessing detected - raw data)")

        # Available data structures
        lines.append("\n### Available Data Structures:")

        if state.get('obsm'):
            lines.append(f"  ✅ adata.obsm: {', '.join(state['obsm'])}")
        else:
            lines.append("  ❌ adata.obsm: (empty)")

        if state.get('obsp'):
            lines.append(f"  ✅ adata.obsp: {', '.join(state['obsp'])}")
        else:
            lines.append("  ❌ adata.obsp: (empty)")

        if state.get('uns'):
            lines.append(f"  ✅ adata.uns: {', '.join(list(state['uns'].keys())[:5])}")
        else:
            lines.append("  ❌ adata.uns: (empty)")

        if state.get('layers'):
            lines.append(f"  ✅ adata.layers: {', '.join(state['layers'])}")
        else:
            lines.append("  ❌ adata.layers: (empty)")

        # High-level status
        lines.append("\n### Analysis Status:")
        lines.append(f"  - Data shape: {self.adata.n_obs} cells × {self.adata.n_vars} genes")
        lines.append(f"  - Preprocessing: {'✅ Complete' if executed_funcs else '❌ Not started'}")
        lines.append(f"  - Dimensionality reduction: {'✅ Done' if any('X_pca' in k or 'X_umap' in k for k in state.get('obsm', [])) else '❌ Not done'}")
        lines.append(f"  - Clustering: {'✅ Done' if any('leiden' in k or 'louvain' in k for k in self.adata.obs.columns) else '❌ Not done'}")

        return "\n".join(lines)

    def _build_function_specific_section(self, target_function: str) -> str:
        """Build function-specific context section.

        Args:
            target_function: Function to generate context for.

        Returns:
            Formatted section with function prerequisites and requirements.
        """
        lines = [f"## Target Function: {target_function}\n"]

        # Validate prerequisites
        result = self.inspector.validate_prerequisites(target_function)

        if result.is_valid:
            lines.append("✅ **All prerequisites satisfied!**\n")
            lines.append(f"You can proceed with executing {target_function}.")
            return "\n".join(lines)

        # Show what's missing
        lines.append("### Prerequisites Status:")

        if result.missing_prerequisites:
            lines.append("\n**Missing Prerequisites:**")
            for prereq in result.missing_prerequisites:
                conf = result.confidence_scores.get(prereq, 0.0)
                lines.append(f"  ❌ {prereq} (confidence: {conf:.2f})")

        if result.executed_functions:
            lines.append("\n**Already Executed:**")
            for func in result.executed_functions:
                conf = result.confidence_scores.get(func, 0.0)
                lines.append(f"  ✅ {func} (confidence: {conf:.2f})")

        # Show missing data structures
        if result.missing_data_structures:
            lines.append("\n**Missing Data Structures:**")
            for structure_type, items in result.missing_data_structures.items():
                for item in items:
                    lines.append(f"  ❌ adata.{structure_type}['{item}']")

        # Add recommendations
        if result.suggestions:
            lines.append("\n### Recommendations:")
            for i, suggestion in enumerate(result.suggestions[:3], 1):
                lines.append(f"\n**{i}. [{suggestion.priority}] {suggestion.description}**")
                if suggestion.code:
                    lines.append(f"   Code: `{suggestion.code[:100]}...`")
                if suggestion.explanation:
                    lines.append(f"   Why: {suggestion.explanation[:150]}...")

        return "\n".join(lines)

    def _build_prerequisite_instructions(self) -> str:
        """Build instructions for handling prerequisites.

        Returns:
            Formatted instructions section.
        """
        return """## IMPORTANT: Prerequisite Handling Instructions

When generating code, you MUST follow these rules:

1. **Check Prerequisites First**
   - Look at the "Executed Functions" list above
   - Verify all required functions have been executed
   - Check that required data structures exist

2. **Auto-Insert Simple Prerequisites (1-2 missing)**
   - If only 1-2 prerequisites are missing AND they are "simple"
   - Simple prerequisites: scale, pca, neighbors, umap
   - Generate code that includes BOTH the prerequisite AND the target function
   - Example:
     ```python
     # Auto-insert neighbors before leiden
     ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
     ov.pp.leiden(adata, resolution=1.0)
     ```

3. **Escalate Complex Workflows (3+ missing)**
   - If 3 or more prerequisites are missing
   - OR if prerequisites include: qc, preprocess, batch_correct
   - Suggest using high-level workflow functions instead
   - Example:
     ```python
     # Complex preprocessing needed - use workflow
     adata = ov.pp.preprocess(adata, mode='shiftlog|pearson', n_HVGs=2000)
     # Now run target function
     ov.pp.leiden(adata, resolution=1.0)
     ```

4. **Generate Complete, Executable Code**
   - Always generate code that will execute successfully
   - Include all prerequisites in the correct order
   - Add helpful print statements explaining what's happening
   - Return the modified adata object if needed

REMEMBER: The user should be able to run your generated code WITHOUT errors!
"""

    def _get_current_state(self) -> Dict[str, Any]:
        """Get current AnnData state.

        Returns:
            Dictionary with current data structures.
        """
        return {
            'shape': (self.adata.n_obs, self.adata.n_vars),
            'obsm': list(self.adata.obsm.keys()),
            'obsp': list(self.adata.obsp.keys()),
            'uns': dict(self.adata.uns) if self.adata.uns else {},
            'layers': list(self.adata.layers.keys()) if self.adata.layers else [],
            'obs_columns': list(self.adata.obs.columns),
            'var_columns': list(self.adata.var.columns),
        }

    def _detect_executed_functions(self) -> Dict[str, float]:
        """Detect which functions have been executed.

        Uses PrerequisiteChecker to detect executed functions with confidence scores.

        Returns:
            Dict mapping function names to confidence scores.
        """
        # Get common function names to check
        common_functions = [
            'qc', 'preprocess', 'scale', 'pca',
            'neighbors', 'leiden', 'louvain', 'umap'
        ]

        executed = {}
        for func_name in common_functions:
            try:
                result = self.inspector.prerequisite_checker.check_function_executed(func_name)
                if result.executed or result.confidence >= 0.5:
                    executed[func_name] = result.confidence
            except Exception:
                # Function not in registry or error checking
                continue

        return executed

    def update_after_execution(self, function_name: str):
        """Update state after a function has been executed.

        Call this after executing any OmicVerse function to keep the context
        injector's state synchronized.

        Args:
            function_name: Name of the function that was just executed.

        Example:
            >>> injector.update_after_execution('pca')
            >>> # Context now reflects that PCA has been executed
        """
        self.conversation_state.add_execution(function_name)

        # Take new snapshot
        state = self._get_current_state()
        self.conversation_state.snapshot_data_state(state)

    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation so far.

        Returns:
            Human-readable summary of executed functions and state changes.

        Example:
            >>> summary = injector.get_conversation_summary()
            >>> print(summary)
            Conversation Summary:
              - 3 functions executed: pca, neighbors, leiden
              - Data evolved from raw → preprocessed → clustered
        """
        lines = ["## Conversation Summary\n"]

        if self.conversation_state.executed_functions:
            lines.append(f"### Functions Executed ({len(self.conversation_state.executed_functions)}):")
            for i, event in enumerate(self.conversation_state.execution_history, 1):
                lines.append(f"  {i}. {event['function']} (at {event['timestamp'].strftime('%H:%M:%S')})")
        else:
            lines.append("No functions executed yet in this conversation.")

        if len(self.conversation_state.data_snapshots) > 1:
            lines.append(f"\n### Data Evolution:")
            lines.append(f"  - Started with {self.conversation_state.data_snapshots[0]['functions_executed']} functions executed")
            lines.append(f"  - Now at {self.conversation_state.data_snapshots[-1]['functions_executed']} functions executed")

        return "\n".join(lines)

    def clear_conversation_state(self):
        """Clear conversation state for a fresh start.

        Resets executed functions and history while keeping the current AnnData state.

        Example:
            >>> injector.clear_conversation_state()
            >>> # Conversation history cleared, but data state detection continues
        """
        self.conversation_state = ConversationState()

        # Take fresh snapshot
        initial_state = self._get_current_state()
        self.conversation_state.snapshot_data_state(initial_state)

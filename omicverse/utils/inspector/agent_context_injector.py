"""
AgentContextInjector - Inject prerequisite state into LLM system prompts.

This module provides the AgentContextInjector class which enhances LLM system
prompts with current data state, executed functions, prerequisite information,
and filesystem-based context, making the agent context-aware and proactive.

The filesystem context integration follows LangChain's context engineering principles:
- Write: Offload information to external storage early and often
- Select: Pull in only relevant context when needed (glob/grep)
- Compress: Summarize using structured schema-driven approaches
- Isolate: Use sub-agent architecture with shared workspaces

Reference: https://blog.langchain.com/how-agents-can-use-filesystems-for-context-engineering/
"""

from typing import Dict, List, Optional, Any, Set, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from anndata import AnnData

from .inspector import DataStateInspector

if TYPE_CHECKING:
    from ..filesystem_context import FilesystemContextManager


@dataclass
class ConversationState:
    """Track state across a conversation with the agent.

    Attributes:
        executed_functions: Set of function names executed in this conversation.
        execution_history: List of execution events with timestamps.
        data_snapshots: Snapshots of data state at different points.
        current_context: Current context string injected into system prompt.
        filesystem_notes: References to notes written to filesystem context.
        filesystem_session_id: Session ID for filesystem context workspace.
    """
    executed_functions: Set[str] = field(default_factory=set)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    data_snapshots: List[Dict[str, Any]] = field(default_factory=list)
    current_context: str = ""
    filesystem_notes: List[str] = field(default_factory=list)
    filesystem_session_id: Optional[str] = None

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
    - Filesystem-based context (notes, plans, intermediate results)

    The injected context makes the LLM agent aware of the current data state
    and enables it to generate complete, working code that handles prerequisites
    automatically.

    Filesystem Context Integration:
        The injector can optionally use a FilesystemContextManager to:
        - Offload intermediate results to disk (reducing context window usage)
        - Search for relevant context using glob/grep patterns
        - Share context between parent and sub-agents
        - Persist execution plans across sessions

    Example:
        >>> injector = AgentContextInjector(adata, registry)
        >>> system_prompt = "You are a helpful bioinformatics assistant."
        >>> enhanced = injector.inject_context(system_prompt)
        >>> # Enhanced prompt now includes data state and prerequisite info
        >>> # Agent knows what's been executed and what's missing

    Example with filesystem context:
        >>> from omicverse.utils.filesystem_context import FilesystemContextManager
        >>> fs_ctx = FilesystemContextManager()
        >>> injector = AgentContextInjector(adata, registry, filesystem_context=fs_ctx)
        >>> # Now context can be offloaded and selectively retrieved
    """

    def __init__(
        self,
        adata: AnnData,
        registry: Any,
        filesystem_context: Optional['FilesystemContextManager'] = None,
        enable_filesystem_context: bool = True,
    ):
        """Initialize the context injector.

        Args:
            adata: AnnData object to inspect and track.
            registry: Function registry with prerequisite metadata.
            filesystem_context: Optional FilesystemContextManager for persistent context.
                If not provided and enable_filesystem_context is True, one will be created.
            enable_filesystem_context: Whether to enable filesystem-based context management.
                Default is True. Set to False for lightweight usage without filesystem I/O.
        """
        self.adata = adata
        self.registry = registry
        self.inspector = DataStateInspector(adata, registry)
        self.conversation_state = ConversationState()

        # Filesystem context management
        self.enable_filesystem_context = enable_filesystem_context
        self._filesystem_context: Optional['FilesystemContextManager'] = None

        if enable_filesystem_context:
            if filesystem_context is not None:
                self._filesystem_context = filesystem_context
            else:
                # Lazy initialization - create on first use
                pass

        # Take initial snapshot
        initial_state = self._get_current_state()
        self.conversation_state.snapshot_data_state(initial_state)

        # Store session ID if filesystem context is available
        if self._filesystem_context is not None:
            self.conversation_state.filesystem_session_id = self._filesystem_context.session_id

    @property
    def filesystem_context(self) -> Optional['FilesystemContextManager']:
        """Get the filesystem context manager, creating one if needed.

        Returns:
            FilesystemContextManager or None if filesystem context is disabled.
        """
        if not self.enable_filesystem_context:
            return None

        if self._filesystem_context is None:
            # Lazy initialization
            try:
                from ..filesystem_context import FilesystemContextManager
                self._filesystem_context = FilesystemContextManager()
                self.conversation_state.filesystem_session_id = self._filesystem_context.session_id
            except ImportError:
                self.enable_filesystem_context = False
                return None

        return self._filesystem_context

    def inject_context(
        self,
        system_prompt: str,
        target_function: Optional[str] = None,
        include_general_state: bool = True,
        include_function_specific: bool = True,
        include_instructions: bool = True,
        include_filesystem_context: bool = True,
        filesystem_query: Optional[str] = None,
        max_filesystem_tokens: int = 1000,
    ) -> str:
        """Inject prerequisite context into system prompt.

        Args:
            system_prompt: Original system prompt for the LLM.
            target_function: Optional specific function being targeted.
            include_general_state: Include general data state summary.
            include_function_specific: Include function-specific context.
            include_instructions: Include prerequisite handling instructions.
            include_filesystem_context: Include relevant filesystem context.
            filesystem_query: Query to search filesystem context. If None, uses
                target_function or a default search.
            max_filesystem_tokens: Maximum tokens for filesystem context section.

        Returns:
            Enhanced system prompt with injected context.

        Example:
            >>> prompt = "You are a helpful assistant."
            >>> enhanced = injector.inject_context(prompt, target_function='leiden')
            >>> # Enhanced prompt now includes:
            >>> # - Current data state (what's been executed)
            >>> # - leiden prerequisites and requirements
            >>> # - Instructions for handling missing prerequisites
            >>> # - Relevant filesystem context (notes, plan, etc.)
        """
        sections = [system_prompt, "\n"]

        if include_general_state:
            sections.append(self._build_general_state_section())
            sections.append("\n")

        if include_function_specific and target_function:
            sections.append(self._build_function_specific_section(target_function))
            sections.append("\n")

        # Include filesystem context if enabled
        if include_filesystem_context and self.enable_filesystem_context:
            fs_section = self._build_filesystem_context_section(
                query=filesystem_query or target_function,
                max_tokens=max_filesystem_tokens,
            )
            if fs_section:
                sections.append(fs_section)
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

    # =========================================================================
    # Filesystem Context Methods
    # =========================================================================

    def _build_filesystem_context_section(
        self,
        query: Optional[str] = None,
        max_tokens: int = 1000,
    ) -> str:
        """Build filesystem context section for prompt injection.

        Args:
            query: Query to search for relevant context.
            max_tokens: Maximum tokens for this section.

        Returns:
            Formatted filesystem context section, or empty string if none available.
        """
        fs_ctx = self.filesystem_context
        if fs_ctx is None:
            return ""

        try:
            # Get relevant context based on query
            if query:
                context = fs_ctx.get_relevant_context(
                    query=query,
                    max_tokens=max_tokens,
                    include_plan=True,
                    include_recent=3,
                )
            else:
                context = fs_ctx.get_relevant_context(
                    query="analysis preprocessing clustering",
                    max_tokens=max_tokens,
                    include_plan=True,
                    include_recent=5,
                )

            if not context or len(context.strip()) < 10:
                return ""

            return f"## Workspace Context\n\n{context}"

        except Exception:
            # Silently fail - filesystem context is optional
            return ""

    def write_to_context(
        self,
        key: str,
        content: Any,
        category: str = "notes",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Write information to the filesystem context.

        Use this to offload intermediate results, observations, or decisions
        from the context window to persistent storage.

        Args:
            key: Unique identifier for this note.
            content: Content to store (string or dict).
            category: Category for organization (notes, results, decisions, etc.).
            metadata: Additional metadata to store.

        Returns:
            Path to the stored note, or None if filesystem context is disabled.

        Example:
            >>> injector.write_to_context(
            ...     "clustering_result",
            ...     {"n_clusters": 8, "resolution": 1.0},
            ...     category="results"
            ... )
        """
        fs_ctx = self.filesystem_context
        if fs_ctx is None:
            return None

        try:
            path = fs_ctx.write_note(key, content, category, metadata)
            self.conversation_state.filesystem_notes.append(f"{category}/{key}")
            return path
        except Exception:
            return None

    def search_context(
        self,
        pattern: str,
        match_type: str = "glob",
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search the filesystem context for relevant notes.

        Args:
            pattern: Search pattern (glob pattern or regex for grep).
            match_type: "glob" for filename matching, "grep" for content search.
            max_results: Maximum results to return.

        Returns:
            List of matching results with key, category, and content preview.

        Example:
            >>> results = injector.search_context("cluster*", match_type="glob")
            >>> results = injector.search_context("resolution", match_type="grep")
        """
        fs_ctx = self.filesystem_context
        if fs_ctx is None:
            return []

        try:
            results = fs_ctx.search_context(pattern, match_type, max_results=max_results)
            return [
                {
                    "key": r.key,
                    "category": r.category,
                    "preview": r.content_preview,
                    "relevance": r.relevance_score,
                }
                for r in results
            ]
        except Exception:
            return []

    def save_execution_plan(self, steps: List[Dict[str, Any]]) -> Optional[str]:
        """Save an execution plan to the filesystem context.

        Args:
            steps: List of step definitions with description and status.

        Returns:
            Path to the plan file, or None if filesystem context is disabled.

        Example:
            >>> injector.save_execution_plan([
            ...     {"description": "Run QC", "status": "pending"},
            ...     {"description": "Normalize data", "status": "pending"},
            ...     {"description": "Cluster cells", "status": "pending"},
            ... ])
        """
        fs_ctx = self.filesystem_context
        if fs_ctx is None:
            return None

        try:
            return fs_ctx.write_plan(steps)
        except Exception:
            return None

    def update_plan_step(
        self,
        step_index: int,
        status: str,
        result: Optional[str] = None,
    ) -> None:
        """Update the status of a plan step.

        Args:
            step_index: Index of the step (0-based).
            status: New status (pending, in_progress, completed, failed).
            result: Optional result or notes for this step.
        """
        fs_ctx = self.filesystem_context
        if fs_ctx is None:
            return

        try:
            fs_ctx.update_plan_step(step_index, status, result)
        except Exception:
            pass

    def save_data_snapshot(
        self,
        step_number: Optional[int] = None,
        description: Optional[str] = None,
    ) -> Optional[str]:
        """Save a snapshot of the current AnnData state to filesystem.

        Args:
            step_number: Optional step number for ordering.
            description: Human-readable description of the snapshot.

        Returns:
            Path to the snapshot file.
        """
        fs_ctx = self.filesystem_context
        if fs_ctx is None:
            return None

        try:
            state = self._get_current_state()
            return fs_ctx.write_snapshot(state, step_number, description)
        except Exception:
            return None

    def get_workspace_summary(self) -> str:
        """Get a summary of the filesystem workspace.

        Returns:
            Markdown-formatted workspace summary.
        """
        fs_ctx = self.filesystem_context
        if fs_ctx is None:
            return "Filesystem context is disabled."

        try:
            return fs_ctx.get_session_summary()
        except Exception:
            return "Error getting workspace summary."

    def create_sub_agent_injector(
        self,
        adata: Optional[AnnData] = None,
    ) -> 'AgentContextInjector':
        """Create a context injector for a sub-agent that shares the workspace.

        Args:
            adata: AnnData to use for the sub-agent. If None, uses the same adata.

        Returns:
            New AgentContextInjector that shares the filesystem workspace.

        Example:
            >>> sub_injector = injector.create_sub_agent_injector()
            >>> # sub_injector shares the same filesystem workspace
        """
        sub_adata = adata if adata is not None else self.adata

        if self.filesystem_context is not None:
            sub_fs_ctx = self.filesystem_context.create_sub_agent_context()
        else:
            sub_fs_ctx = None

        return AgentContextInjector(
            adata=sub_adata,
            registry=self.registry,
            filesystem_context=sub_fs_ctx,
            enable_filesystem_context=self.enable_filesystem_context,
        )

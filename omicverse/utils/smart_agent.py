"""
OmicVerse Smart Agent (internal LLM backend)

This module provides a smart agent that can understand natural language requests
and automatically execute appropriate OmicVerse functions. It now uses a built-in
LLM backend (see `agent_backend.py`) instead of the external Pantheon framework.

Usage:
    import omicverse as ov
    result = ov.Agent("quality control with nUMI>500, mito<0.2", adata)
"""

import sys
import os
import asyncio
import json
import re
import inspect
import ast
import textwrap
import time
import builtins
import warnings
import threading
import traceback
import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

# ---------------------------------------------------------------------------
# Compatibility helpers
# ---------------------------------------------------------------------------

# Some of the test doubles create a lightweight ``omicverse`` package stub that
# does not populate the ``utils`` attribute on the parent package or expose this
# module as ``omicverse.utils.smart_agent``.  Python 3.10 (used in CI) requires
# these attributes to be set manually for ``unittest.mock.patch`` lookups to
# succeed.  When this module is imported in the test suite, we make sure the
# parent references are wired correctly.
_parent_pkg = sys.modules.get("omicverse")
_utils_pkg = sys.modules.get("omicverse.utils")
if _parent_pkg is not None and _utils_pkg is not None:
    # Avoid ``hasattr`` here: the real ``omicverse`` package implements
    # ``__getattr__`` for lazy imports, and probing ``utils`` would re-enter the
    # lazy loader while ``omicverse.utils.smart_agent`` is still importing.
    parent_attrs = getattr(_parent_pkg, "__dict__", {})
    if "utils" not in parent_attrs:
        setattr(_parent_pkg, "utils", _utils_pkg)

    module_name = __name__.split(".")[-1]
    utils_attrs = getattr(_utils_pkg, "__dict__", {})
    if module_name not in utils_attrs:
        setattr(_utils_pkg, module_name, sys.modules[__name__])


# Internal LLM backend (Pantheon replacement)
from .agent_backend import OmicVerseLLMBackend, Usage

# Import registry system and model configuration
from .._registry import _global_registry
from .model_config import ModelConfig, PROVIDER_API_KEYS

# P0-2: Grouped configuration dataclasses
from .agent_config import AgentConfig, SandboxFallbackPolicy

# P0-3: Structured error hierarchy
from .agent_errors import (
    OVAgentError,
    ProviderError,
    ConfigError,
    ExecutionError,
    SandboxDeniedError,
    SecurityViolationError,
)

# P2-4: Sandbox security hardening
from .agent_sandbox import (
    ApprovalMode,
    CodeSecurityScanner,
    SafeOsProxy,
    SecurityConfig,
)

# P1-1: Structured event reporting
from .agent_reporter import (
    AgentEvent,
    EventLevel,
    Reporter,
    make_reporter,
)
from .harness import (
    RunTraceRecorder,
    RunTraceStore,
    build_stream_event,
    coerce_usage_payload,
    hash_code_block,
    make_turn_id,
)
from .harness.runtime_state import runtime_state
from .harness.tool_catalog import (
    get_default_loaded_tool_names,
    get_tool_spec,
    get_visible_tool_schemas,
    normalize_tool_name,
)
from .context_compactor import ContextCompactor
from .ovagent.runtime import OmicVerseRuntime
from .ovagent.auth import (
    ResolvedBackend as _ResolvedBackend,
    resolve_model_and_provider as _resolve_model_and_provider,
    collect_api_key_env as _collect_api_key_env,
    temporary_api_keys as _temporary_api_keys_cm,
    display_backend_info as _display_backend_info,
)
from .ovagent.bootstrap import (
    format_skill_overview as _format_skill_overview,
    initialize_skill_registry as _initialize_skill_registry,
    initialize_notebook_executor as _initialize_notebook_executor,
    initialize_filesystem_context as _initialize_filesystem_context,
    initialize_session_history as _initialize_session_history,
    initialize_tracing as _initialize_tracing,
    initialize_security as _initialize_security,
    initialize_ov_runtime as _initialize_ov_runtime,
    create_llm_backend as _create_llm_backend,
    display_reflection_config as _display_reflection_config,
)
from .ovagent.prompt_builder import (
    PromptBuilder as _PromptBuilder,
    CODE_QUALITY_RULES as _CODE_QUALITY_RULES_EXT,
    build_filesystem_context_instructions as _build_filesystem_context_instructions,
)
from .ovagent.analysis_executor import (
    AnalysisExecutor as _AnalysisExecutor,
    ProactiveCodeTransformer as _ProactiveCodeTransformerExt,
)
from .ovagent.tool_runtime import ToolRuntime as _ToolRuntime
from .ovagent.subagent_controller import SubagentController as _SubagentController
from .ovagent.turn_controller import (
    TurnController as _TurnController,
    FollowUpGate as _FollowUpGate,
)
from .session_history import HistoryEntry, SessionHistory
from .skill_registry import (
    SkillMatch,
    SkillMetadata,
    SkillDefinition,
    SkillRegistry,
    SkillRouter,
    build_skill_registry,
    discover_multi_path_skill_roots,
    build_multi_path_skill_registry,
)

# Import filesystem context management for context engineering
# Reference: https://blog.langchain.com/how-agents-can-use-filesystems-for-context-engineering/
from .filesystem_context import FilesystemContextManager


logger = logging.getLogger(__name__)


# ProactiveCodeTransformer is now in ovagent/analysis_executor.py — keep a
# backward-compatible alias so existing code/tests referencing it still work.
ProactiveCodeTransformer = _ProactiveCodeTransformerExt


class OmicVerseAgent:
    """
    Intelligent agent for OmicVerse function discovery and execution.

    This agent uses an internal LLM backend to understand natural language
    requests and automatically execute appropriate OmicVerse functions.

    Usage:
        agent = ov.Agent(api_key="your-api-key")  # Uses gpt-5.2 by default
        result_adata = agent.run("quality control with nUMI>500, mito<0.2", adata)
    """
    
    def __init__(self, model: str = "gpt-5.2", api_key: Optional[str] = None, endpoint: Optional[str] = None, enable_reflection: bool = True, reflection_iterations: int = 1, enable_result_review: bool = True, use_notebook_execution: bool = True, max_prompts_per_session: int = 5, notebook_storage_dir: Optional[str] = None, keep_execution_notebooks: bool = True, notebook_timeout: int = 600, strict_kernel_validation: bool = True, enable_filesystem_context: bool = True, context_storage_dir: Optional[str] = None, approval_mode: str = "never", agent_mode: str = "agentic", max_agent_turns: int = 15, security_level: Optional[str] = None, *, config: Optional[AgentConfig] = None, reporter: Optional[Reporter] = None, verbose: bool = True):
        """
        Initialize the OmicVerse Smart Agent.

        Parameters
        ----------
        model : str
            LLM model to use for reasoning (default: "gpt-5.2")
        api_key : str, optional
            API key for the model provider. If not provided, will use environment variable
        endpoint : str, optional
            Custom API endpoint. If not provided, will use default for the provider
        enable_reflection : bool, optional
            Enable reflection step to review and improve generated code (default: True)
        reflection_iterations : int, optional
            Maximum number of reflection iterations (default: 1, range: 1-3)
        enable_result_review : bool, optional
            Enable result review to validate output matches user intent (default: True)
        use_notebook_execution : bool, optional
            Execute code in separate Jupyter notebook for isolation and debugging (default: True).
            Set to False to use legacy in-process execution.
        max_prompts_per_session : int, optional
            Number of prompts to execute in same notebook session before restart (default: 5).
            This prevents memory bloat while maintaining context for iterative analysis.
        notebook_storage_dir : str, optional
            Directory to store session notebooks. Defaults to ~/.ovagent/sessions
        keep_execution_notebooks : bool, optional
            Whether to keep session notebooks after execution (default: True)
        notebook_timeout : int, optional
            Execution timeout in seconds (default: 600)
        strict_kernel_validation : bool, optional
            If True, raise error if kernel not found. If False, fall back to python3 kernel (default: True)
        enable_filesystem_context : bool, optional
            Enable filesystem-based context management for offloading intermediate results,
            plans, and notes to disk. This reduces context window usage and enables
            selective context retrieval. Default: True.
        context_storage_dir : str, optional
            Directory for storing context files. Defaults to ~/.ovagent/context/
        config : AgentConfig, optional
            Grouped configuration object.  When provided, its values take priority
            over the flat keyword arguments above.
        reporter : Reporter, optional
            Structured event reporter.  When omitted a default reporter is
            created based on the *verbose* flag.
        verbose : bool, optional
            Whether to emit events to stdout (default: True).
        """

        # --- Build AgentConfig (P0-2) ------------------------------------------
        if config is not None:
            self._config = config
        else:
            if agent_mode != "agentic":
                import warnings
                warnings.warn(
                    "agent_mode='legacy' is deprecated and ignored. "
                    "Agentic mode is now the only execution mode.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            self._config = AgentConfig.from_flat_kwargs(
                model=model,
                api_key=api_key,
                endpoint=endpoint,
                enable_reflection=enable_reflection,
                reflection_iterations=reflection_iterations,
                enable_result_review=enable_result_review,
                use_notebook_execution=use_notebook_execution,
                max_prompts_per_session=max_prompts_per_session,
                notebook_storage_dir=notebook_storage_dir,
                keep_execution_notebooks=keep_execution_notebooks,
                notebook_timeout=notebook_timeout,
                strict_kernel_validation=strict_kernel_validation,
                enable_filesystem_context=enable_filesystem_context,
                context_storage_dir=context_storage_dir,
                approval_mode=approval_mode,
                max_agent_turns=max_agent_turns,
                security_level=security_level,
            )
            self._config.verbose = verbose

        # --- Build Reporter (P1-1) ---------------------------------------------
        self._reporter: Reporter = make_reporter(
            verbose=self._config.verbose,
            reporter=reporter,
        )

        def _emit(level: EventLevel, message: str, category: str = "") -> None:
            self._reporter.emit(AgentEvent(level=level, message=message, category=category))

        self._emit = _emit

        _emit(EventLevel.INFO, "Initializing OmicVerse Smart Agent (internal backend)...", "init")

        # --- Auth & backend resolution (ovagent.auth) --------------------------
        backend = _resolve_model_and_provider(model, api_key, endpoint)
        self.model = backend.model
        self.api_key = api_key
        self.endpoint = backend.endpoint
        self.provider = backend.provider

        # --- Attribute defaults ------------------------------------------------
        self._llm: Optional[OmicVerseLLMBackend] = None
        self.skill_registry: Optional[SkillRegistry] = None
        self._skill_overview_text: str = ""
        self._use_llm_skill_matching: bool = True
        self._managed_api_env: Dict[str, str] = {}
        self.enable_reflection = enable_reflection
        self.reflection_iterations = max(1, min(3, reflection_iterations))
        self.enable_result_review = enable_result_review
        self.use_notebook_execution = use_notebook_execution
        self.max_prompts_per_session = max_prompts_per_session
        self._notebook_executor = None
        self.enable_filesystem_context = enable_filesystem_context
        self._filesystem_context: Optional[FilesystemContextManager] = None
        self.last_usage = None
        self.last_usage_breakdown: Dict[str, Any] = {
            'generation': None,
            'reflection': [],
            'review': [],
            'total': None
        }
        self._session_history: Optional[SessionHistory] = None
        self._trace_store: Optional[RunTraceStore] = None
        self._context_compactor: Optional[ContextCompactor] = None
        self._last_run_trace = None
        self._approval_handler = None
        self._web_session_id = ""
        self._ov_runtime: Optional[OmicVerseRuntime] = None
        self._active_run_id = ""

        # --- API key env collection (ovagent.auth) -----------------------------
        try:
            self._managed_api_env = _collect_api_key_env(
                self.model, self.endpoint, api_key,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to collect API key environment variables: %s", exc)
            self._managed_api_env = {}

        # --- Skill registry (ovagent.bootstrap) --------------------------------
        self.skill_registry, self._skill_overview_text = _initialize_skill_registry()

        # --- Display backend info (ovagent.auth) -------------------------------
        _display_backend_info(
            self.model, self.endpoint, self.provider,
            self.api_key, self._managed_api_env,
        )

        # --- Subsystem bootstrap (ovagent.bootstrap) ---------------------------
        try:
            with self._temporary_api_keys():
                self._setup_agent()
            stats = self._get_registry_stats()
            print(f"   📚 Function registry loaded: {stats['total_functions']} functions in {stats['categories']} categories")

            _display_reflection_config(
                self.enable_reflection,
                self.reflection_iterations,
                self.enable_result_review,
            )

            # Notebook executor
            nb_storage = Path(notebook_storage_dir) if notebook_storage_dir else None
            self.use_notebook_execution, self._notebook_executor = (
                _initialize_notebook_executor(
                    use_notebook=self.use_notebook_execution,
                    storage_dir=nb_storage,
                    max_prompts_per_session=max_prompts_per_session,
                    keep_notebooks=keep_execution_notebooks,
                    timeout=notebook_timeout,
                    strict_kernel_validation=strict_kernel_validation,
                )
            )

            # Filesystem context
            ctx_storage = Path(context_storage_dir) if context_storage_dir else None
            self.enable_filesystem_context, self._filesystem_context = (
                _initialize_filesystem_context(
                    enabled=self.enable_filesystem_context,
                    storage_dir=ctx_storage,
                )
            )

            # Session history
            self._session_history = _initialize_session_history(self._config)

            # Harness tracing & context compaction
            self._trace_store, self._context_compactor = _initialize_tracing(
                self._config, self._llm, self.model,
            )

            # Security scanner
            self._security_config, self._security_scanner = _initialize_security(
                self._config,
            )

            # OmicVerse runtime (workflow / run-store bridge)
            self._ov_runtime = _initialize_ov_runtime(self._detect_repo_root())

            # Extracted module delegates
            self._prompt_builder = _PromptBuilder(self)
            self._analysis_executor = _AnalysisExecutor(self)
            self._tool_runtime = _ToolRuntime(self, self._analysis_executor)
            self._subagent_controller = _SubagentController(
                self, self._prompt_builder, self._tool_runtime,
            )
            self._tool_runtime.set_subagent_controller(self._subagent_controller)
            self._turn_controller = _TurnController(
                self, self._prompt_builder, self._tool_runtime,
            )

            print("✅ Smart Agent initialized successfully!")
        except Exception as e:
            print(f"❌ Agent initialization failed: {e}")
            raise

    def _initialize_skill_registry(self) -> None:
        """Load skills from package install and current working directory.

        Delegates to :func:`ovagent.bootstrap.initialize_skill_registry`.
        """
        self.skill_registry, self._skill_overview_text = _initialize_skill_registry()

    def _get_registry_stats(self) -> dict:
        """Get statistics about the function registry."""
        # Use static AST scan to count all registered functions (not just imported ones)
        static_entries = self._load_static_registry_entries()
        unique_functions = set(e['full_name'] for e in static_entries)
        categories = set(e['category'] for e in static_entries)

        # Fall back to runtime registry if static scan finds nothing
        if not unique_functions:
            for entry in _global_registry._registry.values():
                unique_functions.add(entry['full_name'])
                categories.add(entry['category'])

        return {
            'total_functions': len(unique_functions),
            'categories': len(categories),
            'category_list': list(categories)
        }
    
    def _get_available_functions_info(self) -> str:
        """Get formatted information about all available functions."""
        functions_info = []
        
        # Get all unique functions from registry
        processed_functions = set()
        for entry in _global_registry._registry.values():
            full_name = entry['full_name']
            if full_name in processed_functions:
                continue
            processed_functions.add(full_name)
            
            # Format function information
            info = {
                'name': entry['short_name'],
                'full_name': entry['full_name'],
                'description': entry['description'],
                'aliases': entry['aliases'],
                'category': entry['category'],
                'signature': entry['signature'],
                'examples': entry['examples']
            }
            functions_info.append(info)
        
        return json.dumps(functions_info, indent=2, ensure_ascii=False)
    
    def _collect_api_key_env(self, api_key: Optional[str] = None) -> Dict[str, str]:
        """Collect environment variables required for API authentication.

        Delegates to :func:`ovagent.auth.collect_api_key_env`.
        """
        return _collect_api_key_env(self.model, self.endpoint, api_key)

    def help_short(self) -> str:
        """Return a short, non-expert help string with sample prompts."""
        examples = [
            "basic single-cell QC and clustering",
            "batch integration with harmony on this adata",
            "simple trajectory with DPT, root on paul15_clusters=7MEP, list top genes",
            "find markers for each Leiden cluster (wilcoxon)",
            "doublet check and report rate",
        ]
        return (
            "ov.Agent quick start (use only your provided adata):\n"
            "- " + "\n- ".join(examples) + "\n"
            "Notes: do not create new/dummy AnnData; prefer use_raw=False unless you need raw; "
            "allowed libs: omicverse/scanpy/matplotlib."
        )

    def _build_filesystem_context_instructions(self) -> str:
        """Build instructions for using the filesystem context workspace.

        Delegates to :func:`ovagent.prompt_builder.build_filesystem_context_instructions`.
        """
        session_id = self._filesystem_context.session_id if self._filesystem_context else "N/A"
        return _build_filesystem_context_instructions(session_id)

    @contextmanager
    def _temporary_api_keys(self):
        """Temporarily inject API keys into the environment and clean up afterwards.

        Delegates to :func:`ovagent.auth.temporary_api_keys`.
        """
        with _temporary_api_keys_cm(self._managed_api_env):
            yield

    def _setup_agent(self):
        """Setup the internal agent backend with dynamic instructions."""
        
        # Get current function information dynamically
        functions_info = self._get_available_functions_info()
        
        instructions = """
You are an intelligent OmicVerse assistant that can automatically discover and execute functions based on natural language requests.

## Available OmicVerse Functions

Here are all the currently registered functions in OmicVerse:

""" + functions_info + """

## Your Task

When given a natural language request and an adata object, you should:

Quick-start examples (non-experts can copy/paste):
- "basic single-cell QC and clustering" (uses QC → preprocess → neighbors/UMAP → Leiden → markers)
- "batch integration with harmony on this adata" (uses harmony then neighbors/UMAP/Leiden, use_raw=False)
- "simple trajectory with DPT, root on paul15_clusters=7MEP, list top genes"
- "find markers for each Leiden cluster (wilcoxon)".
- "doublet check and report rate"

1. **Analyze the request** to understand what the user wants to accomplish
2. **Find the most appropriate function** from the available functions above
3. **Extract parameters** from the user's request (e.g., "nUMI>500" means min_genes=500)
4. **Generate and execute Python code** using the appropriate OmicVerse function
5. **Return the modified adata object**

## Parameter Extraction Rules

Extract parameters dynamically based on patterns in the user request:

- For qc function: Create tresh dict with 'mito_perc', 'nUMIs', 'detected_genes'
  - "nUMI>X", "umi>X" → tresh={'nUMIs': X, 'detected_genes': 250, 'mito_perc': 0.15}
  - "mito<X", "mitochondrial<X" → include in tresh dict as 'mito_perc': X
  - "genes>X" → include in tresh dict as 'detected_genes': X
  - Always provide complete tresh dict with all three keys
- "resolution=X" → resolution=X
- "n_pcs=X", "pca=X" → n_pcs=X
- "max_value=X" → max_value=X
- Mode indicators: "seurat", "mads", "pearson" → mode="seurat"
- Boolean indicators: "no doublets", "skip doublets" → doublets=False

## Code Execution Rules

1. **Always import omicverse as ov** at the start
2. **Use the exact function signature** from the available functions
3. **Handle the adata variable** - it will be provided in the context
4. **Update adata in place** when possible
5. **Print success messages** and basic info about the result

## Example Workflow

User request: "quality control with nUMI>500, mito<0.2"

1. Find function: Look for functions with aliases containing "qc", "quality", or "质控"
2. Get function details: Check that qc requires tresh dict with 'mito_perc', 'nUMIs', 'detected_genes'
3. Extract parameters: nUMI>500 → tresh['nUMIs']=500, mito<0.2 → tresh['mito_perc']=0.2
4. Generate code:
   ```python
   import omicverse as ov
   # Execute quality control with complete tresh dict
   adata = ov.pp.qc(adata, tresh={'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250})
   print("QC completed. Dataset shape: " + str(adata.shape[0]) + " cells × " + str(adata.shape[1]) + " genes")
   ```

## Important Notes

- Always work with the provided `adata` variable
- Use the function signatures exactly as shown in the available functions
- Provide helpful feedback about what was executed
- Do not create dummy AnnData objects; operate directly on the provided data
- Prefer `use_raw=False` unless the user explicitly requests raw
- Handle errors gracefully and suggest alternatives if needed

## CRITICAL CODE PATTERNS - MANDATORY RULES

### Print Statements
- ALWAYS use string concatenation: `print("Result: " + str(value))`
- NEVER use f-strings in print statements - they cause format errors

### In-Place Functions (pca, scale, neighbors, leiden, umap, tsne)
- ALWAYS call without assignment: `ov.pp.pca(adata, n_pcs=50)`
- NEVER assign result: `adata = ov.pp.pca(adata)` (returns None!)
- These functions modify adata in-place and return None

### Categorical Column Access
- ALWAYS check dtype before .cat: `if hasattr(col, 'cat'): col.cat.categories`
- NEVER assume column is categorical - it may be string or object dtype
- CORRECT: `adata.obs['col'].value_counts()` (works for any dtype)

### HVG Selection (highly_variable_genes)
- ALWAYS wrap in try/except with seurat fallback:
  ```python
  try:
      sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=2000)
  except ValueError:
      sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=2000)
  ```
- NEVER use seurat_v3 without fallback - fails on small datasets

### Batch Column Handling
- ALWAYS validate batch column before batch operations:
  ```python
  if 'batch' in adata.obs.columns:
      adata.obs['batch'] = adata.obs['batch'].astype(str).fillna('unknown')
  ```
- NEVER assume batch column exists or has valid values
"""

        if self._skill_overview_text:
            instructions += (
                "\n\n## Project Skill Catalog\n"
                "OmicVerse provides curated Agent Skills that capture end-to-end workflows. "
                "Before executing complex tasks, call `_list_project_skills` to view the catalog and `_load_skill_guidance` "
                "to read detailed instructions for relevant skills. Follow the selected skill guidance when planning code "
                "execution.\n\n"
                f"{self._skill_overview_text}"
            )

        # Add filesystem context instructions if enabled
        if self.enable_filesystem_context and self._filesystem_context:
            instructions += self._build_filesystem_context_instructions()
        
        # Prepare API key environment pin if passed (non-destructive)
        if self.api_key:
            required_key = PROVIDER_API_KEYS.get(self.model)
            if required_key and not os.getenv(required_key):
                os.environ[required_key] = self.api_key

        # Create the internal LLM backend
        self._llm = _create_llm_backend(
            system_prompt=instructions,
            model=self.model,
            api_key=self.api_key,
            endpoint=self.endpoint,
        )
    
    def _search_functions(self, query: str) -> str:
        """
        Search for functions in the OmicVerse registry.
        
        Parameters
        ----------
        query : str
            Search query
            
        Returns
        -------
        str
            JSON formatted search results
        """
        try:
            results = _global_registry.find(query)
            
            if not results:
                return json.dumps({"error": f"No functions found for query: '{query}'"})
            
            # Format results for the agent
            formatted_results = []
            for entry in results:
                formatted_results.append({
                    'name': entry['short_name'],
                    'full_name': entry['full_name'],
                    'description': entry['description'],
                    'signature': entry['signature'],
                    'aliases': entry['aliases'],
                    'examples': entry['examples'],
                    'category': entry['category']
                })
            
            return json.dumps({
                "found": len(formatted_results),
                "functions": formatted_results
            }, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Error searching functions: {str(e)}"})
    
    def _get_function_details(self, function_name: str) -> str:
        """
        Get detailed information about a specific function.

        Parameters
        ----------
        function_name : str
            Function name or alias
            
        Returns
        -------
        str
            JSON formatted function details
        """
        try:
            results = _global_registry.find(function_name)
            
            if not results:
                return json.dumps({"error": f"Function '{function_name}' not found"})
            
            entry = results[0]  # Get first match
            
            return json.dumps({
                'name': entry['short_name'],
                'full_name': entry['full_name'],
                'description': entry['description'],
                'signature': entry['signature'],
                'parameters': entry.get('parameters', []),
                'aliases': entry['aliases'],
                'examples': entry['examples'],
                'category': entry['category'],
                'docstring': entry['docstring'],
                'help': f"Function: {entry['full_name']}\nSignature: {entry['signature']}\n\nDescription:\n{entry['description']}\n\nDocstring:\n{entry['docstring']}\n\nExamples:\n" + "\n".join(entry['examples'])
            }, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Error getting function details: {str(e)}"})

    # =====================================================================
    # Agentic Loop: Tool-calling based autonomous execution
    # =====================================================================

    LEGACY_AGENT_TOOLS = [
        {
            "name": "inspect_data",
            "description": "Inspect the AnnData or MuData object without modifying it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "aspect": {
                        "type": "string",
                        "enum": ["shape", "obs", "var", "obsm", "uns", "layers", "full"],
                    }
                },
                "required": ["aspect"],
            },
        },
        {
            "name": "execute_code",
            "description": "Execute Python code against the current dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["code", "description"],
            },
        },
        {
            "name": "run_snippet",
            "description": "Run read-only Python code on a shallow copy of the current dataset.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
        {
            "name": "search_functions",
            "description": "Search the OmicVerse function registry.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {
            "name": "search_skills",
            "description": "Search installed domain-specific skills.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {
            "name": "delegate",
            "description": "Delegate to an explore, plan, or execute subagent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_type": {"type": "string", "enum": ["explore", "plan", "execute"]},
                    "task": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["agent_type", "task"],
            },
        },
        {
            "name": "web_fetch",
            "description": "Fetch a URL and return readable content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                "required": ["url"],
            },
        },
        {
            "name": "web_search",
            "description": "Search the web and return results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "num_results": {"type": "number"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "web_download",
            "description": "Download a file from a URL to disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "filename": {"type": "string"},
                    "directory": {"type": "string"},
                },
                "required": ["url"],
            },
        },
        {
            "name": "finish",
            "description": "Declare the task complete and return the summary.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    ]

    AGENT_TOOLS = get_visible_tool_schemas(get_default_loaded_tool_names()) + LEGACY_AGENT_TOOLS
    _URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)
    _ACTION_REQUEST_PATTERN = re.compile(
        r"\b(analy[sz]e|download|fetch|get|open|read|inspect|load|run|execute|search|lookup|look up|find|process|parse|clone|fix|edit|write)\b",
        re.IGNORECASE,
    )
    _PROMISSORY_PATTERN = re.compile(
        r"\b(let me|i(?:'ll| will| can)|going to|start by|first(?:,)?\s+i(?:'ll| will)|can continue\?|could continue\?|re-?start)\b",
        re.IGNORECASE,
    )
    _BLOCKER_PATTERN = re.compile(
        r"\b(can(?:not|'t)|unable|failed|error|need your|please provide|approval required|missing|not installed|permission denied)\b",
        re.IGNORECASE,
    )
    _RESULT_PATTERN = re.compile(
        r"\b(found|fetched|downloaded|loaded|read|parsed|here (?:is|are)|summary|supplementary|links?)\b",
        re.IGNORECASE,
    )

    def _tool_inspect_data(self, adata: Any, aspect: str) -> str:
        return self._tool_runtime._tool_inspect_data(adata, aspect)

    def _check_code_prerequisites(self, code: str, adata: Any) -> str:
        return self._analysis_executor.check_code_prerequisites(code, adata)

    def _tool_execute_code(self, code: str, description: str, adata: Any) -> dict:
        return self._tool_runtime._tool_execute_code(code, description, adata)

    def _tool_run_snippet(self, code: str, adata: Any) -> str:
        return self._tool_runtime._tool_run_snippet(code, adata)

    def _tool_search_functions(self, query: str) -> str:
        return self._tool_runtime._tool_search_functions(query)

    def _tool_search_skills(self, query: str) -> str:
        return self._tool_runtime._tool_search_skills(query)

    # ----- Web tools -----

    def _tool_web_fetch(self, url: str, prompt: str = None, timeout: int = 15) -> str:
        return self._tool_runtime._tool_web_fetch(url, prompt=prompt, timeout=timeout)

    def _tool_web_search(self, query: str, num_results: int = 5) -> str:
        return self._tool_runtime._tool_web_search(query, num_results=num_results)

    def _tool_web_download(self, url: str, filename: str = None,
                           directory: str = None) -> str:
        return self._tool_runtime._tool_web_download(url, filename=filename, directory=directory)

    # ----- Claude-style tool helpers -----

    def _resolve_local_path(self, file_path: str, *, allow_relative: bool = False) -> Path:
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            if not allow_relative:
                raise ValueError("file_path must be an absolute path")
            path = Path(self._refresh_runtime_working_directory()) / path
        return path.resolve()

    def _ensure_server_tool_mode(self, tool_name: str) -> None:
        harness_config = getattr(self._config, "harness", None)
        if harness_config is None or not getattr(harness_config, "server_tool_mode", True):
            raise RuntimeError(f"{tool_name} is disabled because server_tool_mode is off.")

    def _request_interaction(self, payload: dict[str, Any]) -> Any:
        if self._approval_handler is None:
            # Headless / bench mode — auto-approve tool calls
            return True
        return self._approval_handler(payload)

    def _request_tool_approval(self, tool_name: str, *, reason: str, payload: dict[str, Any]) -> None:
        spec = get_tool_spec(tool_name)
        if spec is None or not spec.requires_approval:
            return
        interaction = {
            "kind": "approval",
            "title": f"{tool_name} approval required",
            "message": reason,
            "tool_name": tool_name,
            "payload": payload,
            "session_id": self._get_runtime_session_id(),
            "trace_id": getattr(getattr(self, "_last_run_trace", None), "trace_id", ""),
        }
        approved = self._request_interaction(interaction)
        if not approved:
            raise PermissionError(f"{tool_name} was not approved by the user.")

    def _tool_tool_search(self, query: str, max_results: int = 5) -> str:
        return self._tool_runtime._tool_tool_search(query, max_results=max_results)

    def _tool_read(self, file_path: str, offset: int = 0, limit: int = 2000, pages: str = "") -> str:
        return self._tool_runtime._tool_read(file_path, offset=offset, limit=limit, pages=pages)

    def _tool_edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        return self._tool_runtime._tool_edit(
            file_path, old_string, new_string, replace_all=replace_all,
        )

    def _tool_write(self, file_path: str, content: str) -> str:
        return self._tool_runtime._tool_write(file_path, content)

    def _tool_glob(self, pattern: str, root: str = "", max_results: int = 200) -> str:
        return self._tool_runtime._tool_glob(pattern, root=root, max_results=max_results)

    def _tool_grep(self, pattern: str, root: str = "", glob: str = "", max_results: int = 200) -> str:
        return self._tool_runtime._tool_grep(pattern, root=root, glob=glob, max_results=max_results)

    def _tool_notebook_edit(self, file_path: str, cell_index: int, source: str, cell_type: str = "") -> str:
        return self._tool_runtime._tool_notebook_edit(
            file_path, cell_index, source, cell_type=cell_type,
        )

    def _tool_create_task(self, title: str, description: str = "", status: str = "pending") -> str:
        return self._tool_runtime._tool_create_task(title, description=description, status=status)

    def _tool_get_task(self, task_id: str) -> str:
        return self._tool_runtime._tool_get_task(task_id)

    def _tool_list_tasks(self, status: str = "") -> str:
        return self._tool_runtime._tool_list_tasks(status=status)

    def _tool_task_output(self, task_id: str, offset: int = 0, limit: int = 200) -> str:
        return self._tool_runtime._tool_task_output(task_id, offset=offset, limit=limit)

    def _tool_task_stop(self, task_id: str) -> str:
        return self._tool_runtime._tool_task_stop(task_id)

    def _tool_task_update(self, task_id: str, status: str, summary: str = "") -> str:
        return self._tool_runtime._tool_task_update(task_id, status, summary=summary)

    def _tool_enter_plan_mode(self, reason: str = "") -> str:
        return self._tool_runtime._tool_enter_plan_mode(reason=reason)

    def _tool_exit_plan_mode(self, summary: str = "") -> str:
        return self._tool_runtime._tool_exit_plan_mode(summary=summary)

    def _detect_repo_root(self, cwd: Optional[Path] = None) -> Optional[Path]:
        current = (cwd or Path(self._refresh_runtime_working_directory())).resolve()
        for candidate in (current, *current.parents):
            if (candidate / ".git").exists():
                return candidate
        return None

    def _tool_enter_worktree(self, branch_name: str = "", path: str = "", base_ref: str = "HEAD") -> str:
        return self._tool_runtime._tool_enter_worktree(
            branch_name=branch_name, path=path, base_ref=base_ref,
        )

    def _tool_skill(self, query: str, mode: str = "search") -> str:
        return self._tool_runtime._tool_skill(query, mode=mode)

    def _tool_list_mcp_resources(self, server: str = "") -> str:
        return self._tool_runtime._tool_list_mcp_resources(server=server)

    def _tool_read_mcp_resource(self, server: str, uri: str) -> str:
        return self._tool_runtime._tool_read_mcp_resource(server, uri)

    def _tool_ask_user_question(self, question: str, header: str = "", options: Optional[list[str]] = None) -> str:
        return self._tool_runtime._tool_ask_user_question(
            question, header=header, options=options,
        )

    def _tool_bash(
        self,
        command: str,
        description: str = "",
        timeout: int = 120000,
        run_in_background: bool = False,
        dangerouslyDisableSandbox: bool = False,
    ) -> str:
        return self._tool_runtime._tool_bash(
            command,
            description=description,
            timeout=timeout,
            run_in_background=run_in_background,
            dangerouslyDisableSandbox=dangerouslyDisableSandbox,
        )

    # ----- Subagent prompt builders -----

    _CODE_QUALITY_RULES = _CODE_QUALITY_RULES_EXT

    # -- Prompt building (delegated to ovagent.prompt_builder) ---------------

    def _build_explore_prompt(self, context: str) -> str:
        return self._prompt_builder.build_explore_prompt(context)

    def _build_plan_prompt(self, context: str) -> str:
        return self._prompt_builder.build_plan_prompt(context)

    def _build_execute_prompt(self, context: str) -> str:
        return self._prompt_builder.build_execute_prompt(context)

    def _build_subagent_system_prompt(self, agent_type: str, context: str = "") -> str:
        return self._prompt_builder.build_subagent_system_prompt(agent_type, context)

    def _build_subagent_user_message(self, task: str, adata: Any) -> str:
        return self._prompt_builder.build_subagent_user_message(task, adata)

    async def _run_subagent(
        self, agent_type: str, task: str, adata: Any, context: str = "",
    ) -> dict:
        return await self._subagent_controller.run_subagent(
            agent_type=agent_type, task=task, adata=adata, context=context,
        )

    def _build_agentic_system_prompt(self) -> str:
        return self._prompt_builder.build_agentic_system_prompt()

    def _build_initial_user_message(self, request: str, adata: Any) -> str:
        return self._prompt_builder.build_initial_user_message(request, adata)

    def _get_harness_session_id(self) -> str:
        """Best-effort session identifier for harness traces/history."""
        web_session_id = getattr(self, "_web_session_id", "")
        if web_session_id:
            return web_session_id
        if self._filesystem_context is not None:
            return self._filesystem_context.session_id
        if self._notebook_executor is not None and self._notebook_executor.current_session:
            return self._notebook_executor.current_session.get("session_id", "")
        return ""

    def _get_runtime_session_id(self) -> str:
        """Return the session key used by the harness runtime registry."""
        return self._get_harness_session_id() or "default"

    def _get_visible_agent_tools(self, *, allowed_names: Optional[set[str]] = None) -> list[dict[str, Any]]:
        """Return the currently visible tool schemas for this session."""
        session_id = self._get_runtime_session_id()
        loaded = runtime_state.get_loaded_tools(session_id)
        tools = get_visible_tool_schemas(loaded) + list(self.LEGACY_AGENT_TOOLS)
        if allowed_names is None:
            return tools
        normalized_allowed = {normalize_tool_name(name) for name in allowed_names}
        return [
            tool for tool in tools
            if tool["name"] in allowed_names or normalize_tool_name(tool["name"]) in normalized_allowed
        ]

    def _get_loaded_tool_names(self) -> list[str]:
        return runtime_state.get_loaded_tools(self._get_runtime_session_id())

    def _refresh_runtime_working_directory(self) -> str:
        """Keep runtime cwd aligned with the active worktree / filesystem context."""
        session_id = self._get_runtime_session_id()
        cwd = runtime_state.get_working_directory(session_id)
        if cwd:
            return cwd
        current = os.getcwd()
        if self._filesystem_context is not None:
            current = str(self._filesystem_context._workspace_dir)
        runtime_state.set_working_directory(session_id, current)
        return current

    def _tool_blocked_in_plan_mode(self, tool_name: str) -> bool:
        spec = get_tool_spec(tool_name)
        session_state = runtime_state.get_summary(self._get_runtime_session_id())
        plan_mode = bool((session_state.get("plan_mode") or {}).get("enabled", False))
        if not plan_mode:
            return False
        if spec is not None:
            return spec.high_risk or spec.name in {"Bash", "Edit", "Write", "NotebookEdit", "EnterWorktree"}
        return tool_name in {"execute_code", "web_download"}

    def _request_requires_tool_action(self, request: str, adata: Any) -> bool:
        return _FollowUpGate.request_requires_tool_action(request, adata)

    def _response_is_promissory(self, content: str) -> bool:
        return _FollowUpGate.response_is_promissory(content)

    def _select_agent_tool_choice(self, *, request, adata, turn_index, had_meaningful_tool_call, forced_retry) -> str:
        return _FollowUpGate.select_tool_choice(
            request=request, adata=adata, turn_index=turn_index,
            had_meaningful_tool_call=had_meaningful_tool_call, forced_retry=forced_retry,
        )

    def _should_continue_after_text_response(self, *, request, response_content, adata, had_meaningful_tool_call) -> bool:
        return _FollowUpGate.should_continue_after_text(
            request=request, response_content=response_content,
            adata=adata, had_meaningful_tool_call=had_meaningful_tool_call,
        )

    def _build_no_tool_follow_up_message(self, request, *, retry_count=0, max_retries=2) -> str:
        return _FollowUpGate.build_no_tool_follow_up(request, retry_count=retry_count, max_retries=max_retries)

    def _persist_harness_history(self, request: str) -> None:
        self._turn_controller._persist_harness_history(request)

    def _append_tool_results(self, messages: list, tool_results: list) -> None:
        self._turn_controller._append_tool_results(messages, tool_results)

    async def _dispatch_tool(self, tool_call, current_adata: Any, request: str):
        return await self._tool_runtime.dispatch_tool(tool_call, current_adata, request)

    async def _run_agentic_loop(self, request: str, adata: Any,
                               event_callback=None,
                               cancel_event=None,
                               history=None,
                               approval_handler=None) -> Any:
        return await self._turn_controller.run_agentic_loop(
            request, adata,
            event_callback=event_callback,
            cancel_event=cancel_event,
            history=history,
            approval_handler=approval_handler,
        )

    def _save_conversation_log(self, messages: list) -> None:
        _TurnController._save_conversation_log(messages)

    def _list_project_skills(self) -> str:
        """Return a JSON catalog of the discovered project skills."""

        if not self.skill_registry or not self.skill_registry.skill_metadata:
            return json.dumps({"skills": [], "message": "No project skills available."}, indent=2)

        skills_payload = [
            {
                "name": skill.name,
                "slug": skill.slug,
                "description": skill.description,
                "path": str(skill.path),
                "metadata": skill.metadata,
            }
            for skill in sorted(self.skill_registry.skill_metadata.values(), key=lambda item: item.name.lower())
        ]
        return json.dumps({"skills": skills_payload}, indent=2)

    def _load_skill_guidance(self, skill_name: str) -> str:
        """Return the detailed instructions for a requested skill.

        This triggers lazy loading of the full skill content if using progressive disclosure.
        """

        if not self.skill_registry or not self.skill_registry.skill_metadata:
            return json.dumps({"error": "No project skills are available."})

        if not skill_name or not skill_name.strip():
            return json.dumps({"error": "Provide a skill name to load guidance."})

        # Lazy load the full skill content
        slug = skill_name.strip().lower()
        definition = self.skill_registry.load_full_skill(slug)
        if not definition:
            return json.dumps({"error": f"Skill '{skill_name}' not found."})

        return json.dumps(
            {
                "name": definition.name,
                "description": definition.description,
                "instructions": definition.prompt_instructions(provider=getattr(self, "provider", None)),
                "path": str(definition.path),
                "metadata": definition.metadata,
            },
            indent=2,
        )

    @staticmethod
    def _merge_usage_stats(usages: List[Optional[Usage]]) -> Optional[Usage]:
        """Merge usage records from multiple lightweight codegen calls."""

        valid = [usage for usage in usages if usage is not None]
        if not valid:
            return None

        return Usage(
            input_tokens=sum(max(0, usage.input_tokens) for usage in valid),
            output_tokens=sum(max(0, usage.output_tokens) for usage in valid),
            total_tokens=sum(max(0, usage.total_tokens) for usage in valid),
            model=valid[-1].model,
            provider=valid[-1].provider,
        )

    def _collect_relevant_registry_entries(
        self,
        request: str,
        max_entries: int = 8,
    ) -> List[Dict[str, Any]]:
        """Return a compact set of registry entries relevant to a free-form request."""

        if max_entries <= 0:
            return []

        self._ensure_runtime_registry_for_codegen()

        runtime_entries = self._collect_runtime_registry_entries(
            request,
            max_entries=max_entries * 3,
        )
        static_entries = self._collect_static_registry_entries(
            request,
            max_entries=max_entries * 3,
        )

        merged: Dict[str, Tuple[float, int, Dict[str, Any]]] = {}
        for raw_entry in [*runtime_entries, *static_entries]:
            entry = self._normalize_registry_entry_for_codegen(raw_entry)
            full_name = entry.get("full_name", "")
            if not full_name:
                continue
            score = self._score_registry_entry_for_codegen(request, entry)
            if score <= 0:
                continue
            # Prefer higher score, then runtime over static when tied.
            source_rank = 1 if entry.get("source") == "runtime" else 0
            current = merged.get(full_name)
            if current is None or (score, source_rank) > (current[0], current[1]):
                merged[full_name] = (score, source_rank, entry)

        ranked = sorted(
            merged.values(),
            key=lambda item: (item[0], item[1], item[2].get("full_name", "")),
            reverse=True,
        )
        return [entry for _, _, entry in ranked[:max_entries]]

    def _ensure_runtime_registry_for_codegen(self) -> None:
        """Hydrate the runtime registry when a partial lazy-import state is insufficient."""

        if getattr(_global_registry, "_registry", None):
            return

        try:
            from ..mcp.manifest import ensure_registry_populated

            ensure_registry_populated()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Runtime registry hydration for claw failed: %s", exc)

    def _collect_runtime_registry_entries(
        self,
        request: str,
        max_entries: int = 8,
    ) -> List[Dict[str, Any]]:
        """Query the in-memory registry when it has been hydrated successfully."""

        if not getattr(_global_registry, "_registry", None):
            return []

        seen: set = set()
        entries: List[Dict[str, Any]] = []

        def _add_matches(query: str) -> None:
            if not query or len(entries) >= max_entries:
                return
            for entry in _global_registry.find(query):
                full_name = entry.get("full_name", "")
                if not full_name or full_name in seen:
                    continue
                seen.add(full_name)
                entries.append(entry)
                if len(entries) >= max_entries:
                    return

        _add_matches(request)

        keywords = [
            token for token in re.findall(r"[A-Za-z_][A-Za-z0-9_\\.\\-]*", request or "")
            if len(token) >= 2
        ]
        for keyword in keywords[:12]:
            _add_matches(keyword)
            if len(entries) >= max_entries:
                break

        return entries

    def _normalize_registry_entry_for_codegen(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Convert registry entries to public-facing ov.* names for code generation."""

        normalized = dict(entry)
        source = normalized.get("source")
        normalized["source"] = source if str(source).startswith("static_ast") else "runtime"

        original_full_name = str(normalized.get("full_name", "") or "")
        normalized["registry_full_name"] = original_full_name

        public_name = original_full_name
        short_name = str(normalized.get("short_name") or normalized.get("name") or "")

        if original_full_name.startswith("omicverse."):
            parts = original_full_name.split(".")
            if len(parts) >= 2:
                domain = parts[1]
                if domain == "_settings":
                    public_name = f"ov.core.{short_name or parts[-1]}"
                elif domain:
                    public_name = f"ov.{domain}.{short_name or parts[-1]}"

        normalized["full_name"] = public_name
        return normalized

    def _score_registry_entry_for_codegen(
        self,
        request: str,
        entry: Dict[str, Any],
    ) -> float:
        """Score a registry entry for lightweight code generation retrieval."""

        query = (request or "").strip().lower()
        if not query:
            return 0.0

        tokens = [
            token for token in re.findall(r"[a-z0-9_\\.\\-]+", query)
            if len(token) >= 2
        ]
        aliases = [str(alias).lower() for alias in (entry.get("aliases") or [])]
        haystack_parts = [
            entry.get("name", ""),
            entry.get("short_name", ""),
            entry.get("full_name", ""),
            entry.get("registry_full_name", ""),
            entry.get("category", ""),
            entry.get("description", ""),
            " ".join(aliases),
            " ".join(entry.get("examples", []) or []),
            " ".join(entry.get("imports", []) or []),
        ]
        haystack = " ".join(str(part) for part in haystack_parts).lower()

        score = 0.0
        if query == str(entry.get("full_name", "")).lower():
            score += 10.0
        if query == str(entry.get("short_name", "")).lower():
            score += 9.0
        if query in haystack:
            score += 4.0

        for alias in aliases:
            if alias == query:
                score += 8.0
            elif alias and alias in query:
                score += 2.0

        for token in tokens:
            if token in haystack:
                score += 1.25

        public_name = str(entry.get("full_name", ""))
        if public_name.startswith(("ov.pp.", "ov.single.", "ov.pl.", "ov.bulk.", "ov.space.")):
            score += 0.5

        if public_name.startswith("ov.datasets.") and not any(
            word in query for word in ("dataset", "download", "read", "load", "example", "demo")
        ):
            score -= 2.0

        if public_name.startswith("ov.core.") and not any(
            word in query for word in ("reference", "table", "gpu", "cpu", "settings")
        ):
            score -= 2.0

        return score

    def _load_static_registry_entries(self) -> List[Dict[str, Any]]:
        """Parse @register_function metadata plus nested method/branch capabilities."""

        cached = getattr(self, "_static_registry_entries_cache", None)
        if cached is not None:
            return cached

        package_root = Path(__file__).resolve().parents[1]
        search_roots = (
            "pp", "pl", "single", "bulk", "space", "utils",
            "io", "alignment", "external", "biocontext", "bulk2single", "datasets",
        )
        entries: List[Dict[str, Any]] = []
        seen: set = set()

        for root_name in search_roots:
            root = package_root / root_name
            if not root.exists():
                continue
            for file_path in sorted(root.rglob("*.py")):
                if file_path.name == "__init__.py":
                    continue
                if "__pycache__" in file_path.parts or ".ipynb_checkpoints" in file_path.parts:
                    continue
                try:
                    source = file_path.read_text(encoding="utf-8")
                    tree = ast.parse(source, filename=str(file_path))
                except Exception:
                    continue

                for node in tree.body:
                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        continue
                    decorator = self._find_register_function_decorator(node)
                    if decorator is None:
                        continue
                    for entry in self._build_static_registry_entries(file_path, node, decorator):
                        full_name = entry.get("full_name", "")
                        if not full_name or full_name in seen:
                            continue
                        seen.add(full_name)
                        entries.append(entry)

        self._static_registry_entries_cache = entries
        return entries

    @staticmethod
    def _find_register_function_decorator(node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]) -> Optional[ast.Call]:
        """Return the register_function decorator call when present."""

        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            if isinstance(func, ast.Name) and func.id == "register_function":
                return decorator
            if isinstance(func, ast.Attribute) and func.attr == "register_function":
                return decorator
        return None

    @staticmethod
    def _literal_eval_or_default(node: Optional[ast.AST], default: Any) -> Any:
        if node is None:
            return default
        try:
            return ast.literal_eval(node)
        except Exception:
            return default

    @staticmethod
    def _derive_static_signature(node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]) -> str:
        """Build a lightweight signature string from AST."""

        if isinstance(node, ast.ClassDef):
            return f"{node.name}(...)"

        arg_names = [arg.arg for arg in node.args.args]
        if arg_names and arg_names[0] == "self":
            arg_names = arg_names[1:]
        return f"{node.name}({', '.join(arg_names)})"

    def _build_static_registry_entries(
        self,
        file_path: Path,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
        decorator: ast.Call,
    ) -> List[Dict[str, Any]]:
        """Build the primary static entry plus nested method/branch entries."""

        base_entry = self._build_static_registry_entry(file_path, node, decorator)
        if not base_entry:
            return []

        entries = [base_entry]
        entries.extend(self._build_nested_static_registry_entries(node, base_entry))
        return entries

    def _build_nested_static_registry_entries(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
        base_entry: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Expand registered classes/functions into searchable method and branch entries."""

        nested: List[Dict[str, Any]] = []
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if child.name.startswith("_"):
                    continue
                method_entry = self._build_static_method_entry(base_entry, node, child)
                if not method_entry:
                    continue
                nested.append(method_entry)
                nested.extend(self._build_static_branch_entries(child, method_entry, base_entry))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nested.extend(self._build_static_branch_entries(node, base_entry, base_entry))
        return nested

    def _build_static_method_entry(
        self,
        base_entry: Dict[str, Any],
        class_node: ast.ClassDef,
        method_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    ) -> Optional[Dict[str, Any]]:
        """Create a searchable entry for a public method on a registered class."""

        aliases: List[str] = [
            method_node.name,
            f"{class_node.name}.{method_node.name}",
            f"{base_entry.get('short_name', class_node.name)} {method_node.name}",
        ]
        for alias in (base_entry.get("aliases") or [])[:8]:
            alias = str(alias).strip()
            if alias:
                aliases.append(f"{alias} {method_node.name}")

        description = ast.get_docstring(method_node) or (
            f"Method `{method_node.name}` on {base_entry.get('full_name', class_node.name)}."
        )

        return {
            "name": method_node.name,
            "short_name": method_node.name,
            "full_name": f"{base_entry.get('full_name', class_node.name)}.{method_node.name}",
            "module": base_entry.get("module", ""),
            "aliases": list(dict.fromkeys(aliases)),
            "category": base_entry.get("category", ""),
            "description": description,
            "examples": self._filter_examples_for_method(base_entry.get("examples", []), method_node.name),
            "related": [base_entry.get("full_name", "")],
            "signature": self._derive_static_signature(method_node),
            "docstring": ast.get_docstring(method_node) or "",
            "prerequisites": base_entry.get("prerequisites", {}) or {},
            "requires": base_entry.get("requires", {}) or {},
            "produces": base_entry.get("produces", {}) or {},
            "source": "static_ast_method",
            "parent_full_name": base_entry.get("full_name", ""),
            "imports": self._collect_import_targets(method_node.body),
        }

    @staticmethod
    def _filter_examples_for_method(examples: List[str], method_name: str) -> List[str]:
        """Keep examples most relevant to a nested method entry."""

        if not examples:
            return []
        matches = [example for example in examples if method_name in str(example)]
        return matches[:3] if matches else list(examples[:2])

    def _build_static_branch_entries(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        parent_entry: Dict[str, Any],
        owner_entry: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Create searchable entries for string-dispatched branches like method='celltypist'."""

        entries: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str]] = set()

        for child in ast.walk(node):
            if not isinstance(child, ast.If):
                continue
            for param_name, branch_value in self._extract_branch_variants(child.test):
                key = (param_name, branch_value)
                if key in seen:
                    continue
                seen.add(key)
                imports = self._collect_import_targets(child.body)
                aliases = [
                    branch_value,
                    f"{node.name} {branch_value}",
                    f"{param_name} {branch_value}",
                    f"{parent_entry.get('short_name', node.name)} {branch_value}",
                ]
                for alias in (owner_entry.get("aliases") or [])[:8]:
                    alias = str(alias).strip()
                    if alias:
                        aliases.append(f"{alias} {branch_value}")
                examples = self._filter_examples_for_branch(
                    parent_entry.get("examples", []),
                    param_name,
                    branch_value,
                )
                description = (
                    f"Variant of `{parent_entry.get('full_name', node.name)}` when "
                    f"`{param_name}='{branch_value}'`."
                )
                if imports:
                    description += " Imports/uses: " + ", ".join(imports) + "."

                entries.append({
                    "name": branch_value,
                    "short_name": branch_value,
                    "full_name": f"{parent_entry.get('full_name', node.name)}[{param_name}={branch_value}]",
                    "module": parent_entry.get("module", ""),
                    "aliases": list(dict.fromkeys(aliases)),
                    "category": parent_entry.get("category", ""),
                    "description": description,
                    "examples": examples,
                    "related": [parent_entry.get("full_name", "")],
                    "signature": parent_entry.get("signature", self._derive_static_signature(node)),
                    "docstring": parent_entry.get("docstring", ""),
                    "prerequisites": parent_entry.get("prerequisites", {}) or {},
                    "requires": parent_entry.get("requires", {}) or {},
                    "produces": parent_entry.get("produces", {}) or {},
                    "source": "static_ast_branch",
                    "parent_full_name": parent_entry.get("full_name", ""),
                    "branch_parameter": param_name,
                    "branch_value": branch_value,
                    "imports": imports,
                })

        return entries

    @staticmethod
    def _filter_examples_for_branch(examples: List[str], param_name: str, branch_value: str) -> List[str]:
        """Keep examples mentioning the relevant branch parameter/value when possible."""

        if not examples:
            return []
        value_lower = branch_value.lower()
        param_lower = param_name.lower()
        matches = [
            example for example in examples
            if value_lower in str(example).lower() or param_lower in str(example).lower()
        ]
        return matches[:3] if matches else list(examples[:2])

    def _extract_branch_variants(self, test: ast.AST) -> List[Tuple[str, str]]:
        """Extract simple string-dispatch branches from an ``if`` test."""

        variants: List[Tuple[str, str]] = []
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
            for value in test.values:
                variants.extend(self._extract_branch_variants(value))
            return variants

        if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
            return variants

        subject = self._branch_subject_name(test.left)
        comparator = test.comparators[0]
        op = test.ops[0]
        values: List[str] = []

        if isinstance(op, ast.Eq):
            values = self._branch_string_values(comparator)
        elif isinstance(op, ast.In):
            values = self._branch_string_values(comparator)

        if subject and values:
            variants.extend((subject, value) for value in values)
        return variants

    @staticmethod
    def _branch_subject_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    @staticmethod
    def _branch_string_values(node: ast.AST) -> List[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value]
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            values: List[str] = []
            for child in node.elts:
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    values.append(child.value)
            return values
        return []

    @staticmethod
    def _collect_import_targets(statements: List[ast.stmt]) -> List[str]:
        """Collect import targets mentioned inside a function or branch body."""

        module = ast.Module(body=list(statements), type_ignores=[])
        imports: List[str] = []
        for child in ast.walk(module):
            if isinstance(child, ast.Import):
                for alias in child.names:
                    if alias.name:
                        imports.append(alias.name.split(".")[0])
            elif isinstance(child, ast.ImportFrom):
                if child.module:
                    imports.append(child.module.split(".")[0])
                for alias in child.names:
                    if alias.name and alias.name != "*":
                        imports.append(alias.name.split(".")[0])
        return list(dict.fromkeys(imports))

    def _build_static_registry_entry(
        self,
        file_path: Path,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
        decorator: ast.Call,
    ) -> Optional[Dict[str, Any]]:
        """Convert one @register_function AST node into a registry-like record."""

        rel = file_path.relative_to(Path(__file__).resolve().parents[1])
        if not rel.parts:
            return None
        domain = rel.parts[0]
        if domain not in {"pp", "pl", "single", "bulk", "space", "utils"}:
            return None

        args = list(decorator.args)
        aliases = self._literal_eval_or_default(args[0], []) if len(args) >= 1 else []
        category = self._literal_eval_or_default(args[1], "") if len(args) >= 2 else ""
        description = self._literal_eval_or_default(args[2], "") if len(args) >= 3 else ""
        examples = self._literal_eval_or_default(args[3], []) if len(args) >= 4 else []

        kw = {item.arg: item.value for item in decorator.keywords if item.arg}
        aliases = self._literal_eval_or_default(kw.get("aliases"), aliases)
        category = self._literal_eval_or_default(kw.get("category"), category)
        description = self._literal_eval_or_default(kw.get("description"), description)
        examples = self._literal_eval_or_default(kw.get("examples"), examples)
        related = self._literal_eval_or_default(kw.get("related"), [])
        prerequisites = self._literal_eval_or_default(kw.get("prerequisites"), {})
        requires = self._literal_eval_or_default(kw.get("requires"), {})
        produces = self._literal_eval_or_default(kw.get("produces"), {})

        full_name = f"ov.{domain}.{node.name}"
        module_name = "omicverse." + ".".join(rel.with_suffix("").parts)

        return {
            "name": node.name,
            "short_name": node.name,
            "full_name": full_name,
            "module": module_name,
            "aliases": aliases or [],
            "category": category or domain,
            "description": description or (ast.get_docstring(node) or ""),
            "examples": examples or [],
            "related": related or [],
            "signature": self._derive_static_signature(node),
            "docstring": ast.get_docstring(node) or "",
            "prerequisites": prerequisites or {},
            "requires": requires or {},
            "produces": produces or {},
            "source": "static_ast",
        }

    def _collect_static_registry_entries(
        self,
        request: str,
        max_entries: int = 8,
    ) -> List[Dict[str, Any]]:
        """Search the static AST-derived registry snapshot."""

        query = (request or "").strip().lower()
        if not query:
            return []

        tokens = [token for token in re.findall(r"[a-z0-9_\\.\\-]+", query) if len(token) >= 2]
        entries = self._load_static_registry_entries()
        scored: List[Tuple[float, Dict[str, Any]]] = []

        for entry in entries:
            aliases = entry.get("aliases", []) or []
            haystack = " ".join(
                [
                    entry.get("name", ""),
                    entry.get("short_name", ""),
                    entry.get("full_name", ""),
                    entry.get("category", ""),
                    entry.get("description", ""),
                    " ".join(aliases),
                    " ".join(entry.get("examples", []) or []),
                    " ".join(entry.get("imports", []) or []),
                ]
            ).lower()

            score = 0.0
            if query == entry.get("name", "").lower():
                score += 8.0
            if query == entry.get("short_name", "").lower():
                score += 8.0
            if query == entry.get("full_name", "").lower():
                score += 9.0
            if query in haystack:
                score += 4.0
            for alias in aliases:
                alias_lower = str(alias).lower()
                if query == alias_lower:
                    score += 8.0
                elif alias_lower and alias_lower in query:
                    score += 2.0
            for token in tokens:
                if token and token in haystack:
                    score += 1.0

            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored[:max_entries]]

    def _select_codegen_skill_matches(self, request: str, top_k: int = 2) -> List[SkillMatch]:
        """Select skills for codegen using the same loaded skill registry as Jarvis."""

        if not self.skill_registry or not self.skill_registry.skill_metadata:
            return []

        try:
            router = SkillRouter(self.skill_registry, min_score=0.1)
            return router.route(request, top_k=top_k)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Skill routing failed for codegen: %s", exc)
            return []

    def _format_registry_context_for_codegen(
        self,
        entries: List[Dict[str, Any]],
    ) -> str:
        """Format a compact registry snippet for code-only generation."""

        if not entries:
            return "No highly relevant registry matches were found. Use the best OmicVerse API you know."

        blocks: List[str] = []
        for entry in entries:
            full_name = entry.get("full_name", "")
            signature = entry.get("signature", "")
            description = entry.get("description", "")
            examples = entry.get("examples", [])[:2]

            block = [
                f"Function: {full_name}",
                f"Signature: {signature}",
                f"Description: {description}",
            ]
            if examples:
                block.append("Examples:")
                for example in examples:
                    block.append(f"- {example}")

            prereq_text = self._format_prerequisites_for_codegen_entry(entry)
            if prereq_text:
                block.append("Prerequisites:")
                block.append(prereq_text)

            blocks.append("\n".join(block))

        return "\n\n".join(blocks)

    def _format_prerequisites_for_codegen_entry(self, entry: Dict[str, Any]) -> str:
        """Format prerequisites from runtime registry or static AST metadata."""

        full_name = entry.get("registry_full_name", entry.get("full_name", ""))
        if getattr(_global_registry, "_registry", None):
            prereq_text = _global_registry.format_prerequisites_for_llm(full_name)
            if prereq_text and "not found" not in prereq_text.lower():
                return prereq_text

        prerequisites = entry.get("prerequisites", {}) or {}
        requires = entry.get("requires", {}) or {}
        produces = entry.get("produces", {}) or {}

        parts: List[str] = []
        required_functions = prerequisites.get("required", []) or prerequisites.get("functions", []) or []
        optional_functions = prerequisites.get("optional", []) or []

        if required_functions:
            parts.append("  - Required functions: " + ", ".join(required_functions))
        if optional_functions:
            parts.append("  - Optional functions: " + ", ".join(optional_functions))

        req_items = []
        for key, values in requires.items():
            for value in values:
                req_items.append(f"adata.{key}['{value}']")
        if req_items:
            parts.append("  - Requires: " + ", ".join(req_items))

        prod_items = []
        for key, values in produces.items():
            for value in values:
                prod_items.append(f"adata.{key}['{value}']")
        if prod_items:
            parts.append("  - Produces: " + ", ".join(prod_items))

        return "\n".join(parts)

    def _build_code_generation_system_prompt(self, adata: Any) -> str:
        """Build the code-only prompt on top of the fully initialized Agent prompt."""

        base_prompt = ""
        if self._llm is not None and getattr(self._llm, "config", None) is not None:
            base_prompt = getattr(self._llm.config, "system_prompt", "") or ""
        if not base_prompt:
            base_prompt = self._build_agentic_system_prompt()

        dataset_line = (
            f"Dataset summary: {adata.shape[0]} cells x {adata.shape[1]} features."
            if adata is not None and hasattr(adata, "shape")
            else "No dataset object was provided."
        )

        return (
            f"{base_prompt}\n\n"
            "## Claw Code-Only Mode\n"
            "Return ONLY executable Python code for the user's request.\n"
            "Do not explain the code. Do not describe future actions. Do not execute tools.\n"
            "Use OmicVerse APIs (`import omicverse as ov`) for all analysis steps.\n"
            "Assume every needed Scanpy-style step has an OmicVerse `ov.*` wrapper available.\n"
            "Do NOT import scanpy and do NOT call `sc.*` anywhere in the output.\n"
            "If no dataset object is provided, assume an AnnData object named `adata` already exists unless the user explicitly asks to load data from disk.\n"
            "When using in-place OmicVerse preprocessing functions, call them without assigning their return value.\n"
            "Include imports that are actually needed.\n"
            "Produce a single coherent snippet, not multiple alternatives.\n"
            f"{dataset_line}\n"
        )

    def _build_code_generation_user_prompt(self, request: str, adata: Any) -> str:
        """Build the lightweight user prompt for code-only generation."""

        dataset_hint = (
            "A live dataset object is available as `adata`."
            if adata is not None
            else "No live dataset object is available. Generate reusable code that assumes `adata` already exists."
        )
        return (
            f"User request: {request}\n\n"
            f"{dataset_hint}\n"
            "Return Python code only."
        )

    @staticmethod
    def _contains_forbidden_scanpy_usage(code: str) -> bool:
        """Disallow raw scanpy usage in registry-first claw generation."""

        if not code:
            return False
        patterns = [
            r"^\s*import\s+scanpy\s+as\s+sc\b",
            r"^\s*from\s+scanpy\b",
            r"\bsc\.",
        ]
        return any(re.search(pattern, code, re.MULTILINE) for pattern in patterns)

    def _rewrite_scanpy_calls_with_registry(
        self,
        code: str,
        entries: List[Dict[str, Any]],
    ) -> str:
        """Best-effort mechanical rewrite from scanpy-style calls to ov.* calls."""

        if not code:
            return code

        lookup: Dict[str, str] = {}
        for raw_entry in entries:
            entry = self._normalize_registry_entry_for_codegen(raw_entry)
            public_name = str(entry.get("full_name", "") or "")
            short_name = str(entry.get("short_name") or entry.get("name") or "").strip()
            if not public_name.startswith("ov.") or not short_name:
                continue
            lookup.setdefault(short_name, public_name)
            for alias in entry.get("aliases", []) or []:
                alias_key = str(alias).strip().split(".")[-1]
                if alias_key:
                    lookup.setdefault(alias_key, public_name)

        rewritten = re.sub(r"^\s*import\s+scanpy\s+as\s+sc\s*$", "", code, flags=re.MULTILINE)
        rewritten = re.sub(r"^\s*from\s+scanpy\b.*$", "", rewritten, flags=re.MULTILINE)

        def _replace(match: re.Match[str]) -> str:
            func_name = match.group(1)
            replacement = lookup.get(func_name)
            if replacement:
                return replacement + "("
            return match.group(0)

        rewritten = re.sub(r"\bsc\.(?:pp|tl|pl)\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", _replace, rewritten)
        rewritten = re.sub(r"\n{3,}", "\n\n", rewritten).strip()
        return rewritten

    async def _rewrite_code_without_scanpy(
        self,
        code: str,
        request: str,
        adata: Any,
        registry_context: str = "",
        skill_guidance: str = "",
    ) -> Tuple[str, Optional[Usage]]:
        """Rewrite code to strict OmicVerse-only style when scanpy slips in."""

        backend = OmicVerseLLMBackend(
            system_prompt=(
                "You rewrite Python snippets to use OmicVerse APIs only.\n"
                "Return ONLY executable Python code.\n"
                "Do not import scanpy. Do not call sc.*.\n"
            ),
            model=self.model,
            api_key=self.api_key,
            endpoint=self.endpoint,
            max_tokens=4096,
            temperature=0.0,
        )

        dataset_line = (
            f"Dataset summary: {adata.shape[0]} cells x {adata.shape[1]} features."
            if adata is not None and hasattr(adata, "shape")
            else "No dataset object is available."
        )

        prompt = (
            f"User request: {request}\n"
            f"{dataset_line}\n\n"
            "Rewrite the code below so that all analysis calls use `ov.*` APIs only.\n"
            "Keep the behavior aligned with the initialized OmicVerse Agent prompt.\n"
        f"```python\n{code}\n```"
        )

        with self._temporary_api_keys():
            response = await backend.run(prompt)

        try:
            rewritten = self._extract_python_code_strict(response)
        except ValueError:
            return code, backend.last_usage
        return rewritten, backend.last_usage

    async def _review_generated_code_lightweight(
        self,
        code: str,
        request: str,
        adata: Any,
    ) -> Tuple[str, Optional[Usage]]:
        """Run a lightweight reflection pass and return improved code if possible."""

        dataset_line = (
            f"Dataset summary: {adata.shape[0]} cells x {adata.shape[1]} features."
            if adata is not None and hasattr(adata, "shape")
            else "No dataset object is available."
        )

        backend = OmicVerseLLMBackend(
            system_prompt=(
                "You are a strict reviewer of OmicVerse Python code.\n"
                "Return ONLY corrected executable Python code.\n"
                "Do not add explanations.\n"
            ),
            model=self.model,
            api_key=self.api_key,
            endpoint=self.endpoint,
            max_tokens=4096,
            temperature=0.0,
        )

        prompt = (
            f"User request: {request}\n"
            f"{dataset_line}\n\n"
            "Review the code below for correctness, OmicVerse API misuse, bad assumptions, "
            "and obvious syntax/runtime issues. Keep it concise and executable.\n\n"
            f"```python\n{code}\n```"
        )

        with self._temporary_api_keys():
            response = await backend.run(prompt)

        try:
            return self._extract_python_code_strict(response), backend.last_usage
        except ValueError:
            return code, backend.last_usage

    def _capture_code_only_snippet(self, code: str, description: str = "") -> None:
        """Store the latest code snippet captured from execute_code in code-only mode."""

        history = getattr(self, "_code_only_captured_history", None)
        if history is None:
            history = []
            self._code_only_captured_history = history
        history.append({
            "code": code,
            "description": description or "",
        })
        self._code_only_captured_code = code

    def _build_code_only_agentic_request(self, request: str, adata: Any) -> str:
        """Wrap a raw claw request so the normal agentic loop produces code instead of running it."""

        dataset_hint = (
            "A live dataset object is available as `adata`. Reuse the normal Jarvis workflow, "
            "but stop at code generation."
            if adata is not None
            else "No live dataset object is available. Unless the user explicitly asks to load data, "
            "assume an AnnData object named `adata` already exists in the generated code."
        )
        return (
            f"{request}\n\n"
            "CLAW REQUEST MODE:\n"
            "- Use the same OmicVerse Agent logic as Jarvis.\n"
            "- Use search_functions and search_skills when helpful.\n"
            "- Produce the final answer by calling execute_code with the final Python script.\n"
            "- In this mode, execute_code captures code without running it.\n"
            "- After execute_code, call finish.\n"
            f"{dataset_hint}"
        )

    async def _generate_code_via_agentic_loop(
        self,
        request: str,
        adata: Any,
        *,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Run the normal Jarvis agentic loop and capture the final execute_code snippet."""

        captured_chunks: List[str] = []
        captured_errors: List[str] = []

        def _progress(message: str) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(message)
            except Exception:
                pass

        async def _event_callback(event: Dict[str, Any]) -> None:
            event_type = str(event.get("type") or "")
            content = event.get("content")
            if event_type == "tool_call":
                tool_name = ""
                if isinstance(content, dict):
                    tool_name = str(content.get("name") or "")
                _progress(f"tool: {tool_name or 'unknown'}")
            elif event_type == "code":
                code = str(content or "")
                if code:
                    captured_chunks.append(code)
                _progress("captured code")
            elif event_type == "error":
                captured_errors.append(str(content or "unknown error"))
                _progress("agent error")
            elif event_type == "done":
                _progress("finish")

        previous_mode = getattr(self, "_code_only_mode", False)
        previous_code = getattr(self, "_code_only_captured_code", "")
        previous_history = getattr(self, "_code_only_captured_history", None)
        self._code_only_mode = True
        self._code_only_captured_code = ""
        self._code_only_captured_history = []
        try:
            _progress("start agentic loop")
            await self._run_agentic_loop(
                self._build_code_only_agentic_request(request, adata),
                adata,
                event_callback=_event_callback,
            )
        finally:
            captured_code = str(getattr(self, "_code_only_captured_code", "") or "")
            captured_history = list(getattr(self, "_code_only_captured_history", []) or [])
            self._code_only_mode = previous_mode
            self._code_only_captured_code = previous_code
            self._code_only_captured_history = previous_history

        if captured_code:
            return captured_code

        for item in reversed(captured_history):
            code = str(item.get("code", "") or "")
            if code:
                return code

        for chunk in reversed(captured_chunks):
            try:
                return self._extract_python_code_strict(chunk)
            except ValueError:
                continue

        if captured_errors:
            raise RuntimeError(
                "Jarvis-style code generation did not reach execute_code. "
                + "; ".join(captured_errors[:3])
            )
        raise RuntimeError(
            "Jarvis-style code generation did not emit executable code. "
            "The agent finished without calling execute_code."
        )

    async def generate_code_async(
        self,
        request: str,
        adata: Any = None,
        *,
        max_functions: int = 8,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Generate OmicVerse Python code without executing it."""

        if not request or not request.strip():
            raise ValueError("request cannot be empty")

        direct_code = self._detect_direct_python_request(request)
        if direct_code:
            self.last_usage = None
            self.last_usage_breakdown = {
                'generation': None,
                'reflection': [],
                'review': [],
                'total': None,
            }
            return direct_code

        code = await self._generate_code_via_agentic_loop(
            request,
            adata,
            progress_callback=progress_callback,
        )
        return code

    def generate_code(
        self,
        request: str,
        adata: Any = None,
        *,
        max_functions: int = 8,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Synchronous wrapper for code-only OmicVerse generation."""

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            result_container: Dict[str, Any] = {}
            error_container: Dict[str, BaseException] = {}

            def _run_in_thread() -> None:
                try:
                    result_container["value"] = asyncio.run(
                        self.generate_code_async(
                            request,
                            adata,
                            max_functions=max_functions,
                            progress_callback=progress_callback,
                        )
                    )
                except BaseException as exc:  # pragma: no cover - propagate to caller
                    error_container["error"] = exc

            thread = threading.Thread(target=_run_in_thread, name="OmicVerseCodegenRunner")
            thread.start()
            thread.join()

            if "error" in error_container:
                raise error_container["error"]

            return result_container.get("value", "")

        return asyncio.run(
            self.generate_code_async(
                request,
                adata,
                max_functions=max_functions,
                progress_callback=progress_callback,
            )
        )

    def _extract_python_code(self, response_text: str) -> str:
        """Extract executable Python code from the agent response using AST validation."""

        candidates = self._gather_code_candidates(response_text)
        if not candidates:
            # Provide detailed diagnostic information
            error_msg = (
                f"Could not extract executable code: no code candidates found in the response.\n"
                f"Response length: {len(response_text)} characters\n"
                f"Response preview (first 500 chars):\n{response_text[:500]}\n"
                f"Response preview (last 300 chars):\n...{response_text[-300:]}"
            )
            logger.error(error_msg)
            # Fallback: return a minimal safe workflow to keep execution moving
            return textwrap.dedent(
                """
                import omicverse as ov
                import scanpy as sc
                # Fallback minimal workflow when code extraction fails
                adata = adata
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
                sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat')
                sc.pp.pca(adata)
                sc.pp.neighbors(adata)
                try:
                    sc.tl.leiden(adata)
                except Exception:
                    pass
                try:
                    sc.tl.umap(adata)
                except Exception:
                    pass
                """
            ).strip()

        logger.debug(f"Found {len(candidates)} code candidate(s) to validate")

        syntax_errors = []
        for i, candidate in enumerate(candidates):
            logger.debug(f"Validating candidate {i+1}/{len(candidates)} (length: {len(candidate)} chars)")
            logger.debug(f"Candidate preview (first 200 chars): {candidate[:200]}")

            try:
                normalized = self._normalize_code_candidate(candidate)
            except ValueError as exc:
                error = f"Candidate {i+1}: normalization failed - {exc}"
                logger.debug(error)
                syntax_errors.append(error)
                continue

            try:
                ast.parse(normalized)
                logger.debug(f"✓ Candidate {i+1} validated successfully")
                # Apply proactive transformations to prevent common LLM errors
                transformer = ProactiveCodeTransformer()
                transformed = transformer.transform(normalized)
                if transformed != normalized:
                    logger.debug("✓ Proactive transformations applied to fix potential errors")
                return transformed
            except SyntaxError as exc:
                error = f"Candidate {i+1}: syntax error - {exc}"
                logger.debug(error)
                syntax_errors.append(error)
                continue

        # All candidates failed - provide detailed error message
        error_msg = (
            f"Could not extract executable code: all {len(candidates)} candidate(s) failed validation.\n"
            f"Errors:\n" + "\n".join(f"  - {err}" for err in syntax_errors)
        )
        logger.error(error_msg)
        # Fallback to the same minimal safe workflow
        return textwrap.dedent(
            """
            import omicverse as ov
            import scanpy as sc
            adata = adata
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat')
            sc.pp.pca(adata)
            sc.pp.neighbors(adata)
            try:
                sc.tl.leiden(adata)
            except Exception:
                pass
            try:
                sc.tl.umap(adata)
            except Exception:
                pass
            """
        ).strip()

    def _gather_code_candidates(self, response_text: str) -> List[str]:
        """Enhanced code extraction with multiple strategies to handle various formats."""

        candidates = []

        # Strategy 1: Standard fenced code blocks with python identifier
        fenced_python = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)
        for match in fenced_python.finditer(response_text):
            code = textwrap.dedent(match.group(1)).strip()
            if code:
                candidates.append(code)

        # Strategy 2: Generic fenced code blocks (```...```)
        if not candidates:
            fenced_generic = re.compile(r"```\s*(.*?)```", re.DOTALL)
            for match in fenced_generic.finditer(response_text):
                code = textwrap.dedent(match.group(1)).strip()
                # Skip if starts with language identifier that's not python
                first_line = code.split('\n')[0].strip().lower()
                if first_line in ['bash', 'shell', 'json', 'yaml', 'xml', 'html', 'css', 'javascript']:
                    continue
                if code and self._looks_like_python(code):
                    candidates.append(code)

        # Strategy 3: Code blocks with alternative language identifiers (py, python3)
        if not candidates:
            fenced_alt = re.compile(r"```(?:py|python3)\s*(.*?)```", re.DOTALL | re.IGNORECASE)
            for match in fenced_alt.finditer(response_text):
                code = textwrap.dedent(match.group(1)).strip()
                if code:
                    candidates.append(code)

        # Strategy 4: Code following "Here's the code:" or similar phrases
        if not candidates:
            code_intro = re.compile(
                r"(?:here'?s? (?:the )?code|code:|solution:)\s*[:\n]\s*```(?:python)?\s*(.*?)```",
                re.DOTALL | re.IGNORECASE
            )
            for match in code_intro.finditer(response_text):
                code = textwrap.dedent(match.group(1)).strip()
                if code:
                    candidates.append(code)

        # Strategy 5: GPT-5 specific - last code block (reasoning may come before)
        # If we have multiple candidates, try the last one first (GPT-5 reasoning is before code)
        if len(candidates) > 1:
            # Reverse order to try last code block first
            candidates = list(reversed(candidates))

        # Strategy 6: Inline extraction as fallback
        if not candidates:
            inline = self._extract_inline_python(response_text)
            if inline:
                candidates.append(inline)

        return candidates

    def _looks_like_python(self, code: str) -> bool:
        """Heuristic check if code snippet looks like Python."""

        # Python indicators to look for
        python_indicators = [
            r'\bimport\b',
            r'\bdef\b',
            r'\bclass\b',
            r'\badata\b',
            r'\bov\.',
            r'\bsc\.',
            r'\breturn\b',
            r'\bfor\b.*\bin\b',
            r'\bif\b.*:',
            r'\.obs\[',
            r'\.var\[',
            r'=\s*\w+\(',
        ]

        matches = sum(1 for pattern in python_indicators if re.search(pattern, code))
        return matches >= 2  # At least 2 Python indicators

    def _extract_inline_python(self, response_text: str) -> str:
        """Heuristically gather inline Python statements for AST validation."""

        python_line_pattern = re.compile(
            r"^\s*(?:async\s+def |def |class |import |from |for |while |if |elif |else:|try:|except |with |return |@|print|adata|ov\.|sc\.)"
        )
        assignment_pattern = re.compile(r"^\s*[\w\.]+\s*=.*")
        call_pattern = re.compile(r"^\s*[\w\.]+\s*\(.*")
        collected: List[str] = []

        for raw_line in response_text.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                continue
            if (
                python_line_pattern.match(line)
                or assignment_pattern.match(line)
                or call_pattern.match(line)
                or stripped.startswith("#")
            ):
                collected.append(line)

        snippet = "\n".join(collected).strip()
        return textwrap.dedent(snippet) if snippet else ""

    def _normalize_code_candidate(self, code: str) -> str:
        """Ensure imports and formatting are in place for execution."""

        dedented = textwrap.dedent(code).strip()
        if not dedented:
            raise ValueError("empty code candidate")

        import_present = re.search(r"^\s*(?:import|from)\s+omicverse", dedented, re.MULTILINE)
        if not import_present:
            dedented = "import omicverse as ov\n" + dedented

        return dedented

    def _extract_python_code_strict(self, response_text: str) -> str:
        """Extract executable Python code without logging errors or falling back."""

        candidates = self._gather_code_candidates(response_text)
        if not candidates:
            raise ValueError("no code candidates found")

        syntax_errors: List[str] = []
        for i, candidate in enumerate(candidates, start=1):
            try:
                normalized = self._normalize_code_candidate(candidate)
            except ValueError as exc:
                syntax_errors.append(f"Candidate {i}: normalization failed - {exc}")
                continue

            try:
                ast.parse(normalized)
                transformer = ProactiveCodeTransformer()
                transformed = transformer.transform(normalized)
                ast.parse(transformed)
                return transformed
            except SyntaxError as exc:
                syntax_errors.append(f"Candidate {i}: syntax error - {exc}")
                continue

        raise ValueError(
            "Could not extract executable code: all "
            f"{len(candidates)} candidate(s) failed validation.\nErrors:\n  - "
            + "\n  - ".join(syntax_errors)
        )

    async def _review_result(self, original_adata: Any, result_adata: Any, request: str, code: str) -> Dict[str, Any]:
        """
        Review the execution result to validate it matches the user's task assignment.

        This method compares the original and result data to verify:
        - Expected transformations occurred
        - Data integrity maintained
        - Result aligns with user intent
        - No unexpected side effects

        Parameters
        ----------
        original_adata : Any
            Original AnnData object before execution
        result_adata : Any
            Result AnnData object after execution
        request : str
            The original user request
        code : str
            The executed code

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'matched': bool - whether result matches user intent
            - 'assessment': str - overall assessment
            - 'changes_detected': List[str] - list of detected changes
            - 'issues': List[str] - list of issues or concerns
            - 'confidence': float - confidence in result correctness (0-1)
            - 'recommendation': str - recommendation (accept/review/retry)
        """
        # Gather comparison data
        original_shape = (original_adata.shape[0], original_adata.shape[1])
        result_shape = (result_adata.shape[0], result_adata.shape[1])

        # Check for new attributes
        original_obs_cols = list(getattr(original_adata, 'obs', {}).columns) if hasattr(original_adata, 'obs') else []
        result_obs_cols = list(getattr(result_adata, 'obs', {}).columns) if hasattr(result_adata, 'obs') else []
        new_obs_cols = [col for col in result_obs_cols if col not in original_obs_cols]

        original_uns_keys = list(getattr(original_adata, 'uns', {}).keys()) if hasattr(original_adata, 'uns') else []
        result_uns_keys = list(getattr(result_adata, 'uns', {}).keys()) if hasattr(result_adata, 'uns') else []
        new_uns_keys = [key for key in result_uns_keys if key not in original_uns_keys]

        # Build review prompt
        review_prompt = f"""You are an expert bioinformatics analyst reviewing the results of an OmicVerse operation.

User Request: "{request}"

Executed Code:
```python
{code}
```

Original Data:
- Shape: {original_shape[0]} cells × {original_shape[1]} genes
- Observation columns: {len(original_obs_cols)} columns
- Uns keys: {len(original_uns_keys)} keys

Result Data:
- Shape: {result_shape[0]} cells × {result_shape[1]} genes
- Observation columns: {len(result_obs_cols)} columns (new: {new_obs_cols if new_obs_cols else 'none'})
- Uns keys: {len(result_uns_keys)} keys (new: {new_uns_keys if new_uns_keys else 'none'})

Changes Detected:
- Cells: {original_shape[0]} → {result_shape[0]} (change: {result_shape[0] - original_shape[0]:+d})
- Genes: {original_shape[1]} → {result_shape[1]} (change: {result_shape[1] - original_shape[1]:+d})
- New observation columns: {new_obs_cols if new_obs_cols else 'none'}
- New uns keys: {new_uns_keys if new_uns_keys else 'none'}

Your task:
1. **Evaluate if the result matches the user's intent**:
   - Does the transformation align with the request?
   - Are the changes expected for this operation?
   - Is the data integrity maintained?

2. **Identify any issues or concerns**:
   - Unexpected data loss (too many cells/genes filtered)
   - Missing expected outputs
   - Suspicious transformations

3. **Provide assessment as JSON**:
{{
  "matched": true,
  "assessment": "Brief assessment of the result quality",
  "changes_detected": ["change 1", "change 2"],
  "issues": ["issue 1"] or [],
  "confidence": 0.92,
  "recommendation": "accept"
}}

Recommendation values:
- "accept": Result looks good, matches intent
- "review": Result may have issues, user should review
- "retry": Result appears incorrect, suggest retry

IMPORTANT:
- Return ONLY the JSON object
- Keep confidence between 0.0 and 1.0
- Be specific about changes and issues
- Consider the context of the user's request
"""

        try:
            with self._temporary_api_keys():
                if not self._llm:
                    raise RuntimeError("LLM backend is not initialized")

                response_text = await self._llm.run(review_prompt)

                # Track review token usage
                if self._llm.last_usage:
                    if 'review' not in self.last_usage_breakdown:
                        self.last_usage_breakdown['review'] = []
                    self.last_usage_breakdown['review'].append(self._llm.last_usage)

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                # Fallback: assume success
                return {
                    'matched': True,
                    'assessment': 'Result review completed (JSON extraction failed)',
                    'changes_detected': [f'Shape changed: {original_shape} → {result_shape}'],
                    'issues': [],
                    'confidence': 0.7,
                    'recommendation': 'accept'
                }

            review_result = json.loads(json_match.group(0))

            # Validate and normalize
            result = {
                'matched': bool(review_result.get('matched', True)),
                'assessment': review_result.get('assessment', 'No assessment provided'),
                'changes_detected': review_result.get('changes_detected', []),
                'issues': review_result.get('issues', []),
                'confidence': max(0.0, min(1.0, float(review_result.get('confidence', 0.8)))),
                'recommendation': review_result.get('recommendation', 'accept')
            }

            return result

        except Exception as exc:
            logger.warning(f"Result review failed: {exc}")
            # Fallback: assume success with low confidence
            return {
                'matched': True,
                'assessment': f'Result review failed: {exc}',
                'changes_detected': [f'Shape: {original_shape} → {result_shape}'],
                'issues': [],
                'confidence': 0.6,
                'recommendation': 'review'
            }

    async def _reflect_on_code(self, code: str, request: str, adata: Any, iteration: int = 1) -> Dict[str, Any]:
        """
        Reflect on generated code to identify issues and improvements.

        This method uses the LLM to review the generated code, checking for:
        - Correctness of function calls
        - Proper parameter formatting
        - Syntax errors
        - Alignment with user request

        Parameters
        ----------
        code : str
            The generated Python code to review
        request : str
            The original user request
        adata : Any
            The AnnData object being processed
        iteration : int, optional
            Current reflection iteration number (default: 1)

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'improved_code': str - improved version of code
            - 'issues_found': List[str] - list of issues identified
            - 'confidence': float - confidence in the code (0-1)
            - 'needs_revision': bool - whether code needs revision
            - 'explanation': str - brief explanation of changes
        """
        reflection_prompt = f"""You are a code reviewer for OmicVerse bioinformatics code.

Original User Request: "{request}"

Generated Code (Iteration {iteration}):
```python
{code}
```

Dataset Information:
{f"- Shape: {adata.shape[0]} cells × {adata.shape[1]} genes" if adata is not None and hasattr(adata, 'shape') else "- No dataset provided (knowledge query mode)"}

Your task is to review this code and provide feedback:

1. **Check for correctness**:
   - Are the function calls correct?
   - Are parameters properly formatted (especially dict parameters like 'tresh')?
   - Are there any syntax errors?
   - Does the code match the user's request?

2. **Common issues to check**:
   - Missing or incorrect imports
   - Wrong parameter types or values
   - Incorrect function selection
   - Parameter extraction errors (e.g., nUMI>500 should map to correct parameter)
   - Missing required parameters
   - Using wrong parameter names

3. **CRITICAL VALIDATION CHECKLIST** (These cause frequent errors!):

   **Parameter Name Validation:**
   - pySCSA.cell_auto_anno() uses `clustertype='leiden'`, NOT `cluster='leiden'`!
   - COSG/rank_genes uses `groupby='leiden'`, NOT `cluster='leiden'`
   - These are DIFFERENT parameters with DIFFERENT meanings!

   **Output Storage Validation:**
   - Cell annotations → stored in `adata.obs['column_name']`
   - Marker gene results (COSG, rank_genes_groups) → stored in `adata.uns['key']`
   - COSG does NOT create `adata.obs['cosg_celltype']` - it stores results in `adata.uns['rank_genes_groups']`!

   **Pandas/DataFrame Pitfalls:**
   - DataFrame uses `.dtypes` (PLURAL) for all column types
   - Series uses `.dtype` (SINGULAR) for single column type
   - `df.dtype` will cause AttributeError - use `df.dtypes` instead!

   **Batch Column Validation:**
   - Before batch operations, check if batch column exists and has no NaN values
   - Use `adata.obs['batch'].fillna('unknown')` to handle missing values

   **Geneset Enrichment:**
   - `pathways_dict` must be a dictionary loaded via `ov.utils.geneset_prepare()`, NOT a file path string!
   - WRONG: `ov.bulk.geneset_enrichment(gene_list, pathways_dict='file.gmt')`
   - CORRECT: First load with `pathways_dict = ov.utils.geneset_prepare('file.gmt')`, then pass dict

   **HVG (Highly Variable Genes) - Small Dataset Pitfalls:**
   - `flavor='seurat_v3'` uses LOESS regression which FAILS on:
     - Small batches (<500 cells per batch)
     - Log-normalized data (expects raw counts)
   - Error message: "ValueError: Extrapolation not allowed with blending"
   - ALWAYS wrap HVG in try/except with fallback:
   ```python
   try:
       sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=2000)
   except ValueError:
       sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=2000)
   ```
   - For batch-aware HVG with small batches, prefer `flavor='cell_ranger'` or `flavor='seurat'`

   **In-Place Function Pitfalls:**
   - OmicVerse preprocessing functions operate IN-PLACE by default!
   - Functions: `ov.pp.pca()`, `ov.pp.scale()`, `ov.pp.neighbors()`, `ov.pp.leiden()`, `ov.pp.umap()`, `ov.pp.tsne()`
   - WRONG: `adata = ov.pp.pca(adata)` → returns None, adata becomes None!
   - CORRECT: `ov.pp.pca(adata)` (call without assignment)
   - Alternative: `adata = ov.pp.pca(adata, copy=True)` (explicit copy)
   - Same pattern for `ov.pp.scale()`, `ov.pp.neighbors()`, `ov.pp.umap()`, etc.

   **Print Statement Pitfalls:**
   - NEVER use f-strings in print statements - they cause format errors with special characters
   - WRONG: `print(f"Value: {{val:.2%}}")` → format code errors
   - CORRECT: `print("Value: " + str(round(val * 100, 2)) + "%")`
   - ALWAYS use string concatenation with str() for print statements

   **Categorical Column Access Pitfalls:**
   - NEVER assume a column is categorical - it may be string/object dtype
   - WRONG: `adata.obs['leiden'].cat.categories` → AttributeError if not categorical
   - CORRECT: `adata.obs['leiden'].value_counts()` (works for any dtype)
   - If you MUST access categories: `if hasattr(adata.obs['col'], 'cat'): ...`

   **AUTOMATIC FIXES REQUIRED** (You MUST apply these fixes if found):
   - If code has f-strings in print() → Convert to string concatenation
   - If code has `adata = ov.pp.func(adata)` → Remove the assignment
   - If code has `.cat.categories` without check → Add hasattr() guard or use value_counts()
   - If code has HVG without try/except → Add seurat fallback wrapper
   - If code has batch operations without validation → Add fillna('unknown') guard

4. **Provide feedback as a JSON object**:
{{
  "issues_found": ["specific issue 1", "specific issue 2"],
  "needs_revision": true,
  "confidence": 0.85,
  "improved_code": "the corrected code here",
  "explanation": "brief explanation of what was fixed"
}}

If no issues are found:
{{
  "issues_found": [],
  "needs_revision": false,
  "confidence": 0.95,
  "improved_code": "{code}",
  "explanation": "Code looks correct"
}}

IMPORTANT:
- Return ONLY the JSON object, nothing else
- Keep confidence between 0.0 and 1.0
- If you fix the code, put the complete corrected code in 'improved_code'
- Be specific about issues found
"""

        try:
            with self._temporary_api_keys():
                if not self._llm:
                    raise RuntimeError("LLM backend is not initialized")

                response_text = await self._llm.run(reflection_prompt)

                # Track reflection token usage
                if self._llm.last_usage:
                    self.last_usage_breakdown['reflection'].append(self._llm.last_usage)

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                # Fallback: no issues found
                return {
                    'improved_code': code,
                    'issues_found': [],
                    'confidence': 0.8,
                    'needs_revision': False,
                    'explanation': 'Reflection completed (JSON extraction failed, assuming code is OK)'
                }

            reflection_result = json.loads(json_match.group(0))

            # Validate and normalize the result
            result = {
                'improved_code': reflection_result.get('improved_code', code),
                'issues_found': reflection_result.get('issues_found', []),
                'confidence': max(0.0, min(1.0, float(reflection_result.get('confidence', 0.8)))),
                'needs_revision': bool(reflection_result.get('needs_revision', False)),
                'explanation': reflection_result.get('explanation', 'No explanation provided')
            }

            return result

        except Exception as exc:
            logger.warning(f"Reflection failed: {exc}")
            # Fallback: return original code
            return {
                'improved_code': code,
                'issues_found': [],
                'confidence': 0.7,
                'needs_revision': False,
                'explanation': f'Reflection failed: {exc}'
            }

    def _apply_execution_error_fix(self, code: str, error_msg: str) -> Optional[str]:
        return self._analysis_executor.apply_execution_error_fix(code, error_msg)

    # ------------------------------------------------------------------
    # Self-repair helpers — delegated to AnalysisExecutor (P3-1)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_package_name(error_msg: str) -> Optional[str]:
        return _AnalysisExecutor.extract_package_name(error_msg)

    def _auto_install_package(self, package_name: str) -> bool:
        return self._analysis_executor.auto_install_package(package_name)

    async def _diagnose_error_with_llm(
        self,
        code: str,
        error_msg: str,
        traceback_str: str,
        adata: Any,
    ) -> Optional[str]:
        return await self._analysis_executor.diagnose_error_with_llm(
            code, error_msg, traceback_str, adata,
        )

    def _validate_outputs(self, code: str, output_dir: Optional[str] = None) -> List[str]:
        return self._analysis_executor.validate_outputs(code, output_dir)

    async def _generate_completion_code(
        self,
        original_code: str,
        missing_files: List[str],
        adata: Any,
        request: str,
    ) -> Optional[str]:
        return await self._analysis_executor.generate_completion_code(
            original_code, missing_files, adata, request,
        )

    def _request_approval(self, code: str, violations: list) -> bool:
        return self._analysis_executor.request_approval(code, violations)

    def _execute_generated_code(self, code: str, adata: Any, capture_stdout: bool = False) -> Any:
        return self._analysis_executor.execute_generated_code(code, adata, capture_stdout)

    @staticmethod
    def _normalize_doublet_obs(adata: Any) -> None:
        _AnalysisExecutor.normalize_doublet_obs(adata)

    def _process_context_directives(self, code: str, local_vars: Dict[str, Any]) -> None:
        self._analysis_executor.process_context_directives(code, local_vars)

    def _handle_context_write(self, directive: str, local_vars: Dict[str, Any]) -> None:
        self._analysis_executor._handle_context_write(directive, local_vars)

    def _handle_context_update(self, directive: str) -> None:
        self._analysis_executor._handle_context_update(directive)

    @staticmethod
    def _parse_plan_step(step_text: str) -> Optional[Dict[str, Any]]:
        return _AnalysisExecutor._parse_plan_step(step_text)

    def _build_sandbox_globals(self) -> Dict[str, Any]:
        return self._analysis_executor.build_sandbox_globals()

    def _detect_direct_python_request(self, request: str) -> Optional[str]:
        """Detect and return user-provided Python code to execute directly."""
        trimmed = (request or "").strip()
        if not trimmed:
            return None

        python_markers = (
            "```",
            "import ",
            "from ",
            "def ",
            "class ",
            "adata",
            "ov.",
            "sc.",
            "pd.",
            "np.",
        )

        # For non-python providers, require explicit Python cues to avoid false positives
        if self.provider != "python" and not any(marker in trimmed for marker in python_markers):
            return None

        candidates = self._gather_code_candidates(trimmed)
        if not candidates and self.provider == "python":
            candidates = [trimmed]

        for candidate in candidates:
            try:
                normalized = self._normalize_code_candidate(candidate)
            except ValueError:
                continue
            try:
                ast.parse(normalized)
                return normalized
            except SyntaxError:
                continue

        return None

    async def run_async(self, request: str, adata: Any) -> Any:
        """
        Process a natural language request using the agentic tool-calling loop.

        The agent autonomously inspects data, searches functions/skills,
        generates and executes code, and delegates subtasks until the request
        is fulfilled or the turn limit is reached.

        Parameters
        ----------
        request : str
            Natural language description of what to do.
        adata : Any
            AnnData/MuData object to process.

        Returns
        -------
        Any
            Processed adata object.

        Examples
        --------
        >>> agent.run("qc with nUMI>500", adata)
        >>> agent.run("complete bulk DEG pipeline", adata)
        """

        print(f"\n{'=' * 70}")
        print(f"🤖 OmicVerse Agent Processing Request")
        print(f"{'=' * 70}")
        print(f"Request: \"{request}\"")
        if adata is not None and hasattr(adata, 'shape'):
            print(f"Dataset: {adata.shape[0]} cells × {adata.shape[1]} genes")
        else:
            print(f"Dataset: None (knowledge query)")
        print(f"{'=' * 70}\n")

        # Direct execution path for explicit Python snippets (no LLM required)
        direct_code = self._detect_direct_python_request(request)
        if direct_code:
            print(f"🧪 Direct Python detected → executing without model calls")
            # Reset usage tracking for clarity
            self.last_usage = None
            self.last_usage_breakdown = {
                'generation': None,
                'reflection': [],
                'review': [],
                'total': None
            }
            try:
                result_adata = self._execute_generated_code(direct_code, adata)
                print(f"✅ Python code executed directly.")
                return result_adata
            except Exception as exc:
                print(f"❌ Direct Python execution failed: {exc}")
                raise

        # If user explicitly selected the Python provider, require executable code
        if self.provider == "python":
            raise ValueError("Python provider requires executable Python code in the request.")

        return await self._run_agentic_mode(request, adata)

    async def _run_agentic_mode(self, request: str, adata: Any) -> Any:
        """Agentic loop mode: LLM autonomously calls tools to complete the task."""
        print(f"🤖 Mode: Agentic Loop (tool-calling)")
        print()

        try:
            result = await self._run_agentic_loop(request, adata)
            self._persist_harness_history(request)

            print()
            print(f"{'=' * 70}")
            print(f"✅ SUCCESS - Agentic loop completed!")
            print(f"{'=' * 70}\n")

            return result
        except Exception as e:
            self._persist_harness_history(request)
            print()
            print(f"{'=' * 70}")
            print(f"❌ ERROR - Agentic loop failed: {e}")
            print(f"{'=' * 70}\n")
            raise

    async def _select_skill_matches_llm(self, request: str, top_k: int = 2) -> List[str]:
        """Use LLM to select relevant skills based on the request (Claude Code approach).

        This is pure LLM reasoning - no algorithmic routing, embeddings, or pattern matching.
        The LLM reads skill descriptions and decides which skills match the user's intent.

        Returns:
            List of skill slugs matched by the LLM
        """
        if not self.skill_registry or not self.skill_registry.skill_metadata:
            return []

        # Format all available skills for LLM
        skills_list = []
        for skill in sorted(self.skill_registry.skill_metadata.values(), key=lambda s: s.name.lower()):
            skills_list.append(f"- **{skill.slug}**: {skill.description}")

        skills_catalog = "\n".join(skills_list)

        # Ask LLM to match skills
        matching_prompt = f"""You are a skill matching system. Given a user request and a list of available skills, determine which skills (if any) are relevant.

User Request: "{request}"

Available Skills:
{skills_catalog}

Your task:
1. Analyze the user request to understand their intent
2. Review the skill descriptions
3. Select the {top_k} most relevant skills (or fewer if not many are relevant)
4. Respond with ONLY the skill slugs as a JSON array, e.g., ["skill-slug-1", "skill-slug-2"]
5. If no skills are relevant, return an empty array: []

IMPORTANT: Respond with ONLY the JSON array, nothing else."""

        try:
            with self._temporary_api_keys():
                if not self._llm:
                    return []
                response = await self._llm.run(matching_prompt)

            # Extract JSON array from response
            import re
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                matched_slugs = json.loads(json_match.group(0))
                return [slug for slug in matched_slugs if slug in self.skill_registry.skill_metadata]
            return []

        except Exception as exc:
            logger.warning(f"LLM skill matching failed: {exc}")
            return []

    def _format_skill_guidance(self, matches: List[SkillMatch]) -> str:
        """Format skill instructions for prompt injection."""

        if not matches:
            return ""
        blocks = []
        for match in matches:
            instructions = match.skill.prompt_instructions(max_chars=2000)
            blocks.append(
                f"- {match.skill.name} (score={match.score:.3f})\n"
                f"{instructions}"
            )
        return "\n\n".join(blocks)

    def _format_skill_overview(self) -> str:
        """Generate a bullet overview of available project skills.

        Delegates to :func:`ovagent.bootstrap.format_skill_overview`.
        """
        return _format_skill_overview(self.skill_registry)

    async def stream_async(self, request: str, adata: Any,
                           cancel_event=None, history=None,
                           approval_handler=None):
        """
        Stream agentic-loop events as the agent processes a request.

        Wraps ``_run_agentic_loop`` with an event callback so that callers
        can observe tool calls, code execution, results, and completion in
        real time.

        Parameters
        ----------
        request : str
            Natural language description of what to do.
        adata : Any
            AnnData/MuData object to process.
        cancel_event : threading.Event, optional
            When set, signals the agentic loop to stop at the next safe
            checkpoint.
        history : list, optional
            Prior conversation messages for multi-turn context.

        Yields
        ------
        dict
            Dictionary with ``'type'`` and ``'content'`` keys. Types:

            - ``'tool_call'``: Agent dispatched a tool (``content`` has ``name`` and ``arguments``).
            - ``'llm_chunk'``: LLM assistant text response.
            - ``'code'``: Python code sent to ``execute_code``.
            - ``'result'``: Updated adata after execution (also has ``'shape'``).
            - ``'done'``: Agent declared the task complete (may have ``'cancelled': True``).
            - ``'error'``: An error occurred.
            - ``'usage'``: Token usage statistics (final event).

        Examples
        --------
        >>> agent = ov.Agent(model="gpt-4o-mini")
        >>> async for event in agent.stream_async("qc with nUMI>500", adata):
        ...     if event['type'] == 'llm_chunk':
        ...         print(event['content'], end='', flush=True)
        ...     elif event['type'] == 'result':
        ...         result_adata = event['content']
        ...     elif event['type'] == 'usage':
        ...         print(f"Tokens used: {event['content'].total_tokens}")
        """
        queue: asyncio.Queue = asyncio.Queue()

        async def _event_callback(event):
            await queue.put(event)

        async def _run_loop():
            try:
                await self._run_agentic_loop(
                    request, adata,
                    event_callback=_event_callback,
                    cancel_event=cancel_event,
                    history=history,
                    approval_handler=approval_handler,
                )
            except Exception as exc:
                trace_id = getattr(getattr(self, "_last_run_trace", None), "trace_id", "")
                turn_id = getattr(getattr(self, "_last_run_trace", None), "turn_id", "")
                await queue.put(build_stream_event(
                    "error",
                    str(exc),
                    turn_id=turn_id,
                    trace_id=trace_id,
                    session_id=self._get_harness_session_id(),
                    category="runtime",
                ))
                # Defense-in-depth: ensure a done event is always emitted
                await queue.put(build_stream_event(
                    "done",
                    f"Error: {exc}",
                    turn_id=turn_id,
                    trace_id=trace_id,
                    session_id=self._get_harness_session_id(),
                    category="lifecycle",
                ) | {"error": True})
            finally:
                await queue.put(None)  # sentinel

        task = asyncio.create_task(_run_loop())

        while True:
            event = await queue.get()
            if event is None:
                break
            yield event

        await task

    def run(self, request: str, adata: Any) -> Any:
        """
        Process a natural language request with the provided adata (main method).
        
        Parameters
        ----------
        request : str
            Natural language description of what to do
        adata : Any
            AnnData object to process
            
        Returns
        -------
        Any
            Processed adata object (modified)
            
        Examples
        --------
        >>> agent = ov.Agent(model="gpt-4o-mini")
        >>> result = agent.run("quality control with nUMI>500, mito<0.2", adata)
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            result_container: Dict[str, Any] = {}
            error_container: Dict[str, BaseException] = {}

            def _run_in_thread() -> None:
                try:
                    result_container["value"] = asyncio.run(self.run_async(request, adata))
                except BaseException as exc:  # pragma: no cover - propagate to caller
                    error_container["error"] = exc

            thread = threading.Thread(target=_run_in_thread, name="OmicVerseAgentRunner")
            thread.start()
            thread.join()

            if "error" in error_container:
                raise error_container["error"]

            return result_container.get("value")

        return asyncio.run(self.run_async(request, adata))

    # ===================================================================
    # Session Management Methods
    # ===================================================================

    def get_current_session_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about current notebook session.

        Returns
        -------
        Optional[Dict[str, Any]]
            Session information dictionary with keys:
            - session_id: Session identifier
            - notebook_path: Path to session notebook
            - prompt_count: Number of prompts executed in session
            - max_prompts: Maximum prompts per session
            - remaining_prompts: Prompts remaining before restart
            - start_time: Session start time (ISO format)
            Returns None if notebook execution is disabled or no session exists.

        Examples
        --------
        >>> agent = ov.Agent(model="gpt-5.2")
        >>> agent.run("preprocess data", adata)
        >>> info = agent.get_current_session_info()
        >>> print(f"Session: {info['session_id']}")
        >>> print(f"Prompts: {info['prompt_count']}/{info['max_prompts']}")
        """
        if not self.use_notebook_execution or not self._notebook_executor:
            return None

        if not self._notebook_executor.current_session:
            return None

        session = self._notebook_executor.current_session
        return {
            'session_id': session['session_id'],
            'notebook_path': str(session['notebook_path']),
            'prompt_count': self._notebook_executor.session_prompt_count,
            'max_prompts': self._notebook_executor.max_prompts_per_session,
            'remaining_prompts': self.max_prompts_per_session - self._notebook_executor.session_prompt_count,
            'start_time': session['start_time'].isoformat()
        }

    def restart_session(self):
        """
        Manually restart notebook session (clear memory, start fresh).

        This forces a new session to be created on the next execution,
        useful for freeing memory or starting with a clean state.

        Examples
        --------
        >>> agent = ov.Agent(model="gpt-5.2")
        >>> agent.run("step 1", adata)
        >>> agent.run("step 2", adata)
        >>> # Force new session
        >>> agent.restart_session()
        >>> agent.run("step 3", adata)  # Runs in new session
        """
        if self.use_notebook_execution and self._notebook_executor:
            if self._notebook_executor.current_session:
                print("⚙ = Manually restarting session...")
                self._notebook_executor._archive_current_session()
                self._notebook_executor.current_session = None
                self._notebook_executor.session_prompt_count = 0
                print("✓ Session cleared. Next prompt will start new session.")
            else:
                print("💡 No active session to restart")
        else:
            print("⚠️  Notebook execution is not enabled")

    def get_session_history(self) -> List[Dict[str, Any]]:
        """
        Get history of all archived notebook sessions.

        Returns
        -------
        List[Dict[str, Any]]
            List of session history dictionaries, each containing:
            - session_id: Session identifier
            - notebook_path: Path to archived notebook
            - prompt_count: Number of prompts executed
            - start_time: Session start time
            - end_time: Session end time
            - executions: List of execution records

        Examples
        --------
        >>> agent = ov.Agent(model="gpt-5.2")
        >>> # ... run several prompts causing session restarts ...
        >>> history = agent.get_session_history()
        >>> for session in history:
        ...     print(f"{session['session_id']}: {session['prompt_count']} prompts")
        """
        if self.use_notebook_execution and self._notebook_executor:
            return self._notebook_executor.session_history
        return []

    # ===================================================================
    # Filesystem Context Management Methods
    # ===================================================================

    @property
    def filesystem_context(self) -> Optional[FilesystemContextManager]:
        """Get the filesystem context manager.

        Returns
        -------
        FilesystemContextManager or None
            The context manager if enabled, None otherwise.
        """
        return self._filesystem_context if self.enable_filesystem_context else None

    def write_note(
        self,
        key: str,
        content: Union[str, Dict[str, Any]],
        category: str = "notes",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Write a note to the filesystem context workspace.

        Use this to offload intermediate results, observations, or decisions
        from the context window to persistent storage. This reduces token usage
        and enables selective context retrieval.

        Parameters
        ----------
        key : str
            Unique identifier for this note. Used for later retrieval.
        content : str or dict
            The note content. Can be free-form text or structured data.
        category : str, optional
            Category for organizing notes (default: "notes").
            Options: notes, results, decisions, snapshots, figures, errors
        metadata : dict, optional
            Additional metadata to store with the note.

        Returns
        -------
        str or None
            Path to the stored note, or None if filesystem context is disabled.

        Examples
        --------
        >>> agent.write_note("qc_stats", {"n_cells": 5000, "mito_pct": 0.05}, category="results")
        >>> agent.write_note("observation", "Cluster 3 shows high mitochondrial content")
        """
        if not self._filesystem_context:
            return None

        try:
            return self._filesystem_context.write_note(key, content, category, metadata)
        except Exception as e:
            logger.warning(f"Failed to write note: {e}")
            return None

    def search_context(
        self,
        pattern: str,
        match_type: str = "glob",
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search the filesystem context for relevant notes.

        Use glob patterns to find notes by key, or grep patterns to search
        within note content.

        Parameters
        ----------
        pattern : str
            Search pattern. For glob: "pca*", "cluster_*". For grep: regex pattern.
        match_type : str, optional
            Type of search: "glob" (filename pattern) or "grep" (content search).
            Default: "glob".
        max_results : int, optional
            Maximum number of results to return (default: 10).

        Returns
        -------
        list of dict
            Matching results with key, category, and content preview.

        Examples
        --------
        >>> results = agent.search_context("cluster*", match_type="glob")
        >>> results = agent.search_context("resolution", match_type="grep")
        """
        if not self._filesystem_context:
            return []

        try:
            results = self._filesystem_context.search_context(pattern, match_type, max_results=max_results)
            return [
                {
                    "key": r.key,
                    "category": r.category,
                    "preview": r.content_preview,
                    "relevance": r.relevance_score,
                }
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Failed to search context: {e}")
            return []

    def get_relevant_context(
        self,
        query: str,
        max_tokens: int = 1000,
    ) -> str:
        """Get context relevant to a query, formatted for LLM injection.

        This method searches the filesystem context for notes relevant to
        the given query and formats them for inclusion in prompts.

        Parameters
        ----------
        query : str
            The current task or query to find relevant context for.
        max_tokens : int, optional
            Approximate maximum tokens to return (default: 1000).

        Returns
        -------
        str
            Formatted context string ready for LLM injection.

        Examples
        --------
        >>> context = agent.get_relevant_context("clustering")
        >>> # Use context in custom prompts
        """
        if not self._filesystem_context:
            return ""

        try:
            return self._filesystem_context.get_relevant_context(query, max_tokens)
        except Exception as e:
            logger.warning(f"Failed to get relevant context: {e}")
            return ""

    def save_plan(self, steps: List[Dict[str, Any]]) -> Optional[str]:
        """Save an execution plan to the filesystem context.

        Plans are persisted and can be tracked across prompts.

        Parameters
        ----------
        steps : list of dict
            List of step definitions. Each step should have:
            - description: What this step does
            - status: pending, in_progress, completed, failed
            - optional: function, parameters, expected_output

        Returns
        -------
        str or None
            Path to the plan file, or None if filesystem context is disabled.

        Examples
        --------
        >>> agent.save_plan([
        ...     {"description": "Run QC", "status": "pending"},
        ...     {"description": "Normalize data", "status": "pending"},
        ...     {"description": "Cluster cells", "status": "pending"},
        ... ])
        """
        if not self._filesystem_context:
            return None

        try:
            return self._filesystem_context.write_plan(steps)
        except Exception as e:
            logger.warning(f"Failed to save plan: {e}")
            return None

    def update_plan_step(
        self,
        step_index: int,
        status: str,
        result: Optional[str] = None,
    ) -> None:
        """Update the status of a plan step.

        Parameters
        ----------
        step_index : int
            Index of the step to update (0-based).
        status : str
            New status: pending, in_progress, completed, failed.
        result : str, optional
            Result or notes for this step.

        Examples
        --------
        >>> agent.update_plan_step(0, "completed", "QC removed 500 low-quality cells")
        >>> agent.update_plan_step(1, "in_progress")
        """
        if not self._filesystem_context:
            return

        try:
            self._filesystem_context.update_plan_step(step_index, status, result)
        except Exception as e:
            logger.warning(f"Failed to update plan step: {e}")

    def get_workspace_summary(self) -> str:
        """Get a summary of the filesystem context workspace.

        Returns
        -------
        str
            Markdown-formatted workspace summary including:
            - Session ID
            - Plan progress (if a plan exists)
            - Notes by category
            - Recent activity

        Examples
        --------
        >>> print(agent.get_workspace_summary())
        """
        if not self._filesystem_context:
            return "Filesystem context is disabled."

        try:
            return self._filesystem_context.get_session_summary()
        except Exception as e:
            logger.warning(f"Failed to get workspace summary: {e}")
            return f"Error getting workspace summary: {e}"

    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about the filesystem context workspace.

        Returns
        -------
        dict
            Workspace statistics including:
            - session_id: Current session ID
            - workspace_dir: Path to workspace directory
            - categories: Notes count and size by category
            - total_notes: Total number of notes
            - total_size_bytes: Total size in bytes

        Examples
        --------
        >>> stats = agent.get_context_stats()
        >>> print(f"Total notes: {stats['total_notes']}")
        """
        if not self._filesystem_context:
            return {"enabled": False}

        try:
            stats = self._filesystem_context.get_workspace_stats()
            stats["enabled"] = True
            return stats
        except Exception as e:
            logger.warning(f"Failed to get context stats: {e}")
            return {"enabled": True, "error": str(e)}

    def __del__(self):
        """Cleanup on agent deletion."""
        if hasattr(self, '_notebook_executor') and self._notebook_executor:
            try:
                self._notebook_executor.shutdown()
            except:
                pass

        # Cleanup filesystem context if needed
        if hasattr(self, '_filesystem_context') and self._filesystem_context:
            try:
                self._filesystem_context.cleanup_session(keep_summary=True)
            except:
                pass


def list_supported_models(show_all: bool = False) -> str:
    """
    List all supported models for OmicVerse Smart Agent.
    
    Parameters
    ----------
    show_all : bool, optional
        If True, show all models. If False, show top 3 per provider (default: False)
        
    Returns
    -------
    str
        Formatted list of supported models with API key status
        
    Examples
    --------
    >>> import omicverse as ov
    >>> print(ov.list_supported_models())
    >>> print(ov.list_supported_models(show_all=True))
    """
    return ModelConfig.list_supported_models(show_all)

def Agent(model: str = "gpt-5.2", api_key: Optional[str] = None, endpoint: Optional[str] = None, enable_reflection: bool = True, reflection_iterations: int = 1, enable_result_review: bool = True, use_notebook_execution: bool = True, max_prompts_per_session: int = 5, notebook_storage_dir: Optional[str] = None, keep_execution_notebooks: bool = True, notebook_timeout: int = 600, strict_kernel_validation: bool = True, enable_filesystem_context: bool = True, context_storage_dir: Optional[str] = None, approval_mode: str = "never", agent_mode: str = "agentic", max_agent_turns: int = 15, security_level: Optional[str] = None, *, config: Optional[AgentConfig] = None, reporter: Optional[Reporter] = None, verbose: bool = True) -> OmicVerseAgent:
    """
    Create an OmicVerse Smart Agent instance.

    This function creates and returns a smart agent that can execute OmicVerse functions
    based on natural language descriptions.

    Parameters
    ----------
    model : str, optional
        LLM model to use (default: "gpt-5.2"). Use list_supported_models() to see all options
    api_key : str, optional
        API key for the model provider. If not provided, will use environment variable
    endpoint : str, optional
        Custom API endpoint. If not provided, will use default for the provider
    enable_reflection : bool, optional
        Enable reflection step to review and improve generated code (default: True)
    reflection_iterations : int, optional
        Maximum number of reflection iterations (default: 1, range: 1-3)
    enable_result_review : bool, optional
        Enable result review to validate output matches user intent (default: True)
    use_notebook_execution : bool, optional
        Execute code in separate Jupyter notebook for isolation and debugging (default: True).
        Set to False to use legacy in-process execution.
    max_prompts_per_session : int, optional
        Number of prompts to execute in same notebook session before restart (default: 5).
        This prevents memory bloat while maintaining context for iterative analysis.
    notebook_storage_dir : str, optional
        Directory to store session notebooks. Defaults to ~/.ovagent/sessions
    keep_execution_notebooks : bool, optional
        Whether to keep session notebooks after execution (default: True)
    notebook_timeout : int, optional
        Execution timeout in seconds (default: 600)
    strict_kernel_validation : bool, optional
        If True, raise error if kernel not found. If False, fall back to python3 kernel (default: True)
    enable_filesystem_context : bool, optional
        Enable filesystem-based context management for offloading intermediate results,
        plans, and notes to disk. This reduces context window usage and enables
        selective context retrieval. Default: True.
    context_storage_dir : str, optional
        Directory for storing context files. Defaults to ~/.ovagent/context/
    approval_mode : str, optional
        When to prompt the user before executing generated code.
        "never" (default): execute immediately.
        "always": always show code and ask for approval.
        "on_violation": ask only when security scanner finds issues.

    Returns
    -------
    OmicVerseAgent
        Configured agent instance ready for use

    Examples
    --------
    >>> import omicverse as ov
    >>> import scanpy as sc
    >>>
    >>> # Create agent instance with full validation (default, session-based execution)
    >>> agent = ov.Agent(model="gpt-5", api_key="your-key")
    >>>
    >>> # Create agent with multiple reflection iterations
    >>> agent = ov.Agent(model="gpt-5", api_key="your-key", reflection_iterations=2)
    >>>
    >>> # Create agent without validation (fastest execution)
    >>> agent = ov.Agent(model="gpt-5", api_key="your-key", enable_reflection=False, enable_result_review=False)
    >>>
    >>> # Disable notebook execution (use legacy in-process execution)
    >>> agent = ov.Agent(model="gpt-5", api_key="your-key", use_notebook_execution=False)
    >>>
    >>> # Maximum isolation (new session per prompt)
    >>> agent = ov.Agent(model="gpt-5", api_key="your-key", max_prompts_per_session=1)
    >>>
    >>> # Longer sessions for complex workflows
    >>> agent = ov.Agent(model="gpt-5", api_key="your-key", max_prompts_per_session=10)
    >>>
    >>> # Custom storage directory
    >>> agent = ov.Agent(model="gpt-5", api_key="your-key", notebook_storage_dir="~/my_project/sessions")
    >>>
    >>> # Load data
    >>> adata = sc.datasets.pbmc3k()
    >>>
    >>> # Use agent for quality control
    >>> adata = agent.run("quality control with nUMI>500, mito<0.2", adata)
    >>>
    >>> # Use agent for preprocessing
    >>> adata = agent.run("preprocess with 2000 highly variable genes", adata)
    >>>
    >>> # Use agent for clustering
    >>> adata = agent.run("leiden clustering resolution=1.0", adata)
    >>>
    >>> # Check session info
    >>> info = agent.get_current_session_info()
    >>> print(f"Session: {info['session_id']}, Prompts: {info['prompt_count']}/{info['max_prompts']}")
    """
    return OmicVerseAgent(
        model=model,
        api_key=api_key,
        endpoint=endpoint,
        enable_reflection=enable_reflection,
        reflection_iterations=reflection_iterations,
        enable_result_review=enable_result_review,
        use_notebook_execution=use_notebook_execution,
        max_prompts_per_session=max_prompts_per_session,
        notebook_storage_dir=notebook_storage_dir,
        keep_execution_notebooks=keep_execution_notebooks,
        notebook_timeout=notebook_timeout,
        strict_kernel_validation=strict_kernel_validation,
        enable_filesystem_context=enable_filesystem_context,
        context_storage_dir=context_storage_dir,
        approval_mode=approval_mode,
        agent_mode=agent_mode,
        max_agent_turns=max_agent_turns,
        security_level=security_level,
        config=config,
        reporter=reporter,
        verbose=verbose,
    )


# Export the main functions
__all__ = ["Agent", "OmicVerseAgent", "list_supported_models"]

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
import mimetypes
import shutil
import subprocess
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union
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
    if not hasattr(_parent_pkg, "utils"):
        setattr(_parent_pkg, "utils", _utils_pkg)

    module_name = __name__.split(".")[-1]
    if not hasattr(_utils_pkg, module_name):
        setattr(_utils_pkg, module_name, sys.modules[__name__])

# Internal LLM backend (Pantheon replacement)
from .agent_backend import OmicVerseLLMBackend

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
    resolve_tool_search,
)
from .context_compactor import ContextCompactor
from .session_history import HistoryEntry, SessionHistory
from .skill_registry import (
    SkillMatch,
    SkillMetadata,
    SkillDefinition,
    SkillRegistry,
    SkillRouter,
    build_skill_registry,
    build_multi_path_skill_registry,
)

# Import filesystem context management for context engineering
# Reference: https://blog.langchain.com/how-agents-can-use-filesystems-for-context-engineering/
from .filesystem_context import FilesystemContextManager


logger = logging.getLogger(__name__)


class ProactiveCodeTransformer:
    """Transform LLM-generated code to prevent common errors before execution.

    This class applies AST-based transformations to fix known error patterns
    that the LLM may generate due to stochasticity, such as:
    - In-place function assignments: `adata = ov.pp.pca(adata)` → `ov.pp.pca(adata)`
    - F-strings in print statements: Converted to string concatenation
    - .cat accessor without validation: Adds hasattr() guards
    """

    # In-place OmicVerse functions that return None (or adata if copy=True)
    INPLACE_FUNCTIONS = {
        'pca', 'scale', 'neighbors', 'leiden', 'umap', 'tsne', 'sude',
        'scrublet', 'mde', 'louvain', 'phate'
    }

    # Known keyword renames: {(module_pattern, old_kwarg): new_kwarg}
    # These fix common LLM hallucinations where the parameter name differs
    # from what the installed library version actually accepts.
    KWARG_RENAMES = {
        # muon 0.1.x: mu.atac.tl.lsi() uses n_comps, not n_components
        (r'mu(?:on)?\.atac\.tl\.lsi', 'n_components'): 'n_comps',
    }

    def transform(self, code: str) -> str:
        """Apply all proactive transformations to the code.

        Parameters
        ----------
        code : str
            The LLM-generated Python code

        Returns
        -------
        str
            Transformed code with error patterns fixed
        """
        try:
            # Apply regex-based transforms first (safer, handles syntax variations)
            code = self._fix_inplace_assignments_regex(code)
            code = self._fix_fstring_print_regex(code)
            code = self._fix_cat_accessor_regex(code)
            code = self._fix_kwarg_renames(code)

            # Validate the transformed code is still valid Python
            ast.parse(code)
            return code
        except SyntaxError:
            # If transformation breaks syntax, return original
            logger.debug("ProactiveCodeTransformer: transformation produced invalid syntax, returning original")
            return code
        except Exception as e:
            logger.debug(f"ProactiveCodeTransformer: unexpected error {e}, returning original")
            return code

    def _fix_inplace_assignments_regex(self, code: str) -> str:
        """Remove assignments from in-place function calls using regex.

        Transform: `adata = ov.pp.pca(adata, ...)` → `ov.pp.pca(adata, ...)`
        """
        inplace_pattern = '|'.join(self.INPLACE_FUNCTIONS)
        # Match: adata = ov.pp.func(adata, ...) or adata=ov.pp.func(adata)
        pattern = r'adata\s*=\s*(ov\.pp\.(?:' + inplace_pattern + r')\s*\([^)]*\))'

        # Replace with just the function call
        fixed = re.sub(pattern, r'\1', code)

        if fixed != code:
            logger.debug(f"ProactiveCodeTransformer: fixed in-place function assignment")

        return fixed

    def _fix_fstring_print_regex(self, code: str) -> str:
        """Convert f-strings in print statements to string concatenation.

        This is a simple heuristic - converts common patterns.
        """
        # Pattern to match print(f"...{var}...") and similar
        # We'll handle simple cases like print(f"Text: {var}")

        lines = code.split('\n')
        fixed_lines = []

        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith('print(f"') or stripped.startswith("print(f'"):
                # Try to convert simple f-string patterns
                try:
                    fixed_line = self._convert_fstring_line(line)
                    if fixed_line != line:
                        logger.debug(f"ProactiveCodeTransformer: converted f-string in print")
                    fixed_lines.append(fixed_line)
                except Exception:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _convert_fstring_line(self, line: str) -> str:
        """Convert a single line with f-string print to concatenation."""
        indent = len(line) - len(line.lstrip())
        indent_str = line[:indent]
        content = line.strip()

        # Match print(f"...") or print(f'...')
        match = re.match(r'print\(f(["\'])(.*)\1\)', content)
        if not match:
            return line

        quote = match.group(1)
        fstring_content = match.group(2)

        # Find {var} patterns and convert
        parts = []
        last_end = 0

        for m in re.finditer(r'\{([^}:]+)(?::[^}]*)?\}', fstring_content):
            # Add text before this variable
            if m.start() > last_end:
                text_part = fstring_content[last_end:m.start()]
                if text_part:
                    parts.append(f'"{text_part}"')

            # Add the variable wrapped in str()
            var_name = m.group(1).strip()
            parts.append(f'str({var_name})')

            last_end = m.end()

        # Add remaining text
        if last_end < len(fstring_content):
            remaining = fstring_content[last_end:]
            if remaining:
                parts.append(f'"{remaining}"')

        if not parts:
            return line

        # Join with +
        concatenated = ' + '.join(parts)
        return f'{indent_str}print({concatenated})'

    def _fix_cat_accessor_regex(self, code: str) -> str:
        """Add guards around .cat accessor usage.

        Transform: `col.cat.categories` → `col.value_counts().index.tolist()`
        (Only for simple patterns where we want category values)
        """
        # Pattern: adata.obs['col'].cat.categories or .cat.codes
        # These often fail if column is not categorical

        # Simple replacement: .cat.categories → .value_counts().index.tolist()
        code = re.sub(
            r"\.cat\.categories",
            ".value_counts().index.tolist()",
            code
        )

        return code

    def _fix_kwarg_renames(self, code: str) -> str:
        """Rename keyword arguments that LLMs hallucinate for known APIs.

        For example, muon 0.1.x uses ``n_comps`` in ``mu.atac.tl.lsi()``
        but GPT-5.2 consistently generates ``n_components``.
        """
        for (func_pat, old_kw), new_kw in self.KWARG_RENAMES.items():
            # Match: func_call(..., old_kw=value, ...)
            # re.DOTALL allows matching across multi-line function calls
            pattern = rf'({func_pat}\s*\([^)]*)\b{old_kw}\s*='
            replacement = rf'\1{new_kw}='
            new_code = re.sub(pattern, replacement, code, flags=re.DOTALL)
            if new_code != code:
                logger.debug(
                    f"ProactiveCodeTransformer: renamed kwarg {old_kw} -> {new_kw}"
                )
                code = new_code
        return code


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
        
        # When using a custom endpoint (proxy), keep the model name as-is.
        # Proxies expect the exact model name the user typed.
        if endpoint:
            print(f"   🔌 Proxy mode: model={model}, endpoint={endpoint}")
        else:
            # Normalize model ID for aliases and variations, then validate
            original_model = model
            try:
                model = ModelConfig.normalize_model_id(model)  # type: ignore[attr-defined]
            except Exception:
                # Older ModelConfig without normalization: proceed as-is
                model = model
            if model != original_model:
                print(f"   📝 Model ID normalized: {original_model} → {model}")

            is_valid, validation_msg = ModelConfig.validate_model_setup(model, api_key)
            if not is_valid:
                print(f"❌ {validation_msg}")
                raise ValueError(validation_msg)
        
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint or ModelConfig.get_endpoint_for_model(model)
        # Store provider to allow provider-aware formatting of skills
        self.provider = ModelConfig.get_provider_from_model(model)
        self._llm: Optional[OmicVerseLLMBackend] = None
        self.skill_registry: Optional[SkillRegistry] = None
        self._skill_overview_text: str = ""
        self._use_llm_skill_matching: bool = True  # Use LLM-based skill matching (Claude Code approach)
        self._managed_api_env: Dict[str, str] = {}
        # Reflection configuration
        self.enable_reflection = enable_reflection
        self.reflection_iterations = max(1, min(3, reflection_iterations))  # Clamp to 1-3
        # Result review configuration
        self.enable_result_review = enable_result_review
        # Notebook execution configuration
        self.use_notebook_execution = use_notebook_execution
        self.max_prompts_per_session = max_prompts_per_session
        self._notebook_executor = None
        # Filesystem context configuration (set early to avoid AttributeError)
        self.enable_filesystem_context = enable_filesystem_context
        self._filesystem_context: Optional[FilesystemContextManager] = None
        # Token usage tracking at agent level
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
        try:
            self._managed_api_env = self._collect_api_key_env(api_key)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to collect API key environment variables: %s", exc)
            self._managed_api_env = {}

        # Discover project skills for progressive disclosure guidance
        self._initialize_skill_registry()

        # Display model info
        provider = ModelConfig.get_provider_from_model(model)
        model_desc = ModelConfig.get_model_description(model)
        print(f"    Model: {model_desc}")
        print(f"    Provider: {provider.title()}")
        print(f"    Endpoint: {self.endpoint}")
        
        # Check API key status
        with self._temporary_api_keys():
            key_available, key_msg = ModelConfig.check_api_key_availability(model)
        if key_available:
            print(f"   ✅ {key_msg}")
        else:
            print(f"   ⚠️  {key_msg}")
        
        try:
            with self._temporary_api_keys():
                self._setup_agent()
            stats = self._get_registry_stats()
            print(f"   📚 Function registry loaded: {stats['total_functions']} functions in {stats['categories']} categories")

            # Display reflection and result review configuration
            if self.enable_reflection:
                print(f"   🔍 Reflection enabled: {self.reflection_iterations} iteration{'s' if self.reflection_iterations > 1 else ''} (code review & validation)")
            else:
                print(f"   ⚡ Reflection disabled (faster execution, no code validation)")

            if self.enable_result_review:
                print(f"   ✅ Result review enabled (output validation & assessment)")
            else:
                print(f"   ⚡ Result review disabled (no output validation)")

            # Initialize notebook execution if enabled
            if self.use_notebook_execution:
                try:
                    from .session_notebook_executor import SessionNotebookExecutor
                    from pathlib import Path

                    storage_dir = Path(notebook_storage_dir) if notebook_storage_dir else None
                    self._notebook_executor = SessionNotebookExecutor(
                        max_prompts_per_session=max_prompts_per_session,
                        storage_dir=storage_dir,
                        keep_notebooks=keep_execution_notebooks,
                        timeout=notebook_timeout,
                        strict_kernel_validation=strict_kernel_validation
                    )

                    print(f"   📓 Session-based notebook execution enabled")
                    print(f"      Conda environment: {self._notebook_executor.conda_env or 'default'}")
                    print(f"      Session limit: {max_prompts_per_session} prompts")
                    print(f"      Storage: {self._notebook_executor.storage_dir}")

                except Exception as e:
                    print(f"   ⚠️  Notebook execution initialization failed: {e}")
                    print(f"   ⚡ Falling back to in-process execution")
                    self.use_notebook_execution = False
                    self._notebook_executor = None
            else:
                print(f"   ⚡ Using in-process execution (no session isolation)")

            # Initialize filesystem context management (attributes already set early in __init__)
            if self.enable_filesystem_context:
                try:
                    base_dir = Path(context_storage_dir) if context_storage_dir else None
                    self._filesystem_context = FilesystemContextManager(base_dir=base_dir)
                    print(f"   📁 Filesystem context enabled")
                    print(f"      Session: {self._filesystem_context.session_id}")
                    print(f"      Storage: {self._filesystem_context._workspace_dir}")
                except Exception as e:
                    logger.warning(f"Filesystem context initialization failed: {e}")
                    self.enable_filesystem_context = False
                    self._filesystem_context = None
                    print(f"   ⚠️  Filesystem context disabled (init failed: {e})")
            else:
                print(f"   ⚡ Filesystem context disabled")

            if getattr(self._config, "history_enabled", False):
                self._session_history = SessionHistory(path=self._config.history_path)
                print(f"   📝 Session history enabled")

            harness_config = getattr(self._config, "harness", None)
            if harness_config is not None and getattr(harness_config, "enable_traces", False):
                self._trace_store = RunTraceStore(root=harness_config.trace_dir)
                print(f"   🧰 Harness tracing enabled")
            if harness_config is not None and getattr(harness_config, "enable_context_compaction", False) and self._llm:
                self._context_compactor = ContextCompactor(self._llm, self.model)
                print(f"   🗜️  Harness context compaction enabled")

            # Initialize security scanner
            self._security_config: SecurityConfig = getattr(
                self._config, "security", SecurityConfig()
            )
            self._security_scanner = CodeSecurityScanner(self._security_config)
            print(f"   🛡️  Security scanner enabled (approval: {self._security_config.approval_mode.value})")

            print(f"✅ Smart Agent initialized successfully!")
        except Exception as e:
            print(f"❌ Agent initialization failed: {e}")
            raise

    def _initialize_skill_registry(self) -> None:
        """Load skills from package install and current working directory and prepare routing helpers."""

        package_root = Path(__file__).resolve().parents[2]
        cwd = Path.cwd()
        try:
            registry = build_multi_path_skill_registry(package_root, cwd)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"⚠️  Failed to load Agent Skills: {exc}")
            registry = None

        if registry and registry.skill_metadata:
            self.skill_registry = registry
            self._skill_overview_text = self._format_skill_overview()

            package_skill_root = package_root / ".claude" / "skills"
            cwd_skill_root = cwd / ".claude" / "skills"
            # Count from metadata instead of full skills
            builtin_count = len([s for s in registry.skill_metadata.values() if str(package_skill_root) in str(s.path)])
            user_count = len([s for s in registry.skill_metadata.values() if str(cwd_skill_root) in str(s.path)])
            total = len(registry.skill_metadata)
            msg = f"   🧭 Loaded {total} skills (progressive disclosure)"
            if builtin_count and user_count:
                msg += f" ({builtin_count} built-in + {user_count} user-created)"
            elif builtin_count:
                msg += f" ({builtin_count} built-in)"
            elif user_count:
                msg += f" ({user_count} user-created)"
            print(msg)
        else:
            self.skill_registry = None
            self._skill_overview_text = ""

    def _get_registry_stats(self) -> dict:
        """Get statistics about the function registry."""
        # Get all unique functions from registry
        unique_functions = set()
        categories = set()
        
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
        """Collect environment variables required for API authentication."""

        if not api_key:
            return {}

        env_mapping: Dict[str, str] = {}
        required_key = PROVIDER_API_KEYS.get(self.model)
        if required_key:
            env_mapping[required_key] = api_key

        provider = ModelConfig.get_provider_from_model(self.model)
        if provider == "openai":
            # Ensure OPENAI_API_KEY is always populated for OpenAI-compatible SDKs
            env_mapping.setdefault("OPENAI_API_KEY", api_key)

        return env_mapping

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

        This teaches LLMs how to use the filesystem-based context management
        system for offloading intermediate results, plans, and notes.

        Returns
        -------
        str
            Instructions for filesystem context usage.
        """
        session_id = self._filesystem_context.session_id if self._filesystem_context else "N/A"

        return f"""

## Context Engineering with Filesystem Workspace

You have access to a **filesystem-based context workspace** that allows you to:
- Offload intermediate results to reduce memory/context usage
- Save and track execution plans across multiple steps
- Search for relevant context using patterns
- Share context with sub-agents

**Current Session**: `{session_id}`

### Why Use the Workspace?

1. **Reduce Context Window Usage**: Instead of keeping all results in memory, write them to disk
2. **Track Multi-Step Workflows**: Save plans and update progress as you complete steps
3. **Retrieve Relevant Context**: Search for notes when you need specific information
4. **Debug and Audit**: All notes are persisted for later review

### Available Context Operations

#### 1. Writing Notes (Offload Results)
Use `# CONTEXT_WRITE:` comments in your code to indicate what should be saved:

```python
# After completing a step, offload the result
# CONTEXT_WRITE: qc_result -> {{"n_cells_before": original_count, "n_cells_after": adata.n_obs, "removed": removed_count}}

# Example: Save intermediate statistics
qc_stats = {{
    "n_cells": adata.n_obs,
    "n_genes": adata.n_vars,
    "mito_pct_mean": float(adata.obs['pct_counts_mt'].mean()) if 'pct_counts_mt' in adata.obs else None
}}
# CONTEXT_WRITE: qc_stats -> qc_stats
```

#### 2. Saving Execution Plans
For multi-step workflows, define a plan upfront:

```python
# CONTEXT_PLAN:
# - Step 1: Quality Control [pending]
# - Step 2: Normalization [pending]
# - Step 3: Feature Selection [pending]
# - Step 4: Dimensionality Reduction [pending]
# - Step 5: Clustering [pending]
```

#### 3. Updating Plan Progress
As you complete steps, update the plan:

```python
# CONTEXT_UPDATE: step=0, status=completed, result="QC removed 500 low-quality cells"
```

#### 4. Searching for Context
When you need to reference previous results:

```python
# CONTEXT_SEARCH: pattern="qc*", type="glob"
# Or for content search:
# CONTEXT_SEARCH: pattern="resolution", type="grep"
```

### Context Categories

Organize your notes by category:
- **notes**: General observations and comments
- **results**: Computation results (statistics, parameters)
- **decisions**: Important choices and their rationale
- **snapshots**: Data state at key points
- **errors**: Error logs and debugging information

### Best Practices

1. **Write Early, Write Often**: Offload results as soon as they're computed
2. **Use Descriptive Keys**: `clustering_leiden_res1.0` is better than `result1`
3. **Include Metadata**: Add function names, parameters, timestamps
4. **Reference Previous Context**: Check workspace before repeating computations
5. **Update Plans Promptly**: Mark steps complete immediately after finishing

### Example: Multi-Step Workflow with Context

```python
import omicverse as ov

# CONTEXT_PLAN:
# - Step 1: Quality Control [in_progress]
# - Step 2: Preprocessing [pending]
# - Step 3: Clustering [pending]

# Step 1: QC
original_cells = adata.n_obs
adata = ov.pp.qc(adata, tresh={{'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250}})
removed = original_cells - adata.n_obs

# CONTEXT_WRITE: qc_result -> {{"original": original_cells, "remaining": adata.n_obs, "removed": removed}}
# CONTEXT_UPDATE: step=0, status=completed, result="Removed " + str(removed) + " cells"

print("QC completed: " + str(adata.n_obs) + " cells remaining")
```

### Automatic Context Injection

The workspace context is automatically searched and injected into prompts when relevant.
You can reference previous results without explicitly searching:

- Recent notes are included automatically
- Plan status is always visible
- Relevant context is retrieved based on the current task
"""

    @contextmanager
    def _temporary_api_keys(self):
        """Temporarily inject API keys into the environment and clean up afterwards."""

        if not self._managed_api_env:
            yield
            return

        previous_values: Dict[str, Optional[str]] = {}
        try:
            for key, value in self._managed_api_env.items():
                previous_values[key] = os.environ.get(key)
                os.environ[key] = value
            yield
        finally:
            for key, previous in previous_values.items():
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous

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
        self._llm = OmicVerseLLMBackend(
            system_prompt=instructions,
            model=self.model,
            api_key=self.api_key,
            endpoint=self.endpoint,
            max_tokens=8192,
            temperature=0.2,
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
        """Handle inspect_data tool call. Returns formatted string."""
        if adata is None:
            return "No dataset is loaded. Use web_download to download a dataset first, then load it with execute_code."
        try:
            parts = []
            dtype = type(adata).__name__
            is_mudata = dtype == "MuData"

            # MuData top-level summary
            if is_mudata and aspect in ("full", "shape"):
                parts.append(f"Type: MuData")
                if hasattr(adata, 'mod'):
                    mod_keys = list(adata.mod.keys())
                    parts.append(f"Modalities: {mod_keys}")
                    for mk in mod_keys:
                        mod = adata.mod[mk]
                        layers_keys = list(mod.layers.keys()) if getattr(mod, 'layers', None) is not None else []
                        parts.append(f"  {mk}: {mod.shape[0]} cells x {mod.shape[1]} features, layers={layers_keys}")
                if hasattr(adata, 'shape'):
                    parts.append(f"Combined shape: {adata.shape[0]} obs x {adata.shape[1]} vars")

            if aspect in ("shape", "full") and not is_mudata:
                parts.append(f"Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
            if aspect in ("obs", "full"):
                cols = list(adata.obs.columns)
                parts.append(f"obs columns ({len(cols)}): {cols}")
                try:
                    parts.append(f"obs.head(3):\n{adata.obs.head(3).to_string()}")
                except Exception:
                    pass
            if aspect in ("var", "full") and not is_mudata:
                cols = list(adata.var.columns)
                parts.append(f"var columns ({len(cols)}): {cols}")
                try:
                    parts.append(f"var.head(3):\n{adata.var.head(3).to_string()}")
                except Exception:
                    pass
            if aspect in ("obsm", "full"):
                keys = list(adata.obsm.keys()) if hasattr(adata, 'obsm') else []
                parts.append(f"obsm keys: {keys}")
                for k in keys:
                    try:
                        parts.append(f"  {k}: shape {adata.obsm[k].shape}")
                    except Exception:
                        pass
            if aspect in ("uns", "full"):
                keys = list(adata.uns.keys()) if hasattr(adata, 'uns') else []
                parts.append(f"uns keys: {keys}")
            if aspect in ("layers", "full"):
                layers = getattr(adata, 'layers', None)
                keys = list(layers.keys()) if layers is not None else []
                parts.append(f"layers: {keys}")
            return "\n".join(parts) if parts else f"Unknown aspect: {aspect}"
        except Exception as e:
            return f"Error inspecting data: {e}"

    def _check_code_prerequisites(self, code: str, adata: Any) -> str:
        """Check if code references functions whose prerequisites are not satisfied.

        Returns a warning string (empty if all satisfied).
        """
        warnings = []
        # Known OmicVerse function patterns to check
        func_patterns = {
            'ov.pp.pca': 'pca',
            'ov.pp.scale': 'scale',
            'ov.pp.neighbors': 'neighbors',
            'ov.pp.umap': 'umap',
            'ov.pp.tsne': 'tsne',
            'ov.pp.leiden': 'leiden',
            'ov.pp.louvain': 'louvain',
            'ov.pp.sude': 'sude',
            'ov.pp.mde': 'mde',
            'ov.single.leiden': 'leiden',
            'ov.single.louvain': 'louvain',
        }
        for pattern, func_name in func_patterns.items():
            if pattern in code:
                try:
                    result = _global_registry.check_prerequisites(func_name, adata)
                    if not result['satisfied']:
                        missing = ', '.join(result['missing_structures'][:3])
                        rec = result['recommendation']
                        warnings.append(f"{func_name}: missing {missing}. {rec}")
                except Exception:
                    pass
        return "; ".join(warnings)

    def _tool_execute_code(self, code: str, description: str, adata: Any) -> dict:
        """Handle execute_code tool call. Returns {"adata": result, "output": stdout_str}."""
        # Apply proactive code transforms
        code = ProactiveCodeTransformer().transform(code)

        # Check prerequisites before execution
        prereq_warnings = self._check_code_prerequisites(code, adata)

        try:
            result = self._execute_generated_code(code, adata, capture_stdout=True)
            stdout = result.get("stdout", "")
            result_adata = result.get("adata", adata)

            # Build output summary
            output_parts = []
            if prereq_warnings:
                output_parts.append(f"PREREQUISITE WARNINGS: {prereq_warnings}")
            if stdout.strip():
                output_parts.append(f"stdout:\n{stdout[:3000]}")
            try:
                output_parts.append(f"Result adata shape: {result_adata.shape[0]} cells x {result_adata.shape[1]} features")
            except Exception:
                output_parts.append(f"Result type: {type(result_adata).__name__}")

            return {
                "adata": result_adata,
                "output": "\n".join(output_parts) if output_parts else "Code executed successfully (no output).",
            }
        except Exception as e:
            original_error = str(e)
            tb_str = traceback.format_exc()

            # Stage A: Pattern-based fix (fast, no LLM call)
            fixed_code = self._apply_execution_error_fix(code, original_error)
            if fixed_code:
                try:
                    result = self._execute_generated_code(fixed_code, adata, capture_stdout=True)
                    stdout = result.get("stdout", "")
                    result_adata = result.get("adata", adata)
                    output_parts = [f"RECOVERED (pattern fix): {original_error}"]
                    if stdout.strip():
                        output_parts.append(f"stdout:\n{stdout[:3000]}")
                    try:
                        output_parts.append(f"Result adata shape: {result_adata.shape[0]} cells x {result_adata.shape[1]} features")
                    except Exception:
                        output_parts.append(f"Result type: {type(result_adata).__name__}")
                    return {
                        "adata": result_adata,
                        "output": "\n".join(output_parts),
                    }
                except Exception:
                    pass  # Fall through to return original error

            error_output = f"ERROR: {e}\n\nTraceback (last 2000 chars):\n{tb_str[-2000:]}"
            if prereq_warnings:
                error_output = f"PREREQUISITE WARNINGS: {prereq_warnings}\n\n{error_output}"
            return {
                "adata": adata,  # Return original on error
                "output": error_output,
            }

    def _tool_run_snippet(self, code: str, adata: Any) -> str:
        """Handle run_snippet tool call. Runs on adata copy, returns stdout only."""
        try:
            # Shallow copy to avoid modifying original
            adata_copy = adata.copy() if hasattr(adata, 'copy') else adata
            result = self._execute_generated_code(code, adata_copy, capture_stdout=True)
            stdout = result.get("stdout", "")
            return stdout if stdout.strip() else "(no stdout output)"
        except Exception as e:
            return f"ERROR: {e}"

    def _tool_search_functions(self, query: str) -> str:
        """Handle search_functions tool call. Searches the OmicVerse function registry."""
        query_lower = query.lower()
        matches = []
        for entry in _global_registry._registry.values():
            # Match against name, description, aliases, category
            searchable = (
                entry.get('short_name', '').lower() + ' ' +
                entry.get('full_name', '').lower() + ' ' +
                entry.get('description', '').lower() + ' ' +
                entry.get('category', '').lower() + ' ' +
                ' '.join(entry.get('aliases', [])).lower()
            )
            if any(word in searchable for word in query_lower.split()):
                matches.append(entry)

        if not matches:
            return f"No functions found matching '{query}'. Try broader keywords."

        # Limit to top 10 unique functions
        results = []
        seen = set()
        for m in matches[:20]:
            fname = m.get('full_name', m.get('short_name', ''))
            if fname in seen:
                continue
            seen.add(fname)
            sig = m.get('signature', '')
            desc = m.get('description', '')[:300]

            entry_text = f"  {fname}({sig})\n    {desc}"

            # Add prerequisite info
            prereqs = m.get('prerequisites', {})
            req_funcs = prereqs.get('functions', [])
            if req_funcs:
                entry_text += "\n    Must run first: " + ", ".join(req_funcs)

            # Add requires/produces
            requires = m.get('requires', {})
            if requires:
                req_items = [f"{k}['{v}']" for k, vals in requires.items() for v in vals]
                entry_text += "\n    Requires: " + ", ".join(req_items)

            produces = m.get('produces', {})
            if produces:
                prod_items = [f"{k}['{v}']" for k, vals in produces.items() for v in vals]
                entry_text += "\n    Produces: " + ", ".join(prod_items)

            # Add one code example
            examples = m.get('examples', [])
            code_examples = [ex for ex in examples
                             if ex.strip().startswith(('ov.', 'sc.'))]
            if code_examples:
                entry_text += "\n    Example: " + code_examples[0]
            elif examples:
                entry_text += "\n    Example: " + examples[0]

            results.append(entry_text)
            if len(results) >= 10:
                break

        return f"Found {len(results)} matching functions:\n" + "\n".join(results)

    def _tool_search_skills(self, query: str) -> str:
        """Handle search_skills tool call. Searches domain-specific skill guidance."""
        if not hasattr(self, 'skill_registry') or not self.skill_registry:
            return "No domain skills available."
        if not self.skill_registry.skill_metadata:
            return "No domain skills loaded."

        query_lower = query.lower()
        scored = []
        for meta in self.skill_registry.skill_metadata.values():
            searchable = f"{meta.name} {meta.description} {meta.slug}".lower()
            score = sum(1 for word in query_lower.split() if word in searchable)
            if score > 0:
                scored.append((meta, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        if not scored:
            slugs = ", ".join(m.slug for m in self.skill_registry.skill_metadata.values())
            return f"No skills matched '{query}'. Available skills: {slugs}"

        # Load top 1-2 matched skills (lazy loading)
        results = []
        for meta, _ in scored[:2]:
            try:
                full_skill = self.skill_registry.load_full_skill(meta.slug)
                if full_skill:
                    # Use provider-aware formatting if available
                    provider = None
                    if hasattr(self, '_llm') and self._llm and hasattr(self._llm, 'config'):
                        provider = self._llm.config.provider
                    body = full_skill.prompt_instructions(max_chars=4000, provider=provider)
                    results.append(f"=== {full_skill.name} ===\n{body}")
            except Exception:
                pass

        if not results:
            return "Skills matched but content could not be loaded."

        return "\n\n".join(results)

    # ----- Web tools -----

    def _tool_web_fetch(self, url: str, prompt: str = None, timeout: int = 15) -> str:
        """Fetch a URL and return content as readable text."""
        import urllib.request
        import urllib.error
        from html.parser import HTMLParser

        # Simple HTML-to-text converter
        class _HTMLToText(HTMLParser):
            def __init__(self):
                super().__init__()
                self._pieces: list = []
                self._skip = False
                self._skip_tags = {"script", "style", "noscript", "svg", "head"}

            def handle_starttag(self, tag, attrs):
                if tag in self._skip_tags:
                    self._skip = True
                elif tag in ("br", "hr"):
                    self._pieces.append("\n")
                elif tag in ("p", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6"):
                    self._pieces.append("\n")

            def handle_endtag(self, tag):
                if tag in self._skip_tags:
                    self._skip = False
                elif tag in ("p", "div", "tr", "h1", "h2", "h3", "h4", "h5", "h6"):
                    self._pieces.append("\n")

            def handle_data(self, data):
                if not self._skip:
                    self._pieces.append(data)

            def get_text(self) -> str:
                raw = "".join(self._pieces)
                # Collapse whitespace
                lines = []
                for line in raw.splitlines():
                    stripped = " ".join(line.split())
                    if stripped:
                        lines.append(stripped)
                return "\n".join(lines)

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "OmicVerseAgent/1.0 (research bot)"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content_type = resp.headers.get("Content-Type", "")
                raw_bytes = resp.read(512_000)  # 512KB max
                charset = "utf-8"
                if "charset=" in content_type:
                    charset = content_type.split("charset=")[-1].split(";")[0].strip()
                body = raw_bytes.decode(charset, errors="replace")

            # Convert HTML to text
            if "html" in content_type.lower() or body.strip().startswith("<"):
                parser = _HTMLToText()
                parser.feed(body)
                text = parser.get_text()
            else:
                text = body

            # Truncate to fit context. Keep this tighter than before because the
            # fetched page is fed back into follow-up tool-calling turns and some
            # OpenAI-compatible proxy chains become unstable with larger tool
            # result payloads.
            max_chars = 4000 if prompt else 6000
            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n... [truncated, page too long]"

            header = f"Content from {url}:\n\n"
            if prompt:
                header = f"Content from {url} (focus: {prompt}):\n\n"
            return header + text

        except urllib.error.HTTPError as e:
            return f"HTTP error fetching {url}: {e.code} {e.reason}"
        except urllib.error.URLError as e:
            return f"URL error fetching {url}: {e.reason}"
        except Exception as e:
            return f"Error fetching {url}: {type(e).__name__}: {e}"

    def _tool_web_search(self, query: str, num_results: int = 5) -> str:
        """Search the web using DuckDuckGo and return results."""
        import urllib.request
        import urllib.parse
        import urllib.error
        from html.parser import HTMLParser

        num_results = max(1, min(int(num_results), 10))

        # DuckDuckGo HTML endpoint (no API key needed)
        encoded_q = urllib.parse.urlencode({"q": query})
        url = f"https://html.duckduckgo.com/html/?{encoded_q}"

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "OmicVerseAgent/1.0 (research bot)",
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = resp.read(256_000).decode("utf-8", errors="replace")
        except Exception as e:
            return f"Search error: {type(e).__name__}: {e}"

        # Parse DuckDuckGo HTML results
        # Results are in <a class="result__a" href="...">title</a>
        # Snippets in <a class="result__snippet" ...>text</a>
        import re

        results = []
        # Extract result blocks
        result_blocks = re.findall(
            r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>'
            r'.*?'
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            body,
            re.DOTALL,
        )

        if not result_blocks:
            # Fallback: try simpler pattern
            links = re.findall(
                r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
                body,
            )
            for href, title in links[:num_results]:
                clean_title = re.sub(r"<[^>]+>", "", title).strip()
                # DuckDuckGo wraps URLs in a redirect
                actual_url = href
                uddg_match = re.search(r'uddg=([^&]+)', href)
                if uddg_match:
                    actual_url = urllib.parse.unquote(uddg_match.group(1))
                results.append(f"- {clean_title}\n  {actual_url}")
        else:
            for href, title, snippet in result_blocks[:num_results]:
                clean_title = re.sub(r"<[^>]+>", "", title).strip()
                clean_snippet = re.sub(r"<[^>]+>", "", snippet).strip()
                actual_url = href
                uddg_match = re.search(r'uddg=([^&]+)', href)
                if uddg_match:
                    actual_url = urllib.parse.unquote(uddg_match.group(1))
                results.append(f"- {clean_title}\n  {actual_url}\n  {clean_snippet}")

        if not results:
            return f"No results found for '{query}'."

        return f"Search results for '{query}':\n\n" + "\n\n".join(results)

    def _tool_web_download(self, url: str, filename: str = None,
                           directory: str = None) -> str:
        """Download a file from a URL and save it to disk."""
        import urllib.request
        import urllib.error
        import urllib.parse

        MAX_SIZE = 2 * 1024 * 1024 * 1024  # 2GB

        # Determine filename
        if not filename:
            path_part = urllib.parse.urlparse(url).path
            filename = os.path.basename(path_part) or "downloaded_file"

        # Determine save directory
        save_dir = directory or os.getcwd()
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "OmicVerseAgent/1.0 (research bot)"},
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                # Check content-length if available
                content_length = resp.headers.get("Content-Length")
                if content_length and int(content_length) > MAX_SIZE:
                    size_gb = int(content_length) / (1024**3)
                    return f"File too large ({size_gb:.1f} GB). Max allowed is 2 GB."

                # Stream download with progress
                downloaded = 0
                chunk_size = 1024 * 1024  # 1MB chunks
                with open(save_path, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        downloaded += len(chunk)
                        if downloaded > MAX_SIZE:
                            f.close()
                            os.remove(save_path)
                            return f"Download aborted: exceeded 2GB limit at {downloaded / (1024**3):.1f} GB."
                        f.write(chunk)

            size_bytes = os.path.getsize(save_path)
            if size_bytes < 1024:
                size_str = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            elif size_bytes < 1024 * 1024 * 1024:
                size_str = f"{size_bytes / (1024**2):.1f} MB"
            else:
                size_str = f"{size_bytes / (1024**3):.2f} GB"

            return (
                f"Downloaded successfully:\n"
                f"  File: {save_path}\n"
                f"  Size: {size_str}\n"
                f"You can now load this file with execute_code, e.g.:\n"
                f"  adata = ov.read('{save_path}')"
            )

        except urllib.error.HTTPError as e:
            return f"HTTP error downloading {url}: {e.code} {e.reason}"
        except urllib.error.URLError as e:
            return f"URL error downloading {url}: {e.reason}"
        except Exception as e:
            # Clean up partial file
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                except OSError:
                    pass
            return f"Download error: {type(e).__name__}: {e}"

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
            raise RuntimeError("Interactive tool requires a connected session bridge.")
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
        payload = resolve_tool_search(
            query,
            loaded_tools=runtime_state.get_loaded_tools(self._get_runtime_session_id()),
            max_results=max_results,
        )
        selected = payload.get("selected_tools", [])
        loaded_tools = ()
        if selected:
            loaded_tools = runtime_state.load_tools(
                self._get_runtime_session_id(),
                selected,
            )
        payload["loaded_tools"] = list(runtime_state.get_loaded_tools(self._get_runtime_session_id()))
        payload["newly_loaded"] = list(loaded_tools)
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _tool_read(self, file_path: str, offset: int = 0, limit: int = 2000, pages: str = "") -> str:
        path = self._resolve_local_path(file_path)
        if not path.exists():
            return f"File not found: {path}"
        if path.is_dir():
            return f"Path is a directory, not a file: {path}"

        suffix = path.suffix.lower()
        if suffix == ".ipynb":
            try:
                import nbformat
            except ImportError:
                return "Notebook reading requires nbformat to be installed."
            nb = nbformat.read(path, as_version=4)
            cells = []
            for idx, cell in enumerate(nb.cells):
                if idx < offset:
                    continue
                if len(cells) >= max(1, limit):
                    break
                source = cell.source if isinstance(cell.source, str) else "".join(cell.source)
                preview = "\n".join(source.splitlines()[:20])
                cells.append(f"[{idx}] {cell.cell_type}\n{preview}")
            return "\n\n".join(cells) if cells else f"No notebook cells in range for {path}"

        if suffix == ".pdf":
            try:
                import pypdf
            except ImportError:
                return "PDF reading requires pypdf to be installed."
            reader = pypdf.PdfReader(str(path))
            page_numbers = []
            if pages:
                for part in pages.split(","):
                    part = part.strip()
                    if "-" in part:
                        start, end = part.split("-", 1)
                        page_numbers.extend(range(max(1, int(start)), min(len(reader.pages), int(end)) + 1))
                    elif part:
                        page_numbers.append(int(part))
            else:
                page_numbers = list(range(1, min(len(reader.pages), 5) + 1))
            snippets = []
            for page_no in page_numbers[:20]:
                text = reader.pages[page_no - 1].extract_text() or ""
                snippets.append(f"## Page {page_no}\n{text[:4000]}")
            return "\n\n".join(snippets) if snippets else f"No readable PDF text in {path}"

        mime = mimetypes.guess_type(path.name)[0] or ""
        if mime.startswith("image/"):
            stat = path.stat()
            return json.dumps(
                {
                    "type": "image",
                    "path": str(path),
                    "mime": mime,
                    "size_bytes": stat.st_size,
                },
                ensure_ascii=False,
                indent=2,
            )

        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        segment = lines[max(0, offset): max(0, offset) + max(1, limit)]
        numbered = [f"{idx + offset + 1:6d}\t{line[:2000]}" for idx, line in enumerate(segment)]
        return "\n".join(numbered) if numbered else f"(empty file) {path}"

    def _tool_edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        self._ensure_server_tool_mode("Edit")
        if self._tool_blocked_in_plan_mode("Edit"):
            return "Edit is blocked while the session is in plan mode."
        path = self._resolve_local_path(file_path)
        if not path.exists():
            return f"File not found: {path}"
        self._request_tool_approval(
            "Edit",
            reason=f"Edit file {path}",
            payload={"file_path": str(path)},
        )
        content = path.read_text(encoding="utf-8", errors="replace")
        occurrences = content.count(old_string)
        if occurrences == 0:
            return f"Edit failed: old_string was not found in {path}"
        updated = content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)
        path.write_text(updated, encoding="utf-8")
        return json.dumps(
            {
                "file_path": str(path),
                "replacements": occurrences if replace_all else 1,
                "status": "updated",
            },
            ensure_ascii=False,
            indent=2,
        )

    def _tool_write(self, file_path: str, content: str) -> str:
        self._ensure_server_tool_mode("Write")
        if self._tool_blocked_in_plan_mode("Write"):
            return "Write is blocked while the session is in plan mode."
        path = self._resolve_local_path(file_path)
        self._request_tool_approval(
            "Write",
            reason=f"Write file {path}",
            payload={"file_path": str(path)},
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return json.dumps(
            {"file_path": str(path), "bytes_written": len(content.encode('utf-8'))},
            ensure_ascii=False,
            indent=2,
        )

    def _tool_glob(self, pattern: str, root: str = "", max_results: int = 200) -> str:
        base = self._resolve_local_path(root, allow_relative=True) if root else Path(self._refresh_runtime_working_directory())
        matches = sorted(str(path) for path in base.glob(pattern))[: max(1, max_results)]
        return json.dumps({"root": str(base), "pattern": pattern, "matches": matches}, ensure_ascii=False, indent=2)

    def _tool_grep(self, pattern: str, root: str = "", glob: str = "", max_results: int = 200) -> str:
        base = self._resolve_local_path(root, allow_relative=True) if root else Path(self._refresh_runtime_working_directory())
        rg_path = shutil.which("rg")
        if rg_path:
            cmd = [rg_path, "-n", pattern, str(base)]
            if glob:
                cmd[1:1] = ["-g", glob]
        else:
            cmd = ["grep", "-R", "-n", pattern, str(base)]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode not in (0, 1):
            return proc.stderr or proc.stdout or f"Grep failed with exit code {proc.returncode}"
        lines = [line for line in proc.stdout.splitlines() if line.strip()][: max(1, max_results)]
        return "\n".join(lines) if lines else f"No matches for {pattern}"

    def _tool_notebook_edit(self, file_path: str, cell_index: int, source: str, cell_type: str = "") -> str:
        self._ensure_server_tool_mode("NotebookEdit")
        if self._tool_blocked_in_plan_mode("NotebookEdit"):
            return "NotebookEdit is blocked while the session is in plan mode."
        path = self._resolve_local_path(file_path)
        self._request_tool_approval(
            "NotebookEdit",
            reason=f"Edit notebook {path}",
            payload={"file_path": str(path), "cell_index": cell_index},
        )
        try:
            import nbformat
        except ImportError:
            return "NotebookEdit requires nbformat to be installed."
        nb = nbformat.read(path, as_version=4)
        if cell_index < 0 or cell_index >= len(nb.cells):
            return f"Notebook cell index out of range: {cell_index}"
        nb.cells[cell_index].source = source
        if cell_type:
            nb.cells[cell_index].cell_type = cell_type
        nbformat.write(nb, path)
        return json.dumps(
            {"file_path": str(path), "cell_index": cell_index, "status": "updated"},
            ensure_ascii=False,
            indent=2,
        )

    def _tool_create_task(self, title: str, description: str = "", status: str = "pending") -> str:
        task = runtime_state.create_task(
            self._get_runtime_session_id(),
            title=title,
            description=description,
            status=status if status else "pending",
        )
        return json.dumps(task.to_dict(), ensure_ascii=False, indent=2)

    def _tool_get_task(self, task_id: str) -> str:
        task = runtime_state.get_task(self._get_runtime_session_id(), task_id)
        payload = task.to_dict() if task is not None else {"error": "Task not found"}
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _tool_list_tasks(self, status: str = "") -> str:
        tasks = runtime_state.list_tasks(self._get_runtime_session_id(), status=status)
        return json.dumps({"tasks": tasks}, ensure_ascii=False, indent=2)

    def _tool_task_output(self, task_id: str, offset: int = 0, limit: int = 200) -> str:
        payload = runtime_state.read_task_output(self._get_runtime_session_id(), task_id, offset=offset, limit=limit)
        return json.dumps(payload or {"error": "Task not found"}, ensure_ascii=False, indent=2)

    def _tool_task_stop(self, task_id: str) -> str:
        self._ensure_server_tool_mode("TaskStop")
        updated = runtime_state.stop_task(self._get_runtime_session_id(), task_id)
        payload = updated.to_dict() if updated is not None else {"error": "Task not found"}
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _tool_task_update(self, task_id: str, status: str, summary: str = "") -> str:
        updated = runtime_state.update_task(
            self._get_runtime_session_id(),
            task_id,
            status=status,
            summary=summary,
        )
        payload = updated.to_dict() if updated is not None else {"error": "Task not found"}
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _tool_enter_plan_mode(self, reason: str = "") -> str:
        payload = runtime_state.enter_plan_mode(self._get_runtime_session_id(), reason=reason)
        return json.dumps(payload.to_dict(), ensure_ascii=False, indent=2)

    def _tool_exit_plan_mode(self, summary: str = "") -> str:
        payload = runtime_state.exit_plan_mode(self._get_runtime_session_id(), reason=summary)
        return json.dumps(payload.to_dict(), ensure_ascii=False, indent=2)

    def _detect_repo_root(self, cwd: Optional[Path] = None) -> Optional[Path]:
        current = (cwd or Path(self._refresh_runtime_working_directory())).resolve()
        for candidate in (current, *current.parents):
            if (candidate / ".git").exists():
                return candidate
        return None

    def _tool_enter_worktree(self, branch_name: str = "", path: str = "", base_ref: str = "HEAD") -> str:
        self._ensure_server_tool_mode("EnterWorktree")
        if self._tool_blocked_in_plan_mode("EnterWorktree"):
            return "EnterWorktree is blocked while the session is in plan mode."
        repo_root = self._detect_repo_root()
        if repo_root is None:
            return json.dumps({"error": "No git repository found for worktree creation."}, ensure_ascii=False, indent=2)
        branch = branch_name.strip() or f"ovagent/{self._get_runtime_session_id()}"
        if path:
            worktree_path = self._resolve_local_path(path, allow_relative=True)
        else:
            worktree_root = Path.home() / ".ovagent" / "worktrees"
            worktree_root.mkdir(parents=True, exist_ok=True)
            worktree_path = worktree_root / branch.replace("/", "_")
        self._request_tool_approval(
            "EnterWorktree",
            reason=f"Create or switch git worktree {worktree_path}",
            payload={"branch_name": branch, "path": str(worktree_path)},
        )
        if not worktree_path.exists():
            proc = subprocess.run(
                ["git", "-C", str(repo_root), "worktree", "add", str(worktree_path), "-b", branch, base_ref],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                # Branch may already exist; try attaching existing branch.
                proc = subprocess.run(
                    ["git", "-C", str(repo_root), "worktree", "add", str(worktree_path), branch],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            if proc.returncode != 0:
                return json.dumps(
                    {"error": proc.stderr.strip() or proc.stdout.strip() or "git worktree add failed"},
                    ensure_ascii=False,
                    indent=2,
                )
        worktree = runtime_state.set_worktree(
            self._get_runtime_session_id(),
            path=str(worktree_path),
            repo_root=str(repo_root),
            branch=branch,
            base_branch=base_ref,
        )
        return json.dumps(worktree.to_dict(), ensure_ascii=False, indent=2)

    def _tool_skill(self, query: str, mode: str = "search") -> str:
        if mode == "load":
            return self._load_skill_guidance(query)
        exact = None
        if self.skill_registry and self.skill_registry.skill_metadata:
            for meta in self.skill_registry.skill_metadata.values():
                if query.strip().lower() in {meta.slug.lower(), meta.name.lower()}:
                    exact = meta.slug
                    break
        if exact:
            return self._load_skill_guidance(exact)
        return self._tool_search_skills(query)

    def _tool_list_mcp_resources(self, server: str = "") -> str:
        manifest_path = os.environ.get("OV_AGENT_MCP_MANIFEST", "").strip()
        if not manifest_path:
            return json.dumps(
                {"available": False, "reason": "OV_AGENT_MCP_MANIFEST is not configured.", "resources": []},
                ensure_ascii=False,
                indent=2,
            )
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        resources = manifest.get("resources", [])
        if server:
            resources = [item for item in resources if item.get("server") == server]
        return json.dumps({"available": True, "resources": resources}, ensure_ascii=False, indent=2)

    def _tool_read_mcp_resource(self, server: str, uri: str) -> str:
        manifest_path = os.environ.get("OV_AGENT_MCP_MANIFEST", "").strip()
        if not manifest_path:
            return json.dumps(
                {"available": False, "reason": "OV_AGENT_MCP_MANIFEST is not configured."},
                ensure_ascii=False,
                indent=2,
            )
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        for item in manifest.get("resources", []):
            if item.get("server") == server and item.get("uri") == uri:
                target = item.get("path", "")
                if not target:
                    return json.dumps(item, ensure_ascii=False, indent=2)
                return self._tool_read(target)
        return json.dumps({"error": "MCP resource not found"}, ensure_ascii=False, indent=2)

    def _tool_ask_user_question(self, question: str, header: str = "", options: Optional[list[str]] = None) -> str:
        record = runtime_state.create_question(
            self._get_runtime_session_id(),
            turn_id=getattr(getattr(self, "_last_run_trace", None), "turn_id", ""),
            trace_id=getattr(getattr(self, "_last_run_trace", None), "trace_id", ""),
            question=question,
            header=header,
            options=list(options or []),
        )
        answer = self._request_interaction(
            {
                "kind": "question",
                "question_id": record.question_id,
                "question": question,
                "header": header,
                "options": list(options or []),
                "session_id": self._get_runtime_session_id(),
                "trace_id": record.trace_id,
                "turn_id": record.turn_id,
            }
        )
        if isinstance(answer, dict):
            resolved = runtime_state.resolve_question(
                self._get_runtime_session_id(),
                record.question_id,
                str(answer.get("answer", "")),
            )
        else:
            resolved = runtime_state.resolve_question(
                self._get_runtime_session_id(),
                record.question_id,
                str(answer or ""),
            )
        payload = resolved.to_dict() if resolved is not None else record.to_dict()
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _background_bash_worker(
        self,
        task_id: str,
        *,
        command: str,
        cwd: str,
        timeout_ms: int,
    ) -> None:
        session_id = self._get_runtime_session_id()
        proc = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        runtime_state.bind_process(session_id, task_id, proc)
        assert proc.stdout is not None
        start = time.time()
        for line in proc.stdout:
            runtime_state.append_task_output(session_id, task_id, line.rstrip("\n"))
            if time.time() - start > timeout_ms / 1000:
                proc.terminate()
                runtime_state.update_task(session_id, task_id, status="failed", summary="Background command timed out")
                return
        return_code = proc.wait()
        status = "completed" if return_code == 0 else "failed"
        runtime_state.update_task(session_id, task_id, status=status, summary=f"Exit code {return_code}")

    def _tool_bash(
        self,
        command: str,
        description: str = "",
        timeout: int = 120000,
        run_in_background: bool = False,
        dangerouslyDisableSandbox: bool = False,
    ) -> str:
        self._ensure_server_tool_mode("Bash")
        if self._tool_blocked_in_plan_mode("Bash"):
            return "Bash is blocked while the session is in plan mode."
        reason = description or f"Run shell command: {command[:120]}"
        self._request_tool_approval(
            "Bash",
            reason=reason,
            payload={
                "command": command,
                "dangerouslyDisableSandbox": dangerouslyDisableSandbox,
            },
        )
        cwd = self._refresh_runtime_working_directory()
        if run_in_background:
            task = runtime_state.create_task(
                self._get_runtime_session_id(),
                title=description or command[:80],
                description=command,
                kind="background_command",
                status="in_progress",
                background=True,
                tool_name="Bash",
                metadata={"cwd": cwd, "command": command},
            )
            proc = subprocess.Popen(
                ["bash", "-lc", command],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            runtime_state.attach_background_process(
                self._get_runtime_session_id(),
                task.task_id,
                proc,
            )
            return json.dumps(
                {
                    "task_id": task.task_id,
                    "status": "in_progress",
                    "cwd": cwd,
                    "command": command,
                },
                ensure_ascii=False,
                indent=2,
            )

        proc = subprocess.run(
            ["bash", "-lc", command],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout)) / 1000,
            check=False,
        )
        output = (proc.stdout or "") + ((proc.stderr or "") if proc.stderr else "")
        return json.dumps(
            {
                "cwd": cwd,
                "command": command,
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "output": output.strip(),
            },
            ensure_ascii=False,
            indent=2,
        )

    # ----- Subagent prompt builders -----

    _CODE_QUALITY_RULES = (
        "MANDATORY CODE QUALITY RULES:\n"
        "- NEVER use f-strings in print() - use string concatenation: "
        "print('Result: ' + str(value))\n"
        "- NEVER assign result of in-place OmicVerse functions. Call them directly:\n"
        "  ov.pp.pca(adata, n_pcs=50)   # CORRECT\n"
        "  adata = ov.pp.pca(adata)      # WRONG - returns None!\n"
        "  In-place functions: pca, scale, neighbors, leiden, umap, tsne, "
        "sude, scrublet, mde, louvain, phate\n"
        "- NEVER use .cat.categories - use .value_counts() instead\n"
        "- ALWAYS wrap HVG selection in try/except with fallback:\n"
        "  try:\n"
        "      sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=2000)\n"
        "  except ValueError:\n"
        "      sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=2000)\n"
        "- ALWAYS validate batch column before batch operations "
        "(check existence, fillna, astype('category'))\n"
    )

    def _build_explore_prompt(self, context: str) -> str:
        """System prompt for the explore subagent (read-only data characterisation)."""
        prompt = (
            "You are a bioinformatics data inspector for OmicVerse. "
            "Your job is to thoroughly characterize the dataset and report findings.\n\n"
            "You CANNOT modify the data. Use inspect_data and run_snippet to explore.\n"
            "Use search_functions to understand what OmicVerse functions are available.\n\n"
            "Report:\n"
            "1. Dataset dimensions (cells x genes)\n"
            "2. Available metadata columns (obs, var) with example values\n"
            "3. Existing embeddings (obsm) and layers\n"
            "4. Data quality indicators (sparsity, mito%, batch columns, cell type annotations)\n"
            "5. Potential issues or recommendations\n\n"
            "Available packages: ov (omicverse), sc (scanpy), np (numpy), pd (pandas)\n\n"
            "Call finish() with a comprehensive summary when done.\n"
        )
        if context:
            prompt += f"\nAdditional context from parent: {context}"
        return prompt

    def _build_plan_prompt(self, context: str) -> str:
        """System prompt for the plan subagent (workflow design, read-only)."""
        prompt = (
            "You are a bioinformatics workflow architect for OmicVerse. "
            "Design a step-by-step execution plan. Do NOT execute code — only plan.\n\n"
            "Use search_functions to find OmicVerse functions and their prerequisites.\n"
            "Use search_skills for domain-specific workflow guidance.\n"
            "Use inspect_data and run_snippet to understand current data state.\n\n"
            "Your plan should include:\n"
            "1. Ordered steps with specific OmicVerse function calls and parameters\n"
            "2. Quality check points between steps\n"
            "3. Fallback strategies for known failure modes\n"
            "4. Expected outputs at each step (obsm, obs columns, layers)\n\n"
        )
        prompt += self._CODE_QUALITY_RULES
        prompt += "\nCall finish() with the complete plan when done.\n"
        if context:
            prompt += f"\nAdditional context from parent: {context}"
        # Append skill catalog
        if hasattr(self, 'skill_registry') and self.skill_registry and self.skill_registry.skill_metadata:
            prompt += "\nAvailable domain skills (use search_skills for details):\n"
            for meta in sorted(self.skill_registry.skill_metadata.values(), key=lambda s: s.slug):
                prompt += f"  - {meta.slug}: {meta.description[:80]}\n"
        return prompt

    def _build_execute_prompt(self, context: str) -> str:
        """System prompt for the execute subagent (focused code execution)."""
        prompt = (
            "You are a focused bioinformatics code executor for OmicVerse. "
            "Complete the specific sub-task described. Execute code step by step.\n\n"
            "Guidelines:\n"
            "- Inspect data before writing code that depends on column names\n"
            "- Execute in logical steps, not one giant block\n"
            "- If code fails, diagnose and try a different approach\n"
            "- Use run_snippet() for exploration, execute_code() for processing\n"
            "- Call finish() when done with a summary of what was accomplished\n\n"
        )
        prompt += self._CODE_QUALITY_RULES
        prompt += (
            "\nAvailable packages: ov (omicverse), sc (scanpy), np (numpy), pd (pandas), "
            "matplotlib, seaborn, scipy, sklearn\n"
            "All OmicVerse functions are called via ov.* (e.g. ov.pp.preprocess, ov.pp.scale)\n"
        )
        if context:
            prompt += f"\nContext from parent: {context}"
        return prompt

    def _build_subagent_system_prompt(self, agent_type: str, context: str = "") -> str:
        """Route to the correct subagent prompt builder."""
        if agent_type == "explore":
            return self._build_explore_prompt(context)
        elif agent_type == "plan":
            return self._build_plan_prompt(context)
        elif agent_type == "execute":
            return self._build_execute_prompt(context)
        raise ValueError(f"Unknown subagent type: {agent_type}")

    def _build_subagent_user_message(self, task: str, adata: Any) -> str:
        """Build the initial user message for a subagent."""
        msg = f"Task: {task}\n\n"
        if adata is not None and hasattr(adata, 'shape'):
            msg += f"Dataset: {adata.shape[0]} cells x {adata.shape[1]} genes\n"
        return msg

    async def _run_subagent(
        self,
        agent_type: str,
        task: str,
        adata: Any,
        context: str = "",
    ) -> dict:
        """Spawn a subagent with restricted tools and its own conversation.

        The subagent gets its own system prompt and message history (isolated
        context window). Only the final result is returned to the parent.

        Returns
        -------
        dict
            ``{"result": str, "adata": AnnData}``
        """
        from .agent_config import SUBAGENT_CONFIGS

        config = SUBAGENT_CONFIGS[agent_type]

        # Filter tools to only those allowed for this subagent type
        subagent_tools = self._get_visible_agent_tools(allowed_names=set(config.allowed_tools))

        # Build independent message history
        messages = [
            {"role": "system", "content": self._build_subagent_system_prompt(agent_type, context)},
            {"role": "user", "content": self._build_subagent_user_message(task, adata)},
        ]

        working_adata = adata

        for turn in range(config.max_turns):
            print(f"      🔄 [{agent_type}] Turn {turn + 1}/{config.max_turns}")

            response = await self._llm.chat(
                messages, tools=subagent_tools, tool_choice="auto"
            )

            # Track usage
            if response.usage:
                self.last_usage = response.usage

            # Append assistant response
            if response.raw_message:
                messages.append(response.raw_message)
            elif response.content:
                messages.append({"role": "assistant", "content": response.content})

            # Text-only response → done
            if not response.tool_calls:
                return {
                    "result": response.content or "",
                    "adata": working_adata,
                }

            # Process tool calls
            for tc in response.tool_calls:
                print(f"      🔧 [{agent_type}] {tc.name}({', '.join(f'{k}=' for k in tc.arguments)})")

                result = await self._dispatch_tool(tc, working_adata, task)

                if tc.name == "execute_code" and isinstance(result, dict) and "adata" in result:
                    working_adata = result["adata"]
                    tool_output = result.get("output", "Code executed.")
                elif tc.name == "finish":
                    summary = tc.arguments.get("summary", "")
                    print(f"      ✅ [{agent_type}] Finished: {summary[:120]}")
                    return {"result": summary, "adata": working_adata}
                elif isinstance(result, str):
                    tool_output = result
                else:
                    tool_output = str(result)

                # Tighter truncation for subagents
                if len(tool_output) > 6000:
                    tool_output = tool_output[:5500] + "\n... (truncated)"

                tool_msg = self._llm.format_tool_result_message(
                    tc.id, tc.name, tool_output
                )
                messages.append(tool_msg)

        return {
            "result": f"Subagent ({agent_type}) reached max turns ({config.max_turns})",
            "adata": working_adata,
        }

    def _build_agentic_system_prompt(self) -> str:
        """Build the system prompt for agentic loop mode."""
        prompt = (
            "You are OmicVerse Agent, an expert bioinformatics assistant that processes "
            "single-cell, bulk RNA-seq, and spatial transcriptomics data.\n\n"
            "You have OmicVerse data tools plus a Claude-style coding tool catalog.\n"
            "Core Claude tools are ToolSearch, Bash, and Read. Deferred Claude tools "
            "must be loaded through ToolSearch before use. High-risk tools require approval.\n\n"
            "Use the legacy OmicVerse tools for dataset inspection and execution, and "
            "use Claude-style tools for files, tasks, planning, worktrees, and questions.\n\n"
            "Do not narrate future actions without taking them in the same turn. "
            "If a tool is needed, call it now instead of saying that you will call it later.\n\n"
            "CODING / FILESYSTEM:\n"
            "- Use ToolSearch when you need deferred tools such as Edit, Write, Glob, Grep, NotebookEdit, Task*, or EnterWorktree\n"
            "- Use Read for local files instead of shelling out to cat/head/sed when possible\n"
            "- Use AskUserQuestion when a turn genuinely needs user clarification\n\n"
            "WEB ACCESS:\n"
            "- Use WebFetch(url) or web_fetch(url) to read any web page (GEO datasets, papers, docs)\n"
            "- Use WebSearch(query) or web_search(query) to search the internet for information\n"
            "- Use web_download(url) to download files (datasets, h5ad, csv, gz) to disk\n"
            "- When the user provides a URL, ALWAYS fetch it first instead of asking them to paste content\n"
            "- When you need background info (gene functions, diseases, methods), search the web\n"
            "- To download GEO data, fetch the GEO page first to find download links, then use web_download\n\n"
            "WORKFLOW:\n"
            "1. If the user provides a URL, use web_fetch to read it\n"
            "2. Use inspect_data to understand the dataset structure\n"
            "3. Use search_functions to find relevant OmicVerse functions "
            "(includes prerequisite chains and examples)\n"
            "4. For complex multi-step workflows, use search_skills for domain guidance\n"
            "5. Execute code step by step, checking results between steps\n"
            "6. Call finish() when the task is complete\n\n"
            "MANDATORY CODE QUALITY RULES:\n"
            "- NEVER use f-strings in print() - use string concatenation: "
            "print('Result: ' + str(value))\n"
            "- NEVER assign result of in-place OmicVerse functions. Call them directly:\n"
            "  ov.pp.pca(adata, n_pcs=50)   # CORRECT\n"
            "  adata = ov.pp.pca(adata)      # WRONG - returns None!\n"
            "  In-place functions: pca, scale, neighbors, leiden, umap, tsne, "
            "sude, scrublet, mde, louvain, phate\n"
            "- NEVER use .cat.categories - use .value_counts() instead\n"
            "- ALWAYS wrap HVG selection in try/except with fallback:\n"
            "  try:\n"
            "      sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=2000)\n"
            "  except ValueError:\n"
            "      sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=2000)\n"
            "- ALWAYS validate batch column before batch operations "
            "(check existence, fillna, astype('category'))\n\n"
            "Guidelines:\n"
            "- ALWAYS inspect data before writing code that depends on column names or structure\n"
            "- Execute code in logical steps, not one giant block\n"
            "- If code fails, read the error carefully, diagnose the issue, and try a different approach\n"
            "- Use run_snippet() for exploration (won't modify data)\n"
            "- Use execute_code() for actual processing (modifies data)\n"
            "- Available packages: ov (omicverse), sc (scanpy), np (numpy), pd (pandas), "
            "matplotlib, seaborn, scipy, sklearn\n"
            "- All OmicVerse functions are called via ov.* (e.g. ov.pp.preprocess, ov.pp.scale)\n"
            "- When saving files, use the output directory or current working directory\n\n"
            "DELEGATION STRATEGY (use the delegate tool for complex tasks):\n"
            "- Agent(subagent_type='explore', ...) or delegate('explore', ...) — read-only data characterization\n"
            "- Agent(subagent_type='plan', ...) or delegate('plan', ...) — design workflow before execution\n"
            "- Agent(subagent_type='execute', ...) or delegate('execute', ...) — focused execution step\n"
            "- For simple single-step operations, execute directly without delegation\n"
            "- Subagents have their own context window (prevents context overflow)\n"
        )

        # Add brief skill catalog if available
        if hasattr(self, 'skill_registry') and self.skill_registry and self.skill_registry.skill_metadata:
            prompt += "\nAvailable domain skills (use search_skills for detailed guidance):\n"
            for meta in sorted(self.skill_registry.skill_metadata.values(), key=lambda s: s.slug):
                prompt += f"  - {meta.slug}: {meta.description[:80]}\n"

        return prompt

    def _build_initial_user_message(self, request: str, adata: Any) -> str:
        """Build the initial user message for the agentic loop."""
        msg = f"Task: {request}\n\n"
        if adata is not None:
            dtype = type(adata).__name__  # "AnnData" or "MuData"
            if hasattr(adata, 'shape'):
                msg += f"Dataset ({dtype}): {adata.shape[0]} cells x {adata.shape[1]} features\n"
            if dtype == "MuData" and hasattr(adata, 'mod'):
                msg += f"Modalities: {list(adata.mod.keys())}\n"
            msg += "Use inspect_data to learn more about the dataset structure before writing code."
        else:
            msg += (
                "No dataset is loaded yet. You can still help with:\n"
                "- Use web_fetch/web_search to look up information\n"
                "- Use web_download to download datasets from URLs\n"
                "- Answer bioinformatics questions\n"
                "- Plan analysis workflows\n"
            )
        return msg

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
        """Best-effort classification for action-oriented requests.

        Codex CLI keeps `tool_choice=auto`, but its turn loop is item-driven and
        prompt-trained to continue acting. Our backend is less structured, so we
        add a small compatibility shim for clearly action-oriented requests.
        """
        text = (request or "").strip()
        if not text:
            return False
        if self._URL_PATTERN.search(text):
            return True
        if adata is not None:
            return True
        lowered = text.lower()
        if any(marker in lowered for marker in ("数据", "dataset", "下载", "分析", "处理", "读取", "打开", "搜索")):
            return True
        return bool(self._ACTION_REQUEST_PATTERN.search(text))

    def _response_is_promissory(self, content: str) -> bool:
        text = (content or "").strip()
        if not text:
            return False
        if self._PROMISSORY_PATTERN.search(text):
            return True
        lowered = text.lower()
        chinese_markers = (
            "我先", "让我", "我会", "我将", "先获取", "先下载", "先读取",
            "先去", "现在开始", "重新开始", "可以继续吗", "继续吗",
        )
        return any(marker in text for marker in chinese_markers) or lowered.startswith("okay, i")

    def _select_agent_tool_choice(
        self,
        *,
        request: str,
        adata: Any,
        turn_index: int,
        had_meaningful_tool_call: bool,
        forced_retry: bool,
    ) -> str:
        if forced_retry:
            return "required"
        if turn_index == 0 and not had_meaningful_tool_call and self._request_requires_tool_action(request, adata):
            return "required"
        return "auto"

    def _should_continue_after_text_response(
        self,
        *,
        request: str,
        response_content: str,
        adata: Any,
        had_meaningful_tool_call: bool,
    ) -> bool:
        text = (response_content or "").strip()
        if not text:
            return False
        if had_meaningful_tool_call:
            return False
        if self._BLOCKER_PATTERN.search(text):
            return False
        if self._response_is_promissory(text):
            return True
        if self._request_requires_tool_action(request, adata) and not self._RESULT_PATTERN.search(text):
            # If the request clearly needs action and the model only produced a
            # vague narrative without evidence of real work, force a follow-up.
            return True
        return False

    def _build_no_tool_follow_up_message(
        self, request: str, *, retry_count: int = 0, max_retries: int = 2,
    ) -> str:
        if retry_count >= max_retries - 1:
            # Final retry — forceful
            base = (
                "IMPORTANT: You MUST call a tool in this response. "
                "Do NOT respond with text only. Use one of your available tools now. "
                "If you cannot proceed, call the 'finish' tool with a summary of what went wrong."
            )
        else:
            base = (
                "Do not describe future actions without taking them. "
                "Either call the appropriate tool now or provide the final answer only if the task is already complete."
            )
        if self._URL_PATTERN.search(request or ""):
            return (
                base
                + " The user provided a URL, so fetch it in this turn with `WebFetch`/`web_fetch` before continuing."
            )
        return base

    def _persist_harness_history(self, request: str) -> None:
        """Persist the latest trace into session history when enabled."""
        if self._session_history is None or self._last_run_trace is None:
            return

        trace = self._last_run_trace
        def _step_type(step):
            return step.get("step_type") if isinstance(step, dict) else step.step_type

        def _step_name(step):
            return step.get("name") if isinstance(step, dict) else step.name

        def _step_data(step):
            return step.get("data", {}) if isinstance(step, dict) else step.data

        generated_code = "\n\n".join(
            _step_data(step).get("code", "")
            for step in trace.steps
            if _step_type(step) == "code" and _step_data(step).get("code")
        )
        tool_names = [
            _step_name(step)
            for step in trace.steps
            if _step_type(step) == "tool_call" and _step_name(step)
        ]
        artifact_refs = [
            artifact.to_dict() if hasattr(artifact, "to_dict") else dict(artifact)
            for artifact in trace.artifacts
        ]
        self._session_history.append(
            HistoryEntry(
                session_id=trace.session_id or self._get_harness_session_id(),
                timestamp=trace.finished_at or trace.started_at,
                request=request,
                trace_id=trace.trace_id,
                generated_code=generated_code,
                result_summary=trace.result_summary,
                tool_names=tool_names,
                artifact_refs=artifact_refs,
                usage=trace.usage or coerce_usage_payload(self.last_usage),
                usage_breakdown=self.last_usage_breakdown,
                success=trace.status == "success",
            )
        )

    def _append_tool_results(self, messages: list, tool_results: list) -> None:
        """Append tool results to the conversation in the correct format.

        For Anthropic Messages API, all tool results from a single assistant
        response must be in ONE ``user`` message with multiple ``tool_result``
        content blocks.  For OpenAI, each result is a separate ``tool`` message.

        Parameters
        ----------
        messages : list
            Conversation message list (mutated in-place).
        tool_results : list of (tool_call_id, tool_name, tool_output)
        """
        if not tool_results:
            return

        provider_info = None
        if self._llm:
            from .model_config import get_provider
            provider_info = get_provider(self._llm.config.provider)

        # Check if the upstream model is Anthropic (native OR proxy to Claude).
        # Anthropic requires all tool results batched in one user message.
        model_name = self._llm.config.model if self._llm else ""
        is_anthropic_wire = (
            (provider_info is not None and provider_info.wire_api.value == "anthropic")
            or "claude" in model_name.lower()
        )

        if is_anthropic_wire:
            # Batch all tool results into a single user message
            content_blocks = []
            for tc_id, tc_name, tc_output in tool_results:
                content_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tc_id,
                    "content": tc_output,
                })
            messages.append({"role": "user", "content": content_blocks})
        else:
            # OpenAI-compatible: each tool result is a separate message
            for tc_id, tc_name, tc_output in tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": tc_name,
                    "content": tc_output,
                })

    async def _dispatch_tool(self, tool_call, current_adata: Any, request: str):
        """Dispatch a tool call and return the result."""
        name = normalize_tool_name(tool_call.name)
        args = tool_call.arguments

        if self._tool_blocked_in_plan_mode(name):
            return f"{name} is blocked because the session is currently in plan mode."

        if name == "ToolSearch":
            return self._tool_tool_search(
                args.get("query", ""),
                max_results=args.get("max_results", 5),
            )
        elif name == "Bash":
            return self._tool_bash(
                args.get("command", ""),
                description=args.get("description", ""),
                timeout=args.get("timeout", 120000),
                run_in_background=bool(args.get("run_in_background", False)),
                dangerouslyDisableSandbox=bool(args.get("dangerouslyDisableSandbox", False)),
            )
        elif name == "Read":
            return self._tool_read(
                args.get("file_path", ""),
                offset=args.get("offset", 0),
                limit=args.get("limit", 2000),
                pages=args.get("pages", ""),
            )
        elif name == "Edit":
            return self._tool_edit(
                args.get("file_path", ""),
                args.get("old_string", ""),
                args.get("new_string", ""),
                replace_all=bool(args.get("replace_all", False)),
            )
        elif name == "Write":
            return self._tool_write(
                args.get("file_path", ""),
                args.get("content", ""),
            )
        elif name == "Glob":
            return self._tool_glob(
                args.get("pattern", ""),
                root=args.get("root", ""),
                max_results=args.get("max_results", 200),
            )
        elif name == "Grep":
            return self._tool_grep(
                args.get("pattern", ""),
                root=args.get("root", ""),
                glob=args.get("glob", ""),
                max_results=args.get("max_results", 200),
            )
        elif name == "NotebookEdit":
            return self._tool_notebook_edit(
                args.get("file_path", ""),
                cell_index=args.get("cell_index", 0),
                source=args.get("source", ""),
                cell_type=args.get("cell_type", ""),
            )
        elif name == "inspect_data":
            return self._tool_inspect_data(current_adata, args.get("aspect", "full"))
        elif name == "execute_code":
            return self._tool_execute_code(
                args.get("code", ""),
                args.get("description", ""),
                current_adata,
            )
        elif name == "run_snippet":
            return self._tool_run_snippet(args.get("code", ""), current_adata)
        elif name == "search_functions":
            return self._tool_search_functions(args.get("query", ""))
        elif name == "search_skills":
            return self._tool_search_skills(args.get("query", ""))
        elif name == "Agent":
            agent_type = args.get("subagent_type", args.get("agent_type", "explore"))
            task = args.get("task", "")
            context = args.get("context", "")
            print(f"   -> Delegating to {agent_type} subagent: {task[:80]}...")
            sub_result = await self._run_subagent(
                agent_type=agent_type,
                task=task,
                adata=current_adata,
                context=context,
            )
            # execute subagents return modified adata
            if agent_type == "execute":
                return {"adata": sub_result["adata"], "output": sub_result["result"]}
            return sub_result["result"]
        elif name == "AskUserQuestion":
            return self._tool_ask_user_question(
                args.get("question", ""),
                header=args.get("header", ""),
                options=args.get("options", []),
            )
        elif name == "TaskCreate":
            return self._tool_create_task(
                args.get("title", ""),
                description=args.get("description", ""),
                status=args.get("status", "pending"),
            )
        elif name == "TaskGet":
            return self._tool_get_task(args.get("task_id", ""))
        elif name == "TaskList":
            return self._tool_list_tasks(status=args.get("status", ""))
        elif name == "TaskOutput":
            return self._tool_task_output(
                args.get("task_id", ""),
                offset=args.get("offset", 0),
                limit=args.get("limit", 200),
            )
        elif name == "TaskStop":
            return self._tool_task_stop(args.get("task_id", ""))
        elif name == "TaskUpdate":
            return self._tool_task_update(
                args.get("task_id", ""),
                args.get("status", ""),
                summary=args.get("summary", ""),
            )
        elif name == "EnterPlanMode":
            return self._tool_enter_plan_mode(reason=args.get("reason", ""))
        elif name == "ExitPlanMode":
            return self._tool_exit_plan_mode(summary=args.get("summary", ""))
        elif name == "EnterWorktree":
            return self._tool_enter_worktree(
                branch_name=args.get("branch_name", ""),
                path=args.get("path", ""),
                base_ref=args.get("base_ref", "HEAD"),
            )
        elif name == "Skill":
            return self._tool_skill(
                args.get("query", ""),
                mode=args.get("mode", "search"),
            )
        elif name == "WebFetch":
            return self._tool_web_fetch(
                args.get("url", ""),
                prompt=args.get("prompt"),
            )
        elif name == "WebSearch":
            return self._tool_web_search(
                args.get("query", ""),
                num_results=args.get("num_results", 5),
            )
        elif name == "ListMcpResourcesTool":
            return self._tool_list_mcp_resources(server=args.get("server", ""))
        elif name == "ReadMcpResourceTool":
            return self._tool_read_mcp_resource(
                args.get("server", ""),
                args.get("uri", ""),
            )
        elif name == "web_download":
            return self._tool_web_download(
                args.get("url", ""),
                filename=args.get("filename"),
                directory=args.get("directory"),
            )
        elif name == "finish":
            return {"finished": True, "summary": args.get("summary", "")}
        else:
            return f"Unknown tool: {name}"

    async def _run_agentic_loop(self, request: str, adata: Any,
                               event_callback=None,
                               cancel_event=None,
                               history=None,
                               approval_handler=None) -> Any:
        """Execute the agentic loop: LLM decides tools to call iteratively.

        Parameters
        ----------
        request : str
            Natural language request.
        adata : Any
            AnnData/MuData object to process.
        event_callback : callable, optional
            Async callback ``await event_callback(event_dict)`` for streaming.
            When provided, events are emitted at key points (tool_call,
            llm_chunk, code, result, done, error, usage). When None
            (default), no events are emitted and behavior is identical to
            the pre-callback version.
        cancel_event : threading.Event, optional
            When set, the loop will stop at the next safe checkpoint (between
            tool calls / LLM rounds), emit a ``done`` event with
            ``cancelled=True``, and return the adata in its last consistent
            state.
        history : list, optional
            List of prior conversation messages as dicts with ``role`` and
            ``content`` keys.  Injected between the system prompt and the
            current user message so the LLM has conversational context.
        """
        config = self._config if hasattr(self, '_config') else None
        max_turns = config.execution.max_agent_turns if config else 15
        recorder = RunTraceRecorder(
            request=request,
            model=self.model,
            provider=self.provider,
            session_id=self._get_harness_session_id(),
        )
        self._last_run_trace = recorder.trace
        final_summary = ""
        runtime_session_id = self._get_runtime_session_id()
        def _sync_runtime_trace() -> None:
            runtime_summary = runtime_state.get_summary(runtime_session_id)
            recorder.trace.loaded_tools = list(runtime_summary.get("loaded_tools", []))
            recorder.trace.plan_mode = bool((runtime_summary.get("plan_mode") or {}).get("enabled", False))
            recorder.trace.worktree = dict(runtime_summary.get("worktree") or {})
            recorder.trace.task_ids = [
                task.get("task_id", "")
                for task in runtime_summary.get("tasks", [])
                if task.get("task_id")
            ]
            recorder.trace.question_ids = [
                question.get("question_id", "")
                for question in runtime_summary.get("pending_questions", [])
                if question.get("question_id")
            ]

        _sync_runtime_trace()

        async def emit(event):
            recorder.add_event(event)
            if event_callback:
                await event_callback(event)

        # Build initial messages
        system_prompt = self._build_agentic_system_prompt()
        session_id = self._get_harness_session_id()
        if self._session_history is not None and session_id:
            history_context = self._session_history.build_context_for_llm(session_id)
            if history_context:
                system_prompt += "\n\n" + history_context

        harness_config = getattr(self._config, "harness", None)
        if (
            self._trace_store is not None
            and harness_config is not None
            and getattr(harness_config, "include_recent_failures_in_prompt", False)
        ):
            failures_context = self._trace_store.build_context_for_prompt(
                limit=getattr(harness_config, "max_recent_failures", 3),
            )
            if failures_context:
                system_prompt += "\n\n" + failures_context

        if self._context_compactor is not None and self._context_compactor.needs_compaction(system_prompt, request):
            compaction = await self._context_compactor.compact_bundle(system_prompt)
            recorder.add_step(
                "context_compaction",
                summary="context compacted for handoff",
                data={
                    "original_tokens": compaction.original_tokens,
                    "compacted_tokens": compaction.compacted_tokens,
                    "summary": compaction.summary,
                },
            )
            if harness_config is not None and getattr(harness_config, "record_artifacts", False):
                recorder.add_artifact(
                    "context_summary",
                    label="compacted_system_prompt",
                    description=compaction.summary[:240],
                    metadata={
                        "original_tokens": compaction.original_tokens,
                        "compacted_tokens": compaction.compacted_tokens,
                    },
                )
            system_prompt = compaction.handoff_text

        messages = [
            {"role": "system", "content": system_prompt},
        ]
        # Inject conversation history (prior turns) for multi-turn context
        if history:
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
        messages.append(
            {"role": "user", "content": self._build_initial_user_message(request, adata)},
        )

        current_adata = adata
        completed_without_tool_calls = False
        meaningful_tool_call_seen = False
        no_tool_retry_count = 0
        # Anthropic benefits from an extra follow-up retry before giving up.
        _is_anthropic_model = "claude" in (self._llm.config.model or "").lower()
        max_no_tool_retries = 3 if _is_anthropic_model else 2
        chat_timeout = float(os.environ.get("OV_AGENT_CHAT_TIMEOUT_SECONDS", "45"))
        self._approval_handler = approval_handler

        def _is_cancelled():
            return cancel_event is not None and cancel_event.is_set()

        try:
            for turn in range(max_turns):
                # --- Cancel checkpoint: before LLM call ---
                if _is_cancelled():
                    print("   ⛔ Cancelled before LLM call")
                    recorder.add_step("status", summary="cancelled_before_llm")
                    recorder.finish(status="cancelled", result_summary="Cancelled before LLM call", usage=self.last_usage)
                    if self._trace_store is not None:
                        recorder.save(self._trace_store)
                    await emit(build_stream_event(
                        "done", "Cancelled", turn_id=recorder.trace.turn_id,
                        trace_id=recorder.trace.trace_id, session_id=recorder.trace.session_id,
                        category="lifecycle",
                    ) | {"cancelled": True})
                    self._save_conversation_log(messages)
                    return current_adata

                print(f"   🔄 Turn {turn + 1}/{max_turns}")
                tool_choice = self._select_agent_tool_choice(
                    request=request,
                    adata=current_adata,
                    turn_index=turn,
                    had_meaningful_tool_call=meaningful_tool_call_seen,
                    forced_retry=no_tool_retry_count > 0,
                )

                # LLM call with tools (stall-guarded)
                logger.info(
                    "agentic_llm_call_start turn=%d/%d tool_choice=%s messages=%d post_tool=%s",
                    turn + 1, max_turns, tool_choice, len(messages),
                    "yes" if meaningful_tool_call_seen else "no",
                )
                t_llm_start = time.time()
                stall_retries = 0
                response = None
                while response is None:
                    try:
                        response = await asyncio.wait_for(
                            self._llm.chat(
                                messages,
                                tools=self._get_visible_agent_tools(),
                                tool_choice=tool_choice,
                            ),
                            timeout=chat_timeout + 15,  # grace above backend timeout
                        )
                    except (asyncio.TimeoutError, Exception) as exc:
                        is_timeout = isinstance(exc, asyncio.TimeoutError) or "timeout" in str(exc).lower()
                        if is_timeout and stall_retries < 1:
                            stall_retries += 1
                            logger.warning(
                                "llm_chat_stall_retry attempt=%d turn=%d",
                                stall_retries, turn + 1,
                            )
                            recorder.add_step("status", summary=f"llm_chat_stall_retry_{stall_retries}")
                            continue
                        # Stall persists or non-timeout error
                        logger.error("llm_chat_stall_final turn=%d error=%s", turn + 1, exc)
                        recorder.add_step("error", summary=f"LLM stall after {stall_retries} retries: {exc}")
                        recorder.finish(
                            status="stall",
                            result_summary=f"LLM stall: {exc}",
                            usage=self.last_usage,
                        )
                        if self._trace_store is not None:
                            recorder.save(self._trace_store)
                        await emit(build_stream_event(
                            "done",
                            f"LLM stall detected after tool execution (retries={stall_retries})",
                            turn_id=recorder.trace.turn_id,
                            trace_id=recorder.trace.trace_id,
                            session_id=recorder.trace.session_id,
                            category="lifecycle",
                        ) | {"stall": True, "error": str(exc)})
                        self._save_conversation_log(messages)
                        self._last_run_trace = recorder.trace
                        return current_adata
                logger.info(
                    "agentic_llm_call_done turn=%d/%d elapsed_s=%.2f has_tool_calls=%s content_len=%d",
                    turn + 1, max_turns, time.time() - t_llm_start,
                    bool(response.tool_calls), len(response.content or ""),
                )

                # Track usage
                if response.usage:
                    self.last_usage = response.usage

                # Append assistant response to conversation
                if response.raw_message:
                    messages.append(response.raw_message)
                elif response.content:
                    messages.append({"role": "assistant", "content": response.content})

                # If text-only response with no tool calls, only treat as done
                # when it looks like an actual terminal answer rather than a
                # promise to act later. This approximates Codex's item-driven
                # follow-up loop on top of our simpler chat backend.
                if not response.tool_calls:
                    if response.content:
                        final_summary = response.content
                        recorder.add_step("llm_chunk", summary=response.content[:200])
                        print(f"   💬 Agent response: {response.content[:200]}")
                        await emit(build_stream_event(
                            "llm_chunk",
                            response.content,
                            turn_id=recorder.trace.turn_id,
                            trace_id=recorder.trace.trace_id,
                            session_id=recorder.trace.session_id,
                            category="assistant",
                        ))
                    should_follow_up = self._should_continue_after_text_response(
                        request=request,
                        response_content=response.content or "",
                        adata=current_adata,
                        had_meaningful_tool_call=meaningful_tool_call_seen,
                    )
                    logger.info(
                        "follow_up_gate turn=%d should_continue=%s retry=%d/%d meaningful_tool=%s",
                        turn + 1, should_follow_up, no_tool_retry_count,
                        max_no_tool_retries, meaningful_tool_call_seen,
                    )
                    if should_follow_up and no_tool_retry_count < max_no_tool_retries:
                        no_tool_retry_count += 1
                        guidance = self._build_no_tool_follow_up_message(
                            request,
                            retry_count=no_tool_retry_count,
                            max_retries=max_no_tool_retries,
                        )
                        messages.append({"role": "user", "content": guidance})
                        recorder.add_step(
                            "status",
                            summary="follow_up_required_after_text_only_response",
                            data={
                                "retry_count": no_tool_retry_count,
                                "max_retries": max_no_tool_retries,
                                "tool_choice": tool_choice,
                            },
                        )
                        await emit(build_stream_event(
                            "status",
                            {
                                "follow_up_required": True,
                                "retry_count": no_tool_retry_count,
                                "max_retries": max_no_tool_retries,
                                "reason": "action_request_text_only_response",
                            },
                            turn_id=recorder.trace.turn_id,
                            trace_id=recorder.trace.trace_id,
                            session_id=recorder.trace.session_id,
                            category="runtime",
                        ))
                        continue
                    # Exhaustion: follow-up retries used up or not needed
                    if should_follow_up and not meaningful_tool_call_seen:
                        logger.warning(
                            "follow_up_exhausted turn=%d retries=%d",
                            turn + 1, no_tool_retry_count,
                        )
                        recorder.add_step("status", summary="follow_up_exhausted_no_tool_calls")
                        await emit(build_stream_event(
                            "status",
                            {
                                "follow_up_exhausted": True,
                                "retry_count": no_tool_retry_count,
                            },
                            turn_id=recorder.trace.turn_id,
                            trace_id=recorder.trace.trace_id,
                            session_id=recorder.trace.session_id,
                            category="runtime",
                        ))
                    completed_without_tool_calls = True
                    break

                # Process each tool call and collect results
                tool_results = []  # (tool_call_id, tool_name, tool_output)
                finished = False
                no_tool_retry_count = 0

                for tc in response.tool_calls:
                    canonical_tool_name = normalize_tool_name(tc.name)
                    # --- Cancel checkpoint: before each tool dispatch ---
                    if _is_cancelled():
                        print("   ⛔ Cancelled before tool dispatch")
                        recorder.add_step("status", summary="cancelled_before_tool_dispatch")
                        recorder.finish(status="cancelled", result_summary="Cancelled before tool dispatch", usage=self.last_usage)
                        if self._trace_store is not None:
                            recorder.save(self._trace_store)
                        await emit(build_stream_event(
                            "done", "Cancelled", turn_id=recorder.trace.turn_id,
                            trace_id=recorder.trace.trace_id, session_id=recorder.trace.session_id,
                            category="lifecycle",
                        ) | {"cancelled": True})
                        self._save_conversation_log(messages)
                        return current_adata

                    tool_step_id = recorder.add_step(
                        "tool_call",
                        name=canonical_tool_name or tc.name,
                        summary=f"{canonical_tool_name or tc.name} dispatched",
                        data={"arguments": tc.arguments},
                    )
                    print(f"   🔧 Tool: {canonical_tool_name or tc.name}({', '.join(f'{k}=' for k in tc.arguments)})")

                    await emit(build_stream_event(
                        "item_started",
                        {"item_type": "tool_call", "name": canonical_tool_name or tc.name},
                        turn_id=recorder.trace.turn_id,
                        trace_id=recorder.trace.trace_id,
                        step_id=tool_step_id,
                        session_id=recorder.trace.session_id,
                        category="item",
                    ))
                    await emit(build_stream_event(
                        "tool_call",
                        {"name": canonical_tool_name or tc.name, "arguments": tc.arguments},
                        turn_id=recorder.trace.turn_id,
                        trace_id=recorder.trace.trace_id,
                        step_id=tool_step_id,
                        session_id=recorder.trace.session_id,
                        category="tool",
                    ))

                    result = await self._dispatch_tool(tc, current_adata, request)
                    if canonical_tool_name not in {"finish", "TaskGet", "TaskList", "TaskOutput", "ToolSearch"}:
                        meaningful_tool_call_seen = True

                    # Handle tools that return dict with adata (execute_code, delegate/execute)
                    if canonical_tool_name in ("execute_code", "Agent", "delegate") and isinstance(result, dict) and "adata" in result:
                        current_adata = result["adata"]
                        tool_output = result.get("output", "Code executed.")
                        if canonical_tool_name == "execute_code":
                            code = tc.arguments.get("code", "")
                            recorder.add_step(
                                "code",
                                name=canonical_tool_name,
                                summary=tc.arguments.get("description", "Code executed"),
                                data={"code": code},
                            )
                            recorder.add_artifact("code", label=tc.arguments.get("description", "execute_code"))
                            print(f"      ✅ {tc.arguments.get('description', 'Code executed')}")
                            await emit(build_stream_event(
                                "code",
                                code,
                                turn_id=recorder.trace.turn_id,
                                trace_id=recorder.trace.trace_id,
                                session_id=recorder.trace.session_id,
                                category="execution",
                            ))
                        else:
                            print(f"      ✅ delegate({tc.arguments.get('agent_type', '')}) completed")

                        shape = (
                            (current_adata.shape[0], current_adata.shape[1])
                            if hasattr(current_adata, "shape") else None
                        )
                        recorder.add_step(
                            "result",
                            name=canonical_tool_name or tc.name,
                            summary="adata updated",
                            data={"shape": shape},
                        )
                        await emit(build_stream_event(
                            "result",
                            current_adata,
                            turn_id=recorder.trace.turn_id,
                            trace_id=recorder.trace.trace_id,
                            session_id=recorder.trace.session_id,
                            category="execution",
                        ) | {"shape": shape})
                    elif canonical_tool_name == "finish":
                        summary = tc.arguments.get("summary", "Task completed")
                        final_summary = summary
                        recorder.add_step("done", name=canonical_tool_name, summary=summary)
                        print(f"   ✅ Finished: {summary}")
                        tool_output = f"Task finished: {summary}"
                        finished = True
                    elif isinstance(result, str):
                        tool_output = result
                    else:
                        tool_output = str(result)

                    # Truncate very long tool outputs before feeding them back
                    # into the next model turn. Web/content fetches are kept
                    # tighter because they otherwise produce oversized follow-up
                    # requests in OpenAI-compatible proxy setups.
                    max_tool_output_chars = 8000
                    if canonical_tool_name in {"WebFetch", "WebSearch", "web_fetch", "web_search"}:
                        max_tool_output_chars = 4000
                    if len(tool_output) > max_tool_output_chars:
                        tool_output = tool_output[: max_tool_output_chars - 500] + "\n... (truncated)"

                    try:
                        parsed_tool_output = json.loads(tool_output)
                    except Exception:
                        parsed_tool_output = None

                    if isinstance(parsed_tool_output, dict):
                        if canonical_tool_name == "ToolSearch":
                            await emit(build_stream_event(
                                "status",
                                {
                                    "loaded_tools": parsed_tool_output.get("loaded_tools", []),
                                    "newly_loaded": parsed_tool_output.get("newly_loaded", []),
                                },
                                turn_id=recorder.trace.turn_id,
                                trace_id=recorder.trace.trace_id,
                                step_id=tool_step_id,
                                session_id=recorder.trace.session_id,
                                category="runtime",
                            ))
                        elif canonical_tool_name in {"EnterPlanMode", "ExitPlanMode"}:
                            await emit(build_stream_event(
                                "status",
                                {"plan_mode": parsed_tool_output},
                                turn_id=recorder.trace.turn_id,
                                trace_id=recorder.trace.trace_id,
                                step_id=tool_step_id,
                                session_id=recorder.trace.session_id,
                                category="runtime",
                            ))
                        elif canonical_tool_name == "EnterWorktree":
                            await emit(build_stream_event(
                                "status",
                                {"worktree": parsed_tool_output},
                                turn_id=recorder.trace.turn_id,
                                trace_id=recorder.trace.trace_id,
                                step_id=tool_step_id,
                                session_id=recorder.trace.session_id,
                                category="runtime",
                            ))
                        elif canonical_tool_name in {"TaskCreate", "TaskUpdate", "TaskStop", "Bash"}:
                            await emit(build_stream_event(
                                "task_update",
                                parsed_tool_output,
                                turn_id=recorder.trace.turn_id,
                                trace_id=recorder.trace.trace_id,
                                step_id=tool_step_id,
                                session_id=recorder.trace.session_id,
                                category="task",
                            ))

                    tool_results.append((tc.id, tc.name, tool_output))
                    _sync_runtime_trace()
                    await emit(build_stream_event(
                        "item_completed",
                        {"item_type": "tool_call", "name": canonical_tool_name or tc.name, "status": "success"},
                        turn_id=recorder.trace.turn_id,
                        trace_id=recorder.trace.trace_id,
                        step_id=tool_step_id,
                        session_id=recorder.trace.session_id,
                        category="item",
                    ))

                # Batch all tool results into conversation messages.
                # For Anthropic: must be a single "user" message with all tool_result blocks.
                # For OpenAI: each tool result is a separate "tool" message.
                self._append_tool_results(messages, tool_results)

                if finished:
                    summary = final_summary or tc.arguments.get("summary", "Task completed")
                    recorder.finish(status="success", result_summary=summary, usage=self.last_usage)
                    if self._trace_store is not None:
                        recorder.save(self._trace_store)
                    await emit(build_stream_event(
                        "done",
                        summary,
                        turn_id=recorder.trace.turn_id,
                        trace_id=recorder.trace.trace_id,
                        session_id=recorder.trace.session_id,
                        category="lifecycle",
                    ))
                    if self.last_usage:
                        await emit(build_stream_event(
                            "usage",
                            self.last_usage,
                            turn_id=recorder.trace.turn_id,
                            trace_id=recorder.trace.trace_id,
                            session_id=recorder.trace.session_id,
                            category="usage",
                        ))
                    self._save_conversation_log(messages)
                    self._last_run_trace = recorder.trace
                    return current_adata

            if completed_without_tool_calls:
                recorder.finish(status="success", result_summary=final_summary or "Completed", usage=self.last_usage)
                if self._trace_store is not None:
                    recorder.save(self._trace_store)
                await emit(build_stream_event(
                    "done",
                    final_summary or "Completed",
                    turn_id=recorder.trace.turn_id,
                    trace_id=recorder.trace.trace_id,
                    session_id=recorder.trace.session_id,
                    category="lifecycle",
                ))
            else:
                logger.warning(
                    "agentic_loop_max_turns max_turns=%d meaningful_tool=%s",
                    max_turns, meaningful_tool_call_seen,
                )
                print(f"   ⚠️  Max turns ({max_turns}) reached, returning current result")
                final_summary = final_summary or f"Reached max turns ({max_turns})"
                recorder.finish(status="max_turns", result_summary=final_summary, usage=self.last_usage)
                if self._trace_store is not None:
                    recorder.save(self._trace_store)
                await emit(build_stream_event(
                    "done",
                    final_summary,
                    turn_id=recorder.trace.turn_id,
                    trace_id=recorder.trace.trace_id,
                    session_id=recorder.trace.session_id,
                    category="lifecycle",
                ) | {"max_turns": True})
            if self.last_usage:
                await emit(build_stream_event(
                    "usage",
                    self.last_usage,
                    turn_id=recorder.trace.turn_id,
                    trace_id=recorder.trace.trace_id,
                    session_id=recorder.trace.session_id,
                    category="usage",
                ))
            self._save_conversation_log(messages)
            self._last_run_trace = recorder.trace
            return current_adata
        except Exception as exc:
            recorder.add_step("error", summary=str(exc))
            recorder.finish(status="error", result_summary=str(exc), usage=self.last_usage)
            if self._trace_store is not None:
                recorder.save(self._trace_store)
            self._last_run_trace = recorder.trace
            try:
                await emit(build_stream_event(
                    "done",
                    f"Error: {exc}",
                    turn_id=recorder.trace.turn_id,
                    trace_id=recorder.trace.trace_id,
                    session_id=recorder.trace.session_id,
                    category="lifecycle",
                ) | {"error": True})
            except Exception:
                pass
            raise
        finally:
            self._approval_handler = None

    def _save_conversation_log(self, messages: list) -> None:
        """Save the full conversation to a JSON file for debugging.

        Activated by setting the ``OV_AGENT_LOG_DIR`` environment variable to a
        directory path.  Each run produces a timestamped JSON file containing
        the full message list (system, user, assistant, tool results).
        """
        log_dir = os.environ.get("OV_AGENT_LOG_DIR")
        if not log_dir:
            return
        try:
            from pathlib import Path
            import datetime as _dt
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_file = log_path / f"agent_conversation_{ts}.json"

            # Serialize messages, converting non-serializable objects
            def _safe(obj):
                if isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                if isinstance(obj, (list, tuple)):
                    return [_safe(v) for v in obj]
                if isinstance(obj, dict):
                    return {k: _safe(v) for k, v in obj.items()}
                return repr(obj)

            out_file.write_text(json.dumps(_safe(messages), indent=2, ensure_ascii=False))
            print(f"   📝 Conversation log saved: {out_file}")
        except Exception as exc:
            logger.debug(f"Failed to save conversation log: {exc}")

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
        """
        Apply targeted fixes to code based on known execution error patterns.

        Parameters
        ----------
        code : str
            The code that failed execution
        error_msg : str
            The error message from execution

        Returns
        -------
        Optional[str]
            Fixed code if a known pattern was matched, None otherwise
        """
        error_str = str(error_msg).lower()

        # Fix 0: Missing package → auto-install and retry same code
        cfg = self._config if hasattr(self, '_config') else None
        auto_install = cfg.execution.auto_install_packages if cfg else True
        if auto_install and ("no module named" in error_str or "modulenotfounderror" in error_str):
            pkg = self._extract_package_name(error_msg)
            if pkg and self._auto_install_package(pkg):
                return code  # same code, package now installed

        # Fix 1: .dtype -> .dtypes for DataFrames
        if "has no attribute 'dtype'" in error_str or "'dtype'" in error_str:
            # Replace .dtype with .dtypes (for DataFrame attribute access)
            import re
            fixed_code = re.sub(r'\.dtype\b', '.dtypes', code)
            if fixed_code != code:
                logger.debug("Applied fix: .dtype -> .dtypes")
                return fixed_code

        # Fix 2: seurat_v3 LOESS error -> seurat fallback
        if "extrapolation" in error_str or "loess" in error_str or "blending" in error_str:
            # Replace seurat_v3 with seurat
            fixed_code = code.replace("flavor='seurat_v3'", "flavor='seurat'")
            fixed_code = fixed_code.replace('flavor="seurat_v3"', 'flavor="seurat"')
            if fixed_code != code:
                logger.debug("Applied fix: seurat_v3 -> seurat for HVG")
                return fixed_code

        # Fix 3: Categorical batch column errors (improved to handle Categorical dtype)
        if ("cannot setitem on a categorical" in error_str or "new category" in error_str or
            ("nan" in error_str and ("batch" in error_str or "categorical" in error_str))):
            # Inject defensive batch preparation code that handles Categorical columns
            prep_code = '''
import pandas as pd
if 'batch' in adata.obs.columns:
    if pd.api.types.is_categorical_dtype(adata.obs['batch']):
        adata.obs['batch'] = adata.obs['batch'].astype(str)
    adata.obs['batch'] = adata.obs['batch'].fillna('unknown')
    adata.obs['batch'] = adata.obs['batch'].astype('category')
'''
            fixed_code = prep_code + "\n" + code
            logger.debug("Applied fix: Categorical batch column handling")
            return fixed_code

        # Fix 4: In-place function assignment error (adata = ov.pp.func(adata) returns None)
        if "'nonetype' object has no attribute" in error_str:
            import re
            # Pattern: adata = ov.pp.func(adata, ...) where func is an in-place function
            inplace_funcs = ['pca', 'scale', 'neighbors', 'leiden', 'umap', 'tsne', 'sude', 'scrublet', 'mde']
            pattern = r'adata\s*=\s*ov\.pp\.(' + '|'.join(inplace_funcs) + r')\s*\('

            if re.search(pattern, code):
                # Remove the assignment, keep just the function call
                fixed_code = re.sub(
                    r'adata\s*=\s*(ov\.pp\.(?:' + '|'.join(inplace_funcs) + r')\s*\([^)]*\))',
                    r'\1',  # Keep just the function call
                    code
                )
                if fixed_code != code:
                    logger.debug("Applied fix: Removed assignment from in-place function call")
                    return fixed_code

        return None

    # ------------------------------------------------------------------
    # Self-repair helpers (P3-1)
    # ------------------------------------------------------------------

    _PACKAGE_ALIASES: Dict[str, str] = {
        "cv2": "opencv-python",
        "sklearn": "scikit-learn",
        "skimage": "scikit-image",
        "yaml": "pyyaml",
        "PIL": "Pillow",
        "Bio": "biopython",
        "umap": "umap-learn",
        "leidenalg": "leidenalg",
        "louvain": "louvain",
        "harmonypy": "harmonypy",
        "scanorama": "scanorama",
        "scvi": "scvi-tools",
        "scarches": "scarches",
        "bbknn": "bbknn",
        "scrublet": "scrublet",
        "magic": "magic-impute",
    }

    @staticmethod
    def _extract_package_name(error_msg: str) -> Optional[str]:
        """Extract the top-level package name from a ModuleNotFoundError message."""
        # "No module named 'leidenalg'"  → leidenalg
        # "No module named 'some.sub.pkg'" → some
        m = re.search(r"No module named ['\"]([^'\"]+)['\"]", str(error_msg))
        if m:
            return m.group(1).split(".")[0]
        return None

    def _auto_install_package(self, package_name: str) -> bool:
        """Attempt to pip-install a missing package.  Returns True on success."""
        import subprocess

        cfg = self._config if hasattr(self, '_config') else None

        # Check blocklist
        blocklist = ["os", "sys", "subprocess", "shutil", "signal", "ctypes"]
        if cfg and hasattr(cfg, 'execution'):
            blocklist = cfg.execution.package_blocklist

        if package_name in blocklist:
            logger.warning("Package %r is on the blocklist — skipping auto-install", package_name)
            return False

        pip_name = self._PACKAGE_ALIASES.get(package_name, package_name)
        print(f"   📦 Auto-installing missing package: {pip_name}")

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pip_name],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                print(f"   ✅ Successfully installed {pip_name}")
                # Force re-import by clearing module cache
                for key in list(sys.modules.keys()):
                    if key == package_name or key.startswith(package_name + "."):
                        del sys.modules[key]
                return True
            else:
                print(f"   ❌ pip install failed: {result.stderr[:200]}")
                return False
        except Exception as exc:
            logger.warning("Auto-install of %r failed: %s", pip_name, exc)
            return False

    async def _diagnose_error_with_llm(
        self,
        code: str,
        error_msg: str,
        traceback_str: str,
        adata: Any,
    ) -> Optional[str]:
        """Use LLM to diagnose an execution error and generate fixed code.

        Called when pattern-based fixes fail.  Returns corrected code or None.
        """
        if self._llm is None:
            return None

        dataset_summary = ""
        if adata is not None and hasattr(adata, "shape"):
            dataset_summary = f"Dataset: {adata.shape[0]} cells × {adata.shape[1]} genes"
            if hasattr(adata, "obs") and hasattr(adata.obs, "columns"):
                cols = list(adata.obs.columns[:20])
                dataset_summary += f"\nobs columns: {cols}"

        diagnosis_prompt = f"""The following OmicVerse agent-generated Python code failed during execution.

--- CODE ---
{code}

--- ERROR ---
{error_msg}

--- TRACEBACK ---
{traceback_str[-1500:]}

--- DATASET ---
{dataset_summary}

Your task:
1. Diagnose the root cause of the error.
2. Generate a CORRECTED version of the full code that fixes the issue.
3. Wrap the corrected code in ```python ... ``` markers.

Important rules:
- Fix ONLY the error. Do not change logic that already works.
- If a variable is undefined, define it or remove the reference.
- If a module is unavailable, use an alternative or add a try/except.
- Preserve all file output operations (savefig, to_csv, json.dump, etc.).
"""

        try:
            print(f"   🔬 LLM diagnosing execution error...")
            response = await self._llm.run(diagnosis_prompt)

            diagnosed_code = self._extract_python_code(response)
            if diagnosed_code and diagnosed_code.strip():
                print(f"   💡 LLM generated fix ({len(diagnosed_code)} chars)")
                return diagnosed_code
        except Exception as exc:
            logger.warning("LLM error diagnosis failed: %s", exc)

        return None

    def _validate_outputs(self, code: str, output_dir: Optional[str] = None) -> List[str]:
        """Check which file-write operations in *code* produced actual files.

        Returns list of file paths that were referenced in code but do NOT exist.
        """
        missing: List[str] = []
        # Patterns for common file-write calls
        file_patterns = [
            # savefig("path") or savefig('path')
            r'\.savefig\s*\(\s*["\']([^"\']+)["\']',
            # to_csv("path")
            r'\.to_csv\s*\(\s*["\']([^"\']+)["\']',
            # write_h5ad("path")
            r'\.write_h5ad\s*\(\s*["\']([^"\']+)["\']',
            # write("path")
            r'\.write\s*\(\s*["\']([^"\']+\.h5ad)["\']',
            # json.dump(..., open("path", "w"))
            r'open\s*\(\s*["\']([^"\']+\.json)["\']',
            # pd.to_excel / to_parquet
            r'\.to_excel\s*\(\s*["\']([^"\']+)["\']',
            r'\.to_parquet\s*\(\s*["\']([^"\']+)["\']',
        ]
        for pattern in file_patterns:
            for m in re.finditer(pattern, code):
                fpath = m.group(1)
                if output_dir and not os.path.isabs(fpath):
                    fpath = os.path.join(output_dir, fpath)
                if not os.path.exists(fpath):
                    missing.append(fpath)
        return missing

    async def _generate_completion_code(
        self,
        original_code: str,
        missing_files: List[str],
        adata: Any,
        request: str,
    ) -> Optional[str]:
        """Generate code that produces only the missing output files."""
        if self._llm is None or not missing_files:
            return None

        prompt = f"""The following code was executed successfully but some output files were NOT created:

--- ORIGINAL CODE ---
{original_code}

--- MISSING FILES ---
{json.dumps(missing_files)}

--- ORIGINAL REQUEST ---
{request}

Generate a SHORT Python snippet that creates ONLY the missing files listed above.
- The `adata` variable is already available with the processed data.
- Reuse any variables/imports from the original code.
- Wrap the code in ```python ... ``` markers.
"""
        try:
            response = await self._llm.run(prompt)
            return self._extract_python_code(response)
        except Exception as exc:
            logger.warning("Completion code generation failed: %s", exc)
            return None

    def _request_approval(self, code: str, violations: list) -> bool:
        """Display generated code and ask the user for execution approval.

        Parameters
        ----------
        code : str
            The generated code about to be executed.
        violations : list
            Security violations found by the scanner (may be empty).

        Returns
        -------
        bool
            True if user approves execution, False otherwise.
        """
        if self._approval_handler is not None:
            trace = getattr(self, "_last_run_trace", None)
            payload = {
                "request_id": make_turn_id(),
                "title": "Execution approval required",
                "message": "Generated code requires approval before execution.",
                "code": code,
                "violations": [v.__dict__ if hasattr(v, "__dict__") else str(v) for v in violations],
                "trace_id": getattr(trace, "trace_id", ""),
                "session_id": self._get_harness_session_id(),
                "approval_mode": self._security_config.approval_mode.value,
            }
            try:
                return bool(self._approval_handler(payload))
            except Exception as exc:
                logger.warning("Approval handler failed, denying execution: %s", exc)
                return False

        print("\n" + "=" * 60)
        print("GENERATED CODE REVIEW")
        print("=" * 60)
        display = code if len(code) < 2000 else code[:2000] + "\n... (truncated)"
        for i, line in enumerate(display.split("\n"), 1):
            print(f"  {i:3d} | {line}")
        if violations:
            print()
            print(self._security_scanner.format_report(violations))
        print("=" * 60)
        try:
            response = input("Execute this code? [y/N]: ").strip().lower()
            return response in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    def _execute_generated_code(self, code: str, adata: Any, capture_stdout: bool = False) -> Any:
        """Execute generated Python code in a sandboxed namespace or session notebook.

        Parameters
        ----------
        code : str
            Python code to execute.
        adata : Any
            AnnData object.
        capture_stdout : bool
            If True, returns dict {"adata": result_adata, "stdout": captured_output}
            instead of just result_adata.

        Notes
        -----
        When use_notebook_execution=True, code runs in a separate Jupyter notebook session
        for better isolation and debugging. Otherwise, executes in a sandboxed namespace.
        The sandbox restricts available built-ins and module imports, but it is not a
        foolproof security boundary. Only run the agent with data and environments you
        trust, and consider additional isolation (e.g., containers) for untrusted input.
        """

        # --- Pre-execution security scan ---
        try:
            violations = self._security_scanner.scan(code)
        except SyntaxError:
            violations = []  # Syntax errors handled downstream by compile()

        if violations:
            report = self._security_scanner.format_report(violations)
            logger.warning("Security scan report:\n%s", report)

            if self._security_scanner.has_critical(violations):
                raise SecurityViolationError(
                    f"Code blocked by security scanner:\n{report}",
                    violations=violations,
                )

        # --- Approval gate ---
        approval_mode = self._security_config.approval_mode
        if approval_mode == ApprovalMode.ALWAYS:
            if not self._request_approval(code, violations):
                raise SecurityViolationError("User declined code execution.")
        elif approval_mode == ApprovalMode.ON_VIOLATION and violations:
            if not self._request_approval(code, violations):
                raise SecurityViolationError(
                    "User declined code execution after security warnings."
                )

        # Use notebook execution if enabled
        if self.use_notebook_execution and self._notebook_executor is not None:
            try:
                result_adata = self._notebook_executor.execute(code, adata)

                # Store session info in result
                if hasattr(result_adata, 'uns'):
                    result_adata.uns['_ovagent_session'] = {
                        'session_id': self._notebook_executor.current_session['session_id'],
                        'notebook_path': str(self._notebook_executor.current_session['notebook_path']),
                        'prompt_number': self._notebook_executor.session_prompt_count
                    }

                # Process context directives from the code (notebook execution path)
                if self.enable_filesystem_context and self._filesystem_context:
                    # For notebook execution, we don't have access to local vars
                    # but we can still process plan and update directives
                    self._process_context_directives(code, {})

                return result_adata

            except Exception as e:
                # P2-3: Notebook fallback policy
                policy = getattr(
                    getattr(self, '_config', None),
                    'execution', None,
                )
                fb = getattr(policy, 'sandbox_fallback_policy', SandboxFallbackPolicy.WARN_AND_FALLBACK)

                if fb == SandboxFallbackPolicy.RAISE:
                    raise SandboxDeniedError(
                        f"Notebook execution failed and fallback is disabled: {e}"
                    ) from e
                elif fb == SandboxFallbackPolicy.WARN_AND_FALLBACK:
                    if hasattr(self, '_emit'):
                        self._emit(EventLevel.WARNING, f"Session execution failed: {e}", "execution")
                        self._emit(EventLevel.INFO, "Falling back to in-process execution...", "execution")
                    else:
                        print(f"\u26a0\ufe0f  Session execution failed: {e}")
                        print(f"   Falling back to in-process execution...")
                # SandboxFallbackPolicy.SILENT: fall through silently

        # Legacy in-process execution
        compiled = compile(code, "<omicverse-agent>", "exec")
        sandbox_globals = self._build_sandbox_globals()
        sandbox_locals = {"adata": adata}
        # Common aliases occasionally emitted by LLM-generated code.
        # Keep these as locals (not globals) so user code can override them normally.
        try:
            if hasattr(adata, "obs_names"):
                sandbox_locals.setdefault("obs_names", adata.obs_names)
            if hasattr(adata, "var_names"):
                sandbox_locals.setdefault("var_names", adata.var_names)
        except Exception:
            pass

        # Skip AnnData-specific preprocessing for MuData objects
        _is_mudata = type(adata).__name__ == "MuData"

        # Normalize HVG column naming so generated code can access either alias
        try:
            if not _is_mudata and hasattr(adata, "var") and adata.var is not None:
                if "highly_variable" not in adata.var.columns and "highly_variable_features" in adata.var.columns:
                    adata.var["highly_variable"] = adata.var["highly_variable_features"]
            # Store initial sizes so LLM summaries don't KeyError
            if hasattr(adata, "uns"):
                adata.uns.setdefault("initial_cells", getattr(adata, "n_obs", None) or getattr(adata, "shape", [None])[0])
                adata.uns.setdefault("initial_genes", getattr(adata, "n_vars", None) or getattr(adata, "shape", [None, None])[1])
                adata.uns.setdefault("omicverse_qc_original_cells", getattr(adata, "n_obs", None))
            # Provide a default raw/ scaled layer so generated code using use_raw=True or layer='scaled' does not crash
            if not _is_mudata and getattr(adata, "raw", None) is None:
                try:
                    adata.raw = adata
                except Exception:
                    pass
            if not _is_mudata and hasattr(adata, "layers") and getattr(adata, "layers", None) is not None and "scaled" not in adata.layers:
                try:
                    adata.layers["scaled"] = adata.X.copy()
                except Exception:
                    pass
            # Fill missing numeric obs values to avoid downstream hist/binning errors (AnnData only)
            try:
                import pandas as _pd  # local import to avoid sandbox collisions
                if not _is_mudata and hasattr(adata, "obs"):
                    for col in adata.obs.columns:
                        col_data = adata.obs[col]
                        if _pd.api.types.is_numeric_dtype(col_data):
                            if col_data.isna().any():
                                adata.obs[col] = col_data.fillna(0)
                    # Provide common QC aliases if missing
                    if "total_counts" not in adata.obs.columns:
                        try:
                            import numpy as _np
                            data_matrix = None
                            if hasattr(adata, "layers") and getattr(adata, "layers", None) is not None and "counts" in adata.layers:
                                data_matrix = adata.layers["counts"]
                            else:
                                data_matrix = adata.X
                            sums = _np.asarray(data_matrix.sum(axis=1)).ravel()
                            adata.obs["total_counts"] = sums
                        except Exception:
                            adata.obs["total_counts"] = 0
                    if "n_counts" not in adata.obs.columns:
                        adata.obs["n_counts"] = adata.obs.get("total_counts", 0)
                    if "n_genes_by_counts" not in adata.obs.columns:
                        try:
                            import numpy as _np
                            data_matrix = None
                            if hasattr(adata, "layers") and getattr(adata, "layers", None) is not None and "counts" in adata.layers:
                                data_matrix = adata.layers["counts"]
                            else:
                                data_matrix = adata.X
                            adata.obs["n_genes_by_counts"] = _np.asarray((data_matrix > 0).sum(axis=1)).ravel()
                        except Exception:
                            adata.obs["n_genes_by_counts"] = 0
                    if "pct_counts_mito" not in adata.obs.columns and "pct_counts_mt" in adata.obs.columns:
                        adata.obs["pct_counts_mito"] = adata.obs["pct_counts_mt"]
                    elif "pct_counts_mito" not in adata.obs.columns:
                        adata.obs["pct_counts_mito"] = 0
                if hasattr(adata, "var") and adata.var is not None:
                    # Provide a canonical mitochondrial column name
                    if "mito" not in adata.var.columns and "mt" in adata.var.columns:
                        adata.var["mito"] = adata.var["mt"]
                    elif "mito" not in adata.var.columns:
                        adata.var["mito"] = False
                # Make pandas.cut tolerant to constant/duplicate bin edges
                try:
                    if not getattr(_pd.cut, "_ov_wrapped", False):
                        _orig_cut = _pd.cut
                        def _safe_cut(*args, **kwargs):
                            kwargs.setdefault("duplicates", "drop")
                            return _orig_cut(*args, **kwargs)
                        _safe_cut._ov_wrapped = True  # type: ignore[attr-defined]
                        _pd.cut = _safe_cut
                except Exception:
                    pass
            except Exception:
                pass
            # Ensure common output directories exist for downloads
            try:
                from pathlib import Path
                Path("genesets").mkdir(exist_ok=True)
            except Exception:
                pass
        except Exception as exc:  # pragma: no cover - defensive guard
            warnings.warn(
                f"Failed to normalize HVG columns for agent execution: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

        warnings.warn(
            "Executing agent-generated code. Ensure the input model and prompts come from a trusted source.",
            RuntimeWarning,
            stacklevel=2,
        )

        import io as _io

        stdout_buffer = _io.StringIO() if capture_stdout else None

        with self._temporary_api_keys():
            if capture_stdout:
                old_stdout = sys.stdout
                sys.stdout = stdout_buffer
                try:
                    exec(compiled, sandbox_globals, sandbox_locals)
                finally:
                    sys.stdout = old_stdout
            else:
                exec(compiled, sandbox_globals, sandbox_locals)

        result_adata = sandbox_locals.get("adata", adata)
        self._normalize_doublet_obs(result_adata)

        # Process context directives from the code
        if self.enable_filesystem_context and self._filesystem_context:
            self._process_context_directives(code, sandbox_locals)

        if capture_stdout:
            stdout_text = stdout_buffer.getvalue()
            return {"adata": result_adata, "stdout": stdout_text}

        return result_adata

    def _normalize_doublet_obs(self, adata: Any) -> None:
        """Harmonize doublet labels so downstream reporting works for non-expert users."""
        try:
            obs = getattr(adata, "obs", None)
            if obs is None:
                return
            # Create a canonical column if a plural form exists
            if "predicted_doublet" not in obs.columns and "predicted_doublets" in obs.columns:
                obs["predicted_doublet"] = obs["predicted_doublets"]
            # Choose the first available doublet flag column
            col = None
            for c in ("predicted_doublet", "predicted_doublets", "doublet", "doublets"):
                if c in obs.columns:
                    col = c
                    break
            if col is None:
                return
            # Store a simple rate summary for easy reporting
            try:
                rate = float(obs[col].mean()) * 100.0
                if hasattr(adata, "uns"):
                    adata.uns.setdefault("doublet_summary", {})["rate_percent"] = rate
            except Exception:
                pass
        except Exception:
            # Keep silent; this is a best-effort harmonization step
            return

    def _process_context_directives(self, code: str, local_vars: Dict[str, Any]) -> None:
        """Process context directives from generated code.

        This method parses special comments in the code that instruct the agent
        to write notes, save plans, or update plan status.

        Supported directives:
        - # CONTEXT_WRITE: key -> value
        - # CONTEXT_PLAN: list of steps
        - # CONTEXT_UPDATE: step=N, status=S, result=R

        Parameters
        ----------
        code : str
            The generated code containing context directives.
        local_vars : dict
            The local namespace after code execution, for resolving variable references.
        """
        if not self._filesystem_context:
            return

        try:
            lines = code.split('\n')

            # Track if we're collecting a multi-line plan
            collecting_plan = False
            plan_steps = []

            for line in lines:
                stripped = line.strip()

                # Handle CONTEXT_WRITE directives
                if stripped.startswith('# CONTEXT_WRITE:'):
                    self._handle_context_write(stripped, local_vars)

                # Handle CONTEXT_PLAN directives
                elif stripped.startswith('# CONTEXT_PLAN:'):
                    collecting_plan = True
                    plan_steps = []

                elif collecting_plan:
                    if stripped.startswith('# - '):
                        # Parse plan step: "# - Step N: Description [status]"
                        step_text = stripped[4:]  # Remove "# - "
                        step_info = self._parse_plan_step(step_text)
                        if step_info:
                            plan_steps.append(step_info)
                    elif stripped.startswith('#'):
                        # Continue collecting if it's still a comment
                        if not stripped.startswith('# CONTEXT_'):
                            continue
                        else:
                            # New directive, stop collecting plan
                            if plan_steps:
                                self._filesystem_context.write_plan(plan_steps)
                                logger.debug(f"Saved plan with {len(plan_steps)} steps")
                            collecting_plan = False
                            plan_steps = []
                    else:
                        # Non-comment line, stop collecting plan
                        if plan_steps:
                            self._filesystem_context.write_plan(plan_steps)
                            logger.debug(f"Saved plan with {len(plan_steps)} steps")
                        collecting_plan = False
                        plan_steps = []

                # Handle CONTEXT_UPDATE directives
                elif stripped.startswith('# CONTEXT_UPDATE:'):
                    self._handle_context_update(stripped)

            # Save any remaining plan
            if collecting_plan and plan_steps:
                self._filesystem_context.write_plan(plan_steps)
                logger.debug(f"Saved plan with {len(plan_steps)} steps")

        except Exception as e:
            logger.debug(f"Error processing context directives: {e}")

    def _handle_context_write(self, directive: str, local_vars: Dict[str, Any]) -> None:
        """Handle a CONTEXT_WRITE directive.

        Format: # CONTEXT_WRITE: key -> value
        where value can be a variable name or a literal dict/string.
        """
        try:
            # Extract the part after "# CONTEXT_WRITE:"
            content = directive.replace('# CONTEXT_WRITE:', '').strip()

            if ' -> ' in content:
                key, value_expr = content.split(' -> ', 1)
                key = key.strip()
                value_expr = value_expr.strip()

                # Try to evaluate the value expression
                try:
                    # First, try as a variable reference
                    if value_expr in local_vars:
                        value = local_vars[value_expr]
                    else:
                        # Try to evaluate as a Python expression
                        value = eval(value_expr, {"__builtins__": {}}, local_vars)
                except Exception:
                    # Fall back to string literal
                    value = value_expr

                # Determine category based on key pattern
                category = "notes"
                if any(kw in key.lower() for kw in ['result', 'stats', 'metrics', 'output']):
                    category = "results"
                elif any(kw in key.lower() for kw in ['decision', 'choice', 'why']):
                    category = "decisions"
                elif any(kw in key.lower() for kw in ['error', 'fail', 'exception']):
                    category = "errors"

                self._filesystem_context.write_note(key, value, category)
                logger.debug(f"Context write: {key} -> {category}")

        except Exception as e:
            logger.debug(f"Failed to process CONTEXT_WRITE: {e}")

    def _handle_context_update(self, directive: str) -> None:
        """Handle a CONTEXT_UPDATE directive.

        Format: # CONTEXT_UPDATE: step=N, status=S, result=R
        """
        try:
            content = directive.replace('# CONTEXT_UPDATE:', '').strip()

            # Parse key=value pairs
            parts = {}
            for part in content.split(','):
                if '=' in part:
                    k, v = part.split('=', 1)
                    parts[k.strip()] = v.strip().strip('"').strip("'")

            step = int(parts.get('step', 0))
            status = parts.get('status', 'completed')
            result = parts.get('result')

            self._filesystem_context.update_plan_step(step, status, result)
            logger.debug(f"Context update: step {step} -> {status}")

        except Exception as e:
            logger.debug(f"Failed to process CONTEXT_UPDATE: {e}")

    def _parse_plan_step(self, step_text: str) -> Optional[Dict[str, Any]]:
        """Parse a plan step from text.

        Format: "Step N: Description [status]" or just "Description [status]"
        """
        try:
            # Extract status if present
            status = "pending"
            if '[' in step_text and ']' in step_text:
                status_start = step_text.rfind('[')
                status_end = step_text.rfind(']')
                status = step_text[status_start + 1:status_end].strip().lower()
                step_text = step_text[:status_start].strip()

            # Remove "Step N:" prefix if present
            if step_text.lower().startswith('step '):
                # Find the colon after step number
                colon_idx = step_text.find(':')
                if colon_idx > 0:
                    step_text = step_text[colon_idx + 1:].strip()

            return {
                "description": step_text,
                "status": status,
            }

        except Exception:
            return None

    def _build_sandbox_globals(self) -> Dict[str, Any]:
        """Create a restricted global namespace for executing agent code."""

        allowed_builtins = [
            "abs",
            "all",
            "any",
            "bool",
            "dict",
            "enumerate",
            "Exception",
            "float",
            "int",
            "isinstance",
            "iter",
            "len",
            "list",
            "map",
            "max",
            "min",
            "next",
            "pow",
            "print",
            "range",
            "round",
            "set",
            "sorted",
            "str",
            "sum",
            "tuple",
            "zip",
            "filter",
            "type",
            # File I/O — needed for json.dump(), csv writing, etc.
            "open",
            # Safe introspection helpers often emitted by the agent
            "hasattr",
            "getattr",
            "setattr",
            "ImportError",
            "AttributeError",
            "IndexError",
            "FileNotFoundError",
            "OSError",
            "Exception",
            "ValueError",
            "RuntimeError",
            "TypeError",
            "KeyError",
            "AssertionError",
        ]
        # Conditionally allow introspection builtins (disabled by default
        # to prevent sandbox escape via globals()["os"].system(...) etc.)
        if not self._security_config.restrict_introspection:
            allowed_builtins.extend(["locals", "globals"])

        safe_builtins = {name: getattr(builtins, name) for name in allowed_builtins if hasattr(builtins, name)}
        allowed_modules = {}
        # Modules/roots explicitly disallowed to preserve safety (networking, shells, etc.)
        deny_roots = {
            "subprocess",
            "socket",
            "ssl",
            "urllib",
            "http",
            "ftplib",
            "smtplib",
            "telnetlib",
            "paramiko",
            "requests",
            "importlib",       # Prevents circumventing limited_import
            "ctypes",          # Prevents native code execution
            "multiprocessing", # Prevents process spawning
        }
        # Merge user-configured extra blocked modules
        deny_roots |= self._security_config.extra_blocked_modules
        core_modules = (
            "omicverse",
            "numpy",
            "pandas",
            "scanpy",
            # "os" is injected below as SafeOsProxy instead of raw os module
            "time",
            "math",
            "json",
            "re",
            "pathlib",
            "itertools",
            "functools",
            "collections",
            "statistics",
            "random",
            "warnings",
            "datetime",
            "typing",
        )
        skill_modules = (
            "openpyxl",
            "reportlab",
            "matplotlib",
            "seaborn",
            "scipy",
            "statsmodels",
            "sklearn",
        )
        for module_name in core_modules + skill_modules:
            try:
                allowed_modules[module_name] = __import__(module_name)
            except ImportError:
                warnings.warn(
                    f"Module '{module_name}' is not available inside the agent sandbox.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Inject SafeOsProxy instead of raw os module
        allowed_modules["os"] = SafeOsProxy()

        def _apply_scvi_shims() -> None:
            """Apply small runtime patches for known scvi-tools API footguns.

            Context (ovbench):
            - In scvi-tools 1.4.x, `scvi.model.MULTIVI.train` accepts `**kwargs` and
              internally forwards early-stopping settings to a TrainRunner.
            - LLM-generated code often passes `early_stopping_patience=...` into
              `MULTIVI.train(...)`, which can raise:
                TypeError: TrainRunner() got multiple values for keyword argument 'early_stopping_patience'

            We defensively drop that kwarg to keep benchmark runs deterministic and
            avoid a purely API-shape failure mode.
            """
            try:
                import functools
                from scvi.model import MULTIVI
            except Exception:
                return

            try:
                already = getattr(MULTIVI.train, "_ovbench_patched", False)
            except Exception:
                already = False
            if already:
                return

            orig_train = MULTIVI.train

            @functools.wraps(orig_train)
            def _train_wrapper(self, *args, **kwargs):
                kwargs.pop("early_stopping_patience", None)
                return orig_train(self, *args, **kwargs)

            _train_wrapper._ovbench_patched = True  # type: ignore[attr-defined]
            MULTIVI.train = _train_wrapper  # type: ignore[assignment]

        def limited_import(name, globals=None, locals=None, fromlist=(), level=0):
            root_name = name.split(".")[0]
            if root_name in deny_roots:
                # Allow omicverse internal modules (e.g. biocontext HTTP client)
                # to import denied modules; agent-generated code has no __package__
                caller_pkg = (globals or {}).get("__package__", "") or ""
                if not caller_pkg.startswith("omicverse"):
                    raise ImportError(
                        f"Module '{name}' is blocked inside the OmicVerse agent sandbox."
                    )
            if root_name not in allowed_modules:
                # Deny-list strategy: any non-denied module is allowed on demand.
                # Security boundary is deny_roots (network/process/shell) + AST scanner.
                allowed_modules[root_name] = __import__(root_name)
            if root_name == "scvi":
                _apply_scvi_shims()
            return __import__(name, globals, locals, fromlist, level)

        safe_builtins["__import__"] = limited_import

        sandbox_globals: Dict[str, Any] = {"__builtins__": safe_builtins}
        sandbox_globals.update(allowed_modules)
        # Friendly aliases commonly emitted by LLM code
        if "pandas" in allowed_modules:
            sandbox_globals.setdefault("pd", allowed_modules["pandas"])
        if "numpy" in allowed_modules:
            sandbox_globals.setdefault("np", allowed_modules["numpy"])
        if "scanpy" in allowed_modules:
            sandbox_globals.setdefault("sc", allowed_modules["scanpy"])
        if "omicverse" in allowed_modules:
            sandbox_globals.setdefault("ov", allowed_modules["omicverse"])
        return sandbox_globals

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
        """Generate a bullet overview of available project skills."""

        if not self.skill_registry or not self.skill_registry.skill_metadata:
            return ""
        lines = [
            f"- **{skill.name}** — {skill.description}"
            for skill in sorted(self.skill_registry.skill_metadata.values(), key=lambda item: item.name.lower())
        ]
        return "\n".join(lines)

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

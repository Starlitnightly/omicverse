"""
OmicVerse Smart Agent — thin facade / composer.

This module provides the public ``OmicVerseAgent`` class and the ``Agent()``
factory.  All substantial logic now lives in the ``ovagent/`` subpackage;
this file wires the subsystems together and exposes the stable public API.

Usage:
    import omicverse as ov
    result = ov.Agent("quality control with nUMI>500, mito<0.2", adata)
"""

import asyncio
import json
import logging
import os
import re
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Internal LLM backend
from .agent_backend import OmicVerseLLMBackend, Usage

# Registry system and model configuration
from .._registry import _global_registry
from .model_config import ModelConfig, PROVIDER_API_KEYS

# Grouped configuration dataclasses
from .agent_config import AgentConfig, SandboxFallbackPolicy

# Structured error hierarchy
from .agent_errors import (
    OVAgentError,
    ProviderError,
    ConfigError,
    ExecutionError,
    SandboxDeniedError,
    SecurityViolationError,
)

# Sandbox security hardening
from .agent_sandbox import (
    ApprovalMode,
    CodeSecurityScanner,
    SafeOsProxy,
    SecurityConfig,
)

# Structured event reporting
from .agent_reporter import (
    AgentEvent,
    EventLevel,
    Reporter,
    make_reporter,
)

# Harness helpers
from .harness import (
    RunTraceRecorder,
    RunTraceStore,
    build_stream_event,
    coerce_usage_payload,
    hash_code_block,
    make_turn_id,
)
from .harness.tool_catalog import (
    get_default_loaded_tool_names,
    get_visible_tool_schemas,
)

# Context compactor
from .context_compactor import ContextCompactor

# OVAgent extracted modules
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
    ProactiveCodeTransformer,
)
from .ovagent.tool_runtime import (
    LEGACY_AGENT_TOOLS as _LEGACY_AGENT_TOOLS,
    ToolRuntime as _ToolRuntime,
)
from .ovagent.codegen_pipeline import CodegenPipeline as _CodegenPipeline
from .ovagent.subagent_controller import SubagentController as _SubagentController
from .ovagent.turn_controller import (
    TurnController as _TurnController,
    FollowUpGate as _FollowUpGate,
)
from .ovagent.session_context import (
    SessionService as _SessionService,
    ContextService as _ContextService,
)
from .ovagent.registry_scanner import RegistryScanner as _RegistryScanner

# Session history
from .session_history import HistoryEntry, SessionHistory

# Skill registry
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

# Filesystem context management
from .filesystem_context import FilesystemContextManager
from ..jarvis.config import load_auth
from ..jarvis.openai_oauth import OPENAI_CODEX_BASE_URL, OpenAIOAuthManager


logger = logging.getLogger(__name__)

_LEGACY_OPENAI_CODEX_BASE_URL = "https://api.openai.com/v1"
OPENAI_CODEX_DEFAULT_MODEL = "gpt-5.3-codex"
_OPENAI_OAUTH_SUPPORTED_MODELS = {
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
    "gpt-5.2-codex",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5.1-codex",
    "gpt-5.1-codex-mini",
    "gpt-5.1-codex-max",
    "gpt-5.4",
    "gpt-5.4-mini",
}
_OPENAI_EXPLICIT_OAUTH_MODELS = {
    model_name
    for model_name in _OPENAI_OAUTH_SUPPORTED_MODELS
    if "codex" in model_name
}


def _normalize_auth_mode(auth_mode: Optional[str]) -> str:
    if auth_mode == "openai_api_key":
        return "saved_api_key"
    if auth_mode == "openai_codex":
        return "openai_oauth"
    return str(auth_mode or "environment")


def _is_openai_oauth_supported_model(model: str) -> bool:
    return str(model or "").strip().lower() in {
        name.lower() for name in _OPENAI_OAUTH_SUPPORTED_MODELS
    }


def _is_explicit_openai_oauth_model(model: str) -> bool:
    return str(model or "").strip().lower() in {
        name.lower() for name in _OPENAI_EXPLICIT_OAUTH_MODELS
    }


def _normalize_model_for_routing(model: str) -> str:
    normalized = str(model or "").strip()
    if not normalized:
        return normalized
    try:
        return ModelConfig.normalize_model_id(normalized)
    except Exception:
        return normalized


def _is_custom_openai_endpoint(endpoint: Optional[str]) -> bool:
    normalized = str(endpoint or "").strip().rstrip("/")
    return bool(normalized) and normalized not in {
        _LEGACY_OPENAI_CODEX_BASE_URL,
        OPENAI_CODEX_BASE_URL,
    }


def _extract_openai_codex_account_id(token: Optional[str]) -> str:
    try:
        return OmicVerseLLMBackend._extract_openai_codex_account_id(str(token or ""))
    except Exception:
        return ""


def _resolve_saved_provider_api_key(provider_name: str, auth_path: Optional[Path]) -> Optional[str]:
    auth = load_auth(auth_path)
    providers = dict(auth.get("providers") or {})

    if provider_name == "openai":
        top_level = str(auth.get("OPENAI_API_KEY") or "").strip()
        if top_level:
            return top_level

    provider_auth = dict(providers.get(provider_name) or {})
    api_key = str(provider_auth.get("api_key") or "").strip()
    return api_key or None


def _resolve_agent_llm_credentials(
    *,
    model: str,
    api_key: Optional[str],
    endpoint: Optional[str],
    auth_mode: Optional[str],
    auth_file: Optional[Union[str, Path]],
) -> Tuple[str, Optional[str], Optional[str], str]:
    """Resolve model/auth settings, including OpenAI Codex OAuth fallback."""

    normalized_mode = _normalize_auth_mode(auth_mode)
    resolved_model = model
    normalized_model = _normalize_model_for_routing(model)
    resolved_api_key = api_key
    api_key_source = "explicit" if resolved_api_key else None
    resolved_endpoint = endpoint
    resolved_auth_path = Path(auth_file).expanduser() if auth_file else None

    provider = ModelConfig.get_provider_from_model(normalized_model or resolved_model, resolved_endpoint)
    wants_codex_oauth = provider == "openai" and (
        normalized_mode == "openai_oauth"
        or _is_explicit_openai_oauth_model(normalized_model)
        or str(resolved_endpoint or "").rstrip("/") == OPENAI_CODEX_BASE_URL
    )

    if provider == "openai" and normalized_mode == "saved_api_key" and not resolved_api_key:
        resolved_api_key = _resolve_saved_provider_api_key("openai", resolved_auth_path)
        api_key_source = "saved_api_key" if resolved_api_key else None

    if not wants_codex_oauth:
        return resolved_model, resolved_api_key, resolved_endpoint, normalized_mode

    normalized_mode = "openai_oauth"
    preserve_model_id = _is_custom_openai_endpoint(resolved_endpoint)
    if not resolved_endpoint or resolved_endpoint.rstrip("/") == _LEGACY_OPENAI_CODEX_BASE_URL:
        resolved_endpoint = OPENAI_CODEX_BASE_URL
        preserve_model_id = False
    if _is_openai_oauth_supported_model(normalized_model):
        if not preserve_model_id:
            resolved_model = normalized_model
    elif not preserve_model_id:
        resolved_model = OPENAI_CODEX_DEFAULT_MODEL

    if resolved_api_key:
        if _extract_openai_codex_account_id(resolved_api_key):
            return resolved_model, resolved_api_key, resolved_endpoint, normalized_mode
        if api_key_source == "explicit":
            raise ValueError(
                "OpenAI Codex models require a ChatGPT OAuth access token with "
                "chatgpt_account_id. Run `omicverse jarvis --codex-login` or "
                "pass a valid OpenAI OAuth access token."
            )
        resolved_api_key = None

    manager = OpenAIOAuthManager(resolved_auth_path)
    resolved_api_key = manager.ensure_access_token_with_codex_fallback(
        refresh_if_needed=True,
        import_codex_if_missing=True,
    )
    if resolved_api_key and _extract_openai_codex_account_id(resolved_api_key):
        return resolved_model, resolved_api_key, resolved_endpoint, normalized_mode
    if resolved_api_key:
        raise ValueError(
            "Saved OpenAI Codex login is missing chatgpt_account_id. "
            "Run `omicverse jarvis --codex-login` again."
        )

    raise ValueError(
        "No saved OpenAI Codex login found. Run `omicverse jarvis --codex-login` "
        "or pass a valid OpenAI OAuth access token."
    )


def _wire_package_import_compatibility() -> None:
    """Expose parent package references without probing lazy __getattr__ hooks."""
    parent_pkg = sys.modules.get("omicverse")
    utils_pkg = sys.modules.get("omicverse.utils")
    module = sys.modules.get(__name__)
    if parent_pkg is None or utils_pkg is None or module is None:
        return

    parent_attrs = getattr(parent_pkg, "__dict__", {})
    if "utils" not in parent_attrs:
        setattr(parent_pkg, "utils", utils_pkg)

    module_name = __name__.rsplit(".", 1)[-1]
    utils_attrs = getattr(utils_pkg, "__dict__", {})
    if module_name not in utils_attrs:
        setattr(utils_pkg, module_name, module)


_wire_package_import_compatibility()


class OmicVerseAgent:
    """
    Intelligent agent for OmicVerse function discovery and execution.

    This agent uses an internal LLM backend to understand natural language
    requests and automatically execute appropriate OmicVerse functions.

    Usage:
        agent = ov.Agent(api_key="your-api-key")  # Uses gpt-5.2 by default
        result_adata = agent.run("quality control with nUMI>500, mito<0.2", adata)
    """
    # Class-level tool schemas (backward-compat surface)
    LEGACY_AGENT_TOOLS = _LEGACY_AGENT_TOOLS
    AGENT_TOOLS = get_visible_tool_schemas(get_default_loaded_tool_names()) + _LEGACY_AGENT_TOOLS

    @property
    def _codegen(self) -> "_CodegenPipeline":
        """Lazily create CodegenPipeline for agents constructed via __new__."""
        pipeline = getattr(self, "_codegen_pipeline", None)
        if pipeline is None:
            pipeline = _CodegenPipeline(self)
            self._codegen_pipeline = pipeline
        return pipeline

    def __init__(self, model: str = "gpt-5.2", api_key: Optional[str] = None, endpoint: Optional[str] = None, auth_mode: str = "environment", auth_file: Optional[str] = None, enable_reflection: bool = True, reflection_iterations: int = 1, enable_result_review: bool = True, use_notebook_execution: bool = True, max_prompts_per_session: int = 5, notebook_storage_dir: Optional[str] = None, keep_execution_notebooks: bool = True, notebook_timeout: int = 600, strict_kernel_validation: bool = True, enable_filesystem_context: bool = True, context_storage_dir: Optional[str] = None, approval_mode: str = "never", agent_mode: str = "agentic", max_agent_turns: int = 15, security_level: Optional[str] = None, *, config: Optional[AgentConfig] = None, reporter: Optional[Reporter] = None, verbose: bool = True):
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
        auth_mode : str, optional
            Authentication mode. Use ``"openai_oauth"`` to reuse Jarvis/Codex login state.
        auth_file : str, optional
            Path to the saved Jarvis auth file. Defaults to ``~/.ovjarvis/auth.json``.
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
                auth_mode=auth_mode,
                auth_file=auth_file,
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
        llm_cfg = self._config.llm
        model, api_key, endpoint, resolved_auth_mode = _resolve_agent_llm_credentials(
            model=llm_cfg.model,
            api_key=llm_cfg.api_key,
            endpoint=llm_cfg.endpoint,
            auth_mode=llm_cfg.auth_mode,
            auth_file=llm_cfg.auth_file,
        )
        llm_cfg.model = model
        llm_cfg.api_key = api_key
        llm_cfg.endpoint = endpoint
        llm_cfg.auth_mode = resolved_auth_mode
        model = llm_cfg.model
        api_key = llm_cfg.api_key
        endpoint = llm_cfg.endpoint
        self.auth_mode = llm_cfg.auth_mode
        self.auth_file = str(llm_cfg.auth_file) if llm_cfg.auth_file else None
        
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
        # Reflection configuration
        self.enable_reflection = self._config.reflection.enabled
        self.reflection_iterations = self._config.reflection.iterations
        # Result review configuration
        self.enable_result_review = self._config.reflection.result_review
        # Notebook execution configuration
        self.use_notebook_execution = self._config.execution.use_notebook
        self.max_prompts_per_session = self._config.execution.max_prompts_per_session
        self._notebook_executor = None
        # Filesystem context configuration (set early to avoid AttributeError)
        self.enable_filesystem_context = self._config.context.enabled
        self._filesystem_context: Optional[FilesystemContextManager] = None
        self.last_usage = None
        self.last_usage_breakdown: Dict[str, Any] = {
            'generation': None,
            'reflection': [],
            'review': [],
            'total': None
        }
        self._code_only_mode: bool = False
        self._code_only_captured_code: str = ""
        self._code_only_captured_history: List[Dict[str, Any]] = []
        self._session_history: Optional[SessionHistory] = None
        self._trace_store: Optional[RunTraceStore] = None
        self._context_compactor: Optional[ContextCompactor] = None
        self._last_run_trace = None
        self._approval_handler = None
        self._web_session_id = ""
        self._ov_runtime: Optional[OmicVerseRuntime] = None
        self._active_run_id = ""
        self._registry_scanner = _RegistryScanner()

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
            self._session_service = _SessionService(self)
            self._context_service = _ContextService(self)
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
            self._codegen_pipeline = _CodegenPipeline(self)

            print("✅ Smart Agent initialized successfully!")
        except Exception as e:
            print(f"❌ Agent initialization failed: {e}")
            raise

    # =====================================================================
    # Internal setup (called once from __init__)
    # =====================================================================

    def _get_registry_stats(self) -> dict:
        """Get statistics about the function registry."""
        static_entries = self._load_static_registry_entries()
        unique_functions = set(e['full_name'] for e in static_entries)
        categories = set(e['category'] for e in static_entries)

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
        processed_functions = set()
        for entry in _global_registry._registry.values():
            full_name = entry['full_name']
            if full_name in processed_functions:
                continue
            processed_functions.add(full_name)
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

    def _setup_agent(self):
        """Setup the internal agent backend with dynamic instructions."""
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

        if self.enable_filesystem_context and self._filesystem_context:
            instructions += self._context_service.build_filesystem_context_instructions() if hasattr(self, '_context_service') else _build_filesystem_context_instructions(self._filesystem_context)

        if self.api_key:
            required_key = PROVIDER_API_KEYS.get(self.model)
            if required_key and not os.getenv(required_key):
                os.environ[required_key] = self.api_key

        self._llm = _create_llm_backend(
            system_prompt=instructions,
            model=self.model,
            api_key=self.api_key,
            endpoint=self.endpoint,
        )

    # =====================================================================
    # Protocol-required methods (called by ovagent modules via AgentContext)
    # =====================================================================

    @contextmanager
    def _temporary_api_keys(self):
        """Temporarily inject API keys into the environment and clean up afterwards."""
        with _temporary_api_keys_cm(self._managed_api_env):
            yield

    def _get_harness_session_id(self) -> str:
        """Best-effort session identifier for harness traces/history."""
        return self._session_service.get_harness_session_id()

    def _get_runtime_session_id(self) -> str:
        """Return the session key used by the harness runtime registry."""
        return self._session_service.get_runtime_session_id()

    def _get_visible_agent_tools(self, *, allowed_names: Optional[set[str]] = None) -> list[dict[str, Any]]:
        """Return the currently visible tool schemas."""
        return self._tool_runtime.get_visible_agent_tools(allowed_names=allowed_names)

    def _get_loaded_tool_names(self) -> list[str]:
        """Return loaded tool names."""
        return self._tool_runtime.get_loaded_tool_names()

    def _refresh_runtime_working_directory(self) -> str:
        """Keep runtime cwd aligned with the active worktree / filesystem context."""
        return self._session_service.refresh_runtime_working_directory()

    def _tool_blocked_in_plan_mode(self, tool_name: str) -> bool:
        """Check plan-mode blocking."""
        return self._tool_runtime.tool_blocked_in_plan_mode(tool_name)

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
            return True
        return self._approval_handler(payload)

    def _request_tool_approval(self, tool_name: str, *, reason: str, payload: dict[str, Any]) -> None:
        from .harness.tool_catalog import get_tool_spec
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

    def _detect_repo_root(self, cwd: Optional[Path] = None) -> Optional[Path]:
        current = (cwd or Path(self._refresh_runtime_working_directory())).resolve()
        for candidate in (current, *current.parents):
            if (candidate / ".git").exists():
                return candidate
        return None

    def _extract_python_code(self, response_text: str) -> str:
        """Extract executable Python code from the agent response using AST validation."""
        return self._codegen.extract_python_code(response_text)

    def _normalize_registry_entry_for_codegen(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Convert registry entries to public-facing ov.* names for code generation."""
        return _RegistryScanner.normalize_entry(entry)

    def _build_agentic_system_prompt(self) -> str:
        return self._prompt_builder.build_agentic_system_prompt()

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

    def _load_skill_guidance(self, skill_name: str) -> str:
        """Return the detailed instructions for a requested skill."""
        if not self.skill_registry or not self.skill_registry.skill_metadata:
            return json.dumps({"error": "No project skills are available."})
        if not skill_name or not skill_name.strip():
            return json.dumps({"error": "Provide a skill name to load guidance."})
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

    # =====================================================================
    # Registry scanner delegates (thin wrappers over RegistryScanner)
    # =====================================================================

    @property
    def _scanner(self) -> _RegistryScanner:
        """Lazily create RegistryScanner for agents constructed via __new__."""
        scanner = getattr(self, "_registry_scanner", None)
        if scanner is None:
            scanner = _RegistryScanner()
            self._registry_scanner = scanner
        return scanner

    def _load_static_registry_entries(self) -> List[Dict[str, Any]]:
        """Parse @register_function metadata plus nested method/branch capabilities."""
        return self._scanner.load_static_entries()

    def _collect_relevant_registry_entries(self, request: str, max_entries: int = 8) -> List[Dict[str, Any]]:
        """Return a compact set of registry entries relevant to a free-form request."""
        return self._scanner.collect_relevant_entries(request, max_entries)

    def _collect_static_registry_entries(self, request: str, max_entries: int = 8) -> List[Dict[str, Any]]:
        """Search the static AST-derived registry snapshot."""
        return self._scanner.collect_static_entries(request, max_entries)

    def _score_registry_entry_for_codegen(self, request: str, entry: Dict[str, Any]) -> float:
        """Score a registry entry for lightweight code generation retrieval."""
        return _RegistryScanner.score_entry(request, entry)

    # =====================================================================
    # FollowUp gate delegates (used by tests via agent instance)
    # =====================================================================

    def _request_requires_tool_action(self, request: str, adata: Any) -> bool:
        return _FollowUpGate.request_requires_tool_action(request, adata)

    def _response_is_promissory(self, content: str) -> bool:
        return _FollowUpGate.response_is_promissory(content)

    def _select_agent_tool_choice(self, *, request, adata, turn_index, had_meaningful_tool_call, forced_retry) -> str:
        return _FollowUpGate.select_tool_choice(
            request=request, adata=adata, turn_index=turn_index,
            had_meaningful_tool_call=had_meaningful_tool_call, forced_retry=forced_retry,
        )

    # =====================================================================
    # Tool dispatch delegates
    # =====================================================================

    async def _dispatch_tool(self, tool_call, current_adata: Any, request: str):
        return await self._tool_runtime.dispatch_tool(tool_call, current_adata, request)

    # =====================================================================
    # Codegen pipeline delegates
    # =====================================================================

    def _capture_code_only_snippet(self, code: str, description: str = "") -> None:
        """Store the latest code snippet captured from execute_code in code-only mode."""
        self._codegen.capture_code_only_snippet(code, description)

    def _select_codegen_skill_matches(self, request: str, top_k: int = 2) -> List[SkillMatch]:
        return self._codegen.select_codegen_skill_matches(request, top_k)

    def _format_registry_context_for_codegen(self, entries: List[Dict[str, Any]]) -> str:
        return self._codegen.format_registry_context_for_codegen(entries)

    @staticmethod
    def _format_prerequisites_for_codegen_entry(entry: Dict[str, Any]) -> str:
        return _CodegenPipeline.format_prerequisites_for_codegen_entry(entry)

    def _build_code_generation_system_prompt(self, adata: Any) -> str:
        return self._codegen.build_code_generation_system_prompt(adata)

    @staticmethod
    def _build_code_generation_user_prompt(request: str, adata: Any) -> str:
        return _CodegenPipeline.build_code_generation_user_prompt(request, adata)

    @staticmethod
    def _contains_forbidden_scanpy_usage(code: str) -> bool:
        return _CodegenPipeline.contains_forbidden_scanpy_usage(code)

    def _rewrite_scanpy_calls_with_registry(self, code: str, entries: List[Dict[str, Any]]) -> str:
        return self._codegen.rewrite_scanpy_calls_with_registry(code, entries)

    async def _rewrite_code_without_scanpy(self, code: str, request: str, adata: Any, registry_context: str = "", skill_guidance: str = "") -> tuple:
        return await self._codegen.rewrite_code_without_scanpy(code, request, adata, registry_context, skill_guidance)

    async def _review_generated_code_lightweight(self, code: str, request: str, adata: Any) -> tuple:
        return await self._codegen.review_generated_code_lightweight(code, request, adata)

    @staticmethod
    def _build_code_only_agentic_request(request: str, adata: Any) -> str:
        return _CodegenPipeline.build_code_only_agentic_request(request, adata)

    async def _generate_code_via_agentic_loop(self, request: str, adata: Any, *, progress_callback: Optional[Callable[[str], None]] = None) -> str:
        return await self._codegen.generate_code_via_agentic_loop(request, adata, progress_callback=progress_callback)

    def _gather_code_candidates(self, response_text: str) -> List[str]:
        return self._codegen.gather_code_candidates(response_text)

    @staticmethod
    def _looks_like_python(code: str) -> bool:
        return _CodegenPipeline.looks_like_python(code)

    @staticmethod
    def _extract_inline_python(response_text: str) -> str:
        return _CodegenPipeline.extract_inline_python(response_text)

    @staticmethod
    def _normalize_code_candidate(code: str) -> str:
        return _CodegenPipeline.normalize_code_candidate(code)

    def _extract_python_code_strict(self, response_text: str) -> str:
        return self._codegen.extract_python_code_strict(response_text)

    async def _review_result(self, original_adata: Any, result_adata: Any, request: str, code: str) -> Dict[str, Any]:
        return await self._codegen.review_result(original_adata, result_adata, request, code)

    async def _reflect_on_code(self, code: str, request: str, adata: Any, iteration: int = 1) -> Dict[str, Any]:
        return await self._codegen.reflect_on_code(code, request, adata, iteration)

    def _detect_direct_python_request(self, request: str) -> Optional[str]:
        return self._codegen.detect_direct_python_request(request)

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

    # =====================================================================
    # Analysis executor delegates
    # =====================================================================

    def _check_code_prerequisites(self, code: str, adata: Any) -> str:
        return self._analysis_executor.check_code_prerequisites(code, adata)

    def _apply_execution_error_fix(self, code: str, error_msg: str) -> Optional[str]:
        return self._analysis_executor.apply_execution_error_fix(code, error_msg)

    @staticmethod
    def _extract_package_name(error_msg: str) -> Optional[str]:
        return _AnalysisExecutor.extract_package_name(error_msg)

    def _auto_install_package(self, package_name: str) -> bool:
        return self._analysis_executor.auto_install_package(package_name)

    async def _diagnose_error_with_llm(self, code: str, error_msg: str, traceback_str: str, adata: Any) -> Optional[str]:
        return await self._analysis_executor.diagnose_error_with_llm(code, error_msg, traceback_str, adata)

    def _validate_outputs(self, code: str, output_dir: Optional[str] = None) -> List[str]:
        return self._analysis_executor.validate_outputs(code, output_dir)

    async def _generate_completion_code(self, original_code: str, missing_files: List[str], adata: Any, request: str) -> Optional[str]:
        return await self._analysis_executor.generate_completion_code(original_code, missing_files, adata, request)

    def _request_approval(self, code: str, violations: list) -> bool:
        return self._analysis_executor.request_approval(code, violations)

    def _execute_generated_code(self, code: str, adata: Any, capture_stdout: bool = False) -> Any:
        return self._analysis_executor.execute_generated_code(code, adata, capture_stdout)

    @staticmethod
    def _normalize_doublet_obs(adata: Any) -> None:
        _AnalysisExecutor.normalize_doublet_obs(adata)

    def _process_context_directives(self, code: str, local_vars: Dict[str, Any]) -> None:
        self._analysis_executor.process_context_directives(code, local_vars)

    def _build_sandbox_globals(self) -> Dict[str, Any]:
        return self._analysis_executor.build_sandbox_globals()

    # =====================================================================
    # Skill helpers
    # =====================================================================

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

    async def _select_skill_matches_llm(self, request: str, top_k: int = 2) -> List[str]:
        """Use LLM to select relevant skills based on the request."""
        if not self.skill_registry or not self.skill_registry.skill_metadata:
            return []
        skills_list = []
        for skill in sorted(self.skill_registry.skill_metadata.values(), key=lambda s: s.name.lower()):
            skills_list.append(f"- **{skill.slug}**: {skill.description}")
        skills_catalog = "\n".join(skills_list)
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
            import re as _re
            json_match = _re.search(r'\[.*?\]', response, _re.DOTALL)
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
        return _format_skill_overview(self.skill_registry)

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

    # =====================================================================
    # Public API: run / stream
    # =====================================================================

    async def run_async(self, request: str, adata: Any) -> Any:
        """
        Process a natural language request using the agentic tool-calling loop.

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
            self.last_usage = None
            self.last_usage_breakdown = {
                'generation': None, 'reflection': [], 'review': [], 'total': None
            }
            try:
                result_adata = self._execute_generated_code(direct_code, adata)
                print(f"✅ Python code executed directly.")
                return result_adata
            except Exception as exc:
                print(f"❌ Direct Python execution failed: {exc}")
                raise

        if self.provider == "python":
            raise ValueError("Python provider requires executable Python code in the request.")

        return await self._run_agentic_mode(request, adata)

    async def _run_agentic_mode(self, request: str, adata: Any) -> Any:
        """Agentic loop mode: LLM autonomously calls tools to complete the task."""
        print(f"🤖 Mode: Agentic Loop (tool-calling)")
        print()

        try:
            result = await self._run_agentic_loop(request, adata)
            self._turn_controller._persist_harness_history(request)
            print()
            print(f"{'=' * 70}")
            print(f"✅ SUCCESS - Agentic loop completed!")
            print(f"{'=' * 70}\n")
            return result
        except Exception as e:
            self._turn_controller._persist_harness_history(request)
            print()
            print(f"{'=' * 70}")
            print(f"❌ ERROR - Agentic loop failed: {e}")
            print(f"{'=' * 70}\n")
            raise

    async def generate_code_async(self, request: str, adata: Any = None, *, max_functions: int = 8, progress_callback: Optional[Callable[[str], None]] = None) -> str:
        """Generate OmicVerse Python code without executing it."""
        return await self._codegen.generate_code_async(
            request, adata, max_functions=max_functions, progress_callback=progress_callback,
        )

    def generate_code(self, request: str, adata: Any = None, *, max_functions: int = 8, progress_callback: Optional[Callable[[str], None]] = None) -> str:
        """Synchronous wrapper for code-only OmicVerse generation."""
        return self._codegen.generate_code(
            request, adata, max_functions=max_functions, progress_callback=progress_callback,
        )

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
            Dictionary with ``'type'`` and ``'content'`` keys.
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
    # Session Management Methods (delegated to SessionService)
    # ===================================================================

    def get_current_session_info(self) -> Optional[Dict[str, Any]]:
        """Get information about current notebook session."""
        return self._session_service.get_current_session_info()

    def restart_session(self):
        """Manually restart notebook session (clear memory, start fresh)."""
        self._session_service.restart_session()

    def get_session_history(self) -> List[Dict[str, Any]]:
        """Get history of all archived notebook sessions."""
        return self._session_service.get_session_history()

    # ===================================================================
    # Filesystem Context Management (delegated to ContextService)
    # ===================================================================

    @property
    def filesystem_context(self) -> Optional[FilesystemContextManager]:
        """Get the filesystem context manager."""
        return self._context_service.filesystem_context

    def write_note(self, key: str, content: Union[str, Dict[str, Any]], category: str = "notes", metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Write a note to the filesystem context workspace."""
        return self._context_service.write_note(key, content, category, metadata)

    def search_context(self, pattern: str, match_type: str = "glob", max_results: int = 10) -> List[Dict[str, Any]]:
        """Search the filesystem context for relevant notes."""
        return self._context_service.search_context(pattern, match_type, max_results)

    def get_relevant_context(self, query: str, max_tokens: int = 1000) -> str:
        """Get context relevant to a query."""
        return self._context_service.get_relevant_context(query, max_tokens)

    def save_plan(self, steps: List[Dict[str, Any]]) -> Optional[str]:
        """Save an execution plan."""
        return self._context_service.save_plan(steps)

    def update_plan_step(self, step_index: int, status: str, result: Optional[str] = None) -> None:
        """Update the status of a plan step."""
        self._context_service.update_plan_step(step_index, status, result)

    def get_workspace_summary(self) -> str:
        """Get a summary of the filesystem context workspace."""
        return self._context_service.get_workspace_summary()

    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about the filesystem context workspace."""
        return self._context_service.get_context_stats()

    # ===================================================================
    # Cleanup
    # ===================================================================

    def __del__(self):
        """Cleanup on agent deletion."""
        if hasattr(self, '_notebook_executor') and self._notebook_executor:
            try:
                self._notebook_executor.shutdown()
            except Exception:
                pass
        if hasattr(self, '_filesystem_context') and self._filesystem_context:
            try:
                self._filesystem_context.cleanup_session(keep_summary=True)
            except Exception:
                pass


# =====================================================================
# Module-level API
# =====================================================================


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

def Agent(model: str = "gpt-5.2", api_key: Optional[str] = None, endpoint: Optional[str] = None, auth_mode: str = "environment", auth_file: Optional[str] = None, enable_reflection: bool = True, reflection_iterations: int = 1, enable_result_review: bool = True, use_notebook_execution: bool = True, max_prompts_per_session: int = 5, notebook_storage_dir: Optional[str] = None, keep_execution_notebooks: bool = True, notebook_timeout: int = 600, strict_kernel_validation: bool = True, enable_filesystem_context: bool = True, context_storage_dir: Optional[str] = None, approval_mode: str = "never", agent_mode: str = "agentic", max_agent_turns: int = 15, security_level: Optional[str] = None, *, config: Optional[AgentConfig] = None, reporter: Optional[Reporter] = None, verbose: bool = True) -> OmicVerseAgent:
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
    auth_mode : str, optional
        Authentication mode. Use ``"openai_oauth"`` to reuse saved Jarvis/Codex login state.
    auth_file : str, optional
        Path to the saved Jarvis auth file. Defaults to ``~/.ovjarvis/auth.json``.
    enable_reflection : bool, optional
        Enable reflection step to review and improve generated code (default: True)
    reflection_iterations : int, optional
        Maximum number of reflection iterations (default: 1, range: 1-3)
    enable_result_review : bool, optional
        Enable result review to validate output matches user intent (default: True)
    use_notebook_execution : bool, optional
        Execute code in separate Jupyter notebook for isolation and debugging (default: True).
    max_prompts_per_session : int, optional
        Number of prompts to execute in same notebook session before restart (default: 5).
    notebook_storage_dir : str, optional
        Directory to store session notebooks. Defaults to ~/.ovagent/sessions
    keep_execution_notebooks : bool, optional
        Whether to keep session notebooks after execution (default: True)
    notebook_timeout : int, optional
        Execution timeout in seconds (default: 600)
    strict_kernel_validation : bool, optional
        If True, raise error if kernel not found. If False, fall back to python3 (default: True)
    enable_filesystem_context : bool, optional
        Enable filesystem-based context management. Default: True.
    context_storage_dir : str, optional
        Directory for storing context files. Defaults to ~/.ovagent/context/
    approval_mode : str, optional
        When to prompt the user before executing generated code.
    config : AgentConfig, optional
        Grouped configuration object.
    reporter : Reporter, optional
        Structured event reporter.
    verbose : bool, optional
        Whether to emit events to stdout (default: True).

    Returns
    -------
    OmicVerseAgent
        Configured agent instance ready for use

    Examples
    --------
    >>> import omicverse as ov
    >>> agent = ov.Agent(model="gpt-5", api_key="your-key")
    >>>
    >>> # Reuse a saved ChatGPT/Codex login
    >>> agent = ov.Agent(model="gpt-5.3-codex", auth_mode="openai_oauth")
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
    """
    return OmicVerseAgent(
        model=model,
        api_key=api_key,
        endpoint=endpoint,
        auth_mode=auth_mode,
        auth_file=auth_file,
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

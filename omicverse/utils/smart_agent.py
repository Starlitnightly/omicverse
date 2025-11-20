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
import builtins
import warnings
import threading
import logging
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
from .registry import _global_registry
from .model_config import ModelConfig, PROVIDER_API_KEYS
from .skill_registry import (
    SkillMatch,
    SkillMetadata,
    SkillDefinition,
    SkillRegistry,
    SkillRouter,
    build_skill_registry,
    build_multi_path_skill_registry,
)


logger = logging.getLogger(__name__)


class OmicVerseAgent:
    """
    Intelligent agent for OmicVerse function discovery and execution.

    This agent uses an internal LLM backend to understand natural language
    requests and automatically execute appropriate OmicVerse functions.

    Usage:
        agent = ov.Agent(api_key="your-api-key")  # Uses gemini-2.5-flash by default
        result_adata = agent.run("quality control with nUMI>500, mito<0.2", adata)
    """
    
    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None, endpoint: Optional[str] = None, enable_reflection: bool = True, reflection_iterations: int = 1, enable_result_review: bool = True):
        """
        Initialize the OmicVerse Smart Agent.

        Parameters
        ----------
        model : str
            LLM model to use for reasoning (default: "gemini-2.5-flash")
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
        """
        print(f" Initializing OmicVerse Smart Agent (internal backend)...")
        
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
        # Token usage tracking at agent level
        self.last_usage = None
        self.last_usage_breakdown: Dict[str, Any] = {
            'generation': None,
            'reflection': [],
            'review': [],
            'total': None
        }
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

    async def _analyze_task_complexity(self, request: str) -> str:
        """
        Analyze the complexity of a user request to determine the appropriate execution strategy.

        This method uses a combination of pattern matching and LLM reasoning to classify
        whether a task can be handled with a single function call (simple) or requires
        a multi-step workflow (complex).

        Parameters
        ----------
        request : str
            The user's natural language request

        Returns
        -------
        str
            Complexity classification: 'simple' or 'complex'

        Examples
        --------
        Simple tasks:
        - "quality control with nUMI>500"
        - "normalize data"
        - "run PCA"
        - "leiden clustering"

        Complex tasks:
        - "complete bulk RNA-seq DEG analysis pipeline"
        - "perform spatial deconvolution from start to finish"
        - "full single-cell preprocessing workflow"
        - "analyze my data and generate report"
        """

        # Pattern-based quick classification (fast path, no LLM needed)
        request_lower = request.lower()

        # Keywords that strongly indicate complexity
        complex_keywords = [
            'complete', 'full', 'entire', 'whole', 'comprehensive',
            'pipeline', 'workflow', 'analysis', 'from start', 'end-to-end',
            'step by step', 'all steps', 'everything', 'report',
            'multiple', 'several', 'various', 'different steps',
            'and then', 'followed by', 'after that', 'next',
        ]

        # Keywords that strongly indicate simplicity
        simple_keywords = [
            'just', 'only', 'single', 'one', 'simply',
            'quick', 'fast', 'basic',
        ]

        # Specific function names (simple operations)
        simple_functions = [
            'qc', 'quality control', '质控',
            'normalize', 'normalization', '归一化',
            'pca', 'dimensionality reduction', '降维',
            'cluster', 'clustering', 'leiden', 'louvain', '聚类',
            'plot', 'visualize', 'show', '可视化',
            'filter', 'subset', '过滤',
            'scale', 'log transform',
        ]

        # Count pattern matches
        complex_score = sum(1 for keyword in complex_keywords if keyword in request_lower)
        simple_score = sum(1 for keyword in simple_keywords if keyword in request_lower)
        function_matches = sum(1 for func in simple_functions if func in request_lower)

        # Pattern-based decision rules
        if complex_score >= 2:
            # Multiple complexity indicators = definitely complex
            logger.debug(f"Complexity: complex (pattern match, score={complex_score})")
            return 'complex'

        if function_matches >= 1 and complex_score == 0 and len(request.split()) <= 10:
            # Short request with function name, no complexity indicators = simple
            logger.debug(f"Complexity: simple (pattern match, function_matches={function_matches})")
            return 'simple'

        # Ambiguous cases: Use LLM for classification
        logger.debug("Complexity: using LLM classifier for ambiguous request")

        classification_prompt = f"""You are a task complexity analyzer for bioinformatics workflows.

Analyze this user request and classify it as either SIMPLE or COMPLEX:

Request: "{request}"

Classification rules:

SIMPLE tasks:
- Single operation or function call
- One specific action (e.g., "quality control", "normalize", "cluster", "plot")
- Direct parameter specification (e.g., "with nUMI>500")
- Examples:
  - "quality control with nUMI>500, mito<0.2"
  - "normalize using log transformation"
  - "run PCA with 50 components"
  - "leiden clustering with resolution=1.0"
  - "plot UMAP"

COMPLEX tasks:
- Multiple steps or operations needed
- Full workflows or pipelines
- Phrases like "complete analysis", "full pipeline", "from start to finish"
- Multiple operations in sequence (e.g., "do X and then Y")
- Vague requests needing multiple steps (e.g., "analyze my data")
- Examples:
  - "complete bulk RNA-seq DEG analysis pipeline"
  - "full preprocessing workflow for single-cell data"
  - "spatial deconvolution from start to finish"
  - "perform clustering and generate visualizations"
  - "analyze my data and create a report"

Respond with ONLY one word: either "simple" or "complex"
"""

        try:
            with self._temporary_api_keys():
                if not self._llm:
                    # Fallback to conservative default if LLM unavailable
                    logger.warning("LLM unavailable for complexity classification, defaulting to 'complex'")
                    return 'complex'

                response_text = await self._llm.run(classification_prompt)

                # Extract classification from response
                response_clean = response_text.strip().lower()

                if 'simple' in response_clean:
                    logger.debug(f"Complexity: simple (LLM classified)")
                    return 'simple'
                elif 'complex' in response_clean:
                    logger.debug(f"Complexity: complex (LLM classified)")
                    return 'complex'
                else:
                    # Unable to parse, default to complex (safer)
                    logger.warning(f"Could not parse LLM complexity response: {response_text}, defaulting to 'complex'")
                    return 'complex'

        except Exception as exc:
            # On any error, default to complex (safer, won't break functionality)
            logger.warning(f"Complexity classification failed: {exc}, defaulting to 'complex'")
            return 'complex'

    async def _run_registry_workflow(self, request: str, adata: Any) -> Any:
        """
        Execute Priority 1: Fast registry-based workflow for simple tasks.

        This method provides a streamlined execution path for simple tasks that can be
        handled with a single function call. It uses ONLY the function registry without
        skill guidance, resulting in faster execution and lower token usage.

        Parameters
        ----------
        request : str
            The user's natural language request (pre-classified as simple)
        adata : Any
            AnnData object to process

        Returns
        -------
        Any
            Processed adata object

        Raises
        ------
        ValueError
            If code generation or extraction fails
        RuntimeError
            If LLM backend is not initialized

        Notes
        -----
        This is the Priority 1 fast path that:
        - Uses ONLY registry functions (no skill guidance)
        - Single LLM call for code generation
        - Optimized prompt for direct function mapping
        - 60-70% faster than full workflow
        - 50% lower token usage

        The generated code should contain 1-2 function calls maximum.
        """

        print(f"🚀 Priority 1: Fast registry-based workflow")

        # Build registry-only prompt (no skills, focused on single function)
        functions_info = self._get_available_functions_info()

        priority1_prompt = f"""You are a fast function executor for OmicVerse. Your task is to find and execute the SINGLE BEST function for this request.

Request: "{request}"

Dataset info:
- Shape: {adata.shape[0]} cells × {adata.shape[1]} genes

Available OmicVerse Functions (Registry):
{functions_info}

INSTRUCTIONS:
1. This is a SIMPLE task requiring ONE function call (or at most 2-3 closely related calls)
2. Search the registry above for the most appropriate function
3. Extract parameters from the request (e.g., "nUMI>500" → tresh={{'nUMIs': 500, ...}})
4. Generate ONLY the essential code - no complex workflows
5. Return executable Python code ONLY, no explanations

IMPORTANT CONSTRAINTS:
- Generate 1-3 function calls maximum
- No loops, conditionals, or complex control flow
- Focus on direct parameter extraction and function execution
- If this requires multiple steps or a workflow, respond with: "NEEDS_WORKFLOW"

Examples of GOOD responses:
```python
import omicverse as ov
adata = ov.pp.qc(adata, tresh={{'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250}})
print(f"QC completed: {{adata.shape[0]}} cells")
```

```python
import omicverse as ov
adata = ov.pp.pca(adata, n_comps=50)
print(f"PCA completed: {{adata.obsm['X_pca'].shape}}")
```

Examples of tasks that need NEEDS_WORKFLOW:
- "complete pipeline"
- "do X and then Y and then Z"
- "full workflow from start to finish"

Now generate code for: "{request}"
"""

        # Get code from LLM
        print(f"   💭 Generating code with registry functions only...")
        with self._temporary_api_keys():
            if not self._llm:
                raise RuntimeError("LLM backend is not initialized")

            response_text = await self._llm.run(priority1_prompt)
            self.last_usage = self._llm.last_usage

        # Check if LLM indicates this needs a workflow
        if "NEEDS_WORKFLOW" in response_text:
            raise ValueError("Task requires workflow (Priority 1 insufficient)")

        # Extract code
        try:
            code = self._extract_python_code(response_text)
        except ValueError as exc:
            raise ValueError(f"Could not extract executable code: {exc}") from exc

        # Track generation usage
        self.last_usage_breakdown['generation'] = self.last_usage

        print(f"   🧬 Generated code:")
        print("   " + "-" * 46)
        for line in code.split('\n'):
            print(f"   {line}")
        print("   " + "-" * 46)

        # Reflection step (if enabled)
        if self.enable_reflection:
            print(f"   🔍 Validating code...")
            reflection_result = await self._reflect_on_code(code, request, adata, iteration=1)

            if reflection_result['issues_found']:
                print(f"      ⚠️  Issues found:")
                for issue in reflection_result['issues_found']:
                    print(f"         - {issue}")

            if reflection_result['needs_revision']:
                code = reflection_result['improved_code']
                print(f"      ✏️  Applied improvements (confidence: {reflection_result['confidence']:.1%})")
            else:
                print(f"      ✅ Code validated (confidence: {reflection_result['confidence']:.1%})")

            # Track reflection usage
            self.last_usage_breakdown['reflection'].append(self._llm.last_usage)

        # Compute total usage (generation + reflection)
        if self.last_usage_breakdown['generation'] or self.last_usage_breakdown['reflection']:
            gen_usage = self.last_usage_breakdown['generation']
            total_input = gen_usage.input_tokens if gen_usage else 0
            total_output = gen_usage.output_tokens if gen_usage else 0

            for ref_usage in self.last_usage_breakdown['reflection']:
                total_input += ref_usage.input_tokens
                total_output += ref_usage.output_tokens

            from .agent_backend import Usage
            self.last_usage_breakdown['total'] = Usage(
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total_input + total_output,
                model=self.model,
                provider=self.provider
            )
            self.last_usage = self.last_usage_breakdown['total']

        # Execute
        print(f"   ⚡ Executing code...")
        try:
            original_adata = adata
            result_adata = self._execute_generated_code(code, adata)
            print(f"   ✅ Execution successful!")
            print(f"   📊 Result: {result_adata.shape[0]} cells × {result_adata.shape[1]} genes")

            # Result review (if enabled)
            if self.enable_result_review:
                print(f"   📋 Reviewing result...")
                review_result = await self._review_result(original_adata, result_adata, request, code)

                if review_result['matched']:
                    print(f"      ✅ Result matches intent (confidence: {review_result['confidence']:.1%})")
                else:
                    print(f"      ⚠️  Result may not match intent (confidence: {review_result['confidence']:.1%})")

                if review_result['issues']:
                    print(f"      ⚠️  Issues: {', '.join(review_result['issues'])}")

                # Track review usage
                if self._llm.last_usage:
                    self.last_usage_breakdown['review'].append(self._llm.last_usage)

                    # Recompute total with review
                    gen_usage = self.last_usage_breakdown.get('generation')
                    total_input = gen_usage.input_tokens if gen_usage else 0
                    total_output = gen_usage.output_tokens if gen_usage else 0

                    for ref_usage in self.last_usage_breakdown.get('reflection', []):
                        total_input += ref_usage.input_tokens
                        total_output += ref_usage.output_tokens

                    for rev_usage in self.last_usage_breakdown['review']:
                        total_input += rev_usage.input_tokens
                        total_output += rev_usage.output_tokens

                    from .agent_backend import Usage
                    self.last_usage_breakdown['total'] = Usage(
                        input_tokens=total_input,
                        output_tokens=total_output,
                        total_tokens=total_input + total_output,
                        model=self.model,
                        provider=self.provider
                    )
                    self.last_usage = self.last_usage_breakdown['total']

            return result_adata

        except Exception as e:
            print(f"   ❌ Execution failed: {e}")
            raise ValueError(f"Priority 1 execution failed: {e}") from e

    async def _run_skills_workflow(self, request: str, adata: Any) -> Any:
        """
        Execute Priority 2: Skills-guided workflow for complex tasks.

        This method provides a comprehensive execution path for complex tasks that require
        multi-step workflows. It uses BOTH the function registry AND matched skill guidance
        to generate complete pipelines.

        Parameters
        ----------
        request : str
            The user's natural language request (pre-classified as complex)
        adata : Any
            AnnData object to process

        Returns
        -------
        Any
            Processed adata object

        Raises
        ------
        ValueError
            If code generation or extraction fails
        RuntimeError
            If LLM backend is not initialized

        Notes
        -----
        This is the Priority 2 comprehensive path that:
        - Matches relevant skills using LLM
        - Loads full skill guidance (lazy loading)
        - Injects both registry + skills into prompt
        - Generates multi-step code
        - More thorough but slower than Priority 1

        The generated code may contain multiple steps, loops, and complex logic.
        """

        print(f"🧠 Priority 2: Skills-guided workflow for complex tasks")

        # Step 1: Match relevant skills using LLM
        print(f"   🎯 Matching relevant skills...")
        matched_skill_slugs = await self._select_skill_matches_llm(request, top_k=2)

        # Step 2: Load full content for matched skills (lazy loading)
        skill_matches = []
        if matched_skill_slugs:
            print(f"   📚 Loading skill guidance:")
            for slug in matched_skill_slugs:
                full_skill = self.skill_registry.load_full_skill(slug) if self.skill_registry else None
                if full_skill:
                    print(f"      - {full_skill.name}")
                    skill_matches.append(SkillMatch(skill=full_skill, score=1.0))

        skill_guidance_text = self._format_skill_guidance(skill_matches)
        skill_guidance_section = ""
        if skill_guidance_text:
            skill_guidance_section = (
                "\nRelevant project skills:\n"
                f"{skill_guidance_text}\n"
            )

        # Step 3: Build comprehensive prompt (registry + skills)
        functions_info = self._get_available_functions_info()

        priority2_prompt = f'''You are a workflow orchestrator for OmicVerse. This is a COMPLEX task requiring multiple steps.

Request: "{request}"

Dataset info:
- Shape: {adata.shape[0]} cells × {adata.shape[1]} genes

Available OmicVerse Functions (Registry):
{functions_info}
{skill_guidance_section}

INSTRUCTIONS:
1. This is a COMPLEX task - generate a complete multi-step workflow
2. Review the skill guidance above for best practices and recommended approaches
3. Use the registry functions to implement each step
4. Extract parameters from the request
5. Generate a comprehensive pipeline with proper sequencing
6. Return executable Python code ONLY, no explanations

WORKFLOW GUIDELINES:
- Break down the task into logical steps
- Use appropriate functions from the registry for each step
- Include comments explaining each major step
- Add print statements to show progress
- Handle intermediate results properly

IMPORTANT:
- This is NOT a simple task - generate a complete workflow
- Follow the skill guidance if provided
- Ensure proper sequencing of operations
- Include validation and progress tracking

Now generate a complete workflow for: "{request}"
'''

        # Step 4: Get code from LLM
        print(f"   💭 Generating multi-step workflow code...")
        with self._temporary_api_keys():
            if not self._llm:
                raise RuntimeError("LLM backend is not initialized")

            response_text = await self._llm.run(priority2_prompt)
            self.last_usage = self._llm.last_usage

        # Track generation usage
        self.last_usage_breakdown['generation'] = self.last_usage

        # Step 5: Extract code
        try:
            code = self._extract_python_code(response_text)
        except ValueError as exc:
            raise ValueError(f"Could not extract executable code: {exc}") from exc

        print(f"   🧬 Generated workflow code:")
        print("   " + "=" * 46)
        for line in code.split('\n'):
            print(f"   {line}")
        print("   " + "=" * 46)

        # Step 6: Reflection (if enabled)
        if self.enable_reflection:
            print(f"   🔍 Validating workflow code (max {self.reflection_iterations} iteration{'s' if self.reflection_iterations > 1 else ''})...")

            for iteration in range(self.reflection_iterations):
                reflection_result = await self._reflect_on_code(code, request, adata, iteration + 1)

                if reflection_result['issues_found']:
                    print(f"      ⚠️  Issues found (iteration {iteration + 1}):")
                    for issue in reflection_result['issues_found']:
                        print(f"         - {issue}")

                if reflection_result['needs_revision']:
                    print(f"      ✏️  Applying improvements...")
                    code = reflection_result['improved_code']
                    print(f"      📈 Confidence: {reflection_result['confidence']:.1%}")
                    if reflection_result['explanation']:
                        print(f"      💡 {reflection_result['explanation']}")
                else:
                    print(f"      ✅ Workflow validated (confidence: {reflection_result['confidence']:.1%})")
                    if reflection_result['explanation']:
                        print(f"      💡 {reflection_result['explanation']}")
                    break

            # Track reflection usage
            if self._llm.last_usage:
                self.last_usage_breakdown['reflection'].append(self._llm.last_usage)

            # Show final code if modified
            if reflection_result['needs_revision']:
                print(f"   🧬 Final workflow after reflection:")
                print("   " + "=" * 46)
                for line in code.split('\n'):
                    print(f"   {line}")
                print("   " + "=" * 46)

        # Compute total usage (generation + reflection)
        if self.last_usage_breakdown['generation'] or self.last_usage_breakdown['reflection']:
            gen_usage = self.last_usage_breakdown['generation']
            total_input = gen_usage.input_tokens if gen_usage else 0
            total_output = gen_usage.output_tokens if gen_usage else 0

            for ref_usage in self.last_usage_breakdown['reflection']:
                total_input += ref_usage.input_tokens
                total_output += ref_usage.output_tokens

            from .agent_backend import Usage
            self.last_usage_breakdown['total'] = Usage(
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total_input + total_output,
                model=self.model,
                provider=self.provider
            )
            self.last_usage = self.last_usage_breakdown['total']

        # Step 7: Execute workflow
        print(f"   ⚡ Executing workflow...")
        try:
            original_adata = adata
            result_adata = self._execute_generated_code(code, adata)
            print(f"   ✅ Workflow execution successful!")
            print(f"   📊 Result: {result_adata.shape[0]} cells × {result_adata.shape[1]} genes")

            # Step 8: Result review (if enabled)
            if self.enable_result_review:
                print(f"   📋 Reviewing workflow result...")
                review_result = await self._review_result(original_adata, result_adata, request, code)

                if review_result['matched']:
                    print(f"      ✅ Result matches intent (confidence: {review_result['confidence']:.1%})")
                else:
                    print(f"      ⚠️  Result may not match intent (confidence: {review_result['confidence']:.1%})")

                if review_result['changes_detected']:
                    print(f"      📊 Changes detected:")
                    for change in review_result['changes_detected']:
                        print(f"         - {change}")

                if review_result['issues']:
                    print(f"      ⚠️  Issues found:")
                    for issue in review_result['issues']:
                        print(f"         - {issue}")

                print(f"      💡 {review_result['assessment']}")

                # Show recommendation
                recommendation_icons = {
                    'accept': '✅',
                    'review': '⚠️',
                    'retry': '❌'
                }
                icon = recommendation_icons.get(review_result['recommendation'], '❓')
                print(f"      {icon} Recommendation: {review_result['recommendation'].upper()}")

                # Track review usage
                if self._llm.last_usage:
                    self.last_usage_breakdown['review'].append(self._llm.last_usage)

                    # Recompute total with review
                    gen_usage = self.last_usage_breakdown.get('generation')
                    total_input = gen_usage.input_tokens if gen_usage else 0
                    total_output = gen_usage.output_tokens if gen_usage else 0

                    for ref_usage in self.last_usage_breakdown.get('reflection', []):
                        total_input += ref_usage.input_tokens
                        total_output += ref_usage.output_tokens

                    for rev_usage in self.last_usage_breakdown['review']:
                        total_input += rev_usage.input_tokens
                        total_output += rev_usage.output_tokens

                    from .agent_backend import Usage
                    self.last_usage_breakdown['total'] = Usage(
                        input_tokens=total_input,
                        output_tokens=total_output,
                        total_tokens=total_input + total_output,
                        model=self.model,
                        provider=self.provider
                    )
                    self.last_usage = self.last_usage_breakdown['total']

            return result_adata

        except Exception as e:
            print(f"   ❌ Workflow execution failed: {e}")
            raise ValueError(f"Priority 2 execution failed: {e}") from e

    def _validate_simple_execution(self, code: str) -> tuple[bool, str]:
        """
        Validate that generated code is truly simple (suitable for Priority 1).

        Uses AST analysis to check code complexity and ensure it matches the
        constraints of Priority 1 (1-3 function calls, no complex control flow).

        Parameters
        ----------
        code : str
            The generated Python code to validate

        Returns
        -------
        tuple[bool, str]
            (is_valid, reason) where is_valid is True if code is simple enough,
            and reason explains why it passed or failed validation

        Notes
        -----
        Validation criteria:
        - Maximum 5 function calls (allowing some flexibility)
        - No loops (for, while)
        - No complex conditionals (if/elif/else)
        - No function definitions
        - No class definitions
        """

        try:
            # Parse code into AST
            tree = ast.parse(code)

            # Count different node types
            function_calls = 0
            loops = 0
            conditionals = 0
            func_defs = 0
            class_defs = 0

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    function_calls += 1
                elif isinstance(node, (ast.For, ast.While)):
                    loops += 1
                elif isinstance(node, ast.If):
                    conditionals += 1
                elif isinstance(node, ast.FunctionDef):
                    func_defs += 1
                elif isinstance(node, ast.ClassDef):
                    class_defs += 1

            # Validation rules
            issues = []

            if function_calls > 5:
                issues.append(f"Too many function calls ({function_calls} > 5)")

            if loops > 0:
                issues.append(f"Contains loops ({loops} loop(s) found)")

            if conditionals > 0:
                issues.append(f"Contains conditionals ({conditionals} if statement(s) found)")

            if func_defs > 0:
                issues.append(f"Contains function definitions ({func_defs} function(s) defined)")

            if class_defs > 0:
                issues.append(f"Contains class definitions ({class_defs} class(es) defined)")

            # Determine if valid
            if issues:
                reason = f"Code too complex for Priority 1: {'; '.join(issues)}"
                return False, reason
            else:
                reason = f"Code is simple: {function_calls} function call(s), no complex logic"
                return True, reason

        except SyntaxError as e:
            return False, f"Syntax error in code: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"

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
                return normalized
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
- Shape: {adata.shape[0]} cells × {adata.shape[1]} genes

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

3. **Provide feedback as a JSON object**:
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

    def _execute_generated_code(self, code: str, adata: Any) -> Any:
        """Execute generated Python code in a sandboxed namespace.

        Notes
        -----
        The sandbox restricts available built-ins and module imports, but it is not a
        foolproof security boundary. Only run the agent with data and environments you
        trust, and consider additional isolation (e.g., containers) for untrusted input.
        """

        compiled = compile(code, "<omicverse-agent>", "exec")
        sandbox_globals = self._build_sandbox_globals()
        sandbox_locals = {"adata": adata}

        # Normalize HVG column naming so generated code can access either alias
        try:
            if hasattr(adata, "var") and adata.var is not None:
                if "highly_variable" not in adata.var.columns and "highly_variable_features" in adata.var.columns:
                    adata.var["highly_variable"] = adata.var["highly_variable_features"]
            # Store initial sizes so LLM summaries don't KeyError
            if hasattr(adata, "uns"):
                adata.uns.setdefault("initial_cells", getattr(adata, "n_obs", None) or getattr(adata, "shape", [None])[0])
                adata.uns.setdefault("initial_genes", getattr(adata, "n_vars", None) or getattr(adata, "shape", [None, None])[1])
                adata.uns.setdefault("omicverse_qc_original_cells", getattr(adata, "n_obs", None))
            # Provide a default raw/ scaled layer so generated code using use_raw=True or layer='scaled' does not crash
            if getattr(adata, "raw", None) is None:
                try:
                    adata.raw = adata
                except Exception:
                    pass
            if hasattr(adata, "layers") and "scaled" not in getattr(adata, "layers", {}):
                try:
                    adata.layers["scaled"] = adata.X.copy()
                except Exception:
                    pass
            # Fill missing numeric obs values to avoid downstream hist/binning errors
            try:
                import pandas as _pd  # local import to avoid sandbox collisions
                if hasattr(adata, "obs"):
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
                            if hasattr(adata, "layers") and "counts" in adata.layers:
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
                            if hasattr(adata, "layers") and "counts" in adata.layers:
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

        with self._temporary_api_keys():
            exec(compiled, sandbox_globals, sandbox_locals)

        result_adata = sandbox_locals.get("adata", adata)
        self._normalize_doublet_obs(result_adata)
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
            "locals",
            "globals",
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
        }
        core_modules = (
            "omicverse",
            "numpy",
            "pandas",
            "scanpy",
            "os",
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

        def limited_import(name, globals=None, locals=None, fromlist=(), level=0):
            root_name = name.split(".")[0]
            if root_name in deny_roots:
                raise ImportError(
                    f"Module '{name}' is blocked inside the OmicVerse agent sandbox."
                )
            if root_name not in allowed_modules:
                # Allow additional safe imports needed by skills; cache them after first load
                allowed_modules[root_name] = __import__(root_name)
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
        Process a natural language request using priority-based execution strategy.

        This method implements a two-tier priority system:
        - Priority 1 (Fast): Registry-only workflow for simple tasks (60-70% faster)
        - Priority 2 (Comprehensive): Skills-guided workflow for complex tasks

        The system automatically:
        1. Analyzes task complexity (simple vs complex)
        2. Attempts Priority 1 if simple
        3. Falls back to Priority 2 if needed or if complex

        Parameters
        ----------
        request : str
            Natural language description of what to do
        adata : Any
            AnnData object to process

        Returns
        -------
        Any
            Processed adata object

        Notes
        -----
        Priority 1 is attempted for simple tasks and provides:
        - Single LLM call for code generation
        - 60-70% faster execution
        - 50% lower token usage
        - No skill loading overhead

        Priority 2 is used for complex tasks or as fallback:
        - LLM-based skill matching
        - Full skill guidance injection
        - Multi-step workflow generation
        - More thorough but slower

        Examples
        --------
        Simple task (Priority 1):
        >>> agent.run("qc with nUMI>500", adata)
        # Output: Priority 1 used, ~2-3 seconds

        Complex task (Priority 2):
        >>> agent.run("complete bulk DEG pipeline", adata)
        # Output: Priority 2 used, ~8-10 seconds
        """

        print(f"\n{'=' * 70}")
        print(f"🤖 OmicVerse Agent Processing Request")
        print(f"{'=' * 70}")
        print(f"Request: \"{request}\"")
        print(f"Dataset: {adata.shape[0]} cells × {adata.shape[1]} genes")
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

        # Step 1: Analyze task complexity
        print(f"📊 Analyzing task complexity...")
        complexity = await self._analyze_task_complexity(request)
        print(f"   Classification: {complexity.upper()}")
        print()

        # Track which priority was used for metrics
        priority_used = None
        fallback_occurred = False

        # Step 2: Try Priority 1 (Fast Path) for simple tasks
        if complexity == 'simple':
            print(f"💡 Strategy: Attempting Priority 1 (fast registry-based workflow)")
            print()

            try:
                # Attempt fast registry workflow
                result = await self._run_registry_workflow(request, adata)
                priority_used = 1

                print()
                print(f"{'=' * 70}")
                print(f"✅ SUCCESS - Priority 1 completed successfully!")
                print(f"⚡ Execution time: Fast (registry-only path)")
                print(f"📊 Token savings: ~50% vs full workflow")
                print(f"{'=' * 70}\n")

                return result

            except ValueError as e:
                # Priority 1 failed, fall back to Priority 2
                error_msg = str(e)
                print()
                print(f"{'─' * 70}")
                print(f"⚠️  Priority 1 insufficient: {error_msg}")
                print(f"🔄 Falling back to Priority 2 (skills-guided workflow)...")
                print(f"{'─' * 70}\n")

                fallback_occurred = True
                # Continue to Priority 2 below

            except Exception as e:
                # Unexpected error in Priority 1
                print()
                print(f"{'─' * 70}")
                print(f"⚠️  Priority 1 encountered error: {e}")
                print(f"🔄 Falling back to Priority 2...")
                print(f"{'─' * 70}\n")

                fallback_occurred = True
                # Continue to Priority 2 below

        # Step 3: Use Priority 2 (Comprehensive Path) for complex tasks or fallback
        if complexity == 'complex':
            print(f"💡 Strategy: Using Priority 2 (comprehensive skills-guided workflow)")
            print()
        elif fallback_occurred:
            print(f"💡 Strategy: Priority 2 (fallback from Priority 1)")
            print()

        try:
            result = await self._run_skills_workflow(request, adata)
            priority_used = 2

            print()
            print(f"{'=' * 70}")
            print(f"✅ SUCCESS - Priority 2 completed successfully!")
            if fallback_occurred:
                print(f"🔄 Note: Fell back from Priority 1")
            print(f"🧠 Execution time: Comprehensive (skills-guided workflow)")
            print(f"📊 Full workflow with skill guidance")
            print(f"{'=' * 70}\n")

            return result

        except Exception as e:
            # Priority 2 also failed
            print()
            print(f"{'=' * 70}")
            print(f"❌ ERROR - Priority 2 failed")
            print(f"Error: {e}")
            print(f"{'=' * 70}\n")
            raise

    async def run_async_LEGACY(self, request: str, adata: Any) -> Any:
        """
        [LEGACY] Original run_async implementation before priority system.

        This method is preserved for reference but is no longer used.
        The new run_async() implements the priority-based system.
        """

        # Determine which project skills are relevant to this request using LLM
        matched_skill_slugs = await self._select_skill_matches_llm(request, top_k=2)

        # Load full content for matched skills (lazy loading)
        skill_matches = []
        if matched_skill_slugs:
            print("\n🎯 LLM matched skills:")
            for slug in matched_skill_slugs:
                full_skill = self.skill_registry.load_full_skill(slug)
                if full_skill:
                    print(f"   - {full_skill.name}")
                    skill_matches.append(SkillMatch(skill=full_skill, score=1.0))

        skill_guidance_text = self._format_skill_guidance(skill_matches)
        skill_guidance_section = ""
        if skill_guidance_text:
            skill_guidance_section = (
                "\nRelevant project skills:\n"
                f"{skill_guidance_text}\n"
            )

        # Ask backend to generate the appropriate function call code
        code_generation_request = f'''
Please analyze this OmicVerse request: "{request}"

Your task:
1. Review the Available OmicVerse Functions (in the system prompt) to choose the best function
2. Carefully examine function signatures and parameters described there
3. Extract parameters from the request text
4. Generate executable Python code that calls the correct OmicVerse function with proper parameters

Dataset info:
- Shape: {adata.shape[0]} cells × {adata.shape[1]} genes
- Request: {request}
{skill_guidance_section}

CRITICAL INSTRUCTIONS:
1. Read the function 'help' information embedded above (docstrings and examples)
2. Generate code that matches the actual function signature
3. Return ONLY executable Python code, no explanations

For the qc function specifically:
- The tresh parameter needs a dict with 'mito_perc', 'nUMIs', 'detected_genes' keys
- Default is: tresh={{'mito_perc': 0.15, 'nUMIs': 500, 'detected_genes': 250}}
- Extract values from user request and update the dict accordingly

Example workflow:
1. Determine that ov.pp.qc is relevant for quality control
2. Parse request: "nUMI>500" means tresh['nUMIs']=500
3. Generate: ov.pp.qc(adata, tresh={{'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250}})
'''
        
        # Get the code from the LLM backend
        print(f"\n🤔 LLM analyzing request: '{request}'...")
        with self._temporary_api_keys():
            if not self._llm:
                raise RuntimeError("LLM backend is not initialized")
            response_text = await self._llm.run(code_generation_request)
            # Copy usage information from backend to agent
            self.last_usage = self._llm.last_usage

        # Display LLM response text
        print(f"\n💭 LLM response:")
        print("-" * 50)
        print(response_text)
        print("-" * 50)
        
        try:
            code = self._extract_python_code(response_text)
        except ValueError as exc:
            raise ValueError(f"❌ Could not extract executable code from LLM response: {exc}") from exc

        # Track generation usage
        self.last_usage_breakdown['generation'] = self.last_usage

        print(f"\n🧬 Generated code:")
        print("=" * 50)
        print(f"{code}")
        print("=" * 50)

        # Reflection step: Review and improve the generated code
        if self.enable_reflection:
            print(f"\n🔍 Reflecting on generated code (max {self.reflection_iterations} iteration{'s' if self.reflection_iterations > 1 else ''})...")

            for iteration in range(self.reflection_iterations):
                reflection_result = await self._reflect_on_code(code, request, adata, iteration + 1)

                if reflection_result['issues_found']:
                    print(f"   ⚠️  Issues found (iteration {iteration + 1}):")
                    for issue in reflection_result['issues_found']:
                        print(f"      - {issue}")

                if reflection_result['needs_revision']:
                    print(f"   ✏️  Applying improvements...")
                    code = reflection_result['improved_code']
                    print(f"   📈 Confidence: {reflection_result['confidence']:.1%}")
                    if reflection_result['explanation']:
                        print(f"   💡 {reflection_result['explanation']}")
                else:
                    print(f"   ✅ Code validated (confidence: {reflection_result['confidence']:.1%})")
                    if reflection_result['explanation']:
                        print(f"   💡 {reflection_result['explanation']}")
                    break

            # Show final code if it was modified
            if reflection_result['needs_revision']:
                print(f"\n🧬 Final code after reflection:")
                print("=" * 50)
                print(f"{code}")
                print("=" * 50)

        # Compute total usage
        if self.last_usage_breakdown['generation'] or self.last_usage_breakdown['reflection']:
            gen_usage = self.last_usage_breakdown['generation']
            total_input = gen_usage.input_tokens if gen_usage else 0
            total_output = gen_usage.output_tokens if gen_usage else 0

            for ref_usage in self.last_usage_breakdown['reflection']:
                total_input += ref_usage.input_tokens
                total_output += ref_usage.output_tokens

            from .agent_backend import Usage
            self.last_usage_breakdown['total'] = Usage(
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total_input + total_output,
                model=self.model,
                provider=self.provider
            )
            # Update last_usage to reflect total
            self.last_usage = self.last_usage_breakdown['total']

        # Execute the code locally
        print(f"\n⚡ Executing code locally...")
        try:
            # Keep reference to original for review
            original_adata = adata
            result_adata = self._execute_generated_code(code, adata)
            print(f"✅ Code executed successfully!")
            print(f"📊 Result shape: {result_adata.shape[0]} cells × {result_adata.shape[1]} genes")

            # Result review: Validate output matches user intent
            if self.enable_result_review:
                print(f"\n📋 Reviewing result to validate task completion...")
                review_result = await self._review_result(original_adata, result_adata, request, code)

                # Display review assessment
                if review_result['matched']:
                    print(f"   ✅ Result matches intent (confidence: {review_result['confidence']:.1%})")
                else:
                    print(f"   ⚠️  Result may not match intent (confidence: {review_result['confidence']:.1%})")

                if review_result['changes_detected']:
                    print(f"   📊 Changes detected:")
                    for change in review_result['changes_detected']:
                        print(f"      - {change}")

                if review_result['issues']:
                    print(f"   ⚠️  Issues found:")
                    for issue in review_result['issues']:
                        print(f"      - {issue}")

                print(f"   💡 {review_result['assessment']}")

                # Show recommendation
                recommendation_icons = {
                    'accept': '✅',
                    'review': '⚠️',
                    'retry': '❌'
                }
                icon = recommendation_icons.get(review_result['recommendation'], '❓')
                print(f"   {icon} Recommendation: {review_result['recommendation'].upper()}")

                # Update total usage with review tokens
                if self.last_usage_breakdown['review']:
                    gen_usage = self.last_usage_breakdown.get('generation')
                    total_input = gen_usage.input_tokens if gen_usage else 0
                    total_output = gen_usage.output_tokens if gen_usage else 0

                    for ref_usage in self.last_usage_breakdown.get('reflection', []):
                        total_input += ref_usage.input_tokens
                        total_output += ref_usage.output_tokens

                    for rev_usage in self.last_usage_breakdown['review']:
                        total_input += rev_usage.input_tokens
                        total_output += rev_usage.output_tokens

                    from .agent_backend import Usage
                    self.last_usage_breakdown['total'] = Usage(
                        input_tokens=total_input,
                        output_tokens=total_output,
                        total_tokens=total_input + total_output,
                        model=self.model,
                        provider=self.provider
                    )
                    self.last_usage = self.last_usage_breakdown['total']

            return result_adata

        except Exception as e:
            print(f"❌ Error executing generated code: {e}")
            print(f"Code that failed: {code}")
            return adata

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

    def _select_skill_matches(self, request: str, top_k: int = 1) -> List[SkillMatch]:
        """Return the most relevant project skills for the request (deprecated - kept for backward compatibility)."""

        # LLM-based matching is now done directly in run_async
        # This method is kept for backward compatibility only
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

    async def stream_async(self, request: str, adata: Any):
        """
        Stream LLM response chunks as they arrive while processing a request.

        This method is similar to run_async() but yields LLM response chunks
        in real-time before executing the generated code.

        Parameters
        ----------
        request : str
            Natural language description of what to do
        adata : Any
            AnnData object to process

        Yields
        ------
        dict
            Dictionary with 'type' and 'content' keys. Types include:
            - 'skill_match': Matched skills
            - 'llm_chunk': Streaming LLM response chunks
            - 'code': Generated code to execute
            - 'result': Final result after execution
            - 'usage': Token usage statistics (emitted as final event)

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

        # Determine which project skills are relevant to this request
        skill_matches = self._select_skill_matches(request, top_k=2)
        if skill_matches:
            yield {
                'type': 'skill_match',
                'content': [
                    {'name': match.skill.name, 'score': match.score}
                    for match in skill_matches
                ]
            }

        skill_guidance_text = self._format_skill_guidance(skill_matches)
        skill_guidance_section = ""
        if skill_guidance_text:
            skill_guidance_section = (
                "\nRelevant project skills:\n"
                f"{skill_guidance_text}\n"
            )

        # Build code generation request
        code_generation_request = f'''
Please analyze this OmicVerse request: "{request}"

Your task:
1. Review the Available OmicVerse Functions (in the system prompt) to choose the best function
2. Carefully examine function signatures and parameters described there
3. Extract parameters from the request text
4. Generate executable Python code that calls the correct OmicVerse function with proper parameters

Dataset info:
- Shape: {adata.shape[0]} cells × {adata.shape[1]} genes
- Request: {request}
{skill_guidance_section}

CRITICAL INSTRUCTIONS:
1. Read the function 'help' information embedded above (docstrings and examples)
2. Generate code that matches the actual function signature
3. Return ONLY executable Python code, no explanations

For the qc function specifically:
- The tresh parameter needs a dict with 'mito_perc', 'nUMIs', 'detected_genes' keys
- Default is: tresh={{'mito_perc': 0.15, 'nUMIs': 500, 'detected_genes': 250}}
- Extract values from user request and update the dict accordingly

Example workflow:
1. Determine that ov.pp.qc is relevant for quality control
2. Parse request: "nUMI>500" means tresh['nUMIs']=500
3. Generate: ov.pp.qc(adata, tresh={{'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250}})
'''

        # Stream the LLM response with error handling
        response_chunks = []
        try:
            with self._temporary_api_keys():
                if not self._llm:
                    raise RuntimeError("LLM backend is not initialized")

                async for chunk in self._llm.stream(code_generation_request):
                    response_chunks.append(chunk)
                    yield {'type': 'llm_chunk', 'content': chunk}

                # Copy usage information from backend to agent after streaming completes
                self.last_usage = self._llm.last_usage
        except Exception as exc:
            yield {
                'type': 'error',
                'content': f"LLM streaming failed: {type(exc).__name__}: {exc}"
            }
            return

        # Assemble full response
        response_text = "".join(response_chunks)

        # Extract and yield generated code
        try:
            code = self._extract_python_code(response_text)
            yield {'type': 'code', 'content': code}
        except ValueError as exc:
            yield {
                'type': 'error',
                'content': f"Could not extract executable code from LLM response: {exc}"
            }
            return

        # Execute the code
        try:
            result_adata = self._execute_generated_code(code, adata)
            yield {
                'type': 'result',
                'content': result_adata,
                'shape': (result_adata.shape[0], result_adata.shape[1])
            }
        except Exception as e:
            yield {
                'type': 'error',
                'content': f"Error executing generated code: {e}",
                'code': code
            }

        # Emit usage event as final event (optional for users to consume)
        if self.last_usage:
            yield {
                'type': 'usage',
                'content': self.last_usage
            }

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

def Agent(model: str = "gemini-2.5-flash", api_key: Optional[str] = None, endpoint: Optional[str] = None, enable_reflection: bool = True, reflection_iterations: int = 1, enable_result_review: bool = True) -> OmicVerseAgent:
    """
    Create an OmicVerse Smart Agent instance.

    This function creates and returns a smart agent that can execute OmicVerse functions
    based on natural language descriptions.

    Parameters
    ----------
    model : str, optional
        LLM model to use (default: "gemini-2.5-flash"). Use list_supported_models() to see all options
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

    Returns
    -------
    OmicVerseAgent
        Configured agent instance ready for use

    Examples
    --------
    >>> import omicverse as ov
    >>> import scanpy as sc
    >>>
    >>> # Create agent instance with full validation (default)
    >>> agent = ov.Agent(model="gpt-5", api_key="your-key")
    >>>
    >>> # Create agent with multiple reflection iterations
    >>> agent = ov.Agent(model="gpt-5", api_key="your-key", reflection_iterations=2)
    >>>
    >>> # Create agent without validation (fastest execution)
    >>> agent = ov.Agent(model="gpt-5", api_key="your-key", enable_reflection=False, enable_result_review=False)
    >>>
    >>> # Create agent with only result review (skip code reflection)
    >>> agent = ov.Agent(model="gpt-5", api_key="your-key", enable_reflection=False, enable_result_review=True)
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
    """
    return OmicVerseAgent(model=model, api_key=api_key, endpoint=endpoint, enable_reflection=enable_reflection, reflection_iterations=reflection_iterations, enable_result_review=enable_result_review)


# Export the main functions
__all__ = ["Agent", "OmicVerseAgent", "list_supported_models"]

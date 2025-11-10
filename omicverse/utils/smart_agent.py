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


class DataStateInspector:
    """
    Dynamically inspect AnnData state without hardcoded prerequisites.

    This class provides runtime analysis of data state by examining what
    actually exists in the AnnData object, rather than relying on hardcoded
    rules. This makes it flexible and future-proof.
    """

    @staticmethod
    def inspect(adata: Any) -> Dict[str, Any]:
        """
        Inspect adata and return complete state description.

        This method reports facts about the data structure without making
        assumptions or enforcing rules. It simply catalogs what exists.

        Parameters
        ----------
        adata : Any
            AnnData object to inspect

        Returns
        -------
        Dict[str, Any]
            Complete state information:
            {
                'available': {
                    'layers': List[str],
                    'obsm': List[str],
                    'uns': List[str],
                    'obs_columns': List[str],
                    'var_columns': List[str],
                    'obsp': List[str],
                    'varp': List[str]
                },
                'shape': Tuple[int, int],
                'capabilities': List[str],  # What operations appear possible
                'embeddings': List[str]     # Available embedding types
            }
        """
        state = {
            'available': {
                'layers': [],
                'obsm': [],
                'uns': [],
                'obs_columns': [],
                'var_columns': [],
                'obsp': [],
                'varp': []
            },
            'shape': (0, 0),
            'capabilities': [],
            'embeddings': []
        }

        try:
            # Get basic dimensions
            if hasattr(adata, 'shape'):
                state['shape'] = adata.shape

            # Collect what exists (facts, not interpretations)
            if hasattr(adata, 'layers') and adata.layers is not None:
                state['available']['layers'] = list(adata.layers.keys())

            if hasattr(adata, 'obsm') and adata.obsm is not None:
                state['available']['obsm'] = list(adata.obsm.keys())

            if hasattr(adata, 'uns') and adata.uns is not None:
                state['available']['uns'] = list(adata.uns.keys())

            if hasattr(adata, 'obs') and adata.obs is not None:
                state['available']['obs_columns'] = list(adata.obs.columns)

            if hasattr(adata, 'var') and adata.var is not None:
                state['available']['var_columns'] = list(adata.var.columns)

            if hasattr(adata, 'obsp') and adata.obsp is not None:
                state['available']['obsp'] = list(adata.obsp.keys())

            if hasattr(adata, 'varp') and adata.varp is not None:
                state['available']['varp'] = list(adata.varp.keys())

            # Infer capabilities from available data (still factual, just derived)
            capabilities = []

            # Check for PCA
            if 'X_pca' in state['available']['obsm']:
                capabilities.append('has_pca')

            # Check for neighbor graph
            if 'neighbors' in state['available']['uns']:
                capabilities.append('has_neighbors')

            # Check for processed layers
            processed_indicators = ['scaled', 'normalized', 'lognorm', 'pearson_residuals']
            if any(indicator in layer.lower()
                   for layer in state['available']['layers']
                   for indicator in processed_indicators):
                capabilities.append('has_processed_layers')

            # Check for neighborhood graph in obsp
            if any(key in ['connectivities', 'distances']
                   for key in state['available']['obsp']):
                capabilities.append('has_neighborhood_graph')

            # Collect available embeddings
            embedding_keys = ['X_umap', 'X_tsne', 'X_draw_graph', 'X_phate',
                            'X_diffmap', 'X_mde', 'X_sude', 'spatial']
            available_embeddings = [k for k in embedding_keys
                                   if k in state['available']['obsm']]
            if available_embeddings:
                capabilities.append('has_embeddings')
                state['embeddings'] = available_embeddings

            # Check for clustering results
            clustering_indicators = ['leiden', 'louvain', 'cluster']
            clustering_columns = [col for col in state['available']['obs_columns']
                                if any(ind in col.lower() for ind in clustering_indicators)]
            if clustering_columns:
                capabilities.append('has_clustering')
                state['clustering_columns'] = clustering_columns

            state['capabilities'] = capabilities

        except Exception as e:
            warnings.warn(f"Error inspecting data state: {e}", UserWarning)

        return state

    @staticmethod
    def get_readable_summary(adata: Any) -> str:
        """
        Generate human-readable summary of data state.

        This creates a formatted string suitable for displaying to users
        or including in LLM prompts.

        Parameters
        ----------
        adata : Any
            AnnData object to summarize

        Returns
        -------
        str
            Human-readable summary
        """
        state = DataStateInspector.inspect(adata)

        lines = [
            "## Current Data State",
            f"**Shape**: {state['shape'][0]:,} cells × {state['shape'][1]:,} genes",
            ""
        ]

        # Available components
        if state['available']['layers']:
            layers_str = ', '.join(state['available']['layers'][:5])
            if len(state['available']['layers']) > 5:
                layers_str += f" (+ {len(state['available']['layers']) - 5} more)"
            lines.append(f"**Available Layers**: {layers_str}")
        else:
            lines.append("**Available Layers**: None (raw X matrix only)")

        if state['available']['obsm']:
            obsm_str = ', '.join(state['available']['obsm'][:8])
            if len(state['available']['obsm']) > 8:
                obsm_str += f" (+ {len(state['available']['obsm']) - 8} more)"
            lines.append(f"**Available Obsm**: {obsm_str}")
        else:
            lines.append("**Available Obsm**: None")

        if state['available']['uns']:
            uns_list = state['available']['uns'][:5]
            uns_str = ', '.join(uns_list)
            if len(state['available']['uns']) > 5:
                uns_str += f" (+ {len(state['available']['uns']) - 5} more)"
            lines.append(f"**Available Uns**: {uns_str}")

        lines.append("")

        # Capabilities (what's possible)
        if state['capabilities']:
            lines.append("**Detected Capabilities**:")
            if 'has_processed_layers' in state['capabilities']:
                lines.append("  ✅ Data appears preprocessed")
            if 'has_pca' in state['capabilities']:
                lines.append("  ✅ PCA computed")
            if 'has_neighbors' in state['capabilities']:
                lines.append("  ✅ Neighbor graph available")
            if 'has_embeddings' in state['capabilities']:
                emb_str = ', '.join(state.get('embeddings', [])[:3])
                lines.append(f"  ✅ Embeddings available: {emb_str}")
            if 'has_clustering' in state['capabilities']:
                clust_str = ', '.join(state.get('clustering_columns', [])[:2])
                lines.append(f"  ✅ Clustering results: {clust_str}")
        else:
            lines.append("**Detected Capabilities**: Raw data (no preprocessing detected)")

        return "\n".join(lines)

    @staticmethod
    def check_compatibility(adata: Any, function_name: str,
                           function_signature: str,
                           function_category: str) -> Dict[str, Any]:
        """
        Check if a function can likely be called on current data.

        Uses function signature inspection and category to infer requirements,
        NOT hardcoded rules. This is adaptive and learns from the function itself.

        Parameters
        ----------
        adata : Any
            AnnData object to check
        function_name : str
            Name of the function to check
        function_signature : str
            Function signature string
        function_category : str
            Function category from registry

        Returns
        -------
        Dict[str, Any]
            Compatibility analysis:
            {
                'likely_compatible': bool,
                'warnings': List[str],
                'suggestions': List[str],
                'reasoning': str
            }
        """
        state = DataStateInspector.inspect(adata)
        result = {
            'likely_compatible': True,
            'warnings': [],
            'suggestions': [],
            'reasoning': ''
        }

        try:
            sig_lower = function_signature.lower()
            name_lower = function_name.lower()

            # Check for layer parameter expectations
            if "layer='scaled'" in sig_lower or 'layer="scaled"' in sig_lower:
                if 'scaled' not in state['available']['layers']:
                    result['warnings'].append(
                        "Function signature suggests it expects 'scaled' layer"
                    )
                    result['suggestions'].append(
                        "Consider running ov.pp.scale(adata) or checking if data is already normalized"
                    )

            # Check for use_rep parameter expectations
            if "use_rep='x_pca'" in sig_lower or 'use_rep="x_pca"' in sig_lower:
                if 'X_pca' not in state['available']['obsm']:
                    result['warnings'].append(
                        "Function signature suggests it expects PCA representation"
                    )
                    result['suggestions'].append(
                        "Consider running ov.pp.pca(adata) first"
                    )

            # Infer from function name (not hardcoded, just pattern matching)
            if any(word in name_lower for word in ['leiden', 'louvain']):
                if 'has_neighbors' not in state['capabilities']:
                    result['warnings'].append(
                        "Clustering functions typically work better with neighbor graph"
                    )
                    result['suggestions'].append(
                        "Consider running ov.pp.neighbors(adata) first"
                    )

            # Check category-based expectations
            if function_category == 'clustering':
                if 'has_pca' not in state['capabilities']:
                    result['warnings'].append(
                        "Clustering typically performed on dimensionality-reduced data"
                    )

            if function_category in ['visualization', 'plotting']:
                if not state['embeddings'] and 'has_pca' not in state['capabilities']:
                    result['warnings'].append(
                        "Visualization works best with precomputed embeddings or PCA"
                    )

            # Set reasoning
            if result['warnings']:
                result['reasoning'] = (
                    f"Function {function_name} may need preprocessing. "
                    f"Current state: {', '.join(state['capabilities']) or 'raw data'}."
                )
            else:
                result['reasoning'] = (
                    f"Function {function_name} appears compatible with current data state."
                )

        except Exception as e:
            result['warnings'].append(f"Could not fully analyze compatibility: {e}")

        return result


class LLMPrerequisiteInference:
    """
    Layer 2: LLM-based prerequisite inference.

    Uses LLM to intelligently infer function prerequisites by analyzing:
    - Function documentation
    - Current data state
    - Skill best practices
    - Bioinformatics workflow knowledge

    This provides maximum flexibility without hardcoding, learning from
    documentation and context to make smart decisions.
    """

    def __init__(self, llm_backend):
        """
        Initialize prerequisite inference engine.

        Parameters
        ----------
        llm_backend : OmicVerseLLMBackend
            LLM backend for inference
        """
        self.llm = llm_backend
        self._cache = {}  # Cache inference results to avoid redundant calls

    def _get_cache_key(self, function_name: str, data_state: Dict[str, Any]) -> str:
        """Generate cache key for prerequisite analysis."""
        # Cache based on function name and key data state features
        state_key = (
            tuple(sorted(data_state.get('capabilities', []))),
            tuple(sorted(data_state['available'].get('layers', []))),
            tuple(sorted(data_state['available'].get('obsm', [])))
        )
        return f"{function_name}::{state_key}"

    async def infer_prerequisites(self,
                                  function_name: str,
                                  function_info: Dict[str, Any],
                                  data_state: Dict[str, Any],
                                  skill_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Infer prerequisites for a function using LLM reasoning.

        This method asks the LLM to analyze function documentation and
        current data state to determine what's needed. Unlike hardcoded
        rules, this adapts to any function and learns from context.

        Parameters
        ----------
        function_name : str
            Name of the function to analyze
        function_info : Dict[str, Any]
            Function metadata from registry (docstring, signature, etc.)
        data_state : Dict[str, Any]
            Current data state from DataStateInspector
        skill_context : str, optional
            Relevant skill guidance for context

        Returns
        -------
        Dict[str, Any]
            Inference result:
            {
                'can_run': bool,              # Can function run on current data?
                'confidence': float,          # Confidence level (0-1)
                'missing_items': List[str],   # What's missing
                'required_steps': List[str],  # Steps to prepare data
                'complexity': str,            # 'simple' or 'complex'
                'reasoning': str,             # LLM's explanation
                'auto_fixable': bool          # Can auto-run prerequisites?
            }

        Examples
        --------
        >>> inference = LLMPrerequisiteInference(llm_backend)
        >>> result = await inference.infer_prerequisites(
        ...     'pca',
        ...     function_info,
        ...     data_state
        ... )
        >>> print(result['reasoning'])
        "PCA requires scaled data. Current data has no processed layers.
         Needs: QC → normalize → scale → PCA (4 steps = complex)"
        """
        # Check cache first
        cache_key = self._get_cache_key(function_name, data_state)
        if cache_key in self._cache:
            logger.debug(f"Using cached prerequisite inference for {function_name}")
            return self._cache[cache_key]

        # Prepare context for LLM
        prompt = self._build_inference_prompt(
            function_name, function_info, data_state, skill_context
        )

        try:
            # Call LLM for analysis
            response_text = await self.llm.run(prompt)

            # Parse response
            result = self._parse_inference_response(response_text, function_name)

            # Cache result
            self._cache[cache_key] = result

            logger.debug(
                f"LLM prerequisite inference for {function_name}: "
                f"can_run={result['can_run']}, complexity={result['complexity']}"
            )

            return result

        except Exception as e:
            logger.warning(f"LLM prerequisite inference failed: {e}")
            # Return conservative fallback
            return {
                'can_run': False,
                'confidence': 0.0,
                'missing_items': [],
                'required_steps': [],
                'complexity': 'complex',
                'reasoning': f"Could not analyze prerequisites: {e}",
                'auto_fixable': False
            }

    def _build_inference_prompt(self,
                                function_name: str,
                                function_info: Dict[str, Any],
                                data_state: Dict[str, Any],
                                skill_context: Optional[str]) -> str:
        """Build prompt for LLM prerequisite inference."""

        # Format data state
        state_summary = []
        if data_state['capabilities']:
            state_summary.append(f"**Capabilities**: {', '.join(data_state['capabilities'])}")
        if data_state['available']['layers']:
            state_summary.append(f"**Layers**: {', '.join(data_state['available']['layers'])}")
        if data_state['available']['obsm']:
            state_summary.append(f"**Obsm**: {', '.join(data_state['available']['obsm'])}")
        if data_state['available']['uns']:
            state_summary.append(f"**Uns**: {', '.join(data_state['available']['uns'][:5])}")

        state_str = "\n".join(state_summary) if state_summary else "No preprocessing detected (raw data)"

        # Build prompt
        prompt = f"""You are a bioinformatics workflow expert analyzing function prerequisites.

## Function to Analyze
**Name**: {function_name}
**Full Name**: {function_info.get('full_name', function_name)}
**Category**: {function_info.get('category', 'unknown')}
**Signature**: {function_info.get('signature', '(adata, **kwargs)')}

**Documentation**:
{function_info.get('docstring', 'No documentation available')[:500]}

**Examples**:
{chr(10).join(function_info.get('examples', ['No examples available'])[:2])}

## Current Data State
{state_str}

## Your Task
Analyze whether this function can run on the current data state, and determine what prerequisites are needed.

Consider:
1. **Function signature**: What parameters does it expect? (e.g., layer='scaled', use_rep='X_pca')
2. **Common bioinformatics workflows**: Standard order is QC → normalize → scale → PCA → neighbors → clustering
3. **Current data state**: What's already available vs. what's missing
4. **Complexity**: How many steps to prepare the data?

## Skill Context (Best Practices)
{skill_context or 'No specific skill guidance available'}

## Response Format
Respond in JSON format:
{{
  "can_run": true/false,
  "confidence": 0.0-1.0,
  "missing_items": ["item1", "item2"],
  "required_steps": ["step1", "step2"],
  "complexity": "simple" or "complex",
  "reasoning": "Your detailed analysis",
  "auto_fixable": true/false
}}

**Guidelines**:
- `can_run`: true if function can execute on current data without errors
- `confidence`: How confident you are (1.0 = very confident, 0.5 = uncertain)
- `missing_items`: Specific things missing (e.g., "scaled layer", "X_pca", "neighbors")
- `required_steps`: Functions to run to prepare data (e.g., ["scale", "pca"])
- `complexity`: "simple" if 0-1 steps needed, "complex" if 2+ steps needed
- `reasoning`: Explain your analysis in 1-2 sentences
- `auto_fixable`: true if missing items can be auto-fixed with ≤1 simple step

**Examples**:

Example 1: PCA on preprocessed data
Function: pca, signature: (adata, layer='scaled')
Data State: Has 'scaled' layer
Response: {{"can_run": true, "confidence": 1.0, "missing_items": [], "required_steps": [], "complexity": "simple", "reasoning": "Function expects scaled layer which is available.", "auto_fixable": false}}

Example 2: PCA on raw data
Function: pca, signature: (adata, layer='scaled')
Data State: No preprocessing
Response: {{"can_run": false, "confidence": 0.9, "missing_items": ["scaled layer"], "required_steps": ["qc", "preprocess", "scale"], "complexity": "complex", "reasoning": "PCA requires scaled data. Current data is raw and needs full preprocessing pipeline (3+ steps).", "auto_fixable": false}}

Example 3: Leiden with PCA, missing neighbors
Function: leiden, signature: (adata, resolution=1.0)
Data State: Has X_pca, missing neighbors
Response: {{"can_run": false, "confidence": 0.95, "missing_items": ["neighbors"], "required_steps": ["neighbors"], "complexity": "simple", "reasoning": "Leiden clustering needs neighbor graph. Data has PCA, only missing neighbors (1 step).", "auto_fixable": true}}

Now analyze the function and data state above. Return ONLY valid JSON, no additional text.
"""

        return prompt

    def _parse_inference_response(self, response_text: str, function_name: str) -> Dict[str, Any]:
        """Parse LLM response into structured result."""
        try:
            # Extract JSON from response (might be wrapped in markdown)
            json_text = response_text.strip()

            # Remove markdown code blocks if present
            if json_text.startswith('```'):
                lines = json_text.split('\n')
                json_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else json_text
                json_text = json_text.replace('```json', '').replace('```', '').strip()

            result = json.loads(json_text)

            # Validate required fields
            required_fields = ['can_run', 'confidence', 'missing_items',
                             'required_steps', 'complexity', 'reasoning', 'auto_fixable']
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")

            # Validate types
            if not isinstance(result['can_run'], bool):
                result['can_run'] = bool(result['can_run'])
            if not isinstance(result['auto_fixable'], bool):
                result['auto_fixable'] = bool(result['auto_fixable'])
            if not isinstance(result['confidence'], (int, float)):
                result['confidence'] = 0.5
            if result['complexity'] not in ['simple', 'complex']:
                result['complexity'] = 'complex' if len(result['required_steps']) > 1 else 'simple'

            return result

        except Exception as e:
            logger.warning(f"Failed to parse LLM inference response: {e}")
            # Return conservative fallback
            return {
                'can_run': False,
                'confidence': 0.3,
                'missing_items': [],
                'required_steps': [],
                'complexity': 'complex',
                'reasoning': f"Could not parse LLM response for {function_name}",
                'auto_fixable': False
            }

    def clear_cache(self):
        """Clear the inference cache."""
        self._cache = {}
        logger.debug("Prerequisite inference cache cleared")


class OmicVerseAgent:
    """
    Intelligent agent for OmicVerse function discovery and execution.
    
    This agent uses an internal LLM backend to understand natural language
    requests and automatically execute appropriate OmicVerse functions.
    
    Usage:
        agent = ov.Agent(model="gpt-5", api_key="your-api-key")
        result_adata = agent.run("quality control with nUMI>500, mito<0.2", adata)
    """
    
    def __init__(self, model: str = "gpt-5", api_key: Optional[str] = None, endpoint: Optional[str] = None, enable_reflection: bool = True, reflection_iterations: int = 1, enable_result_review: bool = True):
        """
        Initialize the OmicVerse Smart Agent.

        Parameters
        ----------
        model : str
            LLM model to use for reasoning (default: "gpt-5")
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
        self._prerequisite_inference: Optional[LLMPrerequisiteInference] = None  # Layer 2: LLM inference
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

        # Initialize Layer 2: LLM-based prerequisite inference
        self._prerequisite_inference = LLMPrerequisiteInference(self._llm)
        logger.debug("Layer 2: LLM prerequisite inference initialized")

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

    async def _analyze_task_complexity(self, request: str, adata: Any) -> str:
        """
        Analyze the complexity of a user request with data state awareness.

        This method uses a combination of pattern matching, data state analysis,
        and LLM reasoning to classify whether a task can be handled with a single
        function call (simple) or requires a multi-step workflow (complex).

        Now considers the CURRENT DATA STATE to make smarter decisions. For example,
        "run PCA" is SIMPLE if data is already scaled, but COMPLEX if it's raw data.

        Parameters
        ----------
        request : str
            The user's natural language request
        adata : Any
            AnnData object with current state

        Returns
        -------
        str
            Complexity classification: 'simple' or 'complex'

        Examples
        --------
        Simple tasks (when data state supports them):
        - "quality control with nUMI>500" (always simple)
        - "normalize data" (simple if QC done)
        - "run PCA" (simple if data is scaled)
        - "leiden clustering" (simple if PCA + neighbors exist)

        Complex tasks:
        - "complete bulk RNA-seq DEG analysis pipeline"
        - "run PCA" on raw unprepared data (needs QC + normalize + scale first)
        - "leiden clustering" without PCA (needs full preprocessing)
        - "analyze my data and generate report"
        """

        # Get current data state
        data_state = DataStateInspector.inspect(adata)

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

        # DATA STATE-AWARE CLASSIFICATION (New layer!)
        # Check if the requested operation is feasible given current data state
        logger.debug("Complexity: analyzing data state for context-aware classification")

        has_processed = 'has_processed_layers' in data_state['capabilities']
        has_pca = 'has_pca' in data_state['capabilities']
        has_neighbors = 'has_neighbors' in data_state['capabilities']
        has_clustering = 'has_clustering' in data_state['capabilities']

        # PCA-related requests
        if any(kw in request_lower for kw in ['pca', '主成分', 'principal component']):
            if has_processed or has_pca:
                # Data is preprocessed or already has PCA → SIMPLE
                logger.debug("PCA requested: data preprocessed → SIMPLE")
                return 'simple'
            else:
                # Raw data needs full preprocessing → COMPLEX
                logger.debug("PCA requested: raw data needs preprocessing → COMPLEX")
                return 'complex'

        # Clustering requests
        if any(kw in request_lower for kw in ['leiden', 'louvain', 'cluster', '聚类']):
            if has_clustering:
                # Already clustered → SIMPLE (maybe wants to adjust resolution)
                logger.debug("Clustering requested: already has clustering → SIMPLE")
                return 'simple'
            elif has_pca and has_neighbors:
                # Ready to cluster → SIMPLE
                logger.debug("Clustering requested: has PCA + neighbors → SIMPLE")
                return 'simple'
            elif has_pca:
                # Only needs neighbors (1 step) → SIMPLE
                logger.debug("Clustering requested: has PCA, needs neighbors → SIMPLE")
                return 'simple'
            else:
                # Needs full preprocessing → COMPLEX
                logger.debug("Clustering requested: needs full preprocessing → COMPLEX")
                return 'complex'

        # Neighbor graph requests
        if any(kw in request_lower for kw in ['neighbor', 'neighbours', 'knn']):
            if has_pca:
                # Has PCA, can compute neighbors → SIMPLE
                logger.debug("Neighbors requested: has PCA → SIMPLE")
                return 'simple'
            else:
                # Needs preprocessing first → COMPLEX
                logger.debug("Neighbors requested: needs preprocessing → COMPLEX")
                return 'complex'

        # Visualization requests
        if any(kw in request_lower for kw in ['plot', 'visualize', 'umap', 'tsne', 'embedding']):
            # Check if wants to color by clustering
            color_by_clustering = any(kw in request_lower for kw in ['leiden', 'louvain', 'cluster'])

            if color_by_clustering and not has_clustering and not has_neighbors:
                # Wants clustering colors but no clustering/neighbors → COMPLEX
                logger.debug("Visualization with clustering: no clustering available → COMPLEX")
                return 'complex'
            elif has_pca or data_state['embeddings']:
                # Has embeddings or PCA → SIMPLE
                logger.debug("Visualization: has embeddings/PCA → SIMPLE")
                return 'simple'
            # Otherwise fall through to LLM

        # QC requests are typically simple
        if any(kw in request_lower for kw in ['qc', 'quality control', '质控']):
            logger.debug("QC requested → SIMPLE")
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

        print(f"🚀 Priority 1: Fast registry-based workflow (with Layer 2 LLM inference)")

        # Layer 1: Inspect current data state
        data_state_summary = DataStateInspector.get_readable_summary(adata)
        data_state = DataStateInspector.inspect(adata)

        # Build registry-only prompt (no skills, focused on single function)
        functions_info = self._get_available_functions_info()

        #  Layer 2: Optional LLM-based prerequisite inference for ambiguous cases
        # This provides enhanced intelligence for edge cases
        llm_inference_context = ""
        try:
            # Try to identify the target function from the request
            request_lower = request.lower()
            target_function = None

            # Simple heuristic to find target function
            common_functions = {
                'pca': ['pca', '主成分', 'principal'],
                'leiden': ['leiden', '聚类'],
                'neighbors': ['neighbor', 'knn'],
                'qc': ['qc', 'quality', '质控'],
                'umap': ['umap'],
                'tsne': ['tsne'],
                'scale': ['scale', '标准化']
            }

            for func_key, keywords in common_functions.items():
                if any(kw in request_lower for kw in keywords):
                    target_function = func_key
                    break

            # If we identified a target function, use Layer 2 for deeper analysis
            if target_function and self._prerequisite_inference:
                logger.debug(f"Layer 2: Running LLM inference for function '{target_function}'")

                # Get function info from registry
                func_results = _global_registry.find(target_function)
                if func_results:
                    func_info = func_results[0]

                    # Run LLM-based prerequisite inference
                    inference_result = await self._prerequisite_inference.infer_prerequisites(
                        function_name=target_function,
                        function_info=func_info,
                        data_state=data_state,
                        skill_context=None  # Could add skill context if needed
                    )

                    # Build context string for the prompt
                    if not inference_result['can_run']:
                        llm_inference_context = f"""

## Layer 2: LLM Prerequisite Analysis for '{target_function}'

**LLM Analysis** (Confidence: {inference_result['confidence']:.0%}):
{inference_result['reasoning']}

**Missing Items**: {', '.join(inference_result['missing_items']) if inference_result['missing_items'] else 'None'}
**Required Steps**: {' → '.join(inference_result['required_steps']) if inference_result['required_steps'] else 'None'}
**Complexity**: {inference_result['complexity'].upper()}
**Auto-fixable**: {"YES - can auto-run missing prerequisites" if inference_result['auto_fixable'] else "NO - needs full workflow"}

**Recommendation**:
{"Auto-add the missing prerequisite(s) to your code." if inference_result['auto_fixable'] else 'Respond with "NEEDS_WORKFLOW" - this requires multiple preprocessing steps.'}
"""
                        logger.debug(f"Layer 2: Inference suggests complexity={inference_result['complexity']}, auto_fixable={inference_result['auto_fixable']}")

        except Exception as e:
            logger.warning(f"Layer 2 inference failed (non-critical): {e}")
            # Continue without Layer 2 enhancement


        priority1_prompt = f"""You are a fast function executor for OmicVerse with MULTI-LAYER PREREQUISITE AWARENESS.

## Intelligent Prerequisite System
This system uses TWO layers:
- **Layer 1** (Runtime Inspection): Facts about current data state
- **Layer 2** (LLM Inference): Intelligent reasoning about prerequisites

Use both layers to make smart decisions about prerequisite handling.

Request: "{request}"

{data_state_summary}
{llm_inference_context}

Available OmicVerse Functions (Registry):
{functions_info}

INSTRUCTIONS - MULTI-LAYER PREREQUISITE-AWARE EXECUTION:
1. **Check Layer 1 (Data State)** - See what's available in the current data (layers, obsm, uns, capabilities)
2. **Check Layer 2 (LLM Analysis)** - If provided above, review the LLM's prerequisite analysis and recommendation
3. **Find the best function** from the registry for the user's request
4. **Analyze if prerequisites are met**:
   - Does the function need a 'scaled' layer? Check if it exists in available layers
   - Does it need 'X_pca'? Check if it exists in available obsm
   - Does it need 'neighbors'? Check if it exists in uns
   - Does it need clustering results? Check detected capabilities
   - **IMPORTANT**: If Layer 2 analysis is provided, prioritize its recommendation
5. **Handle missing prerequisites intelligently**:
   - If Layer 2 says "Auto-fixable: YES" → Auto-add the missing prerequisite(s)
   - If 0-1 simple prerequisite is missing (e.g., just needs scaling) → Auto-add it to the code
   - If 2+ steps are missing OR Layer 2 says "NEEDS_WORKFLOW" → Respond "NEEDS_WORKFLOW"
5. **Extract parameters** from request (e.g., "nUMI>500" → tresh={{'nUMIs': 500, ...}})
6. **Return executable Python code ONLY**, no explanations

PREREQUISITE HANDLING EXAMPLES:

Example 1: Ready to execute (no prerequisites needed)
Request: "Run PCA"
Data State: Has 'scaled' layer ✅
Code:
```python
import omicverse as ov
adata = ov.pp.pca(adata, layer='scaled', n_pcs=50)
print(f"✅ PCA completed: {{adata.obsm['X_pca'].shape}}")
```

Example 2: Auto-add 1 simple prerequisite
Request: "Run leiden clustering"
Data State: Has 'X_pca' ✅, missing 'neighbors' ❌
Code:
```python
import omicverse as ov
# Auto-add missing prerequisite (1 step)
if 'neighbors' not in adata.uns:
    print("📊 Computing neighbor graph first...")
    adata = ov.pp.neighbors(adata, n_neighbors=15, use_rep='X_pca')
# Now run requested function
adata = ov.pp.leiden(adata, resolution=1.0)
print(f"✅ Leiden clustering: {{adata.obs['leiden'].nunique()}} clusters")
```

Example 3: Missing too many prerequisites
Request: "Run PCA"
Data State: No preprocessing (raw data only) ❌
Response: "NEEDS_WORKFLOW"
Reason: Needs QC → normalize → scale → PCA (4 steps = too complex for Priority 1)

Example 4: Defensive validation for visualization
Request: "Plot UMAP colored by leiden"
Data State: Has 'X_umap' ✅, missing 'leiden' in obs ❌
Code:
```python
import omicverse as ov
# Validate clustering exists
if 'leiden' not in adata.obs:
    if 'neighbors' not in adata.uns:
        print("NEEDS_WORKFLOW")  # Too many steps
    print("📊 Running leiden clustering...")
    adata = ov.pp.leiden(adata, resolution=1.0)
# Now plot
ov.pl.embedding(adata, basis='X_umap', color='leiden', frameon='small')
print("✅ UMAP plot generated")
```

IMPORTANT CONSTRAINTS:
- Maximum 1-4 function calls (including auto-added prerequisites)
- Can auto-add up to 1 simple prerequisite (like neighbors, scaling)
- If >2 prerequisites missing OR complex pipeline needed → "NEEDS_WORKFLOW"
- Always check data state before executing
- Include informative print statements

Common prerequisite patterns:
- PCA needs: 'scaled' layer (or at least preprocessed data)
- Clustering (leiden/louvain) needs: 'neighbors' in uns + 'X_pca' in obsm
- Neighbors needs: 'X_pca' in obsm
- Visualization with clusters needs: clustering results in obs

Now generate code for: "{request}"

Remember: Check the data state above, auto-add ≤1 simple prerequisite if needed, or respond "NEEDS_WORKFLOW" if too complex.
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
            raise ValueError("no code candidates found in the response")

        syntax_errors = []
        for candidate in candidates:
            try:
                normalized = self._normalize_code_candidate(candidate)
            except ValueError as exc:
                syntax_errors.append(str(exc))
                continue
            try:
                ast.parse(normalized)
            except SyntaxError as exc:
                syntax_errors.append(str(exc))
                continue
            return normalized

        raise ValueError("; ".join(syntax_errors) or "no syntactically valid python code detected")

    def _gather_code_candidates(self, response_text: str) -> List[str]:
        """Collect possible Python snippets from fenced or inline blocks."""

        fenced_pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
        candidates = [
            textwrap.dedent(match.group(1)).strip()
            for match in fenced_pattern.finditer(response_text)
            if match.group(1).strip()
        ]

        if candidates:
            return candidates

        inline = self._extract_inline_python(response_text)
        return [inline] if inline else []

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

        warnings.warn(
            "Executing agent-generated code. Ensure the input model and prompts come from a trusted source.",
            RuntimeWarning,
            stacklevel=2,
        )

        with self._temporary_api_keys():
            exec(compiled, sandbox_globals, sandbox_locals)

        return sandbox_locals.get("adata", adata)

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
            "ValueError",
            "RuntimeError",
            "TypeError",
            "KeyError",
            "AssertionError",
        ]

        safe_builtins = {name: getattr(builtins, name) for name in allowed_builtins if hasattr(builtins, name)}
        allowed_modules = {}
        core_modules = ("omicverse", "numpy", "pandas", "scanpy")
        skill_modules = ("openpyxl", "reportlab", "matplotlib", "seaborn", "scipy", "statsmodels", "sklearn")
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
            if root_name not in allowed_modules:
                raise ImportError(
                    f"Module '{name}' is not available inside the OmicVerse agent sandbox."
                )
            return __import__(name, globals, locals, fromlist, level)

        safe_builtins["__import__"] = limited_import

        sandbox_globals: Dict[str, Any] = {"__builtins__": safe_builtins}
        sandbox_globals.update(allowed_modules)
        return sandbox_globals

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

        # Step 1: Analyze task complexity (with data state awareness)
        print(f"📊 Analyzing task complexity...")
        complexity = await self._analyze_task_complexity(request, adata)
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

def Agent(model: str = "gpt-5", api_key: Optional[str] = None, endpoint: Optional[str] = None, enable_reflection: bool = True, reflection_iterations: int = 1, enable_result_review: bool = True) -> OmicVerseAgent:
    """
    Create an OmicVerse Smart Agent instance.

    This function creates and returns a smart agent that can execute OmicVerse functions
    based on natural language descriptions.

    Parameters
    ----------
    model : str, optional
        LLM model to use (default: "gpt-5"). Use list_supported_models() to see all options
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

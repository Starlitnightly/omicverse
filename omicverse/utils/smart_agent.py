"""
OmicVerse Smart Agent using Pantheon Framework

This module provides a smart agent that can understand natural language requests
and automatically execute appropriate OmicVerse functions.

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
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

# Add pantheon path if not already in path
try:
    from pantheon.agent import Agent as PantheonAgent
except ImportError:
    PANTHEON_INSTALLED = False
else:
    PANTHEON_INSTALLED = True

# Import registry system and model configuration
from .registry import _global_registry
from .model_config import ModelConfig, PROVIDER_API_KEYS
from .skill_registry import (
    SkillMatch,
    SkillRegistry,
    SkillRouter,
    build_skill_registry,
)


class OmicVerseAgent:
    """
    Intelligent agent for OmicVerse function discovery and execution.
    
    This agent uses the Pantheon framework to understand natural language
    requests and automatically execute appropriate OmicVerse functions.
    
    Usage:
        agent = ov.Agent(model="gpt-4o-mini", api_key="your-api-key")
        result_adata = agent.run("quality control with nUMI>500, mito<0.2", adata)
    """
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None, endpoint: Optional[str] = None):
        """
        Initialize the OmicVerse Smart Agent.
        
        Parameters
        ----------
        model : str
            LLM model to use for reasoning (default: "gpt-4o-mini")
        api_key : str, optional
            API key for the model provider. If not provided, will use environment variable
        endpoint : str, optional
            Custom API endpoint. If not provided, will use default for the provider
        """
        if PANTHEON_INSTALLED:
            print(f" Initializing OmicVerse Smart Agent...")
        else:
            print(f"❌ Pantheon not found. Please install pantheon-agents using `pip install pantheon-agents`")
            raise ImportError("Pantheon not found. Please install pantheon-agents using `pip install pantheon-agents`")
        
        # Validate model
        is_valid, validation_msg = ModelConfig.validate_model_setup(model, api_key)
        if not is_valid:
            print(f"❌ {validation_msg}")
            raise ValueError(validation_msg)
        
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint or ModelConfig.get_endpoint_for_model(model)
        self.agent = None
        self.skill_registry: Optional[SkillRegistry] = None
        self.skill_router: Optional[SkillRouter] = None
        self._skill_overview_text: str = ""
        self._managed_api_env: Dict[str, str] = self._collect_api_key_env(api_key)

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
            print(f"✅ Smart Agent initialized successfully!")
        except Exception as e:
            print(f"❌ Agent initialization failed: {e}")
            raise

    def _initialize_skill_registry(self) -> None:
        """Load the OmicVerse project skills and prepare routing helpers."""

        project_root = Path(__file__).resolve().parents[2]
        try:
            registry = build_skill_registry(project_root)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"⚠️  Failed to load Agent Skills: {exc}")
            registry = None

        if registry and registry.skills:
            self.skill_registry = registry
            self.skill_router = SkillRouter(registry)
            self._skill_overview_text = self._format_skill_overview()
            print(f"   🧭 Loaded {len(self.skill_registry.skills)} project skills from .claude/skills")
        else:
            self.skill_registry = None
            self.skill_router = None
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
        """Setup the pantheon agent with dynamic instructions."""
        
        # Suppress debug logs from Pantheon
        import logging
        import sys
        try:
            from loguru import logger
            logger.remove()  # Remove default handler
            logger.add(sys.stderr, level="INFO")  # Only show INFO and above
        except ImportError:
            # Fallback to standard logging
            logging.getLogger('pantheon').setLevel(logging.WARNING)
        
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
        
        # Set API key as environment variable if provided
        if self.api_key:
            provider = ModelConfig.get_provider_from_model(self.model)
            required_key = PROVIDER_API_KEYS.get(self.model)
            if required_key and not os.getenv(required_key):
                os.environ[required_key] = self.api_key
        
        # Create the pantheon agent with only function discovery tools
        self.agent = PantheonAgent(
            "omicverse_agent",
            instructions,
            model=self.model,
        )
        
        # Add custom tools for function discovery (no Python interpreter needed)
        self.agent.tool(self._search_functions)
        self.agent.tool(self._get_function_details)
        if self.skill_registry:
            self.agent.tool(self._list_project_skills)
            self.agent.tool(self._load_skill_guidance)
    
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

    def _list_project_skills(self) -> str:
        """Return a JSON catalog of the discovered project skills."""

        if not self.skill_registry or not self.skill_registry.skills:
            return json.dumps({"skills": [], "message": "No project skills available."}, indent=2)

        skills_payload = [
            {
                "name": skill.name,
                "description": skill.description,
                "path": str(skill.path),
                "metadata": skill.metadata,
            }
            for skill in sorted(self.skill_registry.skills.values(), key=lambda item: item.name.lower())
        ]
        return json.dumps({"skills": skills_payload}, indent=2)

    def _load_skill_guidance(self, skill_name: str) -> str:
        """Return the detailed instructions for a requested skill."""

        if not self.skill_registry or not self.skill_registry.skills:
            return json.dumps({"error": "No project skills are available."})

        if not skill_name or not skill_name.strip():
            return json.dumps({"error": "Provide a skill name to load guidance."})

        definition = self.skill_registry.skills.get(skill_name.strip().lower())
        if not definition:
            return json.dumps({"error": f"Skill '{skill_name}' not found."})

        return json.dumps(
            {
                "name": definition.name,
                "description": definition.description,
                "instructions": definition.prompt_instructions(),
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

        python_line_pattern = re.compile(r"^\s*(?:import |from |for |while |if |elif |else:|try:|except |with |return |@|print|adata|ov\.)")
        assignment_pattern = re.compile(r"^\s*[\w\.]+\s*=.*")
        collected: List[str] = []

        for raw_line in response_text.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                continue
            if python_line_pattern.match(line) or assignment_pattern.match(line):
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

    def _execute_generated_code(self, code: str, adata: Any) -> Any:
        """Execute generated Python code in a sandboxed namespace."""

        compiled = compile(code, "<omicverse-agent>", "exec")
        sandbox_globals = self._build_sandbox_globals()
        sandbox_locals = {"adata": adata}

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
            "__import__",
        ]

        safe_builtins = {name: getattr(builtins, name) for name in allowed_builtins if hasattr(builtins, name)}
        return {"__builtins__": safe_builtins}

    async def run_async(self, request: str, adata: Any) -> Any:
        """
        Process a natural language request and execute the generated code locally.
        
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
        """
        
        # Determine which project skills are relevant to this request
        skill_matches = self._select_skill_matches(request, top_k=2)
        if skill_matches:
            print("\n🎯 Matched project skills:")
            for match in skill_matches:
                print(f"   - {match.skill.name} (score={match.score:.3f})")

        skill_guidance_text = self._format_skill_guidance(skill_matches)
        skill_guidance_section = ""
        if skill_guidance_text:
            skill_guidance_section = (
                "\nRelevant project skills:\n"
                f"{skill_guidance_text}\n"
            )

        # Ask agent to generate the appropriate function call code
        code_generation_request = f'''
Please analyze this OmicVerse request: "{request}"

Your task:
1. Use the _search_functions tool to find the most appropriate OmicVerse function
2. Use the _get_function_details tool to get the complete function signature and docstring
3. Based on the function details, extract parameters from the request text
4. Generate executable Python code that calls the correct OmicVerse function with proper parameters

Dataset info:
- Shape: {adata.shape[0]} cells × {adata.shape[1]} genes
- Request: {request}
{skill_guidance_section}

CRITICAL INSTRUCTIONS:
1. ALWAYS call _search_functions first to find the right function
2. ALWAYS call _get_function_details to get complete help info before generating code
3. Read the 'help' field carefully to understand all parameters and their defaults
4. Generate code that matches the actual function signature
5. Return ONLY executable Python code, no explanations

For the qc function specifically:
- The tresh parameter needs a dict with 'mito_perc', 'nUMIs', 'detected_genes' keys
- Default is: tresh={{'mito_perc': 0.15, 'nUMIs': 500, 'detected_genes': 250}}
- Extract values from user request and update the dict accordingly

Example workflow:
1. _search_functions("quality control") → finds ov.pp.qc
2. _get_function_details("qc") → read help to see all parameters
3. Parse request: "nUMI>500" means tresh['nUMIs']=500
4. Generate: ov.pp.qc(adata, tresh={{'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250}})
'''
        
        # Get the code from the agent
        print(f"\n🤔 Agent analyzing request: '{request}'...")
        with self._temporary_api_keys():
            response = await self.agent.run(code_generation_request)
        
        # Extract code from the response
        # Check if response has content attribute (Pantheon agent response)
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        # Display agent's response
        print(f"\n💭 Agent response:")
        print("-" * 50)
        # Only show the actual content, not the full object
        if hasattr(response, 'content'):
            print(f"{response.content}")
        else:
            print(response_text)
        print("-" * 50)
        
        try:
            code = self._extract_python_code(response_text)
        except ValueError as exc:
            raise ValueError(f"❌ Could not extract executable code from agent response: {exc}") from exc

        print(f"\n🧬 Generated code to execute:")
        print("=" * 50)
        print(f"{code}")
        print("=" * 50)

        # Execute the code locally
        print(f"\n⚡ Executing code locally...")
        try:
            result_adata = self._execute_generated_code(code, adata)
            print(f"✅ Code executed successfully!")
            print(f"📊 Result shape: {result_adata.shape[0]} cells × {result_adata.shape[1]} genes")

            return result_adata

        except Exception as e:
            print(f"❌ Error executing generated code: {e}")
            print(f"Code that failed: {code}")
            return adata

    def _select_skill_matches(self, request: str, top_k: int = 1) -> List[SkillMatch]:
        """Return the most relevant project skills for the request."""

        if not self.skill_router:
            return []
        try:
            return self.skill_router.route(request, top_k=top_k)
        except Exception as exc:  # pragma: no cover - defensive guard
            print(f"⚠️  Skill routing failed: {exc}")
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

        if not self.skill_registry or not self.skill_registry.skills:
            return ""
        lines = [
            f"- **{skill.name}** — {skill.description}"
            for skill in sorted(self.skill_registry.skills.values(), key=lambda item: item.name.lower())
        ]
        return "\n".join(lines)

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

def Agent(model: str = "gpt-4o-mini", api_key: Optional[str] = None, endpoint: Optional[str] = None) -> OmicVerseAgent:
    """
    Create an OmicVerse Smart Agent instance.
    
    This function creates and returns a smart agent that can execute OmicVerse functions
    based on natural language descriptions.
    
    Parameters
    ----------
    model : str, optional
        LLM model to use (default: "gpt-4o-mini"). Use list_supported_models() to see all options
    api_key : str, optional
        API key for the model provider. If not provided, will use environment variable
    endpoint : str, optional
        Custom API endpoint. If not provided, will use default for the provider
        
    Returns
    -------
    OmicVerseAgent
        Configured agent instance ready for use
        
    Examples
    --------
    >>> import omicverse as ov
    >>> import scanpy as sc
    >>> 
    >>> # Create agent instance
    >>> agent = ov.Agent(model="gpt-4o-mini", api_key="your-key")
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
    return OmicVerseAgent(model=model, api_key=api_key, endpoint=endpoint)


# Export the main functions
__all__ = ["Agent", "OmicVerseAgent", "list_supported_models"]

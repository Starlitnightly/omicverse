"""
Function Registry System for OmicVerse

This module provides a decorator-based function registration system that allows
users to discover functions using natural language queries, aliases, and semantic search.
"""

from typing import Dict, List, Optional, Callable, Any, Union
from functools import wraps
from datetime import datetime, timezone
import inspect
from difflib import get_close_matches
import warnings
import json
from pathlib import Path


class FunctionRegistry:
    """
    Central registry for all decorated functions in OmicVerse.
    
    This class maintains a searchable index of functions with their metadata,
    including aliases, categories, descriptions, and usage examples.
    """
    
    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._function_map: Dict[Callable, str] = {}  # Maps functions to their primary keys
        self._categories: Dict[str, List[str]] = {}  # Category to function mapping
        
    def register(self,
                 func: Callable,
                 aliases: List[str],
                 category: str,
                 description: str,
                 examples: Optional[List[str]] = None,
                 related: Optional[List[str]] = None,
                 prerequisites: Optional[Dict[str, List[str]]] = None,
                 requires: Optional[Dict[str, List[str]]] = None,
                 produces: Optional[Dict[str, List[str]]] = None,
                 auto_fix: str = 'none') -> Callable:
        """
        Register a function with its metadata.

        Parameters
        ----------
        func : Callable
            The function to register
        aliases : List[str]
            List of aliases (including Chinese, abbreviations, etc.)
        category : str
            Function category (e.g., "preprocessing", "analysis", "visualization")
        description : str
            Detailed description of the function
        examples : List[str], optional
            Usage examples
        related : List[str], optional
            Related function names
        prerequisites : Dict[str, List[str]], optional
            Function prerequisites, format:
            {'functions': ['scale'], 'optional_functions': ['qc', 'preprocess']}
        requires : Dict[str, List[str]], optional
            Required data structures, format:
            {'layers': ['scaled'], 'obsm': ['X_pca'], 'uns': ['neighbors'], ...}
        produces : Dict[str, List[str]], optional
            Data structures created by this function, format:
            {'layers': ['scaled'], 'obsm': ['X_pca'], 'uns': ['pca'], ...}
        auto_fix : str, optional
            Auto-fix strategy: 'auto' (can auto-insert prerequisites),
            'escalate' (suggest workflow), 'none' (just warn). Default: 'none'

        Returns
        -------
        Callable
            The original function unchanged
        """
        # Validate metadata completeness
        if not aliases or not all(alias.strip() for alias in aliases):
            raise ValueError("Function registration requires at least one non-empty alias.")

        if not category or not category.strip():
            raise ValueError("Function registration requires a category.")

        if not description or not description.strip():
            raise ValueError("Function registration requires a description.")

        if examples is not None:
            if isinstance(examples, (tuple, set)):
                examples = list(examples)
            elif not isinstance(examples, list):
                raise TypeError("Examples must be provided as a list of strings when specified.")

        if related is not None:
            if isinstance(related, (tuple, set)):
                related = list(related)
            elif not isinstance(related, list):
                raise TypeError("Related entries must be provided as a list of strings when specified.")

        # Validate prerequisite metadata
        if prerequisites is not None:
            if not isinstance(prerequisites, dict):
                raise TypeError("Prerequisites must be a dictionary.")
            valid_keys = {'functions', 'optional_functions'}
            for key in prerequisites.keys():
                if key not in valid_keys:
                    raise ValueError(f"Invalid prerequisites key '{key}'. Valid keys: {valid_keys}")
                if not isinstance(prerequisites[key], list):
                    raise TypeError(f"Prerequisites['{key}'] must be a list.")

        if requires is not None:
            if not isinstance(requires, dict):
                raise TypeError("Requires must be a dictionary.")
            valid_keys = {'layers', 'obsm', 'obsp', 'uns', 'var', 'obs', 'varm'}
            for key in requires.keys():
                if key not in valid_keys:
                    raise ValueError(f"Invalid requires key '{key}'. Valid keys: {valid_keys}")
                if not isinstance(requires[key], list):
                    raise TypeError(f"Requires['{key}'] must be a list.")

        if produces is not None:
            if not isinstance(produces, dict):
                raise TypeError("Produces must be a dictionary.")
            valid_keys = {'layers', 'obsm', 'obsp', 'uns', 'var', 'obs', 'varm'}
            for key in produces.keys():
                if key not in valid_keys:
                    raise ValueError(f"Invalid produces key '{key}'. Valid keys: {valid_keys}")
                if not isinstance(produces[key], list):
                    raise TypeError(f"Produces['{key}'] must be a list.")

        if auto_fix not in ('auto', 'escalate', 'none'):
            raise ValueError(f"Invalid auto_fix value '{auto_fix}'. Valid values: 'auto', 'escalate', 'none'")

        # Generate function key
        module_name = func.__module__
        func_name = func.__name__
        full_name = f"{module_name}.{func_name}"

        # Extract function signature and parameters
        try:
            sig = inspect.signature(func)
            signature = str(sig)
            
            # Extract parameter details with defaults
            params_info = []
            for param_name, param in sig.parameters.items():
                param_str = param_name
                if param.default != inspect.Parameter.empty:
                    param_str += f"={repr(param.default)}"
                params_info.append(param_str)
            
            # Get full docstring for help
            raw_doc = inspect.getdoc(func)
            if not raw_doc:
                warnings.warn(
                    f"Function '{full_name}' is missing a docstring; agent help output may be limited.",
                    UserWarning,
                )
            docstring = raw_doc or "No documentation available"
            
        except Exception:
            signature = "(adata, **kwargs)"
            params_info = ["adata", "**kwargs"]
            docstring = "No documentation available"
        
        # Create registry entry
        entry = {
            'function': func,
            'full_name': full_name,
            'short_name': func_name,
            'module': module_name,
            'aliases': aliases,
            'category': category,
            'description': description,
            'examples': examples or [],
            'related': related or [],
            'signature': signature,
            'parameters': params_info,
            'docstring': docstring,
            # Prerequisite tracking metadata
            'prerequisites': prerequisites or {},
            'requires': requires or {},
            'produces': produces or {},
            'auto_fix': auto_fix
        }
        
        # Register under all aliases
        for alias in aliases:
            self._registry[alias.lower()] = entry
            
        # Register under function name
        self._registry[func_name.lower()] = entry
        self._registry[full_name.lower()] = entry
        
        # Map function to primary key
        self._function_map[func] = full_name
        
        # Update category index
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(full_name)
        
        return func
    
    def find(self, query: str, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Find functions matching the query.
        
        Parameters
        ----------
        query : str
            Search query (can be Chinese, English, abbreviation, etc.)
        threshold : float
            Similarity threshold for fuzzy matching (0-1)
        
        Returns
        -------
        List[Dict[str, Any]]
            List of matching function entries, sorted by relevance
        """
        query_lower = query.lower()
        results = []
        
        # Exact match
        if query_lower in self._registry:
            results.append(self._registry[query_lower])
            
        # Fuzzy match on aliases
        all_keys = list(self._registry.keys())
        close_matches = get_close_matches(query_lower, all_keys, n=5, cutoff=threshold)
        
        for match in close_matches:
            if self._registry[match] not in results:
                results.append(self._registry[match])
        
        # Search in descriptions
        for entry in self._registry.values():
            if entry not in results:
                if query_lower in entry['description'].lower():
                    results.append(entry)
                elif any(query_lower in alias.lower() for alias in entry['aliases']):
                    results.append(entry)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for entry in results:
            if entry['full_name'] not in seen:
                seen.add(entry['full_name'])
                unique_results.append(entry)
        
        return unique_results
    
    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all functions in a specific category.
        
        Parameters
        ----------
        category : str
            Category name
        
        Returns
        -------
        List[Dict[str, Any]]
            List of functions in the category
        """
        if category not in self._categories:
            return []
        
        results = []
        for func_name in self._categories[category]:
            # Find the entry
            for entry in self._registry.values():
                if entry['full_name'] == func_name:
                    results.append(entry)
                    break
        
        return results
    
    def list_categories(self) -> List[str]:
        """List all available categories."""
        return list(self._categories.keys())
    
    def get_function(self, query: str) -> Optional[Callable]:
        """
        Get the actual function object for a query.
        
        Parameters
        ----------
        query : str
            Function name or alias
        
        Returns
        -------
        Optional[Callable]
            The function object if found, None otherwise
        """
        matches = self.find(query)
        if matches:
            return matches[0]['function']
        return None
    
    def format_results(self, results: List[Dict[str, Any]], verbose: bool = False) -> str:
        """
        Format search results for display.
        
        Parameters
        ----------
        results : List[Dict[str, Any]]
            Search results
        verbose : bool
            Whether to include detailed information
        
        Returns
        -------
        str
            Formatted string for display
        """
        if not results:
            return "❌ No matching functions found."
        
        output = []
        output.append(f"🔍 Found {len(results)} matching function(s):\n")
        
        for i, entry in enumerate(results, 1):
            output.append(f"\n{i}. 📦 {entry['full_name']}")
            output.append(f"   📝 {entry['description']}")
            output.append(f"   🏷️  Aliases: {', '.join(entry['aliases'])}")
            output.append(f"   📁 Category: {entry['category']}")
            
            if verbose:
                output.append(f"   🔧 Signature: {entry['signature']}")
                if entry['examples']:
                    output.append(f"   💡 Examples:")
                    for example in entry['examples']:
                        output.append(f"      {example}")
                if entry['related']:
                    output.append(f"   🔗 Related: {', '.join(entry['related'])}")

        return "\n".join(output)

    def get_prerequisites(self, func_name: str) -> Dict[str, Any]:
        """
        Get full prerequisite information for a function.

        Parameters
        ----------
        func_name : str
            Function name or alias

        Returns
        -------
        Dict with keys:
            - required_functions: List[str] - Must run these first
            - optional_functions: List[str] - Recommended to run first
            - requires: Dict - Required data structures
            - produces: Dict - What the function creates
            - auto_fix: str - Auto-fix strategy

        Examples
        --------
        >>> registry.get_prerequisites('pca')
        {
            'required_functions': ['scale'],
            'optional_functions': ['qc', 'preprocess'],
            'requires': {'layers': ['scaled']},
            'produces': {'obsm': ['X_pca'], 'varm': ['PCs'], 'uns': ['pca']},
            'auto_fix': 'escalate'
        }
        """
        matches = self.find(func_name)
        if not matches:
            return {
                'required_functions': [],
                'optional_functions': [],
                'requires': {},
                'produces': {},
                'auto_fix': 'none'
            }

        entry = matches[0]
        prereqs = entry.get('prerequisites', {})

        return {
            'required_functions': prereqs.get('functions', []),
            'optional_functions': prereqs.get('optional_functions', []),
            'requires': entry.get('requires', {}),
            'produces': entry.get('produces', {}),
            'auto_fix': entry.get('auto_fix', 'none')
        }

    def get_prerequisite_chain(self, func_name: str, include_optional: bool = False) -> List[str]:
        """
        Get ordered list of functions to run for prerequisites.

        Parameters
        ----------
        func_name : str
            Target function name
        include_optional : bool
            Whether to include optional prerequisites. Default: False

        Returns
        -------
        List[str]
            Ordered prerequisite chain, e.g., ['scale', 'pca']

        Examples
        --------
        >>> registry.get_prerequisite_chain('pca')
        ['scale', 'pca']

        >>> registry.get_prerequisite_chain('leiden')
        ['neighbors', 'leiden']

        >>> registry.get_prerequisite_chain('pca', include_optional=True)
        ['qc', 'preprocess', 'scale', 'pca']
        """
        prereq_info = self.get_prerequisites(func_name)
        chain = []

        # Add optional functions if requested
        if include_optional:
            chain.extend(prereq_info['optional_functions'])

        # Add required functions
        chain.extend(prereq_info['required_functions'])

        # Add the target function itself
        matches = self.find(func_name)
        if matches:
            chain.append(matches[0]['short_name'])

        return chain

    def check_prerequisites(self, func_name: str, adata) -> Dict[str, Any]:
        """
        Validate if all prerequisites are satisfied for an AnnData object.

        Parameters
        ----------
        func_name : str
            Function to check
        adata : AnnData
            Data object to validate

        Returns
        -------
        Dict with keys:
            - satisfied: bool - All requirements met
            - missing_functions: List[str] - Functions likely not run yet
            - missing_structures: List[str] - Missing data layers/structures
            - recommendation: str - What to do next
            - auto_fixable: bool - Can prerequisites be auto-inserted

        Examples
        --------
        >>> result = registry.check_prerequisites('pca', raw_adata)
        >>> result
        {
            'satisfied': False,
            'missing_functions': ['scale'],
            'missing_structures': ['adata.layers["scaled"]'],
            'recommendation': 'Run ov.pp.scale() first or use ov.pp.preprocess()',
            'auto_fixable': False
        }
        """
        prereq_info = self.get_prerequisites(func_name)
        requires = prereq_info['requires']

        missing_structures = []

        # Check required layers
        if 'layers' in requires:
            for layer in requires['layers']:
                if not hasattr(adata, 'layers') or layer not in adata.layers:
                    missing_structures.append(f'adata.layers["{layer}"]')

        # Check required obsm
        if 'obsm' in requires:
            for key in requires['obsm']:
                if not hasattr(adata, 'obsm') or key not in adata.obsm:
                    missing_structures.append(f'adata.obsm["{key}"]')

        # Check required obsp
        if 'obsp' in requires:
            for key in requires['obsp']:
                if not hasattr(adata, 'obsp') or key not in adata.obsp:
                    missing_structures.append(f'adata.obsp["{key}"]')

        # Check required uns
        if 'uns' in requires:
            for key in requires['uns']:
                if not hasattr(adata, 'uns') or key not in adata.uns:
                    missing_structures.append(f'adata.uns["{key}"]')

        # Check required var columns
        if 'var' in requires:
            for col in requires['var']:
                if not hasattr(adata, 'var') or col not in adata.var.columns:
                    missing_structures.append(f'adata.var["{col}"]')

        # Check required obs columns
        if 'obs' in requires:
            for col in requires['obs']:
                if not hasattr(adata, 'obs') or col not in adata.obs.columns:
                    missing_structures.append(f'adata.obs["{col}"]')

        # Check required varm
        if 'varm' in requires:
            for key in requires['varm']:
                if not hasattr(adata, 'varm') or key not in adata.varm:
                    missing_structures.append(f'adata.varm["{key}"]')

        satisfied = len(missing_structures) == 0
        missing_functions = prereq_info['required_functions'] if not satisfied else []

        # Generate recommendation
        if satisfied:
            recommendation = "All prerequisites satisfied"
        else:
            auto_fix = prereq_info['auto_fix']
            if auto_fix == 'auto' and len(missing_functions) <= 2:
                func_list = ', '.join([f'ov.pp.{f}()' for f in missing_functions])
                recommendation = f"Auto-fixable: Will insert {func_list}"
            elif auto_fix == 'escalate' or len(missing_functions) > 2:
                recommendation = f"Complex prerequisite chain. Consider using a workflow function."
            else:
                func_list = ', '.join([f'ov.pp.{f}()' for f in missing_functions])
                recommendation = f"Run {func_list} first"

        # Auto-fixable if: auto_fix='auto' AND <= 2 missing functions
        auto_fixable = (prereq_info['auto_fix'] == 'auto' and
                       len(missing_functions) > 0 and
                       len(missing_functions) <= 2)

        return {
            'satisfied': satisfied,
            'missing_functions': missing_functions,
            'missing_structures': missing_structures,
            'recommendation': recommendation,
            'auto_fixable': auto_fixable
        }

    def format_prerequisites_for_llm(self, func_name: str) -> str:
        """
        Format prerequisite info for LLM consumption in system prompt.

        Parameters
        ----------
        func_name : str
            Function to format prerequisites for

        Returns
        -------
        str
            Formatted text with prerequisite chain, requirements, and guidance

        Examples
        --------
        >>> print(registry.format_prerequisites_for_llm('pca'))
        Function: ov.pp.pca()
        Prerequisites:
          - Required functions: scale
          - Optional functions: qc, preprocess
          - Requires: adata.layers['scaled']
          - Produces: adata.obsm['X_pca'], adata.varm['PCs'], adata.uns['pca']
        Prerequisite Chain: scale → pca
        Full Chain (with optional): qc → preprocess → scale → pca
        Auto-fix Strategy: ESCALATE (suggest workflow for complex cases)
        """
        matches = self.find(func_name)
        if not matches:
            return f"Function '{func_name}' not found in registry."

        entry = matches[0]
        prereq_info = self.get_prerequisites(func_name)
        chain = self.get_prerequisite_chain(func_name, include_optional=False)
        full_chain = self.get_prerequisite_chain(func_name, include_optional=True)

        output = []
        output.append(f"Function: {entry['full_name']}()")
        output.append("Prerequisites:")

        # Required functions
        if prereq_info['required_functions']:
            output.append(f"  - Required functions: {', '.join(prereq_info['required_functions'])}")
        else:
            output.append("  - Required functions: None")

        # Optional functions
        if prereq_info['optional_functions']:
            output.append(f"  - Optional functions: {', '.join(prereq_info['optional_functions'])}")

        # Required structures
        requires = prereq_info['requires']
        req_list = []
        for key, values in requires.items():
            for val in values:
                req_list.append(f"adata.{key}['{val}']")
        if req_list:
            output.append(f"  - Requires: {', '.join(req_list)}")

        # Produced structures
        produces = prereq_info['produces']
        prod_list = []
        for key, values in produces.items():
            for val in values:
                prod_list.append(f"adata.{key}['{val}']")
        if prod_list:
            output.append(f"  - Produces: {', '.join(prod_list)}")

        # Prerequisite chain
        if len(chain) > 1:
            output.append(f"Prerequisite Chain: {' → '.join(chain)}")
        if len(full_chain) > len(chain):
            output.append(f"Full Chain (with optional): {' → '.join(full_chain)}")

        # Auto-fix strategy
        auto_fix = prereq_info['auto_fix']
        auto_fix_desc = {
            'auto': 'AUTO (can auto-insert simple prerequisites)',
            'escalate': 'ESCALATE (suggest workflow for complex cases)',
            'none': 'NONE (just warn user)'
        }
        output.append(f"Auto-fix Strategy: {auto_fix_desc.get(auto_fix, auto_fix)}")

        return "\n".join(output)


# Global registry instance
_global_registry = FunctionRegistry()


def register_function(aliases: List[str],
                     category: str,
                     description: str,
                     examples: Optional[List[str]] = None,
                     related: Optional[List[str]] = None,
                     prerequisites: Optional[Dict[str, List[str]]] = None,
                     requires: Optional[Dict[str, List[str]]] = None,
                     produces: Optional[Dict[str, List[str]]] = None,
                     auto_fix: str = 'none'):
    """
    Decorator to register a function with metadata.

    Parameters
    ----------
    aliases : List[str]
        List of aliases for the function (Chinese, English, abbreviations)
    category : str
        Function category
    description : str
        Function description
    examples : List[str], optional
        Usage examples
    related : List[str], optional
        Related function names
    prerequisites : Dict[str, List[str]], optional
        Function prerequisites, format:
        {'functions': ['scale'], 'optional_functions': ['qc', 'preprocess']}
    requires : Dict[str, List[str]], optional
        Required data structures, format:
        {'layers': ['scaled'], 'obsm': ['X_pca'], 'uns': ['neighbors'], ...}
    produces : Dict[str, List[str]], optional
        Data structures created by this function, format:
        {'layers': ['scaled'], 'obsm': ['X_pca'], 'uns': ['pca'], ...}
    auto_fix : str, optional
        Auto-fix strategy: 'auto' (can auto-insert prerequisites),
        'escalate' (suggest workflow), 'none' (just warn). Default: 'none'

    Examples
    --------
    >>> @register_function(
    ...     aliases=["质控", "qc", "quality_control"],
    ...     category="preprocessing",
    ...     description="Perform quality control on single-cell data"
    ... )
    ... def qc(adata, min_genes=200):
    ...     pass

    >>> @register_function(
    ...     aliases=["pca", "PCA"],
    ...     category="preprocessing",
    ...     description="Perform Principal Component Analysis",
    ...     prerequisites={'functions': ['scale']},
    ...     requires={'layers': ['scaled']},
    ...     produces={'obsm': ['X_pca'], 'varm': ['PCs'], 'uns': ['pca']},
    ...     auto_fix='escalate'
    ... )
    ... def pca(adata, n_pcs=50):
    ...     pass
    """
    def decorator(func: Callable) -> Callable:
        _global_registry.register(
            func=func,
            aliases=aliases,
            category=category,
            description=description,
            examples=examples,
            related=related,
            prerequisites=prerequisites,
            requires=requires,
            produces=produces,
            auto_fix=auto_fix
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Add registry info to function
        wrapper._registry_info = {
            'aliases': aliases,
            'category': category,
            'description': description
        }

        # If func is a class, copy over class methods and static methods
        import inspect
        if inspect.isclass(func):
            for name, value in inspect.getmembers(func):
                if isinstance(inspect.getattr_static(func, name), (classmethod, staticmethod)):
                    setattr(wrapper, name, value)

        return wrapper
    
    return decorator


# User-friendly API functions
def find_function(query: str, verbose: bool = False) -> Union[str, None]:
    """
    Find functions matching the query.
    
    Parameters
    ----------
    query : str
        Search query (Chinese, English, or abbreviation)
    verbose : bool
        Whether to show detailed information
    
    Returns
    -------
    str
        Formatted search results
    
    Examples
    --------
    >>> import omicverse as ov
    >>> ov.find_function("质控")
    >>> ov.find_function("differential expression")
    >>> ov.find_function("pca")
    """
    results = _global_registry.find(query)
    print(_global_registry.format_results(results, verbose=verbose))
    
    if results:
        return results[0]['function']
    return None


def list_functions(category: Optional[str] = None) -> None:
    """
    List all registered functions or functions in a category.
    
    Parameters
    ----------
    category : str, optional
        Category to filter by
    
    Examples
    --------
    >>> import omicverse as ov
    >>> ov.list_functions()  # List all functions
    >>> ov.list_functions("preprocessing")  # List preprocessing functions
    """
    if category:
        results = _global_registry.get_by_category(category)
        print(f"📚 Functions in category '{category}':")
    else:
        # Get all unique functions
        all_functions = {}
        for entry in _global_registry._registry.values():
            all_functions[entry['full_name']] = entry
        results = list(all_functions.values())
        print(f"📚 All registered functions ({len(results)} total):")
    
    if not results:
        print("   No functions found.")
        return
    
    # Group by category if showing all
    if not category:
        categories = {}
        for entry in results:
            cat = entry['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(entry)
        
        for cat, entries in sorted(categories.items()):
            print(f"\n🏷️  {cat.upper()}:")
            for entry in sorted(entries, key=lambda x: x['short_name']):
                print(f"   • {entry['short_name']}: {entry['description']}")
    else:
        for entry in sorted(results, key=lambda x: x['short_name']):
            print(f"   • {entry['short_name']}: {entry['description']}")


def get_function_help(query: str) -> None:
    """
    Get detailed help for a function.
    
    Parameters
    ----------
    query : str
        Function name or alias
    
    Examples
    --------
    >>> import omicverse as ov
    >>> ov.help("qc")
    >>> ov.help("质控")
    """
    results = _global_registry.find(query)
    
    if not results:
        print(f"❌ No function found for '{query}'")
        return
    
    entry = results[0]
    print(f"📦 {entry['full_name']}")
    print(f"{'=' * 50}")
    print(f"📝 Description: {entry['description']}")
    print(f"🏷️  Aliases: {', '.join(entry['aliases'])}")
    print(f"📁 Category: {entry['category']}")
    print(f"🔧 Signature: {entry['signature']}")
    
    if entry['docstring']:
        print(f"\n📖 Documentation:")
        print(entry['docstring'])
    
    if entry['examples']:
        print(f"\n💡 Examples:")
        for example in entry['examples']:
            print(f"   {example}")
    
    if entry['related']:
        print(f"\n🔗 Related functions: {', '.join(entry['related'])}")


def recommend_function(task_description: str, n: int = 5) -> List[Callable]:
    """
    Recommend functions based on task description.
    
    This is a simple keyword-based recommendation. 
    Could be enhanced with embedding-based search in the future.
    
    Parameters
    ----------
    task_description : str
        Description of what you want to do
    n : int
        Number of recommendations
    
    Returns
    -------
    List[Callable]
        List of recommended functions
    
    Examples
    --------
    >>> import omicverse as ov
    >>> ov.recommend_function("我想过滤低质量细胞")
    >>> ov.recommend_function("perform dimensionality reduction")
    """
    # Extract keywords from description
    keywords = task_description.lower().split()
    
    # Score each function based on keyword matches
    scores = {}
    for entry in _global_registry._registry.values():
        if entry['full_name'] in scores:
            continue
            
        score = 0
        # Check aliases
        for keyword in keywords:
            for alias in entry['aliases']:
                if keyword in alias.lower():
                    score += 2
            # Check description
            if keyword in entry['description'].lower():
                score += 1
        
        if score > 0:
            scores[entry['full_name']] = (score, entry)
    
    # Sort by score and return top n
    sorted_entries = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)[:n]
    
    if not sorted_entries:
        print(f"❌ No recommendations found for: '{task_description}'")
        return []
    
    print(f"🎯 Top {min(n, len(sorted_entries))} recommendations for: '{task_description}'")
    results = [entry[1][1] for entry in sorted_entries]
    print(_global_registry.format_results(results, verbose=False))
    
    return [entry[1][1]['function'] for entry in sorted_entries]


def export_registry(filepath: Optional[str] = None, 
                   format: str = "json",
                   include_source: bool = False) -> Union[str, Dict]:
    """
    Export the current registry to JSON format.
    
    Parameters
    ----------
    filepath : str, optional
        File path to save the registry. If None, returns the data as dict/string
    format : str
        Export format: 'json' or 'dict'
    include_source : bool
        Whether to include source code of functions (warning: large file)
    
    Returns
    -------
    Union[str, Dict]
        Registry data as JSON string or dictionary
        
    Examples
    --------
    >>> # Export to file
    >>> ov.export_registry('registry.json')
    
    >>> # Get as dictionary
    >>> data = ov.export_registry(format='dict')
    
    >>> # Get as JSON string
    >>> json_str = ov.export_registry(format='json')
    """
    export_data = {}
    
    # Get all unique functions
    processed_functions = set()
    
    for entry in _global_registry._registry.values():
        full_name = entry['full_name']
        if full_name in processed_functions:
            continue
        processed_functions.add(full_name)
        
        func_data = {
            'full_name': entry['full_name'],
            'short_name': entry['short_name'],
            'module': entry['module'],
            'aliases': entry['aliases'],
            'category': entry['category'],
            'description': entry['description'],
            'examples': entry['examples'],
            'related': entry['related'],
            'signature': entry['signature'],
            'docstring': entry['docstring']
        }
        
        if include_source:
            try:
                func_data['source_code'] = inspect.getsource(entry['function'])
            except Exception:
                func_data['source_code'] = "Source not available"
        
        export_data[full_name] = func_data
    
    # Add metadata
    export_metadata = {
        'exported_at': datetime.now(timezone.utc).isoformat(),
        'total_functions': len(export_data),
        'categories': list(_global_registry._categories.keys()),
        'omicverse_version': getattr(__import__('omicverse'), '__version__', 'unknown')
    }
    
    final_data = {
        'metadata': export_metadata,
        'functions': export_data
    }
    
    if format == 'dict':
        result = final_data
    else:  # json
        result = json.dumps(final_data, indent=2, ensure_ascii=False)
    
    if filepath:
        filepath = Path(filepath)
        with open(filepath, 'w', encoding='utf-8') as f:
            if format == 'dict':
                json.dump(final_data, f, indent=2, ensure_ascii=False)
            else:
                f.write(result)
        print(f"Registry exported to: {filepath}")
        print(f"Total functions: {len(export_data)}")
        return str(filepath)
    
    return result


def import_registry(filepath: str) -> Dict:
    """
    Import registry data from JSON file.
    
    Parameters
    ----------
    filepath : str
        Path to the JSON file
        
    Returns
    -------
    Dict
        Imported registry data
        
    Examples
    --------
    >>> data = ov.import_registry('registry.json')
    """
    filepath = Path(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Registry imported from: {filepath}")
    print(f"Total functions: {data['metadata']['total_functions']}")
    print(f"Categories: {', '.join(data['metadata']['categories'])}")
    
    return data


# Export registry instance and API functions
__all__ = [
    'register_function',
    'find_function', 
    'list_functions',
    'get_function_help',
    'recommend_function',
    'export_registry',
    'import_registry',
    'FunctionRegistry',
    '_global_registry'
]
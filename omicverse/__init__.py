r"""
OmicVerse: A comprehensive omic framework for multi-omic analysis.

OmicVerse is a Python package that provides a unified framework for analyzing
multi-omic data including single-cell RNA-seq, bulk RNA-seq, spatial transcriptomics,
ATAC-seq, and multi-modal integration. It offers streamlined workflows for
preprocessing, analysis, visualization, and interpretation of complex biological data.

Main modules:
    bulk: Bulk RNA-seq analysis including differential expression, pathway analysis
    single: Single-cell RNA-seq analysis including clustering, trajectory inference
    bulk2single: Deconvolution and mapping between bulk and single-cell data
    space: Spatial transcriptomics analysis and integration
    pp: Preprocessing utilities for quality control and normalization
    pl: Comprehensive plotting and visualization functions
    utils: Utility functions for data handling and analysis
    popv: Population-level variation analysis tools

Key features:
    - Unified API for multiple omics data types
    - GPU acceleration support for large-scale analysis
    - Extensive visualization capabilities
    - Integration with popular bioinformatics tools
    - Comprehensive documentation and tutorials

Examples:
    >>> import omicverse as ov
    >>> adata = ov.read('data.h5ad')
    >>>
    >>> # Traditional approach
    >>> ov.pp.preprocess(adata)
    >>> ov.single.leiden(adata)
    >>> ov.pl.umap(adata, color='leiden')
    >>>
    >>> # Smart Agent approach
    >>> agent = ov.Agent(model="gpt-4o-mini", api_key="your-key")
    >>> adata = agent.run("quality control with nUMI>500, mito<0.2", adata)
    >>> adata = agent.run("preprocess with 2000 HVGs", adata)
    >>> adata = agent.run("leiden clustering resolution=1.0", adata)
"""

# Fix PyArrow compatibility issue
# PyExtensionType was renamed to ExtensionType in newer versions
import os
try:
    import pyarrow
    if hasattr(pyarrow, 'ExtensionType') and not hasattr(pyarrow, 'PyExtensionType'):
        pyarrow.PyExtensionType = pyarrow.ExtensionType
except ImportError:
    pass

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from pkg_resources import get_distribution
    version = lambda name: get_distribution(name).version

# Cache for lazy-loaded modules and attributes
_lazy_modules = {}
_lazy_attrs = {}

# Define which modules should be lazy-loaded
_LAZY_MODULES = {
    'alignment',
    'bulk',
    'single',
    'bulk2single',
    'pp',
    'space',
    'pl',
    'datasets',
    'external',
    'llm',
    'agent',
}

# Lazy attribute mappings: {attribute_name: (module_path, attr_name)}
_LAZY_ATTRS = {
    # From utils._data - MAJOR BOTTLENECK (4.5s)!
    'read': ('.utils._data', 'read'),

    # From utils._plot - Lightweight, but let's be lazy
    'palette': ('.utils._plot', 'palette'),
    'ov_plot_set': ('.utils._plot', 'ov_plot_set'),
    'plot_set': ('.utils._plot', 'plot_set'),
    'style': ('.utils._plot', 'style'),

    # From utils.registry
    'find_function': ('.utils.registry', 'find_function'),
    'list_functions': ('.utils.registry', 'list_functions'),
    'get_function_help': ('.utils.registry', 'get_function_help'),
    'recommend_function': ('.utils.registry', 'recommend_function'),
    'export_registry': ('.utils.registry', 'export_registry'),
    'import_registry': ('.utils.registry', 'import_registry'),

    # From utils.smart_agent
    'Agent': ('.utils.smart_agent', 'Agent'),
    'list_supported_models': ('.utils.smart_agent', 'list_supported_models'),

    # From utils.session_notebook_executor
    'setup_kernel_for_env': ('.utils.session_notebook_executor', 'setup_kernel_for_env'),

    # From _settings - delay to avoid circular import
    'settings': ('._settings', 'settings'),
    'generate_reference_table': ('._settings', 'generate_reference_table'),

    # Common libraries - delay these too!
    'plt': ('matplotlib.pyplot', None),
    'np': ('numpy', None),
    'pd': ('pandas', None),
    'AnnData': ('anndata', 'AnnData'),
    'concat': ('anndata', 'concat'),
}

name = "omicverse"
try:
    __version__ = version(name)
except Exception:
    __version__ = "unknown"

# Delay settings import to avoid circular dependency
# from ._settings import settings, generate_reference_table


def __getattr__(name):
    """Lazy import modules and attributes on first access."""

    # Check if it's a lazy attribute (function or class)
    if name in _LAZY_ATTRS:
        if name in _lazy_attrs:
            return _lazy_attrs[name]

        module_path, attr_name = _LAZY_ATTRS[name]
        try:
            import importlib
            # If no attr_name, return the module itself (for plt, np, pd)
            if attr_name is None:
                module = importlib.import_module(module_path)
                _lazy_attrs[name] = module
                return module
            else:
                # Import from omicverse submodule
                module = importlib.import_module(module_path, package='omicverse')
                attr = getattr(module, attr_name)
                _lazy_attrs[name] = attr
                return attr
        except (ImportError, AttributeError) as e:
            raise AttributeError(
                f"Cannot import '{name}' from {module_path}"
            ) from e

    # Check if it's a lazy module
    if name in _LAZY_MODULES:
        if name in _lazy_modules:
            return _lazy_modules[name]

        # Handle special cases
        if name == 'llm':
            if os.environ.get("OMICVERSE_DISABLE_LLM") == "1":
                _lazy_modules[name] = None
                return None
            try:
                from . import llm
                _lazy_modules[name] = llm
                return llm
            except Exception:
                _lazy_modules[name] = None
                return None

        elif name == 'datacollect':
            try:
                from .external import datacollect
                _lazy_modules[name] = datacollect
                return datacollect
            except ImportError:
                _lazy_modules[name] = None
                return None

        elif name == 'agent':
            try:
                from . import agent
                _lazy_modules[name] = agent
                return agent
            except Exception:
                _lazy_modules[name] = None
                return None

        # utils needs special handling
        elif name == 'utils':
            from . import utils
            _lazy_modules[name] = utils
            return utils

        # Standard module import
        else:
            try:
                import importlib
                module = importlib.import_module(f'.{name}', package='omicverse')
                _lazy_modules[name] = module
                return module
            except ImportError as e:
                raise AttributeError(
                    f"Module 'omicverse' has no attribute '{name}'"
                ) from e

    # If not found, raise AttributeError
    raise AttributeError(f"Module 'omicverse' has no attribute '{name}'")


def __dir__():
    """Provide a complete list of available attributes for tab completion."""
    base_attrs = [
        # Lazy attributes
        "read", "palette", "ov_plot_set", "plot_set", "style",
        "find_function", "list_functions", "get_function_help",
        "recommend_function", "export_registry", "import_registry",
        "Agent", "list_supported_models", "setup_kernel_for_env",

        # Settings
        "settings", "generate_reference_table",

        # Common libraries
        "plt", "np", "pd", "AnnData", "concat",

        # Version
        "__version__",
    ]
    # Add utils to lazy modules for proper handling
    lazy_modules_with_utils = _LAZY_MODULES | {'utils', 'datacollect'}
    return sorted(set(base_attrs + list(lazy_modules_with_utils)))


__all__ = [
    # Lazy-loaded modules
    "alignment",
    "bulk",
    "single",
    "utils",
    "bulk2single",
    "pp",
    "space",
    "pl",
    "datasets",
    "external",
    "llm",
    "datacollect",
    "agent",

    # Essential utilities (lazy-loaded)
    "read",
    "palette",
    "ov_plot_set",
    "plot_set",
    "style",

    # Function registry (lazy-loaded)
    "find_function",
    "list_functions",
    "get_function_help",
    "recommend_function",
    "export_registry",
    "import_registry",

    # Smart Agent (lazy-loaded)
    "Agent",
    "list_supported_models",
    "setup_kernel_for_env",

    # Settings
    "settings",
    "generate_reference_table",

    # Common libraries (lazy-loaded)
    "plt",
    "np",
    "pd",
    "AnnData",
    "concat",

    # Version
    "__version__",
]

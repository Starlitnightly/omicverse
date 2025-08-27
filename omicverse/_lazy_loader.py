"""
Lazy loading utility for omicverse modules.

This module provides a lazy loading mechanism to speed up initial package imports
by deferring the loading of heavy submodules until they are actually accessed.
"""

import sys
import importlib
from types import ModuleType
from typing import List


class LazyLoader:
    """
    Lazy loader that imports modules only when accessed.
    
    This class creates a proxy object that behaves like a module but only 
    actually imports the underlying module when an attribute is accessed.
    """
    
    def __init__(self, name: str, parent_module_globals: dict, 
                 warning_message: str = None):
        """
        Initialize the lazy loader.
        
        Parameters:
        -----------
        name : str
            Name of the module to load lazily (e.g., 'omicverse.bulk')
        parent_module_globals : dict
            Globals dict from the parent module (usually __name__)
        warning_message : str, optional
            Warning message to display when module is first loaded
        """
        self._name = name
        self._parent_module_globals = parent_module_globals
        self._warning_message = warning_message
        self._module = None
    
    def _load_module(self):
        """Load the actual module if not already loaded."""
        if self._module is None:
            if self._warning_message:
                import warnings
                warnings.warn(self._warning_message, UserWarning, stacklevel=3)
            
            # Import the module
            self._module = importlib.import_module(self._name)
            
        return self._module
    
    def __getattr__(self, name: str):
        """Get attribute from the lazily loaded module."""
        module = self._load_module()
        return getattr(module, name)
    
    def __dir__(self) -> List[str]:
        """Return available attributes in the lazily loaded module."""
        module = self._load_module()
        return dir(module)
    
    def __repr__(self) -> str:
        if self._module is None:
            return f"<LazyLoader for '{self._name}' (not loaded)>"
        else:
            return f"<LazyLoader for '{self._name}' (loaded)>"


class LazyAttribute:
    """
    Lazy loader for individual attributes like matplotlib.pyplot.
    
    This is useful for expensive imports that are commonly used but not
    always needed immediately.
    """
    
    def __init__(self, import_path: str, attribute_name: str = None):
        """
        Initialize lazy attribute loader.
        
        Parameters:
        -----------
        import_path : str
            Full import path (e.g., 'matplotlib.pyplot')
        attribute_name : str, optional
            Specific attribute to import. If None, imports the whole module.
        """
        self._import_path = import_path
        self._attribute_name = attribute_name
        self._cached_value = None
        self._is_loaded = False
    
    def __getattr__(self, name: str):
        """Get attribute from the lazily loaded object."""
        if not self._is_loaded:
            self._load()
        return getattr(self._cached_value, name)
    
    def __call__(self, *args, **kwargs):
        """Make the object callable if the underlying object is callable."""
        if not self._is_loaded:
            self._load()
        return self._cached_value(*args, **kwargs)
    
    def __dir__(self) -> List[str]:
        """Return available attributes."""
        if not self._is_loaded:
            self._load()
        return dir(self._cached_value)
    
    def _load(self):
        """Load the actual object."""
        if not self._is_loaded:
            module = importlib.import_module(self._import_path)
            if self._attribute_name:
                self._cached_value = getattr(module, self._attribute_name)
            else:
                self._cached_value = module
            self._is_loaded = True
    
    def __repr__(self) -> str:
        if self._is_loaded:
            return f"<LazyAttribute for '{self._import_path}' (loaded)>"
        else:
            return f"<LazyAttribute for '{self._import_path}' (not loaded)>"


def create_lazy_module(name: str, parent_module_globals: dict) -> LazyLoader:
    """
    Convenience function to create a lazy loader for a module.
    
    Parameters:
    -----------
    name : str
        Name of the module to load lazily
    parent_module_globals : dict
        Globals dict from the parent module
        
    Returns:
    --------
    LazyLoader
        Lazy loader instance for the module
    """
    return LazyLoader(name, parent_module_globals)


def create_lazy_attribute(import_path: str, attribute_name: str = None) -> LazyAttribute:
    """
    Convenience function to create a lazy loader for an attribute.
    
    Parameters:
    -----------
    import_path : str
        Full import path (e.g., 'matplotlib.pyplot')
    attribute_name : str, optional
        Specific attribute to import
        
    Returns:
    --------
    LazyAttribute
        Lazy loader instance for the attribute
    """
    return LazyAttribute(import_path, attribute_name)
"""
Resource utilities for accessing package data files with fallback methods.

This module provides robust methods to access resource files in different
packaging scenarios, including support for pkg_resources and importlib.resources.
"""

import warnings
from pathlib import Path
from typing import Optional, Union


def get_resource_path(package_name: str, filename: str, 
                     validate_pickle: bool = True) -> Optional[Path]:
    """
    Get resource path with fallback methods for different packaging scenarios.
    
    This function tries multiple methods to locate resource files:
    1. Standard Path approach (for development)
    2. pkg_resources (for older packaging systems)
    3. importlib.resources (for modern packaging)
    4. Relative to installed package
    
    Args:
        package_name: Name of the package (e.g., 'omicverse.external.scllm.geneformer')
        filename: Name of the resource file
        validate_pickle: Whether to validate that the file is a real pickle file
        
    Returns:
        Path to the resource file, or None if not found
    """
    
    # Method 1: Try pkg_resources (most reliable for packaged installations)
    try:
        import pkg_resources
        resource_path = pkg_resources.resource_filename(package_name, filename)
        path = Path(resource_path)
        if path.exists() and (not validate_pickle or _is_valid_pickle_file(path)):
            return path
    except (ImportError, Exception) as e:
        warnings.warn(f"pkg_resources method failed: {e}")
    
    # Method 2: Try importlib.resources (modern Python approach)
    try:
        import importlib.resources as resources
        try:
            # Python 3.9+
            with resources.files(package_name) as pkg:
                resource_path = pkg / filename
                if resource_path.is_file() and (not validate_pickle or _is_valid_pickle_file(resource_path)):
                    return Path(str(resource_path))
        except AttributeError:
            # Python 3.7-3.8
            with resources.path(package_name, filename) as resource_path:
                if resource_path.exists() and (not validate_pickle or _is_valid_pickle_file(resource_path)):
                    return resource_path
    except (ImportError, Exception) as e:
        warnings.warn(f"importlib.resources method failed: {e}")
    
    # Method 3: Try standard Path approach (for development environments)
    try:
        # Convert package name to path
        package_parts = package_name.split('.')
        if package_parts[0] == 'omicverse':
            import omicverse
            base_path = Path(omicverse.__file__).parent
            for part in package_parts[1:]:
                base_path = base_path / part
            resource_path = base_path / filename
            if resource_path.exists() and (not validate_pickle or _is_valid_pickle_file(resource_path)):
                return resource_path
    except (ImportError, Exception) as e:
        warnings.warn(f"Standard path method failed: {e}")
    
    return None


def _is_valid_pickle_file(filepath: Union[str, Path]) -> bool:
    """
    Check if file is a valid pickle file (not a Git LFS pointer).
    
    Args:
        filepath: Path to check
        
    Returns:
        bool: True if file appears to be valid pickle file
    """
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            return False
            
        # Check file size (Git LFS pointers are typically very small)
        file_size = filepath.stat().st_size
        if file_size < 1000:  # Less than 1KB is suspicious for dictionary files
            # Check if it's a Git LFS pointer
            with open(filepath, 'rb') as f:
                first_bytes = f.read(50)
                if (b'version https://git-lfs' in first_bytes or 
                    b'oid sha256' in first_bytes or 
                    b'size' in first_bytes):
                    return False
        
        # Try to verify it's a valid pickle file
        import pickle
        with open(filepath, 'rb') as f:
            # Just read the first few bytes to check pickle header
            header = f.read(10)
            # Pickle files typically start with specific bytes
            if (header.startswith(b'\x80') or  # Protocol 2+
                header.startswith(b']') or     # List
                header.startswith(b'(') or     # Tuple
                header.startswith(b'}')):      # Dict
                return True
            return False
            
    except Exception:
        return False


def get_geneformer_resource(filename: str, model_size: str = "104M") -> Optional[Path]:
    """
    Get Geneformer-specific resource files with automatic fallbacks.
    
    Args:
        filename: Base filename (e.g., 'gene_median_dictionary.pkl')
        model_size: Model size ('104M' or '30M')
        
    Returns:
        Path to the resource file, or None if not found
    """
    
    # Construct full filename
    if model_size == "30M":
        if "gene_dictionaries_30m" not in filename:
            full_filename = f"gene_dictionaries_30m/{filename.replace('.pkl', '_gc30M.pkl')}"
    else:
        full_filename = filename.replace('.pkl', '_gc104M.pkl')
    
    # Try to get the resource
    resource_path = get_resource_path('omicverse.external.scllm.geneformer', full_filename)
    
    if resource_path is None:
        warnings.warn(f"Could not find Geneformer resource: {full_filename}")
        print(f"💡 To fix this, you can:")
        print(f"1. Download files using: git lfs pull")
        print(f"2. Provide custom path to _initialize_tokenizer()")
        print(f"3. Ensure package includes actual .pkl files, not LFS pointers")
    
    return resource_path


def show_resource_access_examples():
    """Show examples of how to use resource access methods."""
    
    examples = """
📖 Resource Access Examples:

1. Using pkg_resources (recommended for packaged installations):
   import pkg_resources
   gene_median_file = pkg_resources.resource_filename(
       'omicverse.external.scllm.geneformer', 
       'gene_median_dictionary_gc104M.pkl'
   )

2. Using importlib.resources (modern Python):
   import importlib.resources as resources
   with resources.path('omicverse.external.scllm.geneformer', 
                      'gene_median_dictionary_gc104M.pkl') as path:
       gene_median_file = str(path)

3. Using our utility function:
   from omicverse.external.scllm.resource_utils import get_geneformer_resource
   gene_median_file = get_geneformer_resource('gene_median_dictionary.pkl')

4. Direct path specification:
   model._initialize_tokenizer(
       gene_median_file='/path/to/your/gene_median_dictionary.pkl',
       token_dictionary_file='/path/to/your/token_dictionary.pkl'
   )

5. For packaged distributions, include files in setup.py/pyproject.toml:
   # setup.py
   package_data={
       'omicverse.external.scllm.geneformer': ['*.pkl', 'gene_dictionaries_30m/*.pkl']
   }
   
   # or pyproject.toml
   [tool.setuptools.package-data]
   "omicverse.external.scllm.geneformer" = ["*.pkl", "gene_dictionaries_30m/*.pkl"]
"""
    print(examples)


if __name__ == "__main__":
    # Test the resource access functions
    print("🧪 Testing resource access utilities...")
    
    # Test Geneformer resources
    gene_median = get_geneformer_resource('gene_median_dictionary.pkl')
    print(f"Gene median file: {gene_median}")
    
    token_dict = get_geneformer_resource('token_dictionary.pkl')  
    print(f"Token dictionary file: {token_dict}")
    
    show_resource_access_examples()
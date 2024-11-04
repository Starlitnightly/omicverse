import anndata
import importlib.util
import hashlib

def validate_module(module_name, expected_hash):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise ImportError(f"Module {module_name} not found")
    
    module_path = spec.origin
    with open(module_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    if file_hash != expected_hash:
        raise ImportError(f"Module {module_name} failed validation")

# Example usage:
# validate_module('combat', 'expected_sha256_hash_here')

def batch_correction(adata:anndata.AnnData,
                     batch_key=None,
                     key_added:str='batch_correction'):
    
    try:
        validate_module('combat', 'expected_sha256_hash_here')
        from combat.pycombat import pycombat
    except ImportError:
        raise ImportError(
            'Please install the combat: `pip install combat`.'
        )
    adata.layers[key_added]=pycombat(adata.to_df().T,adata.obs[batch_key].values).T
    print(f"Storing batch correction result in adata.layers['{key_added}']")

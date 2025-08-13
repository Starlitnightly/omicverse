"""
OmicVerse format adapters for datacollect module.

Provides functions to convert collected data to OmicVerse-compatible formats
including AnnData, pandas DataFrames, and MuData objects.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union

try:
    import anndata as ad
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False
    ad = None

try:
    import mudata as md
    MUDATA_AVAILABLE = True
except ImportError:
    MUDATA_AVAILABLE = False
    md = None


def to_pandas(data: Dict[str, Any], data_type: str = "auto") -> pd.DataFrame:
    """
    Convert collected data to pandas DataFrame.
    
    Args:
        data: Data dictionary from API collection
        data_type: Type of data ("expression", "protein", "variant", etc.)
        
    Returns:
        pandas DataFrame
    """
    if data_type == "expression" or "expression" in str(data):
        return _convert_expression_to_pandas(data)
    elif data_type == "protein" or "protein" in str(data):
        return _convert_protein_to_pandas(data)
    elif data_type == "variant" or "variant" in str(data):
        return _convert_variant_to_pandas(data)
    else:
        # Generic conversion
        return pd.json_normalize(data)


def to_anndata(data: Dict[str, Any], **kwargs) -> Optional["ad.AnnData"]:
    """
    Convert expression data to AnnData format.
    
    Args:
        data: Gene expression data from GEO or similar
        **kwargs: Additional arguments for AnnData creation
        
    Returns:
        AnnData object if anndata is available, None otherwise
    """
    if not ANNDATA_AVAILABLE:
        raise ImportError("anndata package is required for AnnData conversion")
    
    # Convert expression data to AnnData format
    if "expression_matrix" in data:
        X = np.array(data["expression_matrix"])
        obs = pd.DataFrame(data.get("samples", {}))
        var = pd.DataFrame(data.get("genes", {}))
        
        return ad.AnnData(X=X, obs=obs, var=var, **kwargs)
    
    # Handle GEO-specific format
    elif "GSE" in str(data) or "samples" in data:
        return _convert_geo_to_anndata(data, **kwargs)
    
    else:
        raise ValueError("Data format not suitable for AnnData conversion")


def to_mudata(data_dict: Dict[str, Dict[str, Any]], **kwargs) -> Optional["md.MuData"]:
    """
    Convert multi-omics data to MuData format.
    
    Args:
        data_dict: Dictionary with multiple omics data types
        **kwargs: Additional arguments for MuData creation
        
    Returns:
        MuData object if mudata is available, None otherwise
    """
    if not MUDATA_AVAILABLE:
        raise ImportError("mudata package is required for MuData conversion")
    
    mod_dict = {}
    
    for modality, data in data_dict.items():
        if modality in ["rna", "expression", "transcriptomics"]:
            mod_dict["rna"] = to_anndata(data)
        elif modality in ["protein", "proteomics"]:
            mod_dict["protein"] = _convert_protein_to_anndata(data)
        elif modality in ["atac", "chromatin", "accessibility"]:
            mod_dict["atac"] = _convert_chromatin_to_anndata(data)
        # Add more modalities as needed
    
    return md.MuData(mod_dict, **kwargs)


def _convert_expression_to_pandas(data: Dict[str, Any]) -> pd.DataFrame:
    """Convert expression data to pandas DataFrame."""
    if "expression_matrix" in data:
        df = pd.DataFrame(data["expression_matrix"])
        if "genes" in data:
            df.index = data["genes"]
        if "samples" in data:
            df.columns = data["samples"]
        return df
    else:
        return pd.json_normalize(data)


def _convert_protein_to_pandas(data: Dict[str, Any]) -> pd.DataFrame:
    """Convert protein data to pandas DataFrame."""
    # Flatten protein data structure
    flat_data = []
    
    if isinstance(data, list):
        for protein in data:
            flat_data.append(_flatten_protein_dict(protein))
    else:
        flat_data.append(_flatten_protein_dict(data))
    
    return pd.DataFrame(flat_data)


def _convert_variant_to_pandas(data: Dict[str, Any]) -> pd.DataFrame:
    """Convert variant data to pandas DataFrame."""
    if "variants" in data:
        return pd.DataFrame(data["variants"])
    else:
        return pd.json_normalize(data)


def _convert_geo_to_anndata(data: Dict[str, Any], **kwargs) -> "ad.AnnData":
    """Convert GEO data to AnnData format."""
    # Extract expression matrix
    if "expression_data" in data:
        X = np.array(data["expression_data"])
    else:
        raise ValueError("No expression data found in GEO data")
    
    # Create obs (samples) dataframe
    obs_data = data.get("sample_metadata", {})
    obs = pd.DataFrame(obs_data) if obs_data else pd.DataFrame(index=range(X.shape[0]))
    
    # Create var (genes) dataframe
    var_data = data.get("gene_metadata", {})
    var = pd.DataFrame(var_data) if var_data else pd.DataFrame(index=range(X.shape[1]))
    
    # Create AnnData object
    adata = ad.AnnData(X=X, obs=obs, var=var, **kwargs)
    
    # Add additional metadata
    if "platform_info" in data:
        adata.uns["platform"] = data["platform_info"]
    if "series_info" in data:
        adata.uns["series"] = data["series_info"]
    
    return adata


def _convert_protein_to_anndata(data: Dict[str, Any]) -> "ad.AnnData":
    """Convert protein data to AnnData-like format."""
    # This is a placeholder - protein data doesn't typically fit AnnData format well
    # but we can create a representation for multi-omics integration
    protein_features = _extract_protein_features(data)
    X = np.array(protein_features).reshape(1, -1)  # Single observation
    
    obs = pd.DataFrame({"protein_id": [data.get("id", "unknown")]})
    var = pd.DataFrame({"feature": list(protein_features.keys())})
    
    return ad.AnnData(X=X, obs=obs, var=var)


def _convert_chromatin_to_anndata(data: Dict[str, Any]) -> "ad.AnnData":
    """Convert chromatin accessibility data to AnnData format."""
    # Placeholder for chromatin data conversion
    # Implementation depends on specific data structure
    pass


def _flatten_protein_dict(protein: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested protein dictionary for pandas conversion."""
    flat = {}
    
    for key, value in protein.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat[f"{key}_{subkey}"] = subvalue
        elif isinstance(value, list):
            flat[key] = "; ".join(str(v) for v in value)
        else:
            flat[key] = value
    
    return flat


def _extract_protein_features(data: Dict[str, Any]) -> Dict[str, float]:
    """Extract numerical features from protein data."""
    features = {}
    
    # Extract numerical features
    if "length" in data:
        features["sequence_length"] = float(data["length"])
    if "molecular_weight" in data:
        features["molecular_weight"] = float(data["molecular_weight"])
    
    # Add more feature extraction as needed
    
    return features

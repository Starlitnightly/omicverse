"""
OmicVerse format adapters for DataCollect.

This module provides utilities to convert collected biological data
into OmicVerse-compatible formats (AnnData, pandas, MuData).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List
import logging

logger = logging.getLogger(__name__)

def to_pandas(data: Dict[str, Any], data_type: str = "generic") -> pd.DataFrame:
    """
    Convert collected data to pandas DataFrame.
    
    Parameters:
    - data: Raw data from API client
    - data_type: Type of data ('protein', 'gene', 'expression', 'pathway', 'variant')
    
    Returns:
    - pandas DataFrame with structured data
    """
    try:
        if isinstance(data, dict):
            if data_type == "protein":
                return _protein_to_pandas(data)
            elif data_type == "gene":
                return _gene_to_pandas(data)
            elif data_type == "expression":
                return _expression_to_pandas(data)
            elif data_type == "pathway":
                return _pathway_to_pandas(data)
            elif data_type == "variant":
                return _variant_to_pandas(data)
            else:
                return _generic_to_pandas(data)
        elif isinstance(data, list):
            return pd.DataFrame(data)
        else:
            return pd.DataFrame([data])
    except Exception as e:
        logger.error(f"Error converting to pandas: {e}")
        return pd.DataFrame()

def to_anndata(data: Dict[str, Any], 
               obs_keys: Optional[List[str]] = None,
               var_keys: Optional[List[str]] = None) -> 'anndata.AnnData':
    """
    Convert collected data to AnnData format for OmicVerse compatibility.
    
    Parameters:
    - data: Raw data from API client
    - obs_keys: Keys to use for observation (cell) metadata
    - var_keys: Keys to use for variable (gene) metadata
    
    Returns:
    - AnnData object
    """
    try:
        import anndata as ad
        
        if 'expression_matrix' in data:
            # Gene expression data
            X = np.array(data['expression_matrix'])
            
            # Create observation metadata
            obs_data = {}
            if 'samples' in data:
                for key, values in data['samples'].items():
                    if obs_keys is None or key in obs_keys:
                        obs_data[key] = values
            obs = pd.DataFrame(obs_data) if obs_data else None
            
            # Create variable metadata
            var_data = {}
            if 'genes' in data:
                for key, values in data['genes'].items():
                    if var_keys is None or key in var_keys:
                        var_data[key] = values
            var = pd.DataFrame(var_data) if var_data else None
            
            return ad.AnnData(X=X, obs=obs, var=var)
        
        else:
            # Convert tabular data to AnnData
            df = to_pandas(data)
            if len(df) == 0:
                return ad.AnnData()
                
            # Create simple AnnData from DataFrame
            if 'gene_expression' in df.columns:
                # Pivot to create expression matrix
                if 'sample_id' in df.columns and 'gene_symbol' in df.columns:
                    pivot = df.pivot_table(
                        index='sample_id', 
                        columns='gene_symbol', 
                        values='gene_expression',
                        aggfunc='mean'
                    ).fillna(0)
                    return ad.AnnData(X=pivot.values, 
                                     obs=pd.DataFrame(index=pivot.index),
                                     var=pd.DataFrame(index=pivot.columns))
            
            # Default: create AnnData with DataFrame as obs
            return ad.AnnData(obs=df)
            
    except ImportError:
        logger.warning("AnnData not available. Install with: pip install anndata")
        return None
    except Exception as e:
        logger.error(f"Error converting to AnnData: {e}")
        return None

def to_mudata(data: Dict[str, Any]) -> 'mudata.MuData':
    """
    Convert collected data to MuData format for multi-omics analysis.
    
    Parameters:
    - data: Raw data from API client, should contain multiple modalities
    
    Returns:
    - MuData object
    """
    try:
        import mudata as md
        
        modalities = {}
        
        if isinstance(data, dict):
            for key, value in data.items():
                if key in ['rna', 'protein', 'atac', 'methylation']:
                    # Convert each modality to AnnData
                    adata = to_anndata(value)
                    if adata is not None:
                        modalities[key] = adata
                elif 'expression' in key.lower():
                    adata = to_anndata(value)
                    if adata is not None:
                        modalities['rna'] = adata
                elif 'protein' in key.lower():
                    adata = to_anndata(value)
                    if adata is not None:
                        modalities['protein'] = adata
        
        if modalities:
            return md.MuData(modalities)
        else:
            # Fallback: create single modality MuData
            adata = to_anndata(data)
            if adata is not None:
                return md.MuData({'data': adata})
            return md.MuData()
            
    except ImportError:
        logger.warning("MuData not available. Install with: pip install mudata")
        return None
    except Exception as e:
        logger.error(f"Error converting to MuData: {e}")
        return None

def _protein_to_pandas(data: Dict[str, Any]) -> pd.DataFrame:
    """Convert protein data to pandas DataFrame."""
    rows = []
    
    if 'entries' in data:
        for entry in data['entries']:
            row = {
                'protein_id': entry.get('accession', ''),
                'name': entry.get('name', ''),
                'gene_name': entry.get('gene_name', ''),
                'organism': entry.get('organism', ''),
                'sequence': entry.get('sequence', ''),
                'length': len(entry.get('sequence', '')),
                'description': entry.get('description', '')
            }
            
            # Add features if available
            if 'features' in entry:
                row['num_features'] = len(entry['features'])
                row['domains'] = '; '.join([f['description'] for f in entry['features'] if f['type'] == 'domain'])
            
            rows.append(row)
    elif 'accession' in data:
        # Single protein entry
        rows.append({
            'protein_id': data.get('accession', ''),
            'name': data.get('name', ''),
            'gene_name': data.get('gene_name', ''),
            'organism': data.get('organism', ''),
            'sequence': data.get('sequence', ''),
            'length': len(data.get('sequence', '')),
            'description': data.get('description', '')
        })
    
    return pd.DataFrame(rows)

def _gene_to_pandas(data: Dict[str, Any]) -> pd.DataFrame:
    """Convert gene data to pandas DataFrame."""
    rows = []
    
    if 'genes' in data:
        for gene in data['genes']:
            row = {
                'gene_id': gene.get('id', ''),
                'symbol': gene.get('display_name', ''),
                'biotype': gene.get('biotype', ''),
                'chromosome': gene.get('seq_region_name', ''),
                'start': gene.get('start', 0),
                'end': gene.get('end', 0),
                'strand': gene.get('strand', 0),
                'description': gene.get('description', '')
            }
            rows.append(row)
    elif 'id' in data:
        # Single gene entry
        rows.append({
            'gene_id': data.get('id', ''),
            'symbol': data.get('display_name', ''),
            'biotype': data.get('biotype', ''),
            'chromosome': data.get('seq_region_name', ''),
            'start': data.get('start', 0),
            'end': data.get('end', 0),
            'strand': data.get('strand', 0),
            'description': data.get('description', '')
        })
    
    return pd.DataFrame(rows)

def _expression_to_pandas(data: Dict[str, Any]) -> pd.DataFrame:
    """Convert expression data to pandas DataFrame."""
    rows = []
    
    if 'samples' in data and 'expression_data' in data:
        # GEO-style data
        samples = data['samples']
        expression = data['expression_data']
        
        for sample_id, sample_data in samples.items():
            for gene_symbol, expression_value in expression.get(sample_id, {}).items():
                row = {
                    'sample_id': sample_id,
                    'gene_symbol': gene_symbol,
                    'expression_value': expression_value,
                    'condition': sample_data.get('condition', ''),
                    'tissue': sample_data.get('tissue', ''),
                    'cell_type': sample_data.get('cell_type', '')
                }
                rows.append(row)
    
    return pd.DataFrame(rows)

def _pathway_to_pandas(data: Dict[str, Any]) -> pd.DataFrame:
    """Convert pathway data to pandas DataFrame."""
    rows = []
    
    if 'pathways' in data:
        for pathway in data['pathways']:
            row = {
                'pathway_id': pathway.get('id', ''),
                'name': pathway.get('name', ''),
                'description': pathway.get('description', ''),
                'num_genes': len(pathway.get('genes', [])),
                'genes': '; '.join(pathway.get('genes', [])),
                'database': pathway.get('database', '')
            }
            rows.append(row)
    elif 'id' in data:
        # Single pathway entry
        rows.append({
            'pathway_id': data.get('id', ''),
            'name': data.get('name', ''),
            'description': data.get('description', ''),
            'num_genes': len(data.get('genes', [])),
            'genes': '; '.join(data.get('genes', [])),
            'database': data.get('database', '')
        })
    
    return pd.DataFrame(rows)

def _variant_to_pandas(data: Dict[str, Any]) -> pd.DataFrame:
    """Convert variant data to pandas DataFrame."""
    rows = []
    
    if 'variants' in data:
        for variant in data['variants']:
            row = {
                'variant_id': variant.get('id', ''),
                'chromosome': variant.get('chr', ''),
                'position': variant.get('pos', 0),
                'ref_allele': variant.get('ref', ''),
                'alt_allele': variant.get('alt', ''),
                'rsid': variant.get('rsid', ''),
                'clinical_significance': variant.get('clinical_significance', ''),
                'consequence': variant.get('most_severe_consequence', '')
            }
            rows.append(row)
    
    return pd.DataFrame(rows)

def _generic_to_pandas(data: Dict[str, Any]) -> pd.DataFrame:
    """Convert generic data to pandas DataFrame."""
    if isinstance(data, dict):
        # Flatten nested dictionaries
        flat_data = {}
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                flat_data[key] = str(value)
            else:
                flat_data[key] = value
        return pd.DataFrame([flat_data])
    else:
        return pd.DataFrame([data])

# Convenience functions for specific data types
def protein_to_anndata(protein_data: Dict[str, Any]) -> 'anndata.AnnData':
    """Convert protein data to AnnData format."""
    return to_anndata(protein_data)

def expression_to_anndata(expression_data: Dict[str, Any]) -> 'anndata.AnnData':
    """Convert expression data to AnnData format."""
    return to_anndata(expression_data)

def pathway_to_pandas(pathway_data: Dict[str, Any]) -> pd.DataFrame:
    """Convert pathway data to pandas DataFrame."""
    return to_pandas(pathway_data, "pathway")
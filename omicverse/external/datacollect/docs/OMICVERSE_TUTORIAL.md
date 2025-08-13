# DataCollect Tutorial for OmicVerse Users

This tutorial demonstrates how to use DataCollect within OmicVerse for comprehensive bioinformatics data collection and analysis workflows.

## Table of Contents
1. [Setup and Basic Usage](#setup-and-basic-usage)
2. [Protein Data Collection](#protein-data-collection)
3. [Gene Expression Analysis](#gene-expression-analysis)
4. [Genomics and Variants](#genomics-and-variants)
5. [Pathway Analysis](#pathway-analysis)
6. [Multi-Omics Integration](#multi-omics-integration)
7. [Advanced Workflows](#advanced-workflows)

## Setup and Basic Usage

### Import and Verify Installation
```python
import omicverse as ov
import pandas as pd
import numpy as np

# Verify DataCollect is available
print("DataCollect available:", hasattr(ov.external, 'datacollect'))
print("Version:", getattr(ov.external.datacollect, '__version__', 'Unknown'))

# List available API clients
from omicverse.external.datacollect.api import *
print("\nAvailable API clients:")
print("- Protein APIs:", ['UniProtClient', 'PDBClient', 'AlphaFoldClient'])
print("- Genomics APIs:", ['EnsemblClient', 'ClinVarClient', 'dbSNPClient'])
print("- Expression APIs:", ['GEOClient', 'OpenTargetsClient'])
print("- Pathway APIs:", ['KEGGClient', 'ReactomeClient'])
```

### Basic Data Collection
```python
# Collect protein data (returns pandas DataFrame by default)
protein_data = ov.external.datacollect.collect_protein_data("P04637")  # p53
print(f"Protein data shape: {protein_data.shape}")
print(protein_data.head())
```

## Protein Data Collection

### Single Protein Analysis
```python
# Collect comprehensive p53 data
p53_uniprot = ov.external.datacollect.collect_protein_data("P04637", source="uniprot", format="pandas")
p53_structure = ov.external.datacollect.collect_protein_data("P04637", source="alphafold", format="pandas")
p53_interactions = ov.external.datacollect.collect_protein_data("TP53", source="string", format="pandas")

print("UniProt data columns:", list(p53_uniprot.columns))
print("Protein length:", p53_uniprot['length'].iloc[0])
print("Number of interactions:", len(p53_interactions))
```

### Batch Protein Collection
```python
from omicverse.external.datacollect.collectors import UniProtCollector

# Collect multiple proteins
protein_list = ["P04637", "P21359", "P53_HUMAN", "NF1_HUMAN"]
collector = UniProtCollector()

# Batch collection with features
proteins_df = collector.collect_batch(
    protein_list, 
    include_features=True,
    include_go_terms=True
)

print(f"Collected {len(proteins_df)} proteins")
print("Columns available:", list(proteins_df.columns))

# Convert to format suitable for downstream analysis
proteins_summary = proteins_df[['protein_id', 'gene_name', 'length', 'organism', 'description']]
print(proteins_summary)
```

## Gene Expression Analysis

### GEO Dataset Collection
```python
# Collect gene expression data from GEO
geo_id = "GSE123456"  # Replace with actual GEO accession
try:
    # Automatic conversion to AnnData for OmicVerse compatibility
    expression_adata = ov.external.datacollect.collect_expression_data(
        geo_id, 
        format="anndata"
    )
    
    print(f"Expression data shape: {expression_adata.shape}")
    print(f"Samples: {expression_adata.n_obs}")
    print(f"Genes: {expression_adata.n_vars}")
    
    # Check sample metadata
    print("\nSample metadata columns:", list(expression_adata.obs.columns))
    print("Gene metadata columns:", list(expression_adata.var.columns))
    
except Exception as e:
    print(f"GEO data collection error: {e}")
    # Create example data for demonstration
    n_samples, n_genes = 100, 2000
    X = np.random.negative_binomial(5, 0.3, size=(n_samples, n_genes))
    
    import anndata as ad
    expression_adata = ad.AnnData(X=X.astype(float))
    expression_adata.obs['condition'] = ['treatment'] * 50 + ['control'] * 50
    expression_adata.var['gene_symbol'] = [f'Gene_{i}' for i in range(n_genes)]
    print("Created example expression data for demonstration")
```

### Integration with OmicVerse Analysis
```python
# Preprocessing with OmicVerse
ov.pp.preprocess(expression_adata, mode='shiftlog|pearson', n_HVGs=2000)

# Quality control
ov.pp.qc(expression_adata, tresh={'mito_perc': 0.2, 'n_UMIs': 500})

# Differential expression analysis
try:
    deg_results = ov.bulk.pyDEG(expression_adata, group='condition')
    print(f"Found {len(deg_results)} differentially expressed genes")
    
    # Display top DEGs
    top_degs = deg_results.head(10)
    print("\nTop 10 DEGs:")
    print(top_degs[['names', 'log2FC', 'pvals_adj']])
    
except Exception as e:
    print(f"DEG analysis note: {e}")
```

## Genomics and Variants

### Gene Information Collection
```python
# Collect gene information from Ensembl
gene_id = "ENSG00000141510"  # TP53 gene ID
gene_data = ov.external.datacollect.collect_gene_data(gene_id, source="ensembl", format="pandas")

print("Gene information:")
print(f"Gene ID: {gene_data['gene_id'].iloc[0]}")
print(f"Symbol: {gene_data['symbol'].iloc[0]}")
print(f"Chromosome: {gene_data['chromosome'].iloc[0]}")
print(f"Biotype: {gene_data['biotype'].iloc[0]}")
```

### Variant Data Collection
```python
# Collect variant information
variant_id = "rs7412"  # APOE variant
try:
    variant_data = ov.external.datacollect.collect_variant_data(
        variant_id, 
        source="dbsnp", 
        format="pandas"
    )
    
    print("Variant information:")
    print(variant_data[['variant_id', 'chromosome', 'position', 'consequence']])
    
except Exception as e:
    print(f"Variant collection note: {e}")

# Clinical variant information
try:
    clinical_variants = ov.external.datacollect.collect_variant_data(
        "BRCA1", 
        source="clinvar", 
        format="pandas"
    )
    
    print(f"\nFound {len(clinical_variants)} clinical variants for BRCA1")
    
except Exception as e:
    print(f"Clinical variant collection note: {e}")
```

### GWAS Data Integration
```python
# Collect GWAS associations
try:
    gwas_data = ov.external.datacollect.collect_gwas_data("diabetes", format="pandas")
    print(f"Found {len(gwas_data)} GWAS associations for diabetes")
    
    # Top associations by p-value
    if len(gwas_data) > 0:
        top_gwas = gwas_data.nsmallest(5, 'pvalue')[['trait', 'gene_symbol', 'pvalue', 'or_beta']]
        print("\nTop GWAS associations:")
        print(top_gwas)
        
except Exception as e:
    print(f"GWAS data collection note: {e}")
```

## Pathway Analysis

### KEGG Pathway Collection
```python
# Collect KEGG pathway data
pathway_id = "hsa04110"  # Cell cycle pathway
pathway_data = ov.external.datacollect.collect_pathway_data(
    pathway_id, 
    source="kegg", 
    format="pandas"
)

print("Pathway information:")
print(f"Pathway ID: {pathway_data['pathway_id'].iloc[0]}")
print(f"Name: {pathway_data['name'].iloc[0]}")
print(f"Number of genes: {pathway_data['num_genes'].iloc[0]}")
print(f"Genes: {pathway_data['genes'].iloc[0][:100]}...")  # First 100 chars
```

### Reactome Pathway Analysis
```python
# Collect Reactome pathway
reactome_id = "R-HSA-68886"  # M Phase
try:
    reactome_data = ov.external.datacollect.collect_pathway_data(
        reactome_id, 
        source="reactome", 
        format="pandas"
    )
    
    print("\nReactome pathway:")
    print(f"ID: {reactome_data['pathway_id'].iloc[0]}")
    print(f"Name: {reactome_data['name'].iloc[0]}")
    
except Exception as e:
    print(f"Reactome collection note: {e}")
```

### Pathway Enrichment with Collected Data
```python
# Use collected pathway data for enrichment analysis
if 'expression_adata' in locals() and 'deg_results' in locals():
    try:
        # Extract significantly expressed genes
        significant_genes = deg_results[deg_results['pvals_adj'] < 0.05]['names'].tolist()
        
        # Collect multiple pathways
        pathway_ids = ["hsa04110", "hsa04151", "hsa04210"]  # Cell cycle related
        
        pathway_gene_sets = {}
        for pid in pathway_ids:
            try:
                p_data = ov.external.datacollect.collect_pathway_data(pid, source="kegg", format="pandas")
                genes = p_data['genes'].iloc[0].split('; ') if len(p_data) > 0 else []
                pathway_gene_sets[p_data['name'].iloc[0]] = genes
            except:
                continue
        
        print(f"\nCollected {len(pathway_gene_sets)} pathway gene sets")
        print("Pathway names:", list(pathway_gene_sets.keys()))
        
        # Use with OmicVerse GSEA
        # gsea_results = ov.bulk.pyGSEA(expression_adata, pathway_gene_sets)
        
    except Exception as e:
        print(f"Pathway enrichment setup note: {e}")
```

## Multi-Omics Integration

### Collecting Multi-Modal Data
```python
# Collect complementary data types
target_gene = "TP53"
gene_id = "ENSG00000141510"

# 1. Protein data
protein_info = ov.external.datacollect.collect_protein_data("P04637", format="pandas")

# 2. Gene expression (from previous section)
# expression_adata already collected

# 3. Drug target information
try:
    target_data = ov.external.datacollect.collect_target_data(gene_id, source="opentargets", format="pandas")
    print(f"Found {len(target_data)} drug target associations")
except Exception as e:
    print(f"Target data collection note: {e}")

# 4. Pathway involvement
pathway_involvement = ov.external.datacollect.collect_pathway_data("hsa04115", format="pandas")  # p53 signaling
```

### Creating Integrated Data Views
```python
# Create integrated summary
integration_summary = {
    'gene_symbol': 'TP53',
    'protein_id': 'P04637',
    'ensembl_id': gene_id,
    'protein_length': protein_info['length'].iloc[0] if len(protein_info) > 0 else None,
    'expression_samples': expression_adata.n_obs if 'expression_adata' in locals() else None,
    'pathway_involvement': pathway_involvement['name'].iloc[0] if len(pathway_involvement) > 0 else None
}

print("Integrated data summary:")
for key, value in integration_summary.items():
    print(f"  {key}: {value}")
```

### MuData Integration
```python
# Create MuData object for multi-omics analysis
try:
    from omicverse.external.datacollect.utils.omicverse_adapters import to_mudata
    
    multi_omics_dict = {
        'expression': expression_adata,
        # Add other modalities as available
    }
    
    # Convert to MuData format
    mudata_obj = to_mudata(multi_omics_dict)
    print("Created MuData object with modalities:", list(mudata_obj.mod.keys()))
    
except ImportError:
    print("MuData not available - install with: pip install mudata")
except Exception as e:
    print(f"MuData creation note: {e}")
```

## Advanced Workflows

### Custom Analysis Pipeline
```python
def comprehensive_gene_analysis(gene_symbol, ensembl_id=None):
    """
    Comprehensive analysis pipeline combining DataCollect with OmicVerse.
    """
    results = {}
    
    # 1. Protein information
    try:
        protein_data = ov.external.datacollect.collect_protein_data(gene_symbol, format="pandas")
        results['protein'] = protein_data
        print(f"✓ Collected protein data for {gene_symbol}")
    except Exception as e:
        print(f"✗ Protein collection failed: {e}")
    
    # 2. Gene information
    if ensembl_id:
        try:
            gene_data = ov.external.datacollect.collect_gene_data(ensembl_id, format="pandas")
            results['gene'] = gene_data
            print(f"✓ Collected gene data for {ensembl_id}")
        except Exception as e:
            print(f"✗ Gene collection failed: {e}")
    
    # 3. Pathway involvement
    try:
        # Search for pathways involving this gene
        pathway_data = ov.external.datacollect.collect_pathway_data("hsa04110", format="pandas")
        results['pathways'] = pathway_data
        print(f"✓ Collected pathway data")
    except Exception as e:
        print(f"✗ Pathway collection failed: {e}")
    
    return results

# Run comprehensive analysis
gene_results = comprehensive_gene_analysis("TP53", "ENSG00000141510")
print(f"\nAnalysis completed. Collected {len(gene_results)} data types.")
```

### Batch Analysis Workflow
```python
def batch_gene_analysis(gene_list):
    """
    Analyze multiple genes in batch.
    """
    from omicverse.external.datacollect.collectors import BatchCollector
    
    collector = BatchCollector()
    
    # Batch collect protein data
    protein_results = collector.collect_proteins(gene_list, format="pandas")
    
    # Create summary report
    summary = protein_results.groupby('organism').agg({
        'protein_id': 'count',
        'length': 'mean'
    }).rename(columns={'protein_id': 'count', 'length': 'avg_length'})
    
    return protein_results, summary

# Example batch analysis
gene_list = ["TP53", "BRCA1", "EGFR", "MYC"]
try:
    batch_proteins, summary = batch_gene_analysis(gene_list)
    print("Batch analysis summary:")
    print(summary)
except Exception as e:
    print(f"Batch analysis note: {e}")
```

### Integration with OmicVerse Visualization
```python
# Visualize collected data with OmicVerse plotting functions
try:
    # If we have expression data
    if 'expression_adata' in locals():
        # PCA plot
        ov.pp.pca(expression_adata)
        ov.pl.embedding(expression_adata, basis='pca', color='condition')
        
        # UMAP plot
        ov.pp.neighbors(expression_adata)
        ov.pp.umap(expression_adata)
        ov.pl.embedding(expression_adata, basis='umap', color='condition')
    
    # Protein length distribution
    if 'protein_results' in locals():
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        protein_results['length'].hist(bins=30)
        plt.xlabel('Protein Length')
        plt.ylabel('Frequency')
        plt.title('Distribution of Protein Lengths')
        plt.show()
        
except Exception as e:
    print(f"Visualization note: {e}")
```

## Best Practices

### 1. Error Handling
```python
def safe_data_collection(identifier, source, format="pandas"):
    """
    Safe data collection with error handling.
    """
    try:
        if source == "protein":
            return ov.external.datacollect.collect_protein_data(identifier, format=format)
        elif source == "expression":
            return ov.external.datacollect.collect_expression_data(identifier, format=format)
        elif source == "pathway":
            return ov.external.datacollect.collect_pathway_data(identifier, format=format)
        else:
            raise ValueError(f"Unknown source: {source}")
    except Exception as e:
        print(f"Collection failed for {identifier} ({source}): {e}")
        return None

# Usage
data = safe_data_collection("P04637", "protein")
if data is not None:
    print(f"Successfully collected data: {data.shape}")
```

### 2. Caching Results
```python
import pickle
import os

def cached_collection(identifier, source, cache_dir="./cache"):
    """
    Collection with local caching.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{source}_{identifier}.pkl")
    
    if os.path.exists(cache_file):
        print(f"Loading from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Collect data
    data = safe_data_collection(identifier, source)
    
    if data is not None:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Cached data: {cache_file}")
    
    return data
```

### 3. Configuration Management
```python
# Set up configuration for production use
config = {
    'api_keys': {
        'uniprot': 'your_uniprot_key',
        'ncbi': 'your_ncbi_key'
    },
    'rate_limits': {
        'default': 5,
        'uniprot': 10,
        'geo': 3
    },
    'formats': {
        'default_expression': 'anndata',
        'default_protein': 'pandas'
    }
}

# Apply configuration
# ov.external.datacollect.configure(config)
```

## Troubleshooting

### Common Issues
1. **Rate Limiting**: Reduce request frequency or add API keys
2. **Format Errors**: Check if AnnData/MuData dependencies are installed
3. **Network Issues**: Check internet connection and API endpoints
4. **Data Not Found**: Verify identifiers are correct and databases are accessible

### Getting Help
- Check `docs/TROUBLESHOOTING.md` for detailed solutions
- Review error logs for specific issues
- Consult API documentation for identifier formats

---

This tutorial demonstrates the powerful integration of DataCollect with OmicVerse, enabling seamless data collection and analysis workflows across 29+ biological databases while maintaining full compatibility with OmicVerse's multi-omics analysis ecosystem.
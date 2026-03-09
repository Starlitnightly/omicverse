---
name: biocontext-knowledge-queries
title: BioContext knowledge queries
description: "BioContext knowledge: UniProt, AlphaFold, STRING, Reactome, GO, PanglaoDB, PubMed, OpenTargets queries via ov.biocontext for gene annotation."
---

# BioContext Knowledge Queries

Use this skill when the user wants to look up gene/protein annotations, query pathway databases, find cell type markers, search biomedical literature, or explore drug-disease associations. BioContext provides programmatic access to 49 biomedical databases through a unified Python API.

This is a knowledge integration layer — use it to annotate analysis results (e.g., annotate DEG lists with protein function, find pathways for gene clusters, validate marker genes against PanglaoDB).

## Available Functions by Category

### Protein & Genomics

| Function | Database | Returns |
|----------|----------|---------|
| `query_uniprot(gene_symbol, species)` | UniProt | Protein function, domains, GO terms, references |
| `get_uniprot_id(protein_symbol, species)` | UniProt | UniProt accession ID |
| `query_alphafold(protein_symbol, species)` | AlphaFold | Predicted 3D structure, confidence scores |
| `get_ensembl_id(gene_symbol, species)` | Ensembl | Ensembl gene ID (ENSG...) |
| `query_interpro(protein_id, source_db)` | InterPro | Protein domains, families, structural info |
| `search_interpro(query, entry_type)` | InterPro | Domain search by keyword |
| `query_string(protein_symbol, species, min_score)` | STRING | Protein-protein interactions |
| `query_hpa(gene_symbol)` | Human Protein Atlas | Tissue expression, subcellular localization |

### Pathways & Functional

| Function | Database | Returns |
|----------|----------|---------|
| `query_reactome(identifier, species)` | Reactome | Pathway membership, reactions, disease links |
| `query_go(gene_name, size)` | Gene Ontology | GO term associations (BP, MF, CC) |

### Cell Biology

| Function | Database | Returns |
|----------|----------|---------|
| `query_panglaodb(species, cell_type, organ)` | PanglaoDB | Cell type marker genes with sensitivity/specificity |

### Literature

| Function | Database | Returns |
|----------|----------|---------|
| `search_literature(query, sort_by, page_size)` | Europe PMC | Publications matching keywords |
| `search_preprints(server, days, category)` | bioRxiv/medRxiv | Recent preprints |
| `get_fulltext(pmc_id)` | PubMed Central | Full-text article content |

### Drug & Clinical

| Function | Database | Returns |
|----------|----------|---------|
| `query_opentargets(query_string, variables)` | OpenTargets | Gene-disease-drug associations |
| `search_clinical_trials(condition, status)` | ClinicalTrials.gov | Active/completed trials |
| `search_drugs(brand_name, generic_name)` | openFDA | Drug information, approval status |

### Ontology

| Function | Database | Returns |
|----------|----------|---------|
| `query_efo(disease_name, size, exact_match)` | EFO | Experimental Factor Ontology terms |
| `query_chebi(chemical_name, size)` | ChEBI | Chemical entities, small molecules |
| `query_cell_ontology(cell_type, size)` | Cell Ontology | Standardized cell type hierarchy |

### Proteomics

| Function | Database | Returns |
|----------|----------|---------|
| `search_pride(keyword, page_size)` | PRIDE | Mass spectrometry proteomics datasets |

### Generic Access

```python
# List all 49 available tools with parameters
ov.biocontext.list_tools()

# Call any tool directly by name
result = ov.biocontext.call_tool("tool_name", param1=value1, ...)
```

## Usage Patterns

### Single gene lookup

```python
import omicverse as ov

# Get protein function and domains
info = ov.biocontext.query_uniprot(gene_symbol='TP53', species='9606')

# Get pathway membership
pathways = ov.biocontext.query_reactome(identifier='TP53', species='Homo sapiens')

# Get GO terms
go_terms = ov.biocontext.query_go(gene_name='TP53', size=20)
```

### Annotate a DEG list

```python
# After differential expression: annotate top genes with biological context
deg_genes = ['TP53', 'BRCA1', 'MYC', 'EGFR', 'KRAS']
annotations = {}
for gene in deg_genes:
    annotations[gene] = {
        'uniprot': ov.biocontext.query_uniprot(gene_symbol=gene),
        'pathways': ov.biocontext.query_reactome(identifier=gene),
        'go': ov.biocontext.query_go(gene_name=gene, size=5),
    }
```

### Find cell type markers

```python
# Get known markers for a cell type
markers = ov.biocontext.query_panglaodb(
    species='Hs',           # 'Hs' (human), 'Mm' (mouse), 'Dr' (zebrafish)
    cell_type='T cells',
    organ='Blood',
    min_sensitivity=0.5,
)
# Returns: DataFrame with gene symbols, sensitivity, specificity scores
```

### Drug target exploration

```python
# Find drugs targeting a gene
targets = ov.biocontext.query_opentargets(
    query_string='{ target(ensemblId: "ENSG00000141510") { associatedDiseases { rows { disease { name } score } } } }'
)

# Search clinical trials
trials = ov.biocontext.search_clinical_trials(condition='breast cancer', status='RECRUITING')
```

### Literature search

```python
# Search for papers
results = ov.biocontext.search_literature(
    query='single-cell RNA-seq BRCA1',
    sort_by='RELEVANCE',
    page_size=5,
)

# Get full text of a specific paper
text = ov.biocontext.get_fulltext(pmc_id='PMC1234567')
```

## Species Codes

Different databases use different species identifiers:

| Species | NCBI Taxon ID | Ensembl | PanglaoDB |
|---------|--------------|---------|-----------|
| Human | 9606 | homo_sapiens | Hs |
| Mouse | 10090 | mus_musculus | Mm |
| Zebrafish | 7955 | danio_rerio | Dr |
| Rat | 10116 | rattus_norvegicus | Rn |

Most functions default to human (9606 or homo_sapiens).

## Critical API Reference

### query_uniprot accepts multiple identifier types

```python
# By gene symbol (most common)
ov.biocontext.query_uniprot(gene_symbol='TP53')

# By UniProt accession
ov.biocontext.query_uniprot(protein_id='P04637')

# By protein name
ov.biocontext.query_uniprot(protein_name='Cellular tumor antigen p53')
```

### query_opentargets uses GraphQL

```python
# OpenTargets requires GraphQL query strings
# See OpenTargets Platform API docs for query syntax
result = ov.biocontext.query_opentargets(
    query_string='{ search(queryString: "BRCA1") { total hits { id name } } }'
)
```

## Troubleshooting

- **Empty results for a known gene**: Check species parameter. Default is human (9606) — pass `species='10090'` for mouse genes.
- **Timeout on large queries**: External API calls have network latency. For batch annotation, add small delays between calls to avoid rate limiting.
- **`ConnectionError`**: Requires internet access. BioContext queries external databases in real-time.
- **Gene symbol not found**: Some databases are case-sensitive. Human genes should be uppercase (TP53), mouse mixed-case (Tp53).
- **OpenTargets query fails**: GraphQL syntax must be exact. Use `ov.biocontext.list_tools()` to see available OpenTargets tool variants with example queries.

## Examples
- "Look up the protein function and pathways for my top 10 DEGs."
- "Find known T-cell markers from PanglaoDB for my annotation."
- "Search for recent papers about spatial transcriptomics and BRCA1."
- "What drugs target EGFR? Check clinical trials status."

## References
- Quick copy/paste commands: [`reference.md`](reference.md)
# BioContext quick commands

## Protein & genomics queries

```python
import omicverse as ov

# --- UniProt protein info ---
# species: NCBI taxon ID as string. '9606' = human, '10090' = mouse
info = ov.biocontext.query_uniprot(gene_symbol='TP53', species='9606')

# Gene symbol → UniProt accession ID
uniprot_id = ov.biocontext.get_uniprot_id(protein_symbol='TP53', species='9606')

# --- AlphaFold structure ---
structure = ov.biocontext.query_alphafold(protein_symbol='TP53', species='9606')

# --- Ensembl ID ---
# species: ensembl-style name ('homo_sapiens', 'mus_musculus')
ensembl_id = ov.biocontext.get_ensembl_id(gene_symbol='TP53', species='homo_sapiens')

# --- InterPro domains ---
domains = ov.biocontext.query_interpro(protein_id='P04637', include_structure_info=True)
search_results = ov.biocontext.search_interpro(query='kinase', entry_type='domain', page_size=10)

# --- STRING interactions ---
# species: NCBI taxon ID as integer. 9606 = human, 10090 = mouse
interactions = ov.biocontext.query_string(protein_symbol='TP53', species=9606, min_score=0.7)

# --- Human Protein Atlas ---
hpa = ov.biocontext.query_hpa(gene_symbol='TP53')
```

## Pathway queries

```python
# --- Reactome pathways ---
# species: full name ('Homo sapiens', 'Mus musculus')
pathways = ov.biocontext.query_reactome(identifier='TP53', species='Homo sapiens', include_disease=True)

# --- Gene Ontology ---
# size: max results to return
go_terms = ov.biocontext.query_go(gene_name='TP53', size=20)
```

## Cell type markers

```python
# --- PanglaoDB markers ---
# species: 'Hs' (human), 'Mm' (mouse), 'Dr' (zebrafish)
markers = ov.biocontext.query_panglaodb(species='Hs', cell_type='T cells', organ='Blood')

# Filter by quality
markers_hq = ov.biocontext.query_panglaodb(
    species='Mm',
    cell_type='Macrophage',
    min_sensitivity=0.5,
    min_specificity=0.5,
)
```

## Literature search

```python
# --- PubMed / Europe PMC ---
# search_type: 'lite' (fast) or 'core' (detailed)
# sort_by: 'RELEVANCE', 'DATE', 'CITED'
results = ov.biocontext.search_literature(
    query='CRISPR single-cell perturbation',
    search_type='lite',
    sort_by='RELEVANCE',
    page_size=10,
)

# --- Preprints ---
# server: 'biorxiv' or 'medrxiv'
preprints = ov.biocontext.search_preprints(
    server='biorxiv',
    days=30,            # Last 30 days
    category='genomics',
    max_results=20,
)

# --- Full text retrieval ---
fulltext = ov.biocontext.get_fulltext(pmc_id='PMC1234567')
```

## Drug & clinical queries

```python
# --- OpenTargets (gene-disease-drug links) ---
# Uses GraphQL query syntax
result = ov.biocontext.query_opentargets(
    query_string='{ search(queryString: "BRCA1") { total hits { id name } } }'
)

# --- Clinical trials ---
trials = ov.biocontext.search_clinical_trials(
    condition='lung cancer',
    status='RECRUITING',
    page_size=10,
)

# --- Drug lookup ---
drugs = ov.biocontext.search_drugs(
    generic_name='pembrolizumab',
    limit=5,
)
```

## Ontology queries

```python
# --- Experimental Factor Ontology (diseases) ---
efo = ov.biocontext.query_efo(disease_name='breast cancer', size=5, exact_match=False)

# --- Chemical Entities (small molecules) ---
chebi = ov.biocontext.query_chebi(chemical_name='aspirin', size=5)

# --- Cell Ontology (cell type hierarchy) ---
cell_ont = ov.biocontext.query_cell_ontology(cell_type='T cell', size=10)
```

## Proteomics

```python
# --- PRIDE (mass spec datasets) ---
pride = ov.biocontext.search_pride(keyword='single cell proteomics', page_size=10)
```

## Batch annotation pattern

```python
import omicverse as ov
import time

# Annotate a list of genes from DEG analysis
deg_genes = ['TP53', 'BRCA1', 'MYC', 'EGFR', 'KRAS']

results = []
for gene in deg_genes:
    info = ov.biocontext.query_uniprot(gene_symbol=gene)
    go = ov.biocontext.query_go(gene_name=gene, size=5)
    pathways = ov.biocontext.query_reactome(identifier=gene)
    results.append({
        'gene': gene,
        'uniprot': info,
        'go_terms': go,
        'pathways': pathways,
    })
    time.sleep(0.5)  # Rate limiting between API calls
```

## Generic tool access

```python
# List ALL 49 available tools with their parameters
tools_df = ov.biocontext.list_tools()
print(tools_df)

# Call any tool directly by name
result = ov.biocontext.call_tool('uniprot_protein_info', gene_symbol='TP53')
```

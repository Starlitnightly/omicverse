"""BioContext — biomedical knowledge tools for OmicVerse.

Access 49 curated biomedical database tools (UniProt, KEGG, Reactome,
PanglaoDB, Gene Ontology, ClinicalTrials, and more) via the
`BioContext <https://biocontext.ai/>`_ MCP server.

Quick start::

    import omicverse as ov

    # Protein information
    info = ov.biocontext.query_uniprot(gene_symbol='TP53')

    # Pathway analysis
    pathways = ov.biocontext.query_reactome('BRCA1')

    # Cell type markers
    markers = ov.biocontext.query_panglaodb(species='Hs', cell_type='T cells')

    # Any tool by name
    result = ov.biocontext.call_tool('get_go_terms_by_gene', gene_name='TP53')

    # List all tools
    tools_df = ov.biocontext.list_tools()
"""

from ._tools import (
    # Generic
    call_tool,
    list_tools,
    # Protein & Genomics
    query_uniprot,
    get_uniprot_id,
    query_alphafold,
    get_ensembl_id,
    query_interpro,
    search_interpro,
    query_string,
    query_hpa,
    # Pathways & Functional
    query_reactome,
    query_go,
    # Single-cell & Markers
    query_panglaodb,
    # Literature
    search_literature,
    search_preprints,
    get_fulltext,
    # Clinical & Drug
    query_opentargets,
    search_clinical_trials,
    search_drugs,
    # Ontologies
    query_efo,
    query_chebi,
    query_cell_ontology,
    # Proteomics
    search_pride,
)

from ._client import BioContextClient, get_client

__all__ = [
    "call_tool",
    "list_tools",
    "query_uniprot",
    "get_uniprot_id",
    "query_alphafold",
    "get_ensembl_id",
    "query_interpro",
    "search_interpro",
    "query_string",
    "query_hpa",
    "query_reactome",
    "query_go",
    "query_panglaodb",
    "search_literature",
    "search_preprints",
    "get_fulltext",
    "query_opentargets",
    "search_clinical_trials",
    "search_drugs",
    "query_efo",
    "query_chebi",
    "query_cell_ontology",
    "search_pride",
    "BioContextClient",
    "get_client",
]

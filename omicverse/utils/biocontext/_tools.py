"""High-level Python wrappers for BioContext MCP tools.

Each function is registered via ``@register_function`` so the
OmicVerse Agent can discover and call them automatically.
"""

from __future__ import annotations

import pandas as pd
from typing import Any, Dict, List, Optional, Union

from ..._registry import register_function
from ._client import get_client


# =====================================================================
# Generic fallback
# =====================================================================

@register_function(
    aliases=["biocontext_call", "bc_call", "BioContext工具调用"],
    category="biocontext",
    description="Call any BioContext MCP tool by name. Use list_tools() to see all 49 available tools.",
    examples=[
        "result = ov.utils.biocontext.call_tool('get_uniprot_protein_info', gene_symbol='TP53')",
        "tools = ov.utils.biocontext.list_tools()",
    ],
    related=["biocontext.list_tools"],
)
def call_tool(tool_name: str, **kwargs: Any) -> Any:
    r"""Call any BioContext MCP tool by name.

    Parameters
    ----------
    tool_name : str
        The BioContext tool name (e.g. ``'get_uniprot_protein_info'``).
    **kwargs
        Tool-specific arguments.

    Returns
    -------
    dict or str
        Tool result, automatically parsed from JSON when possible.
    """
    return get_client().call_tool(tool_name, kwargs)


@register_function(
    aliases=["biocontext_tools", "bc_tools", "BioContext工具列表"],
    category="biocontext",
    description="List all available BioContext tools with their parameters and descriptions.",
    examples=["tools = ov.utils.biocontext.list_tools()"],
)
def list_tools() -> pd.DataFrame:
    r"""List all available BioContext tools.

    Returns
    -------
    pd.DataFrame
        Table with columns ``name``, ``description``, ``parameters``, ``required``.
    """
    raw = get_client().list_tools()
    rows = []
    for t in raw:
        schema = t.get("inputSchema", {})
        rows.append({
            "name": t["name"],
            "description": (t.get("description") or "")[:120],
            "parameters": ", ".join(schema.get("properties", {}).keys()),
            "required": ", ".join(schema.get("required", [])),
        })
    return pd.DataFrame(rows)


# =====================================================================
# Protein & Genomics
# =====================================================================

@register_function(
    aliases=["uniprot_query", "protein_info", "蛋白质信息查询", "query_protein"],
    category="biocontext",
    description="Query UniProt for protein information by gene symbol, protein name, or UniProt ID.",
    examples=[
        "info = ov.utils.biocontext.query_uniprot(gene_symbol='TP53')",
        "info = ov.utils.biocontext.query_uniprot(protein_id='P04637')",
    ],
    related=["biocontext.get_uniprot_id", "biocontext.query_kegg"],
)
def query_uniprot(
    protein_id: Optional[str] = None,
    protein_name: Optional[str] = None,
    gene_symbol: Optional[str] = None,
    species: str = "9606",
    include_references: bool = False,
) -> dict:
    r"""Query UniProt protein information.

    Provide at least one of ``protein_id``, ``protein_name``, or ``gene_symbol``.

    Parameters
    ----------
    protein_id : str, optional
        UniProt accession (e.g. ``'P04637'``).
    protein_name : str, optional
        Protein name to search.
    gene_symbol : str, optional
        Gene symbol (e.g. ``'TP53'``).
    species : str
        NCBI taxonomy ID. Default ``'9606'`` (human).
    include_references : bool
        Include literature references.

    Returns
    -------
    dict
        UniProt protein record.
    """
    args: Dict[str, Any] = {"species": species, "include_references": include_references}
    if protein_id:
        args["protein_id"] = protein_id
    if protein_name:
        args["protein_name"] = protein_name
    if gene_symbol:
        args["gene_symbol"] = gene_symbol
    return get_client().call_tool("get_uniprot_protein_info", args)


@register_function(
    aliases=["uniprot_id", "蛋白质ID查询"],
    category="biocontext",
    description="Convert protein/gene symbol to UniProt accession ID.",
    examples=["uid = ov.utils.biocontext.get_uniprot_id('TP53')"],
    related=["biocontext.query_uniprot"],
)
def get_uniprot_id(protein_symbol: str, species: str = "9606") -> str:
    r"""Get UniProt accession ID from protein symbol.

    Parameters
    ----------
    protein_symbol : str
        Gene or protein symbol (e.g. ``'TP53'``).
    species : str
        NCBI taxonomy ID. Default ``'9606'`` (human).

    Returns
    -------
    str
        UniProt accession ID.
    """
    return get_client().call_tool(
        "get_uniprot_id_by_protein_symbol",
        {"protein_symbol": protein_symbol, "species": species},
    )


@register_function(
    aliases=["alphafold_query", "protein_structure", "蛋白质结构"],
    category="biocontext",
    description="Query AlphaFold database for predicted protein structure information.",
    examples=["af = ov.utils.biocontext.query_alphafold('TP53')"],
    related=["biocontext.query_uniprot"],
)
def query_alphafold(protein_symbol: str, species: str = "9606") -> dict:
    r"""Query AlphaFold DB for predicted protein structure.

    Parameters
    ----------
    protein_symbol : str
        Gene or protein symbol.
    species : str
        NCBI taxonomy ID. Default ``'9606'`` (human).

    Returns
    -------
    dict
        AlphaFold prediction data including confidence scores.
    """
    return get_client().call_tool(
        "get_alphafold_info_by_protein_symbol",
        {"protein_symbol": protein_symbol, "species": species},
    )


@register_function(
    aliases=["ensembl_query", "gene_id_convert", "基因ID转换"],
    category="biocontext",
    description="Convert gene symbol to Ensembl gene ID.",
    examples=["eid = ov.utils.biocontext.get_ensembl_id('TP53')"],
    related=["biocontext.get_uniprot_id"],
)
def get_ensembl_id(gene_symbol: str, species: str = "homo_sapiens") -> Any:
    r"""Convert gene symbol to Ensembl ID.

    Parameters
    ----------
    gene_symbol : str
        Gene symbol (e.g. ``'TP53'``).
    species : str
        Species name. Default ``'homo_sapiens'``.

    Returns
    -------
    str or dict
        Ensembl gene ID.
    """
    return get_client().call_tool(
        "get_ensembl_id_from_gene_symbol",
        {"gene_symbol": gene_symbol, "species": species},
    )


@register_function(
    aliases=["interpro_query", "protein_domains", "蛋白质结构域"],
    category="biocontext",
    description="Query InterPro for protein domain and family information.",
    examples=[
        "domains = ov.utils.biocontext.query_interpro(protein_id='P04637')",
        "results = ov.utils.biocontext.search_interpro('kinase')",
    ],
    related=["biocontext.query_uniprot", "biocontext.query_alphafold"],
)
def query_interpro(
    protein_id: str,
    source_db: Optional[str] = None,
    include_structure_info: bool = False,
) -> dict:
    r"""Query InterPro for protein domains.

    Parameters
    ----------
    protein_id : str
        UniProt accession (e.g. ``'P04637'``).
    source_db : str, optional
        Filter by source database (e.g. ``'pfam'``).
    include_structure_info : bool
        Include 3D structure information.

    Returns
    -------
    dict
        Domain and family annotations.
    """
    args: Dict[str, Any] = {"protein_id": protein_id}
    if source_db:
        args["source_db"] = source_db
    args["include_structure_info"] = include_structure_info
    return get_client().call_tool("get_protein_domains", args)


@register_function(
    aliases=["interpro_search", "domain_search"],
    category="biocontext",
    description="Search InterPro entries by keyword.",
    examples=["results = ov.utils.biocontext.search_interpro('kinase')"],
    related=["biocontext.query_interpro"],
)
def search_interpro(query: str, entry_type: Optional[str] = None, page_size: int = 10) -> dict:
    r"""Search InterPro entries.

    Parameters
    ----------
    query : str
        Search term (e.g. ``'kinase'``).
    entry_type : str, optional
        Filter by type (``'domain'``, ``'family'``, ``'homologous_superfamily'``).
    page_size : int
        Number of results. Default 10.

    Returns
    -------
    dict
        Search results.
    """
    args: Dict[str, Any] = {"query": query, "page_size": page_size}
    if entry_type:
        args["entry_type"] = entry_type
    return get_client().call_tool("search_interpro_entries", args)


@register_function(
    aliases=["string_query", "ppi_biocontext", "蛋白质互作"],
    category="biocontext",
    description="Query STRING database for protein-protein interactions via BioContext.",
    examples=["ppi = ov.utils.biocontext.query_string('TP53', species=9606)"],
    related=["bulk.string_interaction", "biocontext.query_uniprot"],
)
def query_string(
    protein_symbol: str,
    species: int = 9606,
    min_score: Optional[float] = None,
) -> dict:
    r"""Query STRING for protein-protein interactions.

    Parameters
    ----------
    protein_symbol : str
        Protein symbol.
    species : int
        NCBI taxonomy ID. Default 9606 (human).
    min_score : float, optional
        Minimum interaction score (0-1).

    Returns
    -------
    dict
        STRING interaction data.
    """
    args: Dict[str, Any] = {"protein_symbol": protein_symbol, "species": str(species)}
    if min_score is not None:
        args["min_score"] = min_score
    return get_client().call_tool("get_string_interactions", args)


@register_function(
    aliases=["hpa_query", "tissue_expression", "组织表达查询"],
    category="biocontext",
    description="Query Human Protein Atlas for tissue expression data.",
    examples=["expr = ov.utils.biocontext.query_hpa('TP53')"],
    related=["biocontext.query_uniprot"],
)
def query_hpa(gene_symbol: str, gene_id: Optional[str] = None) -> dict:
    r"""Query Human Protein Atlas for tissue-level expression.

    Parameters
    ----------
    gene_symbol : str
        Gene symbol (e.g. ``'TP53'``).
    gene_id : str, optional
        Ensembl gene ID (e.g. ``'ENSG00000141510'``).
        If not provided, it is resolved automatically via ``get_ensembl_id``.

    Returns
    -------
    dict
        HPA expression data across tissues.
    """
    if gene_id is None:
        try:
            eid = get_client().call_tool(
                "get_ensembl_id_from_gene_symbol",
                {"gene_symbol": gene_symbol, "species": "homo_sapiens"},
            )
            if isinstance(eid, list) and eid:
                gene_id = eid[0].get("id", eid[0]) if isinstance(eid[0], dict) else str(eid[0])
            elif isinstance(eid, dict):
                gene_id = eid.get("id", eid.get("ensembl_id", gene_symbol))
            elif isinstance(eid, str):
                gene_id = eid
            else:
                gene_id = gene_symbol
        except Exception:
            gene_id = gene_symbol
    return get_client().call_tool(
        "get_human_protein_atlas_info",
        {"gene_id": gene_id, "gene_symbol": gene_symbol},
    )


# =====================================================================
# Pathways & Functional Annotation
# =====================================================================

@register_function(
    aliases=["reactome_query", "pathway_query", "通路查询"],
    category="biocontext",
    description="Query Reactome for pathway information by gene/protein identifier.",
    examples=["pathways = ov.utils.biocontext.query_reactome('TP53')"],
    related=["biocontext.query_go", "biocontext.query_kegg"],
)
def query_reactome(identifier: str, species: str = "Homo sapiens", include_disease: bool = True) -> dict:
    r"""Query Reactome pathway database.

    Parameters
    ----------
    identifier : str
        Gene symbol, UniProt ID, or Reactome stable ID.
    species : str
        Species name. Default ``'Homo sapiens'``.
    include_disease : bool
        Include disease-related pathways.

    Returns
    -------
    dict
        Reactome pathway analysis results.
    """
    return get_client().call_tool(
        "get_reactome_info_by_identifier",
        {"identifier": identifier, "species": species, "include_disease": include_disease},
    )


@register_function(
    aliases=["go_query", "gene_ontology", "GO查询", "基因本体"],
    category="biocontext",
    description="Query Gene Ontology terms associated with a gene.",
    examples=["go = ov.utils.biocontext.query_go('BRCA1')"],
    related=["biocontext.query_reactome"],
)
def query_go(gene_name: str, size: int = 20) -> dict:
    r"""Query Gene Ontology terms for a gene.

    Parameters
    ----------
    gene_name : str
        Gene symbol (e.g. ``'BRCA1'``).
    size : int
        Maximum number of GO terms to return.

    Returns
    -------
    dict
        GO terms (biological process, molecular function, cellular component).
    """
    return get_client().call_tool(
        "get_go_terms_by_gene", {"gene_name": gene_name, "size": size}
    )


# =====================================================================
# Single-cell & Markers
# =====================================================================

@register_function(
    aliases=["panglaodb_query", "cell_markers", "细胞标记基因", "marker_genes"],
    category="biocontext",
    description="Query PanglaoDB for cell type marker genes.",
    examples=[
        "markers = ov.utils.biocontext.query_panglaodb(species='Hs', cell_type='T cells')",
        "markers = ov.utils.biocontext.query_panglaodb(species='Hs', gene_symbol='CD3D')",
    ],
    related=["single.cellanno", "single.gptcelltype"],
)
def query_panglaodb(
    species: str = "Hs",
    cell_type: Optional[str] = None,
    gene_symbol: Optional[str] = None,
    organ: Optional[str] = None,
    min_sensitivity: Optional[float] = None,
    min_specificity: Optional[float] = None,
) -> pd.DataFrame:
    r"""Query PanglaoDB for cell type marker genes.

    Parameters
    ----------
    species : str
        ``'Hs'`` (human), ``'Mm'`` (mouse), or ``'Mm Hs'`` (both).
    cell_type : str, optional
        Cell type to query (e.g. ``'T cells'``).
    gene_symbol : str, optional
        Gene symbol to look up.
    organ : str, optional
        Organ filter (e.g. ``'Brain'``).
    min_sensitivity : float, optional
        Minimum sensitivity threshold.
    min_specificity : float, optional
        Minimum specificity threshold.

    Returns
    -------
    pd.DataFrame
        Marker gene table with sensitivity and specificity scores.
    """
    args: Dict[str, Any] = {"species": species}
    if cell_type:
        args["cell_type"] = cell_type
    if gene_symbol:
        args["gene_symbol"] = gene_symbol
    if organ:
        args["organ"] = organ
    if min_sensitivity is not None:
        args["min_sensitivity"] = min_sensitivity
    if min_specificity is not None:
        args["min_specificity"] = min_specificity
    result = get_client().call_tool("get_panglaodb_marker_genes", args)
    if isinstance(result, dict) and "markers" in result:
        return pd.DataFrame(result["markers"])
    if isinstance(result, list):
        return pd.DataFrame(result)
    return pd.DataFrame([result] if isinstance(result, dict) else [])


# =====================================================================
# Literature & Publishing
# =====================================================================

@register_function(
    aliases=["literature_search", "pubmed_search", "文献搜索"],
    category="biocontext",
    description="Search biomedical literature via Europe PMC.",
    examples=["papers = ov.utils.biocontext.search_literature('single cell RNA-seq heart')"],
    related=["biocontext.search_preprints", "biocontext.get_fulltext"],
)
def search_literature(
    query: str,
    search_type: str = "lite",
    sort_by: str = "RELEVANCE",
    page_size: int = 10,
) -> dict:
    r"""Search Europe PMC for biomedical literature.

    Parameters
    ----------
    query : str
        Search query string.
    search_type : str
        ``'lite'`` for basic metadata, ``'core'`` for full records.
    sort_by : str
        Sort order: ``'RELEVANCE'`` or ``'DATE'``.
    page_size : int
        Number of results.

    Returns
    -------
    dict
        Search results with article metadata.
    """
    return get_client().call_tool(
        "get_europepmc_articles",
        {"query": query, "search_type": search_type, "sort_by": sort_by, "page_size": page_size},
    )


@register_function(
    aliases=["preprint_search", "biorxiv_search", "预印本搜索"],
    category="biocontext",
    description="Search bioRxiv/medRxiv preprints.",
    examples=["preprints = ov.utils.biocontext.search_preprints(recent_count=10, category='bioinformatics')"],
    related=["biocontext.search_literature"],
)
def search_preprints(
    server: str = "biorxiv",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days: Optional[int] = None,
    recent_count: Optional[int] = None,
    category: Optional[str] = None,
    max_results: int = 100,
) -> dict:
    r"""Search bioRxiv or medRxiv preprints.

    Specify **one** search method: ``start_date``/``end_date`` date range,
    ``days`` for last N days, or ``recent_count`` for most recent N.

    Parameters
    ----------
    server : str
        ``'biorxiv'`` or ``'medrxiv'``.
    start_date, end_date : str, optional
        Date range in ``'YYYY-MM-DD'`` format.
    days : int, optional
        Search last N days (1–365).
    recent_count : int, optional
        Most recent N preprints (1–1000).
    category : str, optional
        Category filter (e.g. ``'bioinformatics'``, ``'cell biology'``,
        ``'genomics'``, ``'neuroscience'``).
    max_results : int
        Maximum results to return (1–500).

    Returns
    -------
    dict
        Preprint metadata.
    """
    args: Dict[str, Any] = {"server": server, "max_results": max_results}
    if start_date:
        args["start_date"] = start_date
    if end_date:
        args["end_date"] = end_date
    if days is not None:
        args["days"] = days
    if recent_count is not None:
        args["recent_count"] = recent_count
    if category:
        args["category"] = category
    return get_client().call_tool("get_recent_biorxiv_preprints", args)


@register_function(
    aliases=["fulltext_search", "全文获取"],
    category="biocontext",
    description="Get full text XML of a publication from Europe PMC.",
    examples=["text = ov.utils.biocontext.get_fulltext('PMC1234567')"],
    related=["biocontext.search_literature"],
)
def get_fulltext(pmc_id: str) -> str:
    r"""Get full text from Europe PMC.

    Parameters
    ----------
    pmc_id : str
        PMC ID (e.g. ``'PMC1234567'``).

    Returns
    -------
    str
        Full text in XML format.
    """
    return get_client().call_tool("get_europepmc_fulltext", {"pmc_id": pmc_id})


# =====================================================================
# Clinical & Drug
# =====================================================================

@register_function(
    aliases=["opentargets_query", "drug_target", "药物靶点查询"],
    category="biocontext",
    description="Query Open Targets GraphQL API for drug target information.",
    examples=["result = ov.utils.biocontext.query_opentargets('{ target(ensemblId: \"ENSG00000141510\") { approvedSymbol } }')"],
    related=["biocontext.search_drugs"],
)
def query_opentargets(query_string: str, variables: Optional[dict] = None) -> dict:
    r"""Query Open Targets via GraphQL.

    Parameters
    ----------
    query_string : str
        GraphQL query string.
    variables : dict, optional
        GraphQL variables.

    Returns
    -------
    dict
        Query results.
    """
    args: Dict[str, Any] = {"query_string": query_string}
    if variables:
        args["variables"] = variables
    return get_client().call_tool("query_open_targets_graphql", args)


@register_function(
    aliases=["clinical_trials", "临床试验搜索"],
    category="biocontext",
    description="Search ClinicalTrials.gov for clinical trials by condition.",
    examples=["trials = ov.utils.biocontext.search_clinical_trials('breast cancer')"],
    related=["biocontext.search_drugs"],
)
def search_clinical_trials(
    condition: str,
    status: Optional[str] = None,
    page_size: int = 10,
) -> dict:
    r"""Search ClinicalTrials.gov by condition.

    Parameters
    ----------
    condition : str
        Medical condition (e.g. ``'breast cancer'``).
    status : str, optional
        Trial status filter (e.g. ``'RECRUITING'``).
    page_size : int
        Number of results.

    Returns
    -------
    dict
        Clinical trial summaries.
    """
    args: Dict[str, Any] = {"condition": condition, "page_size": page_size}
    if status:
        args["status"] = status
    return get_client().call_tool("get_studies_by_condition", args)


@register_function(
    aliases=["drug_search", "fda_search", "药物搜索"],
    category="biocontext",
    description="Search FDA drug database.",
    examples=["drugs = ov.utils.biocontext.search_drugs(active_ingredient='ibuprofen')"],
    related=["biocontext.search_clinical_trials"],
)
def search_drugs(
    brand_name: Optional[str] = None,
    generic_name: Optional[str] = None,
    active_ingredient: Optional[str] = None,
    limit: int = 10,
) -> dict:
    r"""Search FDA drug database.

    Parameters
    ----------
    brand_name : str, optional
        Brand name to search.
    generic_name : str, optional
        Generic name to search.
    active_ingredient : str, optional
        Active ingredient to search.
    limit : int
        Maximum results.

    Returns
    -------
    dict
        Drug information from OpenFDA.
    """
    args: Dict[str, Any] = {"limit": limit}
    if brand_name:
        args["brand_name"] = brand_name
    if generic_name:
        args["generic_name"] = generic_name
    if active_ingredient:
        args["active_ingredient"] = active_ingredient
    return get_client().call_tool("search_drugs_fda", args)


# =====================================================================
# Ontologies
# =====================================================================

@register_function(
    aliases=["efo_query", "disease_ontology", "疾病本体查询"],
    category="biocontext",
    description="Query Experimental Factor Ontology for disease terms.",
    examples=["efo = ov.utils.biocontext.query_efo('diabetes')"],
    related=["biocontext.query_go"],
)
def query_efo(disease_name: str, size: int = 10, exact_match: bool = False) -> dict:
    r"""Query EFO for disease ontology terms.

    Parameters
    ----------
    disease_name : str
        Disease name to search.
    size : int
        Maximum results.
    exact_match : bool
        Require exact match.

    Returns
    -------
    dict
        EFO terms with IDs and descriptions.
    """
    return get_client().call_tool(
        "get_efo_id_by_disease_name",
        {"disease_name": disease_name, "size": size, "exact_match": exact_match},
    )


@register_function(
    aliases=["chebi_query", "chemical_ontology", "化合物查询"],
    category="biocontext",
    description="Query ChEBI for chemical entity information.",
    examples=["chebi = ov.utils.biocontext.query_chebi('aspirin')"],
    related=["biocontext.search_drugs"],
)
def query_chebi(chemical_name: str, size: int = 10, exact_match: bool = False) -> dict:
    r"""Query ChEBI for chemical entities.

    Parameters
    ----------
    chemical_name : str
        Chemical compound name.
    size : int
        Maximum results.
    exact_match : bool
        Require exact match.

    Returns
    -------
    dict
        ChEBI terms.
    """
    return get_client().call_tool(
        "get_chebi_terms_by_chemical",
        {"chemical_name": chemical_name, "size": size, "exact_match": exact_match},
    )


@register_function(
    aliases=["cell_ontology", "细胞本体查询"],
    category="biocontext",
    description="Query Cell Ontology for cell type terms.",
    examples=["terms = ov.utils.biocontext.query_cell_ontology('T cell')"],
    related=["biocontext.query_panglaodb"],
)
def query_cell_ontology(cell_type: str, size: int = 10) -> dict:
    r"""Query Cell Ontology for cell type terms.

    Parameters
    ----------
    cell_type : str
        Cell type name (e.g. ``'T cell'``).
    size : int
        Maximum results.

    Returns
    -------
    dict
        Cell Ontology terms.
    """
    return get_client().call_tool(
        "get_cell_ontology_terms", {"cell_type": cell_type, "size": size}
    )


# =====================================================================
# Proteomics (PRIDE)
# =====================================================================

@register_function(
    aliases=["pride_search", "proteomics_search", "蛋白质组学搜索"],
    category="biocontext",
    description="Search PRIDE proteomics repository for projects.",
    examples=["projects = ov.utils.biocontext.search_pride('single cell proteomics')"],
    related=["biocontext.query_uniprot"],
)
def search_pride(keyword: str, page_size: int = 10) -> dict:
    r"""Search PRIDE proteomics repository.

    Parameters
    ----------
    keyword : str
        Search keyword.
    page_size : int
        Maximum results.

    Returns
    -------
    dict
        PRIDE project summaries.
    """
    return get_client().call_tool(
        "search_pride_projects", {"keyword": keyword, "page_size": page_size}
    )

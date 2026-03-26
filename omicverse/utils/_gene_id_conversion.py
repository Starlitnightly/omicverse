from typing import Iterable, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData

from .._registry import register_function

try:
    from ..external.pyensembl import EnsemblRelease
except Exception:  # pragma: no cover
    EnsemblRelease = None


def _infer_species_and_release(input_names: List[str]) -> str:
    """Infer species from Ensembl ID prefix.

    Parameters
    ----------
    input_names : List[str]
        List of Ensembl IDs to sample from.

    Returns
    -------
    str
        Inferred species name. One of: 'human', 'mouse', 'rat', 'zebrafish',
        'fly', 'chicken', 'dog', 'pig', 'cow', 'macaque'. Defaults to
        ``'human'`` when the prefix is unrecognised.

    Notes
    -----
    Supported Ensembl ID prefixes:

    ========  ==========  ========================
    Prefix    Species     Scientific name
    ========  ==========  ========================
    ENSMUSG   mouse       Mus musculus
    ENSRNOG   rat         Rattus norvegicus
    ENSDARG   zebrafish   Danio rerio
    ENSGALG   chicken     Gallus gallus
    ENSCAFG   dog         Canis familiaris
    ENSSSCG   pig         Sus scrofa
    ENSBTAG   cow         Bos taurus
    ENSMMUG   macaque     Macaca mulatta
    ENSG      human       Homo sapiens
    FBGN      fly         Drosophila melanogaster
    ========  ==========  ========================

    Longer prefixes are tested first to avoid ambiguous matches (e.g.
    ``ENSGALG`` must be checked before ``ENSG``).
    """
    sample_id = next((x for x in input_names if isinstance(x, str) and len(x) > 0), None)

    if not sample_id:
        return "human"

    clean_id = sample_id.split('.')[0].upper()

    species_map = [
        ("ENSMUSG", "mouse"),
        ("ENSRNOG", "rat"),
        ("ENSDARG", "zebrafish"),
        ("ENSGALG", "chicken"),
        ("ENSCAFG", "dog"),
        ("ENSSSCG", "pig"),
        ("ENSBTAG", "cow"),
        ("ENSMMUG", "macaque"),
        ("ENSG",    "human"),
        ("FBGN",    "fly"),
    ]

    for prefix, species in species_map:
        if clean_id.startswith(prefix):
            return species

    print(f"Warning: Could not infer species from ID '{sample_id}'. Defaulting to 'human'.")
    return "human"


def _validate_database(data, species: str) -> bool:
    """Check that the local Ensembl database is indexed and not corrupted.

    Runs a single test query against a known gene ID for the given species.
    Returns ``True`` if the database is healthy, ``False`` otherwise.
    """
    test_gene_ids = {
        'human':     'ENSG00000141510',
        'mouse':     'ENSMUSG00000059552',
        'rat':       'ENSRNOG00000019450',
        'zebrafish': 'ENSDARG00000035559',
        'chicken':   'ENSGALG00000000003',
        'dog':       'ENSCAFG00000000002',
        'pig':       'ENSSSCG00000000018',
        'cow':       'ENSBTAG00000000011',
        'macaque':   'ENSMMUG00000000018',
    }
    test_id = test_gene_ids.get(species)
    if not test_id:
        return True
    try:
        gene = data.gene_by_id(test_id)
        if '"' in gene.gene_id:
            print("Database contains corrupted data (quoted gene IDs). Rebuilding...")
            return False
        return True
    except Exception:
        return False


@register_function(
    aliases=["ensembl_to_symbol", "ens2symbol", "基因ID转符号"],
    category="utils",
    description="Convert a list of Ensembl gene IDs to official gene symbols using the bundled pyensembl database.",
    prerequisites={"optional_functions": []},
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "df = ov.utils.convert2gene_symbol(['ENSG00000141510', 'ENSG00000012048'])",
        "df = ov.utils.convert2gene_symbol(list(adata.var_names), species='mouse')",
        "df = ov.utils.convert2gene_symbol(list(adata.var_names), ensembl_release=109)",
    ],
    related=["utils.convert2symbol", "utils.convert2gene_id", "utils.symbol2id"],
)
def convert2gene_symbol(
    input_names: List[str],
    scopes: Union[List[str], None] = "ensembl.gene",
    ensembl_release: Optional[int] = None,
    species: Optional[str] = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Convert Ensembl gene IDs to official gene symbols using pyensembl.

    Parameters
    ----------
    input_names : List[str]
        List of Ensembl gene IDs, optionally including version suffixes
        (e.g. ``'ENSG00000141510.12'`` is handled correctly).
    scopes : list of str or None, optional
        Kept for API compatibility with older code. Not used internally.
        Default is ``'ensembl.gene'``.
    ensembl_release : int or None, optional
        Ensembl release number (e.g. ``109``). If ``None``, defaults to
        release ``77``, which is broadly compatible with most datasets.
    species : str or None, optional
        Target species. Supported values: ``'human'``, ``'mouse'``,
        ``'rat'``, ``'zebrafish'``, ``'fly'``, ``'chicken'``, ``'dog'``,
        ``'pig'``, ``'cow'``, ``'macaque'``. If ``None``, species is
        inferred automatically from the Ensembl ID prefix.
    force_rebuild : bool, optional
        If ``True``, force re-download and re-index the local database even
        if it already exists. Useful after a failed index or suspected
        corruption. Default is ``False``.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by ``'query'`` (original Ensembl ID) with columns:

        * ``'symbol'`` — official gene symbol, or the original ID when no
          match is found.
        * ``'_score'`` — always ``1.0``, kept for downstream compatibility.

    Examples
    --------
    >>> df = ov.utils.convert2gene_symbol(list(adata.var_names))
    >>> df = ov.utils.convert2gene_symbol(
    ...     list(adata.var_names),
    ...     species='mouse',
    ...     ensembl_release=102,
    ... )
    """
    if EnsemblRelease is None:
        raise ImportError(
            "pyensembl is bundled with omicverse.external; please check your installation."
        )

    if species is None:
        species = _infer_species_and_release(input_names)
        print(f"Auto-detected species: {species}")

    if ensembl_release is None:
        ensembl_release = 77

    data = EnsemblRelease(ensembl_release, species=species)

    needs_rebuild = force_rebuild
    try:
        _ = data.db
        if not needs_rebuild and not _validate_database(data, species):
            needs_rebuild = True
    except Exception:
        needs_rebuild = True

    if needs_rebuild:
        if force_rebuild:
            print(f"Force rebuilding database for release {ensembl_release} ({species})...")
        else:
            print(f"Release {ensembl_release} ({species}) not found locally or corrupted.")
        try:
            print(f"Downloading Ensembl release {ensembl_release} ({species})... (this may take several minutes)")
            data.download()
            print(f"Download complete. Indexing database...")
            data.index(overwrite=True)
            print(f"Database ready: {data.db.local_db_path}")
        except Exception as e:
            raise ValueError(
                f"Failed to setup Ensembl DB: {e}.\n"
                f"Try running in terminal: "
                f"pyensembl install --release {ensembl_release} --species {species}"
            )

    clean_ids = [i.split(".")[0] for i in input_names]
    unique_ids = list(set(clean_ids))

    species_prefixes = {
        'human':     'ENSG',
        'mouse':     'ENSMUSG',
        'rat':       'ENSRNOG',
        'zebrafish': 'ENSDARG',
        'fly':       'FBGN',
        'chicken':   'ENSGALG',
        'dog':       'ENSCAFG',
        'pig':       'ENSSSCG',
        'cow':       'ENSBTAG',
        'macaque':   'ENSMMUG',
    }
    expected_prefix = species_prefixes.get(species)

    def _matches_species_prefix(ens_id, species, expected_prefix):
        if not expected_prefix:
            return True
        id_upper = ens_id.upper()
        if species == 'human' and id_upper.startswith('ENSGALG'):
            return False
        return id_upper.startswith(expected_prefix)

    results = {}
    found_count = 0
    for ens_id in unique_ids:
        if not _matches_species_prefix(ens_id, species, expected_prefix):
            results[ens_id] = ens_id
            continue
        symbol = ens_id
        try:
            gene_obj = data.gene_by_id(ens_id)
            gene_name = gene_obj.gene_name
            if gene_name and gene_name.strip():
                symbol = gene_name
                found_count += 1
        except ValueError:
            pass
        except Exception:
            pass
        results[ens_id] = symbol

    var_pd = pd.DataFrame.from_dict(results, orient='index', columns=['symbol'])
    var_pd.index.name = 'query'
    var_pd['_score'] = 1.0
    var_pd['converted'] = var_pd['symbol'] != var_pd.index

    print(f"Conversion finished. Found {found_count}/{len(unique_ids)} symbols.")
    return var_pd


@register_function(
    aliases=["ens2symbol_adata", "转换基因ID为符号"],
    category="utils",
    description="Convert Ensembl IDs in adata.var_names to official gene symbols in-place.",
    prerequisites={"optional_functions": []},
    requires={},
    produces={"var": ["symbol", "query", "scopes"]},
    auto_fix="none",
    examples=[
        "adata = ov.utils.convert2symbol(adata)",
        "adata = ov.utils.convert2symbol(adata, subset=False)",
        "adata = ov.utils.convert2symbol(adata, scopes='ensembl.transcript')",
    ],
    related=["utils.convert2gene_symbol", "utils.symbol2id", "utils.convert2gene_id"],
)
def convert2symbol(
    adata: AnnData,
    scopes: Union[str, Iterable, None] = None,
    subset: bool = True,
) -> AnnData:
    """Convert Ensembl IDs in ``adata.var_names`` to official gene symbols.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object whose ``var_names`` contain Ensembl gene or
        transcript IDs (e.g. ``ENSG*``, ``ENSMUSG*``, ``ENST*``).
    scopes : str, iterable, or None, optional
        Identifier type passed to :func:`convert2gene_symbol`. Common values
        are ``'ensembl.gene'`` and ``'ensembl.transcript'``. If ``None``,
        the scope is inferred from the first entry of ``adata.var_names``:

        * ``ENSG*`` / ``ENSMUSG*`` → ``'ensembl.gene'``
        * ``ENST*`` / ``ENSMUST*`` → ``'ensembl.transcript'``

    subset : bool, optional
        If ``True`` (default), genes that could not be converted are dropped
        from ``adata``. If ``False``, unconverted genes retain their original
        Ensembl ID.

    Returns
    -------
    AnnData
        Updated AnnData with ``var_names`` replaced by official gene symbols.
        The original Ensembl IDs are preserved in ``adata.var['query']``.

    Raises
    ------
    Exception
        If ``scopes`` is ``None`` and the gene IDs cannot be recognised as
        ``ensembl.gene`` or ``ensembl.transcript``.

    Examples
    --------
    >>> adata = ov.utils.convert2symbol(adata)
    >>> adata = ov.utils.convert2symbol(adata, subset=False)
    """
    if np.all(adata.var_names.str.startswith("ENS")) or scopes is not None:
        print("Converting Ensembl IDs to official gene symbols...")

        prefix = adata.var_names[0]
        if scopes is None:
            if prefix[:4] == "ENSG" or prefix[:7] == "ENSMUSG":
                scopes = "ensembl.gene"
            elif prefix[:4] == "ENST" or prefix[:7] == "ENSMUST":
                scopes = "ensembl.transcript"
            else:
                raise Exception(
                    "Your adata object uses non-official gene names as gene index.\n"
                    "omicverse finds those IDs are neither from ensembl.gene or "
                    "ensembl.transcript and thus cannot convert them automatically.\n"
                    "Please pass the correct scopes or first convert the Ensembl ID "
                    "to gene short name.\nSee also ov.utils.convert2gene_symbol"
                )

        adata.var["query"] = [i.split(".")[0] for i in adata.var.index]
        if scopes is str:
            adata.var[scopes] = adata.var.index
        else:
            adata.var["scopes"] = adata.var.index

        official_gene_df = convert2gene_symbol(adata.var_names, scopes)
        merge_df = adata.var.merge(
            official_gene_df, left_on="query", right_on="query", how="left"
        ).set_index(adata.var.index)

        if "notfound" in merge_df.columns:
            valid_ind = np.where(merge_df["notfound"] != True)[0]  # noqa: E712
            merge_df.pop("notfound")
        else:
            valid_ind = np.arange(len(merge_df))

        merge_df['converted'] = False
        merge_df.iloc[valid_ind, merge_df.columns.get_loc('converted')] = (
            merge_df.iloc[valid_ind]['symbol'] != merge_df.iloc[valid_ind]['query']
        )
        adata.var = merge_df

        if subset is True:
            adata._inplace_subset_var(valid_ind)
            adata.var.index = adata.var["symbol"].values.copy()
        else:
            indices = np.array(adata.var.index)
            indices[valid_ind] = adata.var.iloc[valid_ind]["symbol"].values.copy()
            adata.var.index = indices

        if np.sum(adata.var_names.isnull()) > 0:
            print("Subsetting adata and removing NaN columns when converting gene names.")
            adata._inplace_subset_var(adata.var_names.notnull())

    return adata


#: Alias for :func:`convert2symbol`.
id2symbol = convert2symbol


@register_function(
    aliases=["symbol_to_ensembl", "symbol2ensembl", "基因符号转ID"],
    category="utils",
    description="Convert a list of gene symbols to Ensembl gene IDs using the bundled pyensembl database.",
    prerequisites={"optional_functions": []},
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "df = ov.utils.convert2gene_id(['TP53', 'GAPDH', 'BRCA1'])",
        "df = ov.utils.convert2gene_id(['Trp53', 'Gapdh'], species='mouse')",
        "df = ov.utils.convert2gene_id(['TP53'], multi='all')",
    ],
    related=["utils.symbol2id", "utils.convert2gene_symbol", "utils.convert2symbol"],
)
def convert2gene_id(
    input_names: List[str],
    species: Optional[str] = None,
    ensembl_release: Optional[int] = None,
    force_rebuild: bool = False,
    multi: Literal["first", "all", "join"] = "first",
) -> pd.DataFrame:
    """Convert official gene symbols to Ensembl gene IDs using pyensembl.

    Parameters
    ----------
    input_names : List[str]
        List of official gene symbols (e.g. ``['TP53', 'GAPDH', 'BRCA1']``).
    species : str or None, optional
        Target species. Supported values: ``'human'``, ``'mouse'``,
        ``'rat'``, ``'zebrafish'``, ``'fly'``, ``'chicken'``, ``'dog'``,
        ``'pig'``, ``'cow'``, ``'macaque'``. Defaults to ``'human'`` when
        ``None``.
    ensembl_release : int or None, optional
        Ensembl release number. Defaults to ``77`` when ``None``.
    force_rebuild : bool, optional
        Force re-download and re-index the local database. Default is
        ``False``.
    multi : {'first', 'all', 'join'}, optional
        Strategy when a symbol maps to multiple Ensembl IDs (e.g. due to
        gene duplication):

        * ``'first'`` — return only the first ID (default).
        * ``'all'``   — return a Python list of all IDs.
        * ``'join'``  — return all IDs concatenated with ``'|'``.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by ``'query'`` (original symbol) with column:

        * ``'gene_id'`` — Ensembl gene ID, or the original symbol when no
          match is found.

    Examples
    --------
    >>> df = ov.utils.convert2gene_id(['TP53', 'GAPDH'])
    >>> df = ov.utils.convert2gene_id(
    ...     ['Trp53', 'Gapdh'],
    ...     species='mouse',
    ...     ensembl_release=102,
    ... )
    >>> df = ov.utils.convert2gene_id(['TP53'], multi='join')
    """
    if EnsemblRelease is None:
        raise ImportError(
            "pyensembl is bundled with omicverse.external; please check your installation."
        )

    if species is None:
        species = "human"
        print("No species specified, defaulting to 'human'.")

    if ensembl_release is None:
        ensembl_release = 77

    data = EnsemblRelease(ensembl_release, species=species)

    needs_rebuild = force_rebuild
    try:
        _ = data.db
        if not needs_rebuild and not _validate_database(data, species):
            needs_rebuild = True
    except Exception:
        needs_rebuild = True

    if needs_rebuild:
        if force_rebuild:
            print(f"Force rebuilding database for release {ensembl_release} ({species})...")
        else:
            print(f"Release {ensembl_release} ({species}) not found locally or not indexed.")
        try:
            print(f"Downloading Ensembl release {ensembl_release} ({species})... (this may take several minutes)")
            data.download()
            print(f"Download complete. Indexing database...")
            data.index(overwrite=True)
            print(f"Database ready: {data.db.local_db_path}")
        except Exception as e:
            raise ValueError(
                f"Failed to setup Ensembl DB: {e}.\n"
                f"Try running in terminal: "
                f"pyensembl install --release {ensembl_release} --species {species}"
            )

    unique_symbols = list(set(input_names))
    results = {}
    found_count = 0

    for symbol in unique_symbols:
        try:
            gene_ids = data.gene_ids_of_gene_name(symbol)
            if gene_ids:
                found_count += 1
                if multi == "first":
                    results[symbol] = gene_ids[0]
                elif multi == "all":
                    results[symbol] = gene_ids
                else:
                    results[symbol] = "|".join(gene_ids)
            else:
                results[symbol] = symbol
        except (ValueError, Exception):
            results[symbol] = symbol

    var_pd = pd.Series(results, name="gene_id").to_frame()
    var_pd.index.name = "query"
    var_pd['converted'] = var_pd['gene_id'] != var_pd.index

    print(f"Conversion finished. Found {found_count}/{len(unique_symbols)} Ensembl IDs.")
    return var_pd


@register_function(
    aliases=["symbol_to_id_adata", "转换基因符号为ID"],
    category="utils",
    description="Convert gene symbols in adata.var_names to Ensembl gene IDs in-place.",
    prerequisites={"optional_functions": []},
    requires={},
    produces={"var": ["gene_id", "symbol"]},
    auto_fix="none",
    examples=[
        "adata = ov.utils.symbol2id(adata, species='human')",
        "adata = ov.utils.symbol2id(adata, species='mouse', subset=True)",
        "adata = ov.utils.symbol2id(adata, multi='join')",
    ],
    related=["utils.convert2gene_id", "utils.convert2symbol", "utils.convert2gene_symbol"],
)
def symbol2id(
    adata: AnnData,
    species: Optional[str] = None,
    ensembl_release: Optional[int] = None,
    force_rebuild: bool = False,
    multi: Literal["first", "all", "join"] = "first",
    subset: bool = False,
) -> AnnData:
    """Convert gene symbols in ``adata.var_names`` to Ensembl gene IDs.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object whose ``var_names`` are official gene symbols.
    species : str or None, optional
        Target species. Supported values: ``'human'``, ``'mouse'``,
        ``'rat'``, ``'zebrafish'``, ``'fly'``, ``'chicken'``, ``'dog'``,
        ``'pig'``, ``'cow'``, ``'macaque'``. Defaults to ``'human'``.
    ensembl_release : int or None, optional
        Ensembl release number. Defaults to ``77`` when ``None``.
    force_rebuild : bool, optional
        Force re-download and re-index the local database. Default is
        ``False``.
    multi : {'first', 'all', 'join'}, optional
        Strategy when a symbol maps to multiple Ensembl IDs:

        * ``'first'`` — use only the first ID (default).
        * ``'all'``   — store a list of all IDs in ``adata.var['gene_id']``.
        * ``'join'``  — store all IDs joined by ``'|'``.

    subset : bool, optional
        If ``True``, drop genes that could not be converted. If ``False``
        (default), unconverted genes keep their original symbol as the index.

    Returns
    -------
    AnnData
        Updated AnnData with ``var_names`` replaced by Ensembl gene IDs.
        Original symbols are preserved in ``adata.var['symbol']``.

    Examples
    --------
    >>> adata = ov.utils.symbol2id(adata, species='human')
    >>> adata = ov.utils.symbol2id(adata, species='mouse', subset=True)
    """
    print("Converting gene symbols to Ensembl IDs...")

    id_df = convert2gene_id(
        list(adata.var_names),
        species=species,
        ensembl_release=ensembl_release,
        force_rebuild=force_rebuild,
        multi=multi,
    )

    adata.var["symbol"] = adata.var.index.tolist()

    merge_df = adata.var.merge(id_df, left_index=True, right_on="query", how="left")
    merge_df.index = adata.var.index

    converted_mask = merge_df["gene_id"] != merge_df.index
    valid_ind = np.where(converted_mask)[0]
    merge_df['converted'] = converted_mask

    adata.var = merge_df

    if subset:
        adata._inplace_subset_var(valid_ind)
        adata.var.index = adata.var["gene_id"].values.copy()
    else:
        indices = np.array(adata.var.index, dtype=object)
        indices[valid_ind] = adata.var.iloc[valid_ind]["gene_id"].values
        adata.var.index = indices

    if np.sum(adata.var_names.isnull()) > 0:
        adata._inplace_subset_var(adata.var_names.notnull())

    return adata

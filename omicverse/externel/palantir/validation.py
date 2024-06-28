from typing import Union, List, Dict
import numpy as np
import pandas as pd
import scanpy as sc


def _validate_obsm_key(ad, key, as_df=True):
    """
    Validates and retrieves the data associated with a specified key from the provided AnnData object.

    Parameters
    ----------
    ad : sc.AnnData
        The annotated data matrix from which the data is to be retrieved.
    key : str
        The key for accessing the data from the AnnData object's obsm.
    as_df : bool, optional
        If True, the data will be returned as pandas DataFrame with pseudotime as column names.
        If False, the data will be returned as numpy array.
        Default is True.

    Returns
    -------
    data : pd.DataFrame
        A DataFrame containing the data associated with the specified key.
    data_names : List[str]
        A list of column names for the DataFrame.

    Raises
    ------
    KeyError
        If the key or its corresponding columns are not found in the AnnData object.
    """
    if key not in ad.obsm:
        raise KeyError(f"{key} not found in ad.obsm")
    data = ad.obsm[key]
    if not isinstance(data, pd.DataFrame):
        if key + "_columns" not in ad.uns:
            raise KeyError(
                f"{key}_columns not found in ad.uns and ad.obsm[key] is not a DataFrame."
            )
        data_names = list(ad.uns[key + "_columns"])
        if as_df:
            data = pd.DataFrame(data, columns=data_names, index=ad.obs_names)
    else:
        data_names = list(data.columns)
        if not as_df:
            data = data.values
    return data, data_names


def _validate_varm_key(ad, key, as_df=True):
    """
    Validates and retrieves the data associated with a specified key from the provided AnnData object's varm attribute.

    Parameters
    ----------
    ad : sc.AnnData
        The annotated data matrix from which the data is to be retrieved.
    key : str
        The key for accessing the data from the AnnData object's varm.
    as_df : bool, optional
        If True, the trends will be returned as pandas DataFrame with pseudotime as column names.
        If False, the trends will be returned as numpy array.
        Default is True.

    Returns
    -------
    data : Union[pd.DataFrame, np.ndarray]
        A DataFrame or numpy array containing the data associated with the specified key.
    data_names : np.ndarray
        A an array of pseudotimes.

    Raises
    ------
    KeyError
        If the key or its corresponding columns are not found in the AnnData object.
    """
    if key not in ad.varm:
        raise KeyError(f"{key} not found in ad.varm")
    data = ad.varm[key]
    if not isinstance(data, pd.DataFrame):
        if key + "_pseudotime" not in ad.uns:
            raise KeyError(
                f"{key}_pseudotime not found in ad.uns and ad.varm[key] is not a DataFrame."
            )
        data_names = np.array(ad.uns[key + "_pseudotime"])
        if as_df:
            data = pd.DataFrame(data, columns=data_names, index=ad.var_names)
    else:
        data_names = np.array(data.columns.astype(float))
        if not as_df:
            data = data.values
    return data, data_names


def _validate_gene_trend_input(
    data: Union[sc.AnnData, Dict],
    gene_trend_key: str = "gene_trends",
    branch_names: Union[str, List[str]] = "branch_masks",
) -> Dict:
    """
    Validates the input for gene trend plots, and converts it into a dictionary of gene trends.

    Parameters
    ----------
    data : Union[sc.AnnData, Dict]
        An AnnData object or a dictionary containing gene trends.
    gene_trend_key : str, optional
        Key to access gene trends in the varm of the AnnData object. Default is 'gene_trends'.
    branch_names : Union[str, List[str]], optional
        Key to retrieve branch names from the AnnData object or a list of branch names. If a string is provided,
        it is assumed to be a key in AnnData.uns. Default is 'branch_masks'.

    Returns
    -------
    gene_trends : Dict
        A dictionary containing gene trends.

    Raises
    ------
    KeyError
        If 'branch_names' is a string that is not found in .uns, or if 'gene_trend_key + "_" + branch_name'
        is not found in .varm.
    ValueError
        If 'data' is neither an AnnData object nor a dictionary.
    """
    if isinstance(data, sc.AnnData):
        if isinstance(branch_names, str):
            if branch_names in data.uns.keys():
                branch_names = data.uns[branch_names]
            elif branch_names in data.obsm.keys() and isinstance(
                data.obsm[branch_names], pd.DataFrame
            ):
                branch_names = list(data.obsm[branch_names].columns)
            elif branch_names + "_columns" in data.uns.keys():
                branch_names = data.uns[branch_names + "_columns"]
            else:
                raise KeyError(
                    f"The provided key '{branch_names}' is not found in AnnData.uns or as a DataFrame in AnnData.obsm. "
                    "Please ensure the 'branch_names' either exists in AnnData.uns or is a list of branch names."
                )

        gene_trends = dict()
        for branch in branch_names:
            trends, pt_grid = _validate_varm_key(data, gene_trend_key + "_" + branch)
            gene_trends[branch] = {"trends": trends}
    elif isinstance(data, Dict):
        gene_trends = data
    else:
        raise ValueError(
            "The input 'data' must be an instance of either AnnData object or dictionary."
        )

    return gene_trends

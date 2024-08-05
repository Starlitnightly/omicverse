import numpy as np
import scanpy as sc
import pandas as pd

import io
import anndata

def ligand_receptor_database(
    database = "CellChat",
    species = "mouse",
    heteromeric_delimiter = "_",
    signaling_type = "Secreted Signaling" # or "Cell-Cell contact" or "ECM-Receptor" or None
):
    """
    Extract ligand-receptor pairs from LR database.

    Parameters
    ----------
    database
        The name of the ligand-receptor database. Use 'CellChat' for CellChatDB [Jin2021]_ of 'CellPhoneDB_v4.0' for CellPhoneDB_v4.0 [Efremova2020]_.
    species
        The species of the ligand-receptor pairs. Choose between 'mouse' and 'human'.
    heteromeric_delimiter
        The character to separate the heteromeric units of heteromeric ligands and receptors. 
        For example, if the heteromeric receptor (TGFbR1, TGFbR2) will be represented as 'TGFbR1_TGFbR2' if this parameter is set to '_'.
    signaling_type
        The type of signaling. Choose from 'Secreted Signaling', 'Cell-Cell Contact', and 'ECM-Receptor' for CellChatDB or 'Secreted Signaling' and 'Cell-Cell Contact' for CellPhoneDB_v4.0. 
        If None, all pairs in the database are returned.

    Returns
    -------
    df_ligrec : pandas.DataFrame
        A pandas DataFrame of the LR pairs with the three columns representing the ligand, receptor, and the signaling pathway name, respectively.

    References
    ----------

    .. [Jin2021] Jin, S., Guerrero-Juarez, C. F., Zhang, L., Chang, I., Ramos, R., Kuan, C. H., ... & Nie, Q. (2021). 
        Inference and analysis of cell-cell communication using CellChat. Nature communications, 12(1), 1-20.
    .. [Efremova2020] Efremova, M., Vento-Tormo, M., Teichmann, S. A., & Vento-Tormo, R. (2020). 
        CellPhoneDB: inferring cell–cell communication from combined expression of multi-subunit ligand–receptor complexes. Nature protocols, 15(4), 1484-1506.

    """
    import pkgutil
    if database == "CellChat":
        data = pkgutil.get_data(__name__, "_data/LRdatabase/CellChat/CellChatDB.ligrec."+species+".csv")
        df_ligrec = pd.read_csv(io.BytesIO(data), index_col=0)
        if not signaling_type is None:
            df_ligrec = df_ligrec[df_ligrec.iloc[:,3] == signaling_type]
    elif database == 'CellPhoneDB_v4.0':
        data = pkgutil.get_data(__name__, "_data/LRdatabase/CellPhoneDB_v4.0/CellPhoneDBv4.0."+species+".csv")
        df_ligrec = pd.read_csv(io.BytesIO(data), index_col=0)
        if not signaling_type is None:
            df_ligrec = df_ligrec[df_ligrec.iloc[:,3] == signaling_type]
        
    return df_ligrec

def filter_lr_database(
    df_ligrec: pd.DataFrame,
    adata: anndata.AnnData,
    heteromeric: bool = True,
    heteromeric_delimiter: str = "_",
    heteromeric_rule: str = "min",
    filter_criteria: str = "min_cell_pct",
    min_cell: int = 100,
    min_cell_pct: float = 0.05
):
    """
    Filter ligand-receptor pairs.

    Parameters
    ----------
    df_ligrec
        The pandas dataframe of ligand-receptor database with three columns being ligand, receptor, and pathway name respectively.
    adata
        The AnnData object of gene expression. Unscaled data (minimum being zero) is expected.
    heteromeric
        Whether the ligands and receptors are described as heteromeric.
    heteromeric_delimiter
        If heteromeric notations are used for ligands and receptors, the character separating the heteromeric units.
    heteromeric_rule
        When  heteromeric is True, the rule to quantify the level of a heteromeric ligand or receptor. Choose from minimum ('min') and average ('ave').
    filter_criteria
        Use either cell percentage ('min_cell_pct') or cell numbers (min_cell) to filter genes.
    min_cell
        If filter_criteria is 'min_cell', the LR-pairs with both ligand and receptor detected in greater than or equal to min_cell cells are kept.
    min_cell_pct
        If filter_criteria is 'min_cell_pct', the LR-pairs with both ligand and receptor detected in greater than or equal to min_cell_pct percentage of cells are kepts.

    Returns
    -------
    df_ligrec_filtered: pd.DataFrame
        A pandas DataFrame of the filtered ligand-receptor pairs.

    """

    data_genes = set(adata.var_names)
    ncell = adata.shape[0]
    all_genes = list(adata.var_names)
    gene_ncell = np.array( (adata.X > 0).sum(axis=0) ).reshape(-1)
    gene_mean = np.array( adata.X.mean(axis=0) )
    ligrec_list = []
    genes_keep = []
    if not heteromeric:
        tmp_genes = set(df_ligrec.iloc[:,0]).union(set(df_ligrec.iloc[:,1]))
        tmp_genes = list(tmp_genes.intersection(data_genes))
        for gene in tmp_genes:
            if not gene in all_genes: continue
            if filter_criteria == 'min_cell_prc':
                if gene_ncell[all_genes.index(gene)] / ncell >= min_cell_pct:
                    genes_keep.append(gene)
            elif filter_criteria == 'min_cell':
                if gene_ncell[all_genes.index(gene)] >= min_cell:
                    genes_keep.append(gene)
    elif heteromeric:
        tmp_genes = list(set(df_ligrec.iloc[:,0]).union(set(df_ligrec.iloc[:,1])))
        for het_gene in tmp_genes:
            genes = het_gene.split(heteromeric_delimiter)
            gene_found = True
            for gene in genes:
                if not gene in all_genes:
                    gene_found = False
            if not gene_found: continue
            keep = True
            if filter_criteria == 'min_cell_pct' and heteromeric_rule == 'min':
                for gene in genes:
                    if gene_ncell[all_genes.index(gene)] / ncell < min_cell_pct:
                        keep = False
            elif filter_criteria == 'min_cell' and heteromeric_rule == 'min':
                for gene in genes:
                    if gene_ncell[all_genes.index(gene)] < min_cell:
                        keep = False
            elif heteromeric_rule == 'ave':
                ave_ncell = []
                for gene in genes:
                    ave_ncell.append( gene_ncell[all_genes.index(gene)] )
                if filter_criteria == 'min_cell_pct':
                    if np.mean(ave_ncell) / ncell < min_cell_pct:
                        keep = False
                elif filter_criteria == 'min_cell':
                    if np.mean(ave_ncell) < min_cell:
                        keep = False
            if keep:
                genes_keep.append(het_gene)
    for i in range(df_ligrec.shape[0]):
        if df_ligrec.iloc[i,0] in genes_keep and df_ligrec.iloc[i,1] in genes_keep:
            ligrec_list.append(list(df_ligrec.iloc[i,:]))
    
    return pd.DataFrame(data=ligrec_list)
            
                
                
                

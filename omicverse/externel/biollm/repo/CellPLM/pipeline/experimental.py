def ensembl_to_symbol(gene_list):
    import mygene
    mg = mygene.MyGeneInfo()
    return mg.querymany(gene_list, scopes='ensembl.gene', fields='symbol', as_dataframe=True,
                 species='human').reset_index().drop_duplicates(subset='query')['symbol'].fillna('0').tolist()

def symbol_to_ensembl(gene_list):
    import mygene
    mg = mygene.MyGeneInfo()
    return mg.querymany(gene_list, scopes='symbol', fields='ensembl.gene', as_dataframe=True,
                 species='human').reset_index().drop_duplicates(subset='query')['ensembl.gene'].fillna('0').tolist()
import pandas as pd
import scanpy as sc
from scanpy import read


import os
from scipy.io import loadmat
from datetime import datetime
def _zebrahub(foldername="./", use_velocity:bool = False):
    """Load Zebrahub data as AnnData object (2Gb). Optioally load with velocity matrices (10Gb) by setting use_velocity = True

    The data has been filtered and log-normed as follows:
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=5)
        print('after filtering', adata)
        sc.pp.normalize_total(adata, target_sum=1e4)
        adata.raw = adata  # saving the raw counts
        sc.pp.log1p(adata)

    Args:
        foldername (string): foldername (string): path to directory where you want to store the dataset (or read it from if it's already been downloaded. './' current directory is default

    Returns:
        AnnData object

    .. image:: https://github.com/ShobiStassen/VIA/blob/master/Figures/AtlasGallery/zebrahub_labeled.png?raw=true
       :width="200px"
    """
    # read files as pandas objects
    if use_velocity: data_path = foldername + "Zebrahub_data_via_obsmVelocity.h5ad"
    else: data_path = foldername + "zebrahub_data_via.h5ad"


    if not os.path.isfile(data_path):

        if use_velocity:
            print(f'{datetime.now()}\tStart downloading data... This could take a few minutes (10 Gb file)')
            data_url = "https://drive.google.com/file/d/1eOJ734HufRlz2uGTXZE7DZkQ1a5FpRGk/view?usp=drive_link"
        else:
            print(f'{datetime.now()}\tStart downloading data... This could take a few minutes (2.5 Gb file)')
            data_url = "https://drive.google.com/file/d/1Pr_-5JDJYbpaUFwP5BvDrtfEQa_kb3Zq/view?usp=drive_link"
        #wget.download(data_url, data_path)
        adata = read(data_path, backup_url=data_url, sparse=True, cache=True)
        print(f'{datetime.now()}\tFinished downloading data. Saved to {data_path}')
    else:adata=sc.read_h5ad( filename=data_path)


    #adata=sc.read_h5ad( filename=data_path)
    return adata

def _mouse_gastrulation_sala(foldername="./"):
    """Load Mouse Gastrulation 2019 Pijuan Sala data. This anndata object includes

    Args:
        foldername (string): foldername (string): path to directory where you want to store the dataset (or read it from if it's already been downloaded. './' current directory is default

    Returns:
        AnnData object

    .. image:: https://github.com/ShobiStassen/VIA/blob/master/Figures/AtlasGallery/mouseGastrSala.png?raw=true
       :width="200px"
    """
    # read files as pandas objects
    data_path = foldername + "pijuan_gastrulation_via.h5ad"


    if not os.path.isfile(data_path):
        print(f'{datetime.now()}\tStart downloading data... This could take a few minutes')
        data_url = "https://drive.google.com/file/d/1rvH04WAF97nXd0UiHfcVIdIF6sxS3QhL/view"
        data_url='https://drive.google.com/file/d/1rvH04WAF97nXd0UiHfcVIdIF6sxS3QhL/view?usp=drive_link'
        #wget.download(data_url, data_path)
        print(f'{datetime.now()}\tFinished downloading data. Saved to {data_path}')
    adata = read(data_path, backup_url=data_url, sparse=True, cache=True)
    #adata=sc.read_h5ad( filename=data_path)
    return adata

def toy_multifurcating(foldername="./"):
    """Load Toy_Multifurcating data as AnnData object

    To access obs (label) as list, use AnnData.obs['group_id'].values.tolist()

    Args:
        foldername (string): foldername (string): path to directory where you want to store the dataset './' current directory is default

    Returns:
        AnnData object

    .. image:: https://github.com/ShobiStassen/VIA/blob/master/Figures/toy3_streamvia.png?raw=true
       :width="200px"
    """
    # read files as pandas objects
    data_path = foldername + "toy_multifurcating_M8_n1000d1000.csv"
    ids_path = foldername + "toy_multifurcating_M8_n1000d1000_ids_with_truetime.csv"
    import wget
    if not os.path.isfile(data_path):
        data_url = "https://raw.githubusercontent.com/ShobiStassen/VIA/master/Datasets/toy_multifurcating_M8_n1000d1000.csv"
        wget.download(data_url, data_path)

    if not os.path.isfile(ids_path):
        print(f'{datetime.now()}\tStart downloading data...')
        ids_url = "https://raw.githubusercontent.com/ShobiStassen/VIA/master/Datasets/toy_multifurcating_M8_n1000d1000_ids_with_truetime.csv"
        wget.download(ids_url, ids_path)
        print(f'{datetime.now()}\tFinished downloading data. Saved to {data_path}')

    df_counts = pd.read_csv(data_path)
    df_ids = pd.read_csv(ids_path)

    # rearrange df_ids in ascending order of cell_id
    df_ids['cell_id_num'] = [int(s[1::]) for s in df_ids['cell_id']]
    df_counts = df_counts.drop('Unnamed: 0', axis=1)
    df_ids = df_ids.sort_values(by=['cell_id_num'])
    df_ids = df_ids.reset_index(drop=True)
    true_label = df_ids[['group_id', 'true_time']]

    # create AnnData object
    adata = sc.AnnData(df_counts, obs=true_label)#, dtype='float32')
    return adata

def toy_disconnected(foldername="./"):
    """Load Toy_Disconnected data as AnnData object

    To access obs (label) as list, use AnnData.obs['group_id'].values.tolist()

    Args:
        foldername (string): Default current directory. path to directory where you want to store the dataset

    Returns:
        AnnData object

    .. image:: https://github.com/ShobiStassen/VIA/blob/master/Figures/stream_plot_toy4.png?raw=true
       :width="200px"
    """
    # read files as pandas objects
    data_path = foldername + "toy_disconnected_M9_n1000d1000.csv"
    ids_path = foldername + "toy_disconnected_M9_n1000d1000_ids_with_truetime.csv"
    import wget
    if not os.path.isfile(data_path):
        data_url = "https://raw.githubusercontent.com/ShobiStassen/VIA/master/Datasets/toy_disconnected_M9_n1000d1000.csv"
        wget.download(data_url, data_path)

    if not os.path.isfile(ids_path):
        ids_url = "https://raw.githubusercontent.com/ShobiStassen/VIA/master/Datasets/toy_disconnected_M9_n1000d1000_ids_with_truetime.csv"
        wget.download(ids_url, ids_path)

    df_counts = pd.read_csv(data_path)
    df_ids = pd.read_csv(ids_path)

    # rearrange df_ids in ascending order of cell_id
    df_ids['cell_id_num'] = [int(s[1::]) for s in df_ids['cell_id']]
    df_counts = df_counts.drop('Unnamed: 0', axis=1)
    df_ids = df_ids.sort_values(by=['cell_id_num'])
    df_ids = df_ids.reset_index(drop=True)
    true_label = df_ids[['group_id', 'true_time']]

    # create AnnData object
    adata = sc.AnnData(df_counts, obs=true_label)#, dtype='float32')
    return adata

def cell_cycle_cyto_data(foldername="./"):
    '''
    Load cell cycle imagine based flow-cyto features
    AnnData object with n_obs × n_vars = 2036 × 38
    obs: 'cell_cycle_phase'
    :param foldername (string) Default current directory. path to directory where you want to store the dataset

    :return: anndata
    '''
    import wget
    data_path = foldername + "cell_cycle_cyto.h5ad"
    if not os.path.isfile(data_path):
        ids_url = "https://raw.githubusercontent.com/ShobiStassen/VIA/master/Datasets/cell_cycle_cyto.h5ad"
        wget.download(ids_url, data_path)

    adata=sc.read_h5ad(filename=data_path)
    print(adata)
    return adata

def scRNA_hematopoiesis(foldername="./"):
    """Load scRNA seq Hematopoiesis data as AnnData object

    Args:
        foldername (string): Directory of dataset

    Returns:
        AnnData object

    .. image:: https://github.com/ShobiStassen/VIA/blob/master/Figures/humancd34_streamplot.png?raw=true
       :width="200px"

    """
    import gdown
    # read files as pandas objects
    data_path = foldername + "human_cd34_bm_rep1.h5ad"
    ids_path = foldername + "Nover_Cor_PredFine_notLogNorm.csv"
    import wget
    if not os.path.isfile(data_path):
        data_url = "https://docs.google.com/uc?id=1ZSZbMeTQQPfPBGcnfUNDNL4om98UiNcO"
        gdown.download(data_url, data_path, quiet=False)

    if not os.path.isfile(ids_path):
        ids_url = "https://raw.githubusercontent.com/ShobiStassen/VIA/master/Datasets/Nover_Cor_PredFine_notLogNorm.csv"
        wget.download(ids_url, ids_path)

    ad = sc.read(data_path)
    nover_labels = pd.read_csv(ids_path)['x'].values.tolist()

    dict_abb = {'Basophils': 'BASO1', 'CD4+ Effector Memory': 'TCEL7', 'Colony Forming Unit-Granulocytes': 'GRAN1',
                'Colony Forming Unit-Megakaryocytic': 'MEGA1', 'Colony Forming Unit-Monocytes': 'MONO1',
                'Common myeloid progenitors': "CMP", 'Early B cells': "PRE_B2", 'Eosinophils': "EOS2",
                'Erythroid_CD34- CD71+ GlyA-': "ERY2", 'Erythroid_CD34- CD71+ GlyA+': "ERY3",
                'Erythroid_CD34+ CD71+ GlyA-': "ERY1", 'Erythroid_CD34- CD71lo GlyA+': 'ERY4',
                'Granulocyte/monocyte progenitors': "GMP", 'Hematopoietic stem cells_CD133+ CD34dim': "HSC1",
                'Hematopoietic stem cells_CD38- CD34+': "HSC2",
                'Mature B cells class able to switch': "B_a2", 'Mature B cells class switched': "B_a4",
                'Mature NK cells_CD56- CD16- CD3-': "Nka3", 'Monocytes': "MONO2",
                'Megakaryocyte/erythroid progenitors': "MEP", 'Myeloid Dendritic Cells': 'mDC (cDC)',
                'Naïve B cells': "B_a1",
                'Plasmacytoid Dendritic Cells': "pDC", 'Pro B cells': 'PRE_B3'}
    # NOTE: Myeloid DCs are now called Conventional Dendritic Cells cDCs

    nover_labels = [dict_abb[i] for i in nover_labels]
    for i in list(set(nover_labels)):
        print('Cell type', i, 'has ', nover_labels.count(i), 'cells')
    # tsnem = ad.obsm['tsne']
    true_label = nover_labels
    # create AnnData object
    ad.obs['label'] = [i for i in nover_labels]
    #adata = sc.AnnData(ad.X)
    #adata.obs['label'] = true_label
    #adata.obs_names = ad.obs_names
    #adata.var_names = ad.var_names
    return ad


def scATAC_hematopoiesis(foldername="./"):
    """Load scATAC seq Hematopoiesis data as AnnData object

    Args:
        foldername (string): Directory of dataset

    Returns:
        AnnData object
    """
    # read files as pandas objects
    data_path = foldername + "scATAC_hemato_Buenrostro.csv"
    import wget
    if not os.path.isfile(data_path):
        data_url = "https://raw.githubusercontent.com/ShobiStassen/VIA/master/Datasets/scATAC_hemato_Buenrostro.csv"
        wget.download(data_url, data_path)

    df = pd.read_csv(data_path)
    print('number cells', df.shape[0])

    cell_types = ['GMP', 'HSC', 'MEP', 'CLP', 'CMP', 'LMuPP', 'MPP', 'pDC', 'mono', 'UNK']
    cell_annot = df['cellname'].values

    true_label = []
    found_annot = False
    for annot in cell_annot:
        for cell_type_i in cell_types:
            if cell_type_i in annot:
                true_label.append(cell_type_i)
                found_annot = True

        if found_annot == False:
            true_label.append('unknown')
        found_annot = False

    PCcol = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']

    X_in = df[PCcol].values

    # create AnnData object
    adata = sc.AnnData(X_in)
    adata.obs['cell_type'] = true_label
    return adata


def cell_cycle(foldername="./"):
    """Load cell cycle data as AnnData object

    Args:
        foldername (string): Directory of dataset

    Returns:
        AnnData object
    .. image:: https://github.com/ShobiStassen/VIA/blob/master/Figures/mb231_overall_300dpi.png?raw=true
       :width="200px"
    """
    # read files as pandas objects
    data_path = foldername + "mcf7_38features.csv"
    ids_path = foldername + "mcf7_phases.csv"
    import wget
    if not os.path.isfile(data_path):
        data_url = "https://raw.githubusercontent.com/ShobiStassen/VIA/master/Datasets/mcf7_38features.csv"
        wget.download(data_url, data_path)

    if not os.path.isfile(ids_path):
        ids_url = "https://raw.githubusercontent.com/ShobiStassen/VIA/master/Datasets/mcf7_phases.csv"
        wget.download(ids_url, ids_path)

    df = pd.read_csv(data_path)
    df = df.drop('Unnamed: 0', 1)

    true_label = pd.read_csv(ids_path)
    true_label = list(true_label['phase'].values.flatten())
    print('There are ', len(true_label), 'MCF7 cells and ', df.shape[1], 'features')

    adata = sc.AnnData(df)
    adata.obs["phase"] = true_label
    adata.var_names = df.columns
    return adata


def embryoid_body(foldername="./"):
    """Load embryoid body data as AnnData object

    Args:
        foldername (string): Directory to save dataset

    Returns:
        AnnData object
    """
    import gdown
    # read files as pandas objects
    data_path = foldername + "EBdata.mat"
    emb_path = foldername + "EB_phate_embedding.csv"
    import wget
    if not os.path.isfile(data_path):
        data_url = "https://docs.google.com/uc?id=1yz3zR1KAmghjYB_nLLUZoIlKN9Ew4RHf"
        gdown.download(data_url, data_path, quiet=False)

    if not os.path.isfile(emb_path):
        emb_url = "https://raw.githubusercontent.com/ShobiStassen/VIA/master/Datasets/EB_phate_embedding.csv"
        wget.download(emb_url, emb_path)

    annots = loadmat(data_path)
    data = annots[
        'data'].toarray()  # has been filtered but not yet normed (by library size) nor other subsequent pre-processing steps
    time_labels = annots['cells'].flatten().tolist()
    time_labels = ['Day ' + str(i) for i in time_labels]
    adata = sc.AnnData(data)

    # Load in Phate embedding (can also use Umap/tsne embedding if desired)
    Y_phate = pd.read_csv(emb_path)
    Y_phate = Y_phate.values

    # construct AnnData object
    gene_names = []
    gene_names_raw = annots['EBgenes_name']
    for i in gene_names_raw:
        gene_names.append(i[0][0])
    adata.var_names = gene_names
    adata.obs['time'] = time_labels  # ['Day '+str(i) for i in time_labels]
    return adata

def moffitt_preoptic(foldername="./"):
    """Load preoptic hypothalamus mouse data from moffitt et al.,m as AnnData object

    Args:
        foldername (string): foldername (string): path to directory where you want to store the dataset './' current directory is default

    Returns:
        AnnData object

    .. image:: https://github.com/ShobiStassen/VIA/blob/master/Figures/Bregma29_tissue.png?raw=true
       :width="200px"
    """
    # read files as pandas objects
    data_path = foldername + "anndata_moffit.h5ad"

    data_url="https://ndownloader.figshare.com/files/28169379"
    #data_url = 'https://github.com/ShobiStassen/VIA/blob/2cb4085c4a660f0410c4d8725a4322818387e19d/Datasets/anndata_moffit.h5ad' #same file as in figshare. using github url doesnt work for h5ad
    adata = sc.read(filename=data_path,backup_url=data_url)
    #adata = sc.read_h5ad(data_path) #
    return adata

def zesta(foldername="./"):
    '''

    :return:
    '''

    # read files as pandas objects
    data_path = foldername + "anndata_moffit.h5ad"

    data_url = 'https://figshare.com/s/191076ef460ac933071e'
    adata = sc.read(filename=data_path, backup_url=data_url)

    return adata
# scRNA processing workflow

## Usage

**We provide two demo cases in the `demo.ipynb`.**

### Step0. Download Dataset
Use `down.sh` to download the raw data files. The demo usage is in the `demo.sh`:
```
bash demo.sh
```

### Step1. load data and uniform the gene name
```
from scRNA_workflow import *

sc.settings.figdir='./figures_new/' # set figure folder

path = 'your/scRNA/counts/file'

#adata = read_from_csv(path) # read from csv file
adata = read_from_10x_mtx(path) # read from 10x h5 file

```

### Step2. Uniform the gene name
```
X_df= pd.DataFrame(sparse.csr_matrix.toarray(adata.X),index=adata.obs.index.tolist(),columns=adata.var.index.tolist()) # read from csv file
gene_list_df = pd.read_csv('./OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
gene_list = list(gene_list_df['gene_name'])
X_df, to_fill_columns, var = main_gene_selection(X_df, gene_list)
adata_uni = sc.AnnData(X_df)
adata_uni.obs = adata.obs
adata_uni.uns = adata.uns
```

### step3. Quality control and save adata with h5ad
```
adata_uni = BasicFilter(adata_uni,qc_min_genes=200,qc_min_cells=0) # filter cell and gene by lower limit
adata_uni = QC_Metrics_info(adata_uni)
save_path = '/where/file/store/demo.h5ad'
save_adata_h5ad(adata,save_path)
```
**Prerequisites**: Ensure you have obtained the API token for scFoundation/xTrimoGene from https://api.biomap.com/xTrimoGene/apply

### How to Use the scFoundation/xTrimoGene API:
#### Overview

Our scFoundation/xTrimoGene API is designed to process gene expression data, either from single cells or in bulk, and return the corresponding cell or gene context embeddings.

**Performance Estimates**:
- **Cell Embedding**:
  - For single-cell data: Approximately 0.5-1 minute for 1,000 cells.
  - For bulk data: Approximately 3-4 minutes for 100 bulk data.
  
- **Gene Embedding**:
  - For single-cell data: Approximately 2-3 minutes for 1,000 cells.
  - For bulk data: Approximately 4 minutes for 100 bulk data.

**Usage Limitations**:
To ensure a broad user experience and maintain optimal performance, the API has a maximum call duration of 10 minutes. Based on this, we recommend adhering to the following data thresholds per API call:
- **Single-Cell Embedding**: Up to 10,000 single cells.
- **Single-Cell Gene Embedding**: Up to 2,000 single cells.
- **Bulk Embedding**: Up to 100 bulk data.
- **Bulk Gene Embedding**: Up to 100 bulk data.
  
**Required Package**:
To run the provided Python script, ensure you have the following packages installed:
```
requests
argparse
numpy
pandas
json
os
scipy
```

#### 1. Convert Gene Symbol: (If you use the demo data, skip this step)
- Convert the gene symbol in your data to match our list `OS_scRNA_gene_index.19264.tsv`.
- For Python users, you can use the `main_gene_selection` function in `client.py`:
  ```python
  # X_df represents your single cell data with cells in rows and genes in columns
  gene_list_df = pd.read_csv('../OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
  gene_list = list(gene_list_df['gene_name'])
  X_df, to_fill_columns, var = main_gene_selection(X_df, gene_list)
  ```
- Save your data `X_df` in either `npy` or `csv` format.

#### 2. Inference:
- We offer three demos for inferring single cell, bulk, and gene embeddings in `example.sh`. If you're using Windows, copy the Python command from the script, replace the `${}` variables, and paste it into your command prompt or terminal to execute.
- Example for cell embedding inference:
  ```bash
  ### Cell embedding
  taskname=Baron
  tgthighres=a5
  mkdir -p ./pert/${taskname}/${tgthighres}
  python ./client.py --input_type singlecell --output_type cell --pool_type all --pre_normalized F --version 0.2 --tgthighres $tgthighres --data_path ./data/baron_human_samp_19264_fromsaver_demo.csv --save_path ./pert/${taskname}/${tgthighres}/
  ```
  
**Troubleshooting Tips:**
1. Ensure your data is formatted according to our gene symbol list.
2. If errors persist, try reducing the number of cells processed in a single API call.
  

#### 3. Adjusting Arguments:
For a quick start, simply modify the `--data_path` to point to your data.
Below are detailed descriptions for each argument:

- **input_type**: Specifies the type of input. 
  - Choices: `singlecell`, `bulk`
  - Default: `singlecell`

- **output_type**: Determines the type of output.
  - Choices: `cell`, `gene`, `gene_batch`
  - Default: `cell`
  - Note: In `cell` mode, The output shape is (N,h), where N is the number of cells, h is the hidden dimension. In `gene*` mode, The output shape is (N,19264,h),where N is the number of cells,19264 is the gene number, and h is the hidden dimension. In `gene` mode, gene embedding of each cell is processed individually. In `gene_batch` mode, all cells in your data are treated as a single batch and processed together. Ensure the number of input cells doesn't exceed 5 in this mode.

- **pool_type**: Defines the pooling types for cell embedding.
  - Choices: `all`, `max`
  - Default: `all`
  - The method of getting cell embeddings. Applicable only when `output_type` is set to `cell`.

- **tgthighres**: Sets the value of token T. 
  - Default: `t4`
  - Note: Can be set in three ways - targeted high resolution which means T=number (starting with 't'), fold change of high resolution which means T/S=number (starting with 'f'), or addition of high resolution which means T=S+number (starting with 'a'). Only valid when `input_type` is `singlecell`.

- **pre_normalized**: Controls the computation method for the S token.
  - Choices: `F`, `T`, `A`
  - Default: `F`
  - Note: When `input_type` is `singlecell`, `T` or `F` indicates if the input gene expression data is already normalized+log1p. `A` means data is normalized+log1p with the total count appended at the end, resulting in a data shape of N*19265. This mode is used for the GEARS task. For `bulk` input type, `F` means the T and S token values are log10(sum of gene expression), while `T` means they are the sum without log transformation. This is useful for bulk data with few sequenced genes.

- **version**: Model versions for generating cell embeddings.
  - Default: `0.1`
  - Note: Use `0.2` for read depth enhancement and `0.1` otherwise. Only valid when `output_type` is `cell`.

- **data_path**: Path to the input data.
  - Default: `./`

- **save_path**: Path to save the output.
  - Default: `./`

- **url**: Endpoint for the xTrimoGene model inference.
  - do not change this.

- **token**: User's API token.
  - Your token here

Use the provided arguments as needed to customize your API calls.
"""
Script for Evaluating a Single AnnData

Parameters:
----------
- `adata_path` (str):
    Full path to the AnnData you want to embed.
- `dir` (str):
    Working folder where all files will be saved.
- `species` (str):
    Species of the AnnData.
- `filter` (bool):
    Additional gene/cell filtering on the AnnData.
- `skip` (bool):
    Skip datasets that appear to have already been created.
- `model_loc` (str):
    Location of pretrained UCE model's weights in a `.torch` file.
- `batch_size` (int):
    Batch size for processing.
- `CXG` (bool):
    Use CXG model.
- `nlayers` (int):
    Number of transformer layers.
- `output_dim` (int):
    Desired output dimension.
- `d_hid` (int):
    Hidden dimension for processing.
- `token_dim` (int):
    Token dimension.
- `spec_chrom_csv_path` (str):
    CSV file mapping genes from each species to their respective chromosomes
    and genomic start positions.
- `token_file` (str):
    `.torch` file containing token/protein embeddings for all tokens.
- `protein_embeddings_dir` (str):
    Directory containing protein embedding `.pt` files for all species.
- `offset_pkl_path` (str):
    `.pkl` file mapping between species and their gene's locations in the `token_file`.
- `pad_length` (int):
    Length to pad the cell sentence to.
- `pad_token_idx` (int):
    Index of the padding token in the `token_file`.
- `chrom_token_left_idx` (int):
    Left chromosome token index
- `chrom_token_right_idx` (int):
    Right chromosome token index
- `cls_token_idx` (int):
    CLS token index in the `token_file`.
- `CHROM_TOKEN_OFFSET` (int):
    Offset index, tokens after this mark are chromosome identifiers.
- `sample_size` (int):
    Number of genes sampled for cell sentence.
- `multi_gpu` (bool):
    Run evaluation on multiple GPUs (using accelerator)    

Returns:
-------
- `dir/{dataset_name}_proc.h5ad`:
    The processed AnnData. Processing involves subsetting it to genes which
    have protein embeddings and then refiltering the dataset by minimum counts.
- `dir/{dataset_name}_chroms.pkl`:
    File mapping the genes in the dataset to their corresponding chromosome
    indices.
- `dir/{dataset_name}_counts.npz`:
    File containing the counts of the AnnData in an easily accessible format.
- `dir/{dataset_name}_shapes_dict.pkl`:
    File containing the shape (ncell x ngene) of the AnnData, used to read the
    `.npz` file.
- `dir/{dataset_name}_pe_idx.torch`:
    File mapping between the genes in the dataset and their index in the tokens file.
- `dir/{dataset_name}_starts.pkl`:
    File mapping between the genes in the dataset and their genomic start locations.

"""


import argparse
from .evaluate import AnndataProcessor
from accelerate import Accelerator

def main(args, accelerator):
    processor = AnndataProcessor(args, accelerator)
    processor.preprocess_anndata()
    processor.generate_idxs()
    processor.run_evaluation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Embed a single anndata using UCE.')

    # Anndata Processing Arguments
    parser.add_argument('--adata_path', type=str,
                        default=None,
                        help='Full path to the anndata you want to embed.')
    parser.add_argument('--dir', type=str,
                        default="./",
                        help='Working folder where all files will be saved.')
    parser.add_argument('--species', type=str, default="human",
                        help='Species of the anndata.')
    parser.add_argument('--filter', type=bool, default=True,
                        help='Additional gene/cell filtering on the anndata.')
    parser.add_argument('--skip', type=bool, default=True,
                        help='Skip datasets that appear to have already been created.')

    # Model Arguments
    parser.add_argument('--model_loc', type=str,
                        default=None,
                        help='Location of the model.')
    parser.add_argument('--batch_size', type=int, default=25,
                        help='Batch size.')
    parser.add_argument('--pad_length', type=int, default=1536,
                        help='Batch size.')
    parser.add_argument("--pad_token_idx", type=int, default=0,
                        help="PAD token index")
    parser.add_argument("--chrom_token_left_idx", type=int, default=1,
                        help="Chrom token left index")
    parser.add_argument("--chrom_token_right_idx", type=int, default=2,
                        help="Chrom token right index")
    parser.add_argument("--cls_token_idx", type=int, default=3,
                        help="CLS token index")
    parser.add_argument("--CHROM_TOKEN_OFFSET", type=int, default=143574,
                        help="Offset index, tokens after this mark are chromosome identifiers")
    parser.add_argument('--sample_size', type=int, default=1024,
                        help='Number of genes sampled for cell sentence')
    parser.add_argument('--CXG', type=bool, default=True,
                        help='Use CXG model.')
    parser.add_argument('--nlayers', type=int, default=4,
                        help='Number of transformer layers.')
    parser.add_argument('--output_dim', type=int, default=1280,
                        help='Output dimension.')
    parser.add_argument('--d_hid', type=int, default=5120,
                        help='Hidden dimension.')
    parser.add_argument('--token_dim', type=int, default=5120,
                        help='Token dimension.')
    parser.add_argument('--multi_gpu', type=bool, default=False,
                        help='Use multiple GPUs')

    # Misc Arguments
    parser.add_argument("--spec_chrom_csv_path",
                        default="./model_files/species_chrom.csv", type=str,
                        help="CSV Path for species genes to chromosomes and start locations.")
    parser.add_argument("--token_file",
                        default="./model_files/all_tokens.torch", type=str,
                        help="Path for token embeddings.")
    parser.add_argument("--protein_embeddings_dir",
                        default="./model_files/protein_embeddings/", type=str,
                        help="Directory where protein embedding .pt files are stored.")
    parser.add_argument("--offset_pkl_path",
                        default="./model_files/species_offsets.pkl", type=str,
                        help="PKL file which contains offsets for each species.")

    args = parser.parse_args()
    accelerator = Accelerator(project_dir=args.dir)
    main(args, accelerator)

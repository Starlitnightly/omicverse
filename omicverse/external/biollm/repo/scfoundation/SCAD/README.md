# Prepare Data
We processed the embeddings of all data used in the paper and saved them. You need to download the gene expression data for reproducing the baseline results from Figshare: https://doi.org/10.6084/m9.figshare.24049200 . Then you need to used the `split_data_SCAD_5fold_norm.py` to generate the split data for 5-fold cross validation. 
```
cd /data/split_norm/
## with outembedding
python split_data_SCAD_5fold_norm.py --drug Sorafenib --emb 0
## with embedding
python split_data_SCAD_5fold_norm.py --drug Sorafenib --emb 1
```
The directroy with drug name such as `split_norm/Sorafenib/` will be created. You can change the druge name `--drug` to generate other drugs' split data.

# Demo Usage
```
# cd to the SCAD folder
# Sorafenib
## without embedding
CUDA_VISIBLE_DEVICES=1 python model/SCAD_train_binarized_5folds-pub.py -e FX -d Sorafenib -g _norm -s 42 -h_dim 512 -z_dim 128 -ep 20 -la1 5 -mbS 8 -mbT 8 -emb 0

## with embedding
CUDA_VISIBLE_DEVICES=1 python model/SCAD_train_binarized_5folds-pub.py -e FX -d Sorafenib -g _norm -s 42 -h_dim 1024 -z_dim 256 -ep 80 -la1 0.2 -mbS 32 -mbT 32 -emb 1
```
Follow the command in `run.sh` for more results. It will take several minutes to train the model.

## Expected output

|Drug | Method | Single Cell AUROC |
|-----|--------|-------------------|
|Sorafenib|Baseline|0.56|
|Sorafenib|scFoundation|0.84|
|NVP-TAE684|Baseline|0.62|
|NVP-TAE684|scFoundation|0.84|
|PLX4720_451Lu|Baseline|0.38|
|PLX4720_451Lu|scFoundation|0.66|
|Etoposide|Baseline|0.66|
|Etoposide|scFoundation|0.68|

## requirements
```
scanpy
pandas
pytorch
scikit-learn
```


The `plot-publish.ipynb` is the file of reproducing results. If you want to run this notebook by your self, you can download the processed data from figshare: https://dx.doi.org/10.6084/m9.figshare.24049200 .
## Data
We now provide the processed datasets used in our experiments. You can download all h5ad files from Figshare https://doi.org/10.6084/m9.figshare.24049200 .

We now provide demo data in the `data` folder. The demo data is a subset of the Dixit dataset. 

Then you need to  download the `go.csv.zip` from https://www.dropbox.com/s/wl0uxiz5z9dbliv/go.csv.zip?dl=0 and unzip it to the `data/DATASETNAME/` folder. For instance, put it into the `data/demo` or `data/gse133344_k562gi_oe_pert227_84986_19264_withtotalcount` folder. 

## Installation
Install `PyG`, and then do `pip install cell-gears`. Actually we do not use the gears PyPI packages but installing it will install all the dependencies.

## Demo Usage
```
# cd to the GEARS folder
## no embedding
bash run_sh/run_singlecell_maeautobin-demo-baseline.sh

## using embedding from API call 
bash run_sh/run_singlecell_maeautobin-demo-emb.sh

## using embedding from local model
bash run_sh/run_singlecell_maeautobin-demo-train.sh
```

For baseline results, run `run_sh/run_singlecell_maeautobin-demo-baseline.sh`.
For scFoundation results, choose either `run_sh/run_singlecell_maeautobin-demo-emb.sh` (API based) or `run_sh/run_singlecell_maeautobin-demo-train.sh` (local model based). We now offer two approaches for obtaining embeddings:

1. API-Based Embeddings: Modify gears/model.py at Line 117 to invoke the scFoundation API for embedding retrieval. Set API parameters --output_type gene_batch and --pre_normalized A. This will yield a gene context embedding NumPy array for each training batch, shaped as batch*19264*hidden. This array should be returned for further training.
2. Local Model Embeddings: For using embeddings derived from our model, use these settings in your training bash file:
```
model_type=maeautobin
bin_set=autobin_resolution_append
finetune_method='frozen'
singlecell_model_path=../model/models/models.ckpt
```
Here you can change the "finetune_method" as other type such as "finetune_lr_1" to finetune both the scFoundation and GEARS model. Due to the GPU memory limiation, we used "forzen" in our experiments.

The output text in the terminal will be redirected into the `train.log` file.

And the results will be saved in the `results` folder.  

## Experiment datasets
All h5ad files required for the experiments are available for download from Figshare at this link https://doi.org/10.6084/m9.figshare.24049200 . After downloading, execute the corresponding training bash scripts to start your experiments. For example, to obtain baseline results on the Norman dataset, run `run_singlecell_norman.sh`. For results using scFoundation embeddings on the Norman dataset, run `run_singlecell_maeautobin-0.1B-res0-norman.sh`. Note that upon the first execution of any training script, a folder will be automatically created based on the h5ad file.

## Expected output
You will get a folder named `results/DATASETNAME/0.75/xxx` with the following files:
```
config.pkl
model.pt
params.csv
train.log
```

The `Plot.ipynb` is the jupyter notebook of reproducing the figures. 
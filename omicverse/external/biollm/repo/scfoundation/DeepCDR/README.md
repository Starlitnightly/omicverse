The commands in the `DeepCDR/prog/run.sh` can be used to train the CDR prediction model. It will take several minutes. You can check the `plot.ipynb` for reproducing the results shown in the paper. The trained model is saved in `published`.

The scFoundation embeddings of gene expression data are at `DeepCDR/data/50M-0.1B-res_embedding.npy`

## Demo Usage
```
# cd to the DeepCDR folder
mkdir log
mkdir checkpoint
cd ./prog/
## baseline model
CUDA_VISIBLE_DEVICES=0 python run_DeepCDR.py -use_gexp > ../log/Base_rep1.log 2>&1
## embedding based model
CUDA_VISIBLE_DEVICES=0 python run_DeepCDR.py --ckpt_name 50M-0.1B-res -use_gexp > ../log/50M-0.1B-res_rep1.log 2>&1

```
The terminal output will be redirected into a log file, and it will be save at `log/50M-0.1B-res_rep1.log`. The trained model will be saved at `./checkpoint/` folder.

## Expected output
For baseline model, the output will be like:`The overall Pearson's correlation is 0.8371.`

For embedding based model, the output will be like:`The overall Pearson's correlation is 0.8783.`

## Requirements
```
Keras==2.1.4
TensorFlow==1.13.1
hickle >= 2.1.0
```
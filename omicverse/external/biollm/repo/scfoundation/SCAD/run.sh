## get embedding
# CUDA_VISIBLE_DEVICES=1 python run_embedding_sc.py --ckpt_path ../0.1B-trans-pGAU-shuffle5-autobin100-mask0.3-bts1024-0226-bin100-k8s-lr1e-4-resume/models/model_step=35999.ckpt --ckpt_name 50M-0.1B-res --data_path ./data/split_norm/Target_expr_resp_19264.{durg}.csv

# CUDA_VISIBLE_DEVICES=1 python run_embedding_bulk.py --ckpt_path ../0.1B-trans-pGAU-shuffle5-autobin100-mask0.3-bts1024-0226-bin100-k8s-lr1e-4-resume/models/model_step=35999.ckpt --ckpt_name 50M-0.1B-res --data_path ./data/split_norm/Source_exprs_resp_19264.{durg}.csv


# Sorafenib
CUDA_VISIBLE_DEVICES=1 python model/SCAD_train_binarized_5folds-pub.py -e FX -d Sorafenib -g _norm -s 42 -h_dim 512 -z_dim 128 -ep 20 -la1 5 -mbS 8 -mbT 8 -emb 0

CUDA_VISIBLE_DEVICES=1 python model/SCAD_train_binarized_5folds-pub.py -e FX -d Sorafenib -g _norm -s 42 -h_dim 1024 -z_dim 256 -ep 80 -la1 0.2 -mbS 32 -mbT 32 -emb 1


# NVP-TAE684
CUDA_VISIBLE_DEVICES=1 python model/SCAD_train_binarized_5folds-pub.py -e FX -d NVP-TAE684 -g _norm -s 42 -h_dim 1024 -z_dim 128 -ep 10 -la1 2 -mbS 8 -mbT 8 -emb 0

CUDA_VISIBLE_DEVICES=1 python model/SCAD_train_binarized_5folds-pub.py -e FX -d NVP-TAE684 -g _norm -s 42 -h_dim 1024 -z_dim 256 -ep 80 -la1 0 -mbS 8 -mbT 8 -emb 1


# PLX4720 (451Lu)
CUDA_VISIBLE_DEVICES=1 python model/SCAD_train_binarized_5folds-pub.py -e FX -d PLX4720_451Lu -g _norm -s 42 -h_dim 512 -z_dim 256 -ep 100 -la1 1 -mbS 8 -mbT 8 -emb 0

CUDA_VISIBLE_DEVICES=1 python model/SCAD_train_binarized_5folds-pub.py -e FX -d PLX4720_451Lu -g _norm -s 42 -h_dim 1024 -z_dim 512 -ep 80 -la1 0 -mbS 8 -mbT 8 -emb 1


# Etoposide
CUDA_VISIBLE_DEVICES=1 python model/SCAD_train_binarized_5folds-pub.py -e FX -d Etoposide -g _norm -s 42 -h_dim 512 -z_dim 128 -ep 10 -la1 2 -mbS 8 -mbT 8 -emb 0

CUDA_VISIBLE_DEVICES=1 python model/SCAD_train_binarized_5folds-pub.py -e FX -d Etoposide -g _norm -s 42 -h_dim 1024 -z_dim 512 -ep 40 -la1 0.6 -mbS 16 -mbT 16 -emb 1

## get embedding embedding already in the data folder
# CUDA_VISIBLE_DEVICES=0 python run_pytorch_embedding.py --ckpt_path ../0.1B-trans-pGAU-shuffle5-autobin100-mask0.3-bts1024-0226-bin100-k8s-lr1e-4-resume/models/model_step=35999.ckpt --ckpt_name 50M-0.1B-res

# ## Embedding
CUDA_VISIBLE_DEVICES=0 python run_DeepCDR.py --ckpt_name 50M-0.1B-res -use_gexp > ../log/50M-0.1B-res_rep0.log 2>&1
CUDA_VISIBLE_DEVICES=0 python run_DeepCDR.py --ckpt_name 50M-0.1B-res -use_gexp > ../log/50M-0.1B-res_rep1.log 2>&1
CUDA_VISIBLE_DEVICES=0 python run_DeepCDR.py --ckpt_name 50M-0.1B-res -use_gexp > ../log/50M-0.1B-res_rep2.log 2>&1

# ## Baseline
CUDA_VISIBLE_DEVICES=0 python run_DeepCDR.py -use_gexp > ../log/Base_rep0.log 2>&1
CUDA_VISIBLE_DEVICES=0 python run_DeepCDR.py -use_gexp > ../log/Base_rep1.log 2>&1
CUDA_VISIBLE_DEVICES=0 python run_DeepCDR.py -use_gexp > ../log/Base_rep2.log 2>&1

## Leave Drug
CUDA_VISIBLE_DEVICES=0 python run_DeepCDR_leave_drug.py -use_gexp --drugname 65110 --ckpt_name 50M-0.1B-res > ../log/65110_50M-0.1B-res_rep0.log 2>&1

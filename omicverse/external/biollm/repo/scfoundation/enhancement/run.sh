# Baron
# res=1
# while [ $(echo "$res < 5.5"|bc) = 1 ]; do
#   echo "Fold = $res"
#   CUDA_VISIBLE_DEVICES=2 python run_embedding_sc.py --ckpt_path ../0.1B-trans-pGAU-shuffle5-autobin100-mask0.3-bts1024-0226-bin100-k8s-lr1e-4-resume/models/model_step\=36999.ckpt --ckpt_name 50M-0.1B-res --tgthighres $res --data_path ./SAVER/baron/baron_human_samp_19264_fromsaver.csv
#   res=`echo "$res + 0.5"|bc`
# done
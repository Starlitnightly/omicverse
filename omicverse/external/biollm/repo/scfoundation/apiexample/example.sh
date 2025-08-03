token=biomap  ## change to yours
### Cell embedding
taskname=Baron_demo
tgthighres=a5
mkdir -p ./demo/${taskname}/${tgthighres}

python ./client.py --input_type singlecell --output_type cell --pool_type all --pre_normalized F --version 0.2 --tgthighres $tgthighres --data_path ./data/baron_human_samp_19264_fromsaver_demo.csv --save_path ./demo/${taskname}/${tgthighres}/ --token ${token}

### Bulk embedding
taskname=SCAD_bulk_Etoposide_demo
tgthighres=f1
mkdir -p ./demo/${taskname}/${tgthighres}

python ./client.py --input_type bulk --output_type cell --pool_type all --pre_normalized F --version 0.1 --tgthighres $tgthighres --data_path ./data/Source_exprs_resp_19264.Etoposide_demo.csv --save_path ./demo/${taskname}/${tgthighres}/ --token ${token}

### Gene embedding
taskname=GEARS_demo_batch
tgthighres=f1
mkdir -p ./demo/${taskname}/${tgthighres}

python ./client.py --input_type singlecell --output_type gene --pool_type all --pre_normalized A --version 0.1 --tgthighres $tgthighres --data_path ./data/gene_batch.npy --save_path ./demo/${taskname}/${tgthighres}/ --token ${token}
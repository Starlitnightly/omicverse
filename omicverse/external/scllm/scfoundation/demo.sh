# Copyright 2023 BioMap (Beijing) Intelligence Technology Limited

# Enhancement
python get_embedding.py --task_name Baron --input_type singlecell --output_type cell --pool_type all --tgthighres a5 --data_path ./examples/enhancement/Baron_enhancement.csv --save_path ./examples/enhancement/ --pre_normalized F --version rde

## different foldchange change tgthighres to f1, f1.5, f2 ...
# ! python get_embedding.py --task_name Baron --input_type singlecell --output_type cell --pool_type all --tgthighres f1 --data_path ./examples/Baron_enhancement.csv --save_path ./examples/ --pre_normalized F --version rde
# ! python get_embedding.py --task_name Baron --input_type singlecell --output_type cell --pool_type all --tgthighres f1.5 --data_path ./examples/Baron_enhancement.csv --save_path ./examples/ --pre_normalized F --version rde
# ! python get_embedding.py --task_name Baron --input_type singlecell --output_type cell --pool_type all --tgthighres f2 --data_path ./examples/Baron_enhancement.csv --save_path ./examples/ --pre_normalized F --version rde

python get_embedding.py --task_name Zheng68K --input_type singlecell --output_type cell --pool_type all --tgthighres f1 --data_path ./examples/enhancement/pbmc68ksorted_count.npz --save_path ./examples/enhancement/ --pre_normalized F --version rde --demo

# DeepCDR
python get_embedding.py --task_name DeepCDR_bulkdemo --input_type bulk --output_type cell --pool_type max --tgthighres f1 --data_path ./examples/DeepCDR/normalized19264.npy --save_path ./examples/DeepCDR/ --pre_normalized T --version ce --demo

# SCAD
python get_embedding.py --task_name SCAD_bulk_Sorafenib --input_type bulk --output_type cell --pool_type all --tgthighres f1 --data_path ./examples/SCAD/Source_exprs_resp_19264.Sorafenib.csv --save_path ./examples/SCAD/ --pre_normalized F --version ce --demo

python get_embedding.py --task_name SCAD_bulk_NVP-TAE684 --input_type bulk --output_type cell --pool_type all --tgthighres f1 --data_path ./examples/SCAD/Source_exprs_resp_19264.NVP-TAE684.csv --save_path ./examples/SCAD/ --pre_normalized F --version ce --demo

python get_embedding.py --task_name SCAD_bulk_PLX4720_451Lu --input_type bulk --output_type cell --pool_type all --tgthighres f1 --data_path ./examples/SCAD/Source_exprs_resp_19264.PLX4720_451Lu.csv --save_path ./examples/SCAD/ --pre_normalized F --version ce --demo

python get_embedding.py --task_name SCAD_bulk_Etoposide --input_type bulk --output_type cell --pool_type all --tgthighres f1 --data_path ./examples/SCAD/Source_exprs_resp_19264.Etoposide.csv --save_path ./examples/SCAD/ --pre_normalized F --version ce --demo

python get_embedding.py --task_name SCAD_sc_Sorafenib --input_type singlecell --output_type cell --pool_type all --tgthighres t4 --data_path ./examples/SCAD/Target_expr_resp_19264.Sorafenib.csv --save_path ./examples/SCAD/ --pre_normalized T --version ce --demo

python get_embedding.py --task_name SCAD_sc_NVP-TAE684 --input_type singlecell --output_type cell --pool_type all --tgthighres t4 --data_path ./examples/SCAD/Target_expr_resp_19264.NVP-TAE684.csv --save_path ./examples/SCAD/ --pre_normalized T --version ce --demo

python get_embedding.py --task_name SCAD_sc_PLX4720_451Lu --input_type singlecell --output_type cell --pool_type all --tgthighres t4 --data_path ./examples/SCAD/Target_expr_resp_19264.PLX4720_451Lu.csv --save_path ./examples/SCAD/ --pre_normalized T --version ce --demo

python get_embedding.py --task_name SCAD_sc_Etoposide --input_type singlecell --output_type cell --pool_type all --tgthighres t4 --data_path ./examples/SCAD/Target_expr_resp_19264.Etoposide.csv --save_path ./examples/SCAD/ --pre_normalized T --version ce --demo


# GEARS

python get_embedding.py --task_name GEARS_demo_batch --input_type singlecell --output_type gene_batch --pool_type all --tgthighres f1 --data_path ./examples/GEARS/pre_in.npy --save_path ./examples/GEARS/ --pre_normalized A

# Genemodule
python get_embedding.py --task_name genemodule --input_type singlecell --output_type gene --pool_type all --tgthighres f1 --data_path ./examples/genemodule/zheng_subset_cd8t_b_mono.csv --save_path ./examples/genemodule/ --pre_normalized F --demo

# Mapping

python get_embedding.py --task_name mapping --input_type singlecell --output_type cell --pool_type all --tgthighres t4.5 --data_path ./examples/mapping/merged_count_19264.csv --save_path ./examples/mapping/ --pre_normalized F --version rde --demo
#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: stereo_miner_parse.py
@time: 2024/1/5 16:26
"""
import scanpy as sc
import numpy as np
from scipy.sparse import issparse
import os
import pandas as pd
import sys
from .data_utils import *



def stat_stereo_miner_data(indir, sample_df, outdir, human_list, node_rank=None, step=None):
    result = []
    all_files = os.listdir(indir)
    files = all_files if node_rank is None else all_files[node_rank*step: (node_rank + 1) * step]
    files = ["PRO000000005_EPT000000005_SCT000000005.h5ad", "PRO000000003_EPT000000003_SCT000000003.h5ad", "PRO000000356_EPT000000386_SCT000000351.h5ad", "PRO000000001_EPT000000001_SCT000000001.h5ad", "PRO000000335_EPT000000361_SCT000000324.h5ad", "PRO000000344_EPT000000372_SCT000000335.h5ad", "PRO000000006_EPT000000006_SCT000000006.h5ad", "PRO000000358_EPT000000388_SCT000000353.h5ad", "PRO000000382_EPT000000421_SCT000000386.h5ad"]

    for i in files:
        if i.split('_')[-1].split('.h5ad')[0].startswith('SCT') and i.split('_')[-1].split('.h5ad')[0] in human_list:
            try:
                human_path = os.path.join(indir, i)
                cells = get_cells_number(human_path)
                if cells > 300000:
                    organs_adata = sc.read(human_path, backed='r')
                    for j in range(0, cells, 100000):
                        sub_data = organs_adata[j: j+100000, :].to_memory()
                        res = stat_adata(sub_data, sample_df, human_path, f'{i}_{j}')
                        result.extend(res)
                else:
                    organs_adata = sc.read(human_path)
                    res = stat_adata(organs_adata, sample_df, human_path, i)
                    result.extend(res)
                raw_obs_keys = organs_adata.obs.columns
                organs_adata.obs = (pd.merge(organs_adata.obs, sample_df, on=['sample_id'], how='left')).set_index(organs_adata.obs.index)
                organs = organs_adata.obs['root_organ'].unique()
                for organ in organs:
                    adata = organs_adata[organs_adata.obs['root_organ'] == organ, :]
                    adata = cal_data_distribute(adata)
                    if not os.path.exists(os.path.join(outdir, adata.obs['root_organ'][0])):
                        os.makedirs(os.path.join(outdir, adata.obs['root_organ'][0]))
                    adata.uns['organ'] = ';'.join(adata.obs['root_organ'].unique())
                    adata.uns['platform'] = ';'.join(adata.obs['technology_company'].unique())
                    adata.uns['species'] = adata.obs['species_id'][0]
                    adata.uns['sequencer'] = ';'.join(adata.obs['sequencer_name'].unique())
                    adata.uns['dataset'] = os.path.basename(human_path)
                    adata.uns['source_dataset_id'] = adata.obs['source_sample_id'][0].split(':')[-1].split()[0] if adata.obs['source_sample_id'][0] else 'None'
                    adata.uns['path'] = os.path.join(outdir, adata.uns['organ'], i)
                    # adata.uns['path'] = os.path.join(outdir, adata.uns['organ'], f"{adata.uns['source_dataset_id']}.h5ad")
                    adata.uns['source'] = 'stereo_miner'
                    adata.obs = adata.obs[raw_obs_keys]
                    # adata.uns['celltype_values'] = adata.obs.value_counts()
                    adata.write(adata.uns['path'])
                    result.append(adata.uns)
                print(i)
            except Exception as e:
                print('error: ', i)
                print(e)
    stat_df = pd.DataFrame(result)
    if node_rank:
        stat_df.to_csv(os.path.join(outdir, f'dataset_stat_{node_rank}.csv'))
    else:
        stat_df.to_csv(os.path.join(outdir, f'dataset_stat.csv'))


def stat_adata(organs_adata, sample_df, origin_path, origin_file_name):
    raw_obs_keys = organs_adata.obs.columns
    organs_adata.obs = (pd.merge(organs_adata.obs, sample_df, on=['sample_id'], how='left')).set_index(
        organs_adata.obs.index)
    organs = organs_adata.obs['root_organ'].unique()
    result = []
    for organ in organs:
        adata = organs_adata[organs_adata.obs['root_organ'] == organ, :]
        adata = cal_data_distribute(adata)
        if not os.path.exists(os.path.join(outdir, adata.obs['root_organ'][0])):
            os.makedirs(os.path.join(outdir, adata.obs['root_organ'][0]))
        adata.uns['organ'] = ';'.join(adata.obs['root_organ'].unique())
        adata.uns['platform'] = ';'.join(adata.obs['technology_company'].unique())
        adata.uns['species'] = adata.obs['species_id'][0]
        adata.uns['sequencer'] = ';'.join(adata.obs['sequencer_name'].unique())
        adata.uns['dataset'] = os.path.basename(origin_path)
        adata.uns['source_dataset_id'] = adata.obs['source_sample_id'][0].split(':')[-1].split()[0] if \
        adata.obs['source_sample_id'][0] else 'None'
        adata.uns['path'] = os.path.join(outdir, adata.uns['organ'], origin_file_name)
        # adata.uns['path'] = os.path.join(outdir, adata.uns['organ'], f"{adata.uns['source_dataset_id']}.h5ad")
        adata.uns['source'] = 'stereo_miner'
        adata.obs = adata.obs[raw_obs_keys]
        # adata.uns['celltype_values'] = adata.obs.value_counts()
        adata.write(adata.uns['path'])
        result.append(adata.uns)
    return result


def get_cells_number(adata_path):
    adata = sc.read(adata_path, backed='r')
    cells = adata.obs.shape[0]
    adata.file.close()
    return cells


if __name__ == '__main__':
    import math
    path = '/home/share/huada/home/qiuping1/workspace/llm/data/stereo_miner_db/stomics_h5ad_file_final'
    human_list = ["SCT000000001", "SCT000000003", "SCT000000005", "SCT000000006", "SCT000000007", "SCT000000009",
                  "SCT000000010", "SCT000000011", "SCT000000013", "SCT000000014", "SCT000000016", "SCT000000017",
                  "SCT000000019", "SCT000000020", "SCT000000021", "SCT000000022", "SCT000000023", "SCT000000027",
                  "SCT000000029", "SCT000000030", "SCT000000031", "SCT000000034", "SCT000000042", "SCT000000044",
                  "SCT000000047", "SCT000000048", "SCT000000049", "SCT000000051", "SCT000000053", "SCT000000059",
                  "SCT000000062", "SCT000000063", "SCT000000064", "SCT000000065", "SCT000000067", "SCT000000068",
                  "SCT000000069", "SCT000000072", "SCT000000073", "SCT000000075", "SCT000000077", "SCT000000081",
                  "SCT000000085", "SCT000000088", "SCT000000092", "SCT000000093", "SCT000000102", "SCT000000105",
                  "SCT000000107", "SCT000000114", "SCT000000115", "SCT000000116", "SCT000000119", "SCT000000123",
                  "SCT000000124", "SCT000000129", "SCT000000131", "SCT000000132", "SCT000000135", "SCT000000139",
                  "SCT000000143", "SCT000000144", "SCT000000147", "SCT000000148", "SCT000000149", "SCT000000151",
                  "SCT000000156", "SCT000000157", "SCT000000159", "SCT000000160", "SCT000000161", "SCT000000165",
                  "SCT000000166", "SCT000000167", "SCT000000168", "SCT000000169", "SCT000000170", "SCT000000171",
                  "SCT000000173", "SCT000000174", "SCT000000175", "SCT000000176", "SCT000000180", "SCT000000182",
                  "SCT000000184", "SCT000000187", "SCT000000189", "SCT000000194", "SCT000000196", "SCT000000200",
                  "SCT000000203", "SCT000000206", "SCT000000207", "SCT000000210", "SCT000000213", "SCT000000215",
                  "SCT000000216", "SCT000000217", "SCT000000220", "SCT000000225", "SCT000000226", "SCT000000227",
                  "SCT000000229", "SCT000000239", "SCT000000240", "SCT000000247", "SPT000000007", "SPT000000008",
                  "SPT000000009", "SPT000000010", "SPT000000011", "SPT000000014", "SPT000000015", "SPT000000016",
                  "SPT000000017", "SPT000000018", "SPT000000019", "SPT000000020", "SPT000000021", "SPT000000022",
                  "SPT000000023", "SPT000000024", "SPT000000025", "SPT000000026", "SPT000000027", "SPT000000028",
                  "SPT000000029", "SPT000000030", "SPT000000031", "SPT000000032", "SCT000000253", "SCT000000254",
                  "SCT000000255", "SCT000000256", "SCT000000257", "SCT000000258", "SCT000000259", "SCT000000260",
                  "SCT000000261", "SCT000000262", "SCT000000263", "SCT000000266", "SCT000000267", "SCT000000268",
                  "SCT000000269", "SCT000000271", "SCT000000273", "SCT000000274", "SCT000000275", "SCT000000276",
                  "SCT000000277", "SCT000000278", "SCT000000279", "SCT000000280", "SCT000000281", "SCT000000282",
                  "SCT000000283", "SCT000000286", "SCT000000288", "SCT000000289", "SCT000000293", "SCT000000296",
                  "SCT000000298", "SCT000000299", "SCT000000300", "SCT000000301", "SCT000000304", "SCT000000305",
                  "SCT000000312", "SCT000000313", "SCT000000314", "SCT000000318", "SCT000000319", "SCT000000320",
                  "SCT000000321", "SCT000000323", "SCT000000324", "SCT000000327", "SCT000000328", "SCT000000329",
                  "SCT000000330", "SCT000000331", "SCT000000332", "SCT000000334", "SCT000000335", "SCT000000337",
                  "SCT000000338", "SCT000000339", "SCT000000340", "SCT000000341", "SCT000000342", "SCT000000345",
                  "SCT000000346", "SCT000000348", "SCT000000351", "SCT000000352", "SCT000000353", "SCT000000354",
                  "SCT000000355", "SCT000000356", "SCT000000359", "SCT000000361", "SCT000000362", "SCT000000363",
                  "SCT000000364", "SCT000000366", "SCT000000367", "SCT000000368", "SCT000000369", "SCT000000371",
                  "SCT000000372", "SCT000000373", "SCT000000376", "SCT000000377", "SCT000000379", "SCT000000381",
                  "SCT000000382", "SCT000000383", "SCT000000384", "SCT000000385", "SCT000000386", "SCT000000387",
                  "SCT000000388", "SCT000000391", "SCT000000392", "SCT000000399", "SCT000000402", "SCT000000403",
                  "SCT000000404", "SCT000000405", "SCT000000406", "SCT000000407", "SCT000000408", "SCT000000409",
                  "SCT000000410", "SCT000000411", "SCT000000414", "SCT000000417", "SCT000000417", "SCT000000417",
                  "SCT000000417", "SCT000000417", "SCT000000417", "SCT000000417", "SCT000000417", "SCT000000417",
                  "SCT000000417", "SCT000000417", "SCT000000417", "SCT000000417", "SCT000000417", "SCT000000417",
                  "SCT000000417", "SCT000000417", "SCT000000417", "SCT000000417", "SCT000000417", "SCT000000417",
                  "SCT000000417", "SCT000000417", "SCT000000417", "SCT000000417", "SCT000000417", "SCT000000417",
                  "SCT000000417", "SCT000000417", "SCT000000417", "SCT000000417", "SCT000000417", "SCT000000417",
                  "SCT000000417", "SCT000000417", "SCT000000417", "SCT000000417", "SCT000000417", "SPT000000199",
                  "SPT000000200", "SPT000000201", "SPT000000202", "SPT000000203", "SCT000000419", "SCT000000420",
                  "SCT000000421", "SCT000000422", "SCT000000426", "SCT000000428", "SCT000000430", "SCT000000432",
                  "SCT000000438", "SCT000000439", "SCT000000440", "SCT000000443", "SCT000000446", "SCT000000448",
                  "SCT000000449", "SCT000000454", "SCT000000455", "SCT000000456", "SCT000000460", "SCT000000462",
                  "SCT000000465", "SCT000000470", "SCT000000471", "SCT000000475", "SCT000000478", "SCT000000480",
                  "SCT000000483", "SCT000000485", "SCT000000487", "SCT000000488", "SCT000000492", "SCT000000494",
                  "SCT000000496", "SCT000000498", "SCT000000501", "SCT000000502", "SCT000000503", "SCT000000504",
                  "SCT000000505", "SCT000000506", "SCT000000508", "SCT000000510"]
    sample_df = pd.read_csv(
        '/home/share/huada/home/qiuping1/workspace/llm/data/stereo_miner_db/stereo_miner_samples.csv', index_col=0)
    outdir = '/home/share/huada/home/qiuping1/workspace/llm/data/stereo_miner_db/human'
    dsub = False
    if dsub:
        args = sys.argv[1:]
        n_nodes = int(args[0])
        node_rank = int(args[1])
        step = math.ceil(len(os.listdir(path)) / n_nodes)
        print(n_nodes, node_rank, step, len(os.listdir(path)))
        stat_stereo_miner_data(path, sample_df, outdir, human_list, node_rank, step)
    else:
        stat_stereo_miner_data(path, sample_df, outdir, human_list)

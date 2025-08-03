#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: pw_gene_graph.py
@time: 2024/3/3 15:10
"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :pw_gene_graph.py
# @Time      :2024/2/28 16:32
# @Author    :Luni Hu


from gseapy.parser import read_gmt
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations


def get_all_genes(gene_sets):
    """Get a set of all unique genes across all gene sets."""
    all_genes = set()
    for genes in gene_sets.values():
        all_genes.update(genes)
    return all_genes


def make_gene_setinfo(genes, gene_sets):
    res = []
    for gene in genes:
        print(gene)
        setinfo = {i for i in gene_sets if gene in gene_sets[i]}
        res.append((gene, setinfo))
    return res


def cal_jaccard_value(gene_a, gene_b, gene_setinfo):
    gene_sets_a = gene_setinfo[gene_a]
    gene_sets_b = gene_setinfo[gene_b]
    intersection = gene_sets_a.intersection(gene_sets_b)
    union = gene_sets_a.union(gene_sets_b)
    jaccard_index = len(intersection) / len(union) if len(union) > 0 else 0
    return jaccard_index


def main(gmt_file, output_path, n_process=10):
    """
    make the gene_set distribution for genes with multiple processes.
    """
    gene_sets = read_gmt(gmt_file)
    all_genes = list(get_all_genes(gene_sets))
    all_genes = all_genes
    print('the genes num: ', len(all_genes))
    step = int(len(all_genes) / n_process)
    setinfo = []
    with ProcessPoolExecutor(max_workers=n_process) as executor:
        futures = []
        for i in range(0, len(all_genes), step):
            genes = all_genes[i: i+step]
            futures.append(executor.submit(make_gene_setinfo, genes, gene_sets))
        for future in as_completed(futures):
            setinfo.extend(future.result())
    print('all the process is end. setinfo num: ', len(setinfo))
    with open(output_path, 'wb') as fd:
        pickle.dump(dict(setinfo), fd)


def cal_gene_pair_jacard(pkl_file, output):
    with open(pkl_file, 'rb') as pkfd, open(output, 'w') as w:
        setinfo = pickle.load(pkfd)
        for gene_pair in combinations(list(setinfo.keys()), 2):
            score = cal_jaccard_value(gene_pair[0], gene_pair[1], setinfo)
            w.write('{}\t{}\t{}\n'.format(gene_pair[0], gene_pair[1], score))


def cal_score(pkl_file, gene_a, gene_b):
    """
    cal the jacard score for a gene pair.
    """
    with open(pkl_file, 'rb') as fd:
        gene_setinfo = pickle.load(fd)
    if gene_a in gene_setinfo and gene_b in gene_setinfo:
        score = cal_jaccard_value(gene_a, gene_b, gene_setinfo)
        return score
    return -1


if __name__ == '__main__':
    gmt_file = "c5.all.v2023.2.Hs.symbols.gmt"
    output = './gene_setinfo.pkl'
    main(gmt_file, output, 60)

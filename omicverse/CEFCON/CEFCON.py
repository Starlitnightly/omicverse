import argparse
from os import fspath
from pathlib import Path
import pandas as pd

import sys
from .cell_lineage_GRN import NetModel
from .cefcon_result_object import CefconResults
from .utils import data_preparation


def main():
    parser = argparse.ArgumentParser(prog='CEFCON', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = add_main_args(parser)
    args = parser.parse_args()

    ## output dir
    p = Path(args.out_dir)
    if not p.exists():
        Path.mkdir(p)

    ## load data
    data = data_preparation(args.input_expData, args.input_priorNet, genes_DE=args.input_genesDE,
                            additional_edges_pct=args.additional_edges_pct)
    data = data['all']

    ## GRN construction
    cefcon_GRN_model = NetModel(hidden_dim=args.hidden_dim,
                                output_dim=args.output_dim,
                                heads=args.heads,
                                attention_type=args.attention,
                                miu=args.miu,
                                epochs=args.epochs,
                                repeats=args.repeats,
                                seed=args.seed,
                                cuda=args.cuda,
                                )
    cefcon_GRN_model.run(data, showProgressBar=True)
    G_predicted = cefcon_GRN_model.get_network(keep_self_loops=~args.remove_self_loops,
                                               edge_threshold_avgDegree=args.edge_threshold_param,
                                               edge_threshold_zscore=None,
                                               output_file=fspath(p / 'cell_lineage_GRN.csv'))
    node_embeddings = cefcon_GRN_model.get_gene_embedding(output_file=fspath(p / 'gene_embs.csv'))
    cefcon_results = CefconResults(adata=cefcon_GRN_model._adata,
                                   network=G_predicted,
                                   gene_embedding=node_embeddings)

    ## Driver regulators
    cefcon_results.gene_influence_score()
    cefcon_results.driver_regulators(topK=args.topK_drivers, output_file=fspath(p / 'driver_regulators.csv'))

    ## RGMs
    RGMs_results_dict = cefcon_results.RGM_activity(return_value=True)
    RGMs = pd.DataFrame([{'Driver_Regulator': r.name, 'Members': list(r.gene2weight.keys())} for r in RGMs_results_dict['RGMs']])
    RGMs.to_csv(fspath(p / 'RGMs.csv'))
    RGMs_results_dict['aucell'].to_csv(fspath(p / 'AUCell_mtx.csv'))

    print('[Done!] Please check the results in "%s/"' % args.out_dir)


def add_main_args(parser: argparse.ArgumentParser):
    # Input data
    input_parser = parser.add_argument_group(title='Input data options')
    input_parser.add_argument('--input_expData', type=str, required=True, metavar='PATH',
                              help='path to the input gene expression data')
    input_parser.add_argument('--input_priorNet', type=str, required=True, metavar='PATH',
                              help='path to the input prior gene interaction network')
    input_parser.add_argument('--input_genesDE', type=str, default=None, metavar='PATH',
                              help='path to the input gene differential expression score')
    input_parser.add_argument('--additional_edges_pct', type=float, default=0.01,
                              help='proportion of high co-expression interactions to be added')

    # GRN
    grn_parser = parser.add_argument_group(title='Cell-lineage-specific GRN construction options')
    grn_parser.add_argument('--cuda', type=int, default=0,
                            help="an integer greater than -1 indicates the GPU device number and -1 indicates the CPU device")
    grn_parser.add_argument('--seed', type=int, default=2023,
                            help="random seed (set to -1 means no random seed is assigned)")

    grn_parser.add_argument("--hidden_dim", type=int, default=128,
                            help="hidden dimension of the GNN encoder")
    grn_parser.add_argument("--output_dim", type=int, default=64,
                            help="output dimension of the GNN encoder")
    grn_parser.add_argument("--heads", type=int, default=4,
                            help="number of heads")
    grn_parser.add_argument("--attention", type=str, default='COS', choices=['COS', 'AD', 'SD'],
                            help="type of attention scoring function")
    grn_parser.add_argument('--miu', type=float, default=0.5,
                            help='parameter for considering the importance of attention coefficients of the first GNN layer')
    grn_parser.add_argument('--epochs', type=int, default=350,
                            help='number of epochs for one run')
    grn_parser.add_argument('--repeats', type=int, default=5,
                            help='number of run repeats')

    grn_parser.add_argument("--edge_threshold_param", type=int, default=8,
                            help="threshold for selecting top-weighted edges (larger values means more edges)")
    grn_parser.add_argument("--remove_self_loops", action="store_true",
                            help="remove self loops")

    # Driver regulators
    driver_parser = parser.add_argument_group(title='Driver regulator identification options')
    driver_parser.add_argument('--topK_drivers', type=int, default=100,
                               help="number of top-ranked candidate driver genes according to their influence scores")

    # Output dir
    parser.add_argument("--out_dir", type=str, required=True, default='./output',
                        help="results output path")

    return parser


if __name__ == "__main__":
    main()

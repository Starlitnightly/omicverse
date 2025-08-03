import os
import time
import argparse
import pandas as pd
import scanpy as sc
from os.path import join as pjoin

from gears import PertData, GEARS

def main(parser):
    args = parser.parse_args()

    # get data
    pert_data = PertData(args.data_dir)
    # load dataset in paper: norman, adamson, dixit.
    try:
        if args.data_name in ['norman', 'adamson', 'dixit']:
            pert_data.load(data_name = args.data_name)
        else:
            print('load data')
            pert_data.load(data_path = pjoin(args.data_dir, args.data_name))
    except:
        adata = sc.read_h5ad(pjoin(args.data_dir, args.data_name+'.h5ad'))
        adata.uns['log1p'] = {}
        adata.uns['log1p']['base'] = None
        pert_data.new_data_process(dataset_name=args.data_name, adata=adata)
        
    # specify data split
    pert_data.prepare_split(split = args.split, seed = args.seed, train_gene_set_size=args.train_gene_set_size)
    # get dataloader with batch size
    pert_data.get_dataloader(batch_size = args.batch_size, test_batch_size = args.test_batch_size)

    # set up and train a model
    gears_model = GEARS(pert_data, device = args.device)
    gears_model.model_initialize(hidden_size = args.hidden_size, 
                                 model_type = args.model_type,
                                 bin_set=args.bin_set,
                                 load_path=args.singlecell_model_path,
                                 finetune_method=args.finetune_method,
                                 accumulation_steps=args.accumulation_steps,
                                 mode=args.mode,
                                 highres=args.highres)
    gears_model.train(epochs = args.epochs, result_dir=args.result_dir,lr=args.lr)

    # save model
    gears_model.save_model(args.result_dir)

    # save params
    param_pd = pd.DataFrame(vars(args), index=['params']).T
    param_pd.to_csv(f'{args.result_dir}/params.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GEARS')

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--data_name', type=str, default='norman')
    parser.add_argument('--split', type=str, default='simulation')
    parser.add_argument('--result_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--train_gene_set_size', type=float, default=0.75)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_type', type=str, default=None)
    parser.add_argument('--bin_set', type=str, default=None)
    parser.add_argument('--singlecell_model_path', type=str, default=None)
    parser.add_argument('--finetune_method', type=str, default=None)
    parser.add_argument('--mode', type=str, default='v1')
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--highres', type=int, default=0)

    parser.add_argument('--lr', type=float, default=1e-3)
    


    main(parser)
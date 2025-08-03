import requests
import argparse
import numpy as np
import pandas as pd
import json
import os
import scipy.sparse

#####################  align gene symbol
def main_gene_selection(X_df, gene_list):
    """
    Describe:
        rebuild the input adata to unified gene symbol
    """
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    padding_df = pd.DataFrame(np.zeros((X_df.shape[0], len(to_fill_columns))), 
                              columns=to_fill_columns, 
                              index=X_df.index)
    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1), 
                        index=X_df.index, 
                        columns=list(X_df.columns) + list(padding_df.columns))
    X_df = X_df[gene_list]
    
    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns,var

# usage
# gene_list_df = pd.read_csv('../OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
# gene_list = list(gene_list_df['gene_name'])
# X_df, to_fill_columns,var = main_gene_selection(X_df,gene_list)
#####################


gateway_headers = {
    "Accept-Encoding" : "gzip,deflate,br"
     # B-Authorization
}

inner_heads = {
    "Host" : "xtrimogene.inference.example.com"
}

parser = argparse.ArgumentParser(description='get scfoundation/xTrimoGene embeddings')
parser.add_argument('--input_type', type=str, default='singlecell',choices=['singlecell','bulk'], help='the input type; default: singlecell')
parser.add_argument('--output_type', type=str, default='cell',choices=['cell','gene','gene_batch'], help='the output type; default: cell; The difference between gene and gene_batch: In gene mode, the gene embedding will be processed one by one. While in gene_batch mode, the cells in your data will be regard as one batch, and be processed together. So in this mode, the number of input cell cannot exceed 5. for GEARS task, we use gene_batch mode.')
parser.add_argument('--pool_type', type=str, default='all',choices=['all','max'], help='pooling types of cell embedding; default: all; This argument is only valid when output_type=cell')
parser.add_argument('--tgthighres', type=str, default='t4', help='Set the value of token T; It can be set as three types: the targeted high resolution (start with t) or the fold change of the high resolution (start with f), or the addtion (start with a) of the high resoultion; This argument is only valid when input_type=singlecell.')
parser.add_argument('--pre_normalized', type=str, default='F',choices=['F','T','A'], help='It controls the ways of computing S token; When input_type=singlecell: pre_normalized=T or F means the input gene expression data are already normalized+log1p or not. pre_normalized=A means gene expression data are normalized+log1p and the total count is appended to the end of the gene expression matrix. So the input shape of data will be N*19265 in this case. We use this mode for GEARS task. When input_type=bulk: pre_normalized=F means for each sample, the T and S token values will be the log10(sum of gene expression); pre_normalized=T means the token T and S will be the sum of gene expression without log transformation. It is useful when your bulk data only have 1000-2000 sequenced genes.')
parser.add_argument('--version',  type=str, default='0.1', help='model versions for generating cell embeddings. This argument is only valid when output_type=cell. For read depth enhancemnet, version=0.2; For others, version=0.1')
parser.add_argument('--data_path', type=str, default='./', help='input data path')
parser.add_argument('--save_path', type=str, default='./', help='save path')
parser.add_argument('--url', type=str, default='https://api.biomap.com/inference/xtrimogene', help='url do not change')
parser.add_argument('--token', type=str, default='', help='token of user')

args = parser.parse_args()

def main():
    if args.data_path[-3:]=='npz':
        gexpr_feature = scipy.sparse.load_npz(args.data_path)
        gexpr_feature = gexpr_feature.toarray()
    elif args.data_path[-3:]=='npy':
        gexpr_feature = np.load(args.data_path)
    else:
        gexpr_feature=pd.read_csv(args.data_path,index_col=0)
        gexpr_feature = gexpr_feature.to_numpy()
    data_numpy = np.array(gexpr_feature, dtype=np.float32)
    inputs = [
        {
            "data":[
                args.input_type
            ],
            "name":"input_type",
            "shape":[
                1
            ],
            "datatype":"BYTES"
        },
        {
            "data":[
                args.output_type
            ],
            "name":"output_type",
            "shape":[
                1
            ],
            "datatype":"BYTES"
        },
        {
            "data":[
                args.pool_type
            ],
            "name":"pool_type",
            "shape":[
                1
            ],
            "datatype":"BYTES"
        }, 
        {
            "data":[
                args.tgthighres
            ],
            "name":"tgthighres",
            "shape":[
                1
            ],
            "datatype":"BYTES"
        }, 
         {
            "data":[
                args.pre_normalized
            ],
            "name":"pre_normalized",
            "shape":[
                1
            ],
            "datatype":"BYTES"
        }, 
         {
            "data":[
                args.version
            ],
            "name":"version",
            "shape":[
                1
            ],
            "datatype":"BYTES"
        },
         {
            "data":[
                data_numpy.tolist()
            ],
            "name":"data_numpy",
            "shape": data_numpy.shape,
            "datatype":"FP32"
        }                        
    ]
    payload = {
        "id" : "test",
        "inputs" : inputs
    }
    url = args.url
    if url != "https://api.biomap.com/inference/xtrimogene":
        headers = inner_heads
    else:
        headers = gateway_headers
        if len(args.token) > 0:
            headers["B-Authorization"] = args.token 
        print(f"begin request {args} {url} {headers}")
    r = requests.post(url, data=json.dumps(payload), headers = headers)
    code = r.status_code
    if code == 504:
        print(f"code:{code} reason:{r.reason} Currently, the API service is occupied by other users. Please try it later.")
    elif code != 200:
        print(f"code:{code} text:{r.text} reason:{r.reason}")
    else:
        resp = r.json()
        shape = resp["outputs"][0]["shape"]
        print(f"request {args} success,shape:{shape}")
        result_np = np.array(resp["outputs"][0]["data"]).reshape(shape)
        url = result_np.tolist()[0]
        print(f"result: {url} {type(url)}")
        result_path = os.path.join(args.save_path, "result.npy")
        with open(result_path, "wb") as f:
            r = requests.get(url)
            f.write(r.content)

        print(f"request {args} success, save to path:{result_path} result shape:{shape}")

if __name__=='__main__':
    main()
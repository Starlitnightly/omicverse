import anndata
import pandas as pd
import numpy as np
import random
import torch


# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print("seed is fixed, seed is {}".format(seed))
set_seed()
    

from .utils import ProcessInputData
#from OmicsTweezer import model
from .model import OmicsTweezer
from .simulation import generate_simulated_data



def mian(necessary_data, real_bulk ,ot_weight=1,sep='\t', sparse=True,
                        batch_size=128, epochs=128, device=None):
    print("begin")
    if type(necessary_data) is str:
        postfix = necessary_data.split('.')[-1]
        if postfix == 'txt':
            simudata = generate_simulated_data(sc_data=necessary_data, samplenum=5000, sparse=sparse)

        elif postfix == 'h5ad':
            simudata = anndata.read_h5ad(necessary_data)

        elif postfix == 'pth':
            raise Exception('Do not accept a model as input')
        else:
            raise Exception('Please give the correct input')
    else:
        if type(necessary_data) is pd.DataFrame:
            #simudata = generate_simulated_data(sc_data=necessary_data, samplenum=5000, sparse=sparse)
            simudata = necessary_data
            print("Dataframe")
        elif type(necessary_data) is anndata.AnnData:
            #simudata = generate_simulated_data(sc_data=necessary_data, samplenum=5000, sparse=sparse)
            simudata = necessary_data
            print("Anndata")
        else:
            raise Exception('Please give the correct input')

    train_x, train_y, test_x, genename, celltypes, samplename = \
        ProcessInputData(simudata, pd.DataFrame(real_bulk.X, columns=real_bulk.var.index), 
        sep=sep, datatype='counts')
    print('training data shape is ', train_x.shape, '\ntest data shape is ', test_x.shape)

    # Auto-detect if real_bulk has ground truth
    has_ground_truth = len(real_bulk.obs.columns) > 0 and all(ct in real_bulk.obs.columns for ct in celltypes)
    target_type = "simulated" 
    print(f"Target data type: {target_type} (ground truth {'available' if has_ground_truth else 'not available'})")

    pred, groudT = test(train_x,train_y,test_x,real_bulk.obs,celltypes,target_type,ot_weight, 
                batch_size=batch_size,epochs=epochs, device=device)

    return pred, groudT

def test(train_x,train_y,test_x,test_y,celltypes,target_type,ot_weight,batch_size=128,epochs=128, device=None):
    if device is None:
        device = torch.device('cpu')
    architectures = {'m256': ([256,128,64,32],[0,0,0,0]),
                     'm512': ([512,256,128,64],[0, 0.3, 0.2, 0.1]),
                     'm1024': ([1024, 512, 256, 128],[0, 0.6, 0.3, 0.1])}
    final_p, final_g = [], []

    # Use celltypes from ProcessInputData
    cell_type_names = np.array(celltypes)

    # If test_y has no cell type columns (no ground truth), create empty DataFrame with cell type columns
    # Note: For "real" target type, the model will generate random values internally
    if len(test_y.columns) == 0 or not all(ct in test_y.columns for ct in cell_type_names):
        test_y = pd.DataFrame(np.zeros((test_x.shape[0], len(cell_type_names))),
                             columns=cell_type_names,
                             index=test_y.index if hasattr(test_y, 'index') else None)

    for key in ["m256", "m512", "m1024"]:
        model_da = OmicsTweezer(architectures[key], epochs, batch_size, target_type, 0.0001, device=device)
        train_y_df = pd.DataFrame(train_y, columns=cell_type_names)

        train, test = anndata.AnnData(train_x, obs=train_y_df), anndata.AnnData(test_x, obs=test_y)
        train.uns = {'cell_types': cell_type_names}
        test.uns = {'cell_types': cell_type_names}
        print(train,test,ot_weight)
        model_da.train(train, test,ot_weight)

        print("{} model training finished!".format(key))
        final_preds_target, ground_truth_target = model_da.prediction()
        final_p.append(final_preds_target)
        final_g.append(ground_truth_target)

    return final_p, final_g
    

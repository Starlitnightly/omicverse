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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("seed is fixed, seed is {}".format(seed))
set_seed()
    

from .utils import ProcessInputData
#from OmicsTweezer import model
from .model import OmicsTweezer
from .simulation import generate_simulated_data


def mian(necessary_data, real_bulk ,ot_weight=1,sep='\t', sparse=True,
                        batch_size=128, epochs=128):
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
        ProcessInputData(simudata, pd.DataFrame(real_bulk.X, columns=real_bulk.var.index), sep=sep, datatype='counts')
    print('training data shape is ', train_x.shape, '\ntest data shape is ', test_x.shape)
    pred, groudT = test(train_x,train_y,test_x,real_bulk.obs,ot_weight, batch_size=batch_size,epochs=epochs)
    
    return pred, groudT

def test(train_x,train_y,test_x,test_y, ot_weight,batch_size=128,epochs=128):
    architectures = {'m256': ([256,128,64,32],[0,0,0,0]),
                     'm512': ([512,256,128,64],[0, 0.3, 0.2, 0.1]),
                     'm1024': ([1024, 512, 256, 128],[0, 0.6, 0.3, 0.1])}
    final_p, final_g = [], []
    for key in ["m256", "m512", "m1024"]:
        model_da = OmicsTweezer(architectures[key], epochs, batch_size, "simulated", 0.0001)
        train_y = pd.DataFrame(train_y, columns=np.array(test_y.columns))
        
        train, test = anndata.AnnData(train_x, obs=train_y), anndata.AnnData(test_x, obs=test_y)
        train.uns = {'cell_types': np.array(test.obs.columns)}
        test.uns = {'cell_types': np.array(test.obs.columns)}
        print(train,test,ot_weight)
        model_da.train(train, test,ot_weight)
        
        print("{} model training finished!".format(key))
        final_preds_target, ground_truth_target = model_da.prediction()
        final_p.append(final_preds_target)
        final_g.append(ground_truth_target)
    
    return final_p, final_g
    


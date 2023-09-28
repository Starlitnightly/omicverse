import anndata
import pandas as pd
from .simulation import generate_simulated_data
from .utils import ProcessInputData
from .train import train_model, predict, reproducibility
from .model import scaden, AutoEncoder

def Deconvolution(necessary_data, real_bulk, sep='\t', variance_threshold=0.98,
                  scaler='mms',
                  datatype='counts', genelenfile=None, d_prior=None,
                  mode='overall', adaptive=True,
                  save_model_name=None, sparse=True,
                  batch_size=128, epochs=128, seed=0):
    """
    :param necessary_data: for single-cell data, txt file and dataframe are supported. for simulated data, file location
                           and the h5ad variable are supported. for a trained model, model location(saved with pth) and
                           the model are supported.
    :param real_bulk: an expression file path, index is sample, columns is gene name
    :param variance_threshold: value from 0 to 1. Filter out genes with variance low than this rank.
    :param scaler: using MinMaxScaler ("mms") or StandardScaler ("ss") to process data.
    :param sep: used to read bulk data, depends on the format
    :param datatype: FPKM or TPM, if bulk RNA-seq normalization type is RPKM, please just use FPKM.
    :param genelenfile: specify the location of gene length file for transforming counts data to TPM or FPKM
                        this file should in txt format and
                        contain three columns: [Gene name,Transcript start (bp),Transcript end (bp)]
    :param d_prior: prior knowledge about cell fractions, used to generate cell fractions, if this param is None, then the
                    fractions is generated as a random way.
    :param mode: 'high-resolution' means this will apply adaptive stage to every single sample to generate signature matrix,
                 'overall' means that it will deconvolve all the samples at the same time
    :param adaptive: it has to be True, if model is 'high-resolution'
    :param save_model_name: the name used to save model, if it was not provided, it would not be saved
    :return: depends on the mode or adaptive
             there are three combinations:
             1. high-resolution and adaptive deconvolution
                this will return a dictionary and predicted cell fractions in pandas dataframe format
                the keys of the dict are the pre-defined cell type names in the single cell reference data
                the values of the dict are the dataframe of gene expression and samples
             2. overall and adaptive deconvolution
                this will return a signature matrix and a cell fraction
                the rows of the signature matrix is the gene expression in each cell types
                both of the variables are in dataframe format
             3. overall and non-adaptive deconvolution
                this will return a cell fraction directly
                the signature matrix in this mode is None
    """
    if type(necessary_data) is str:
        postfix = necessary_data.split('.')[-1]
        if postfix == 'txt':
            simudata = generate_simulated_data(sc_data=necessary_data, samplenum=5000, d_prior=d_prior, sparse=sparse)

        elif postfix == 'h5ad':
            simudata = anndata.read_h5ad(necessary_data)

        elif postfix == 'pth':
            raise Exception('Do not accept a model as input')
        else:
            raise Exception('Please give the correct input')
    else:
        if type(necessary_data) is pd.DataFrame:
            simudata = generate_simulated_data(sc_data=necessary_data, samplenum=5000, d_prior=d_prior, sparse=sparse)

        elif type(necessary_data) is anndata.AnnData:
            simudata = necessary_data

        elif type(necessary_data) is AutoEncoder:
            raise Exception('Do not accept a model as input')
        else:
            raise Exception('Please give the correct input')

    train_x, train_y, test_x, genename, celltypes, samplename = \
        ProcessInputData(simudata, real_bulk, sep=sep, datatype=datatype, genelenfile=genelenfile,
                         variance_threshold=variance_threshold, scaler=scaler)
    print('training data shape is ', train_x.shape, '\ntest data shape is ', test_x.shape)
    if save_model_name is not None:
        reproducibility(seed)
        model = train_model(train_x, train_y, save_model_name, batch_size=batch_size, epochs=epochs)
    else:
        reproducibility(seed)
        model = train_model(train_x, train_y, batch_size=batch_size, epochs=epochs)
    print('Notice that you are using parameters: mode=' + str(mode) + ' and adaptive=' + str(adaptive))
    if adaptive is True:
        if mode == 'high-resolution':
            CellTypeSigm, Pred = \
                predict(test_x=test_x, genename=genename, celltypes=celltypes, samplename=samplename,
                        model=model, model_name=save_model_name,
                        adaptive=adaptive, mode=mode)
            return CellTypeSigm, Pred

        elif mode == 'overall':
            Sigm, Pred = \
                predict(test_x=test_x, genename=genename, celltypes=celltypes, samplename=samplename,
                        model=model, model_name=save_model_name,
                        adaptive=adaptive, mode=mode)
            return Sigm, Pred
    else:
        Pred = predict(test_x=test_x, genename=genename, celltypes=celltypes, samplename=samplename,
                       model=model, model_name=save_model_name,
                       adaptive=adaptive, mode=mode)
        Sigm = None
        return Sigm, Pred

def ScadenDeconvolution(necessary_data, real_bulk, sep='\t', sparse=True,
                        batch_size=128, epochs=128):
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
            simudata = generate_simulated_data(sc_data=necessary_data, samplenum=5000, sparse=sparse)

        elif type(necessary_data) is anndata.AnnData:
            simudata = necessary_data

        elif type(necessary_data) is AutoEncoder:
            raise Exception('Do not accept a model as input')
        else:
            raise Exception('Please give the correct input')

    train_x, train_y, test_x, genename, celltypes, samplename = \
        ProcessInputData(simudata, real_bulk, sep=sep, datatype='counts')
    print('training data shape is ', train_x.shape, '\ntest data shape is ', test_x.shape)
    pred = test_scaden(train_x,train_y,test_x,batch_size=batch_size,epochs=epochs)
    pred = pd.DataFrame(pred, columns=celltypes, index=samplename)
    return pred

def test_scaden(train_x,train_y,test_x,batch_size=128,epochs=128):
    architectures = {'m256': ([256,128,64,32],[0,0,0,0]),
                     'm512': ([512,256,128,64],[0, 0.3, 0.2, 0.1]),
                     'm1024': ([1024, 512, 256, 128],[0, 0.6, 0.3, 0.1])}
    model = scaden(architectures, train_x, train_y, batch_size=batch_size, epochs=epochs)
    model.train()
    pred = model.predict(test_x)
    return pred


import os
import anndata
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.sparse import issparse


#### NEEDED FILES
# 1. GeneLength.txt
def counts2FPKM(counts, genelen):
    genelen = pd.read_csv(genelen, sep=',')
    genelen['TranscriptLength'] = genelen['Transcript end (bp)'] - genelen['Transcript start (bp)']
    genelen = genelen[['Gene name', 'TranscriptLength']]
    genelen = genelen.groupby('Gene name').max()
    # intersection
    inter = counts.columns.intersection(genelen.index)
    samplename = counts.index
    counts = counts[inter].values
    genelen = genelen.loc[inter].T.values
    # transformation
    totalreads = counts.sum(axis=1)
    counts = counts * 1e9 / (genelen * totalreads.reshape(-1, 1))
    counts = pd.DataFrame(counts, columns=inter, index=samplename)
    return counts


def FPKM2TPM(fpkm):
    genename = fpkm.columns
    samplename = fpkm.index
    fpkm = fpkm.values
    total = fpkm.sum(axis=1).reshape(-1, 1)
    fpkm = fpkm * 1e6 / total
    fpkm = pd.DataFrame(fpkm, columns=genename, index=samplename)
    return fpkm


def counts2TPM(counts, genelen):
    fpkm = counts2FPKM(counts, genelen)
    tpm = FPKM2TPM(fpkm)
    return tpm


def ProcessInputData(train_data, test_data, sep=None, datatype='TPM', variance_threshold=0.98,
                     scaler="mms",
                     genelenfile=None):
    ### read train data
    print('Reading training data')
    if type(train_data) is anndata.AnnData:
        pass
    elif type(train_data) is str:
        train_data = anndata.read_h5ad(train_data)
    # train_data.var_names_make_unique()
    if issparse(train_data.X):
        train_x = pd.DataFrame(train_data.X.toarray(), columns=train_data.var.index)
    else:
        train_x = pd.DataFrame(train_data.X, columns=train_data.var.index)

    train_y = train_data.obs
    print('Reading is done')
    ### read test data
    print('Reading test data')
    if type(test_data) is str:
        test_x = pd.read_csv(test_data, index_col=0, sep=sep)
    elif type(test_data) is pd.DataFrame:
        test_x = test_data
    print('Reading test data is done')
    ### transform to datatype
    if datatype == 'FPKM':
        if genelenfile is None:
            raise Exception("Please add gene length file!")
        print('Transforming to FPKM')
        train_x = counts2FPKM(train_x, genelenfile)
    elif datatype == 'TPM':
        if genelenfile is None:
            raise Exception("Please add gene length file!")
        print('Transforming to TPM')
        train_x = counts2TPM(train_x, genelenfile)
    elif datatype == 'counts':
        print('Using counts data to train model')
    ### variance cutoff
    print('Cutting variance...')
    var_cutoff = train_x.var(axis=0).sort_values(ascending=False)[int(train_x.shape[1] * variance_threshold)]
    train_x = train_x.loc[:, train_x.var(axis=0) > var_cutoff]

    var_cutoff = test_x.var(axis=0).sort_values(ascending=False)[int(test_x.shape[1] * variance_threshold)]
    test_x = test_x.loc[:, test_x.var(axis=0) > var_cutoff]

    ### find intersected genes
    print('Finding intersected genes...')
    inter = train_x.columns.intersection(test_x.columns)
    train_x = train_x[inter]
    test_x = test_x[inter]

    genename = list(inter)
    celltypes = train_y.columns
    samplename = test_x.index

    print('Intersected gene number is ', len(inter))
    ### MinMax process
    print('Scaling...')
    train_x = np.log(train_x + 1)
    test_x = np.log(test_x + 1)

    colors = sns.color_palette('RdYlBu', 10)
    fig = plt.figure()
    sns.histplot(data=np.mean(train_x, axis=0), kde=True, color=colors[3],edgecolor=None)
    sns.histplot(data=np.mean(test_x, axis=0), kde=True, color=colors[7],edgecolor=None)
    plt.legend(title='datatype', labels=['trainingdata', 'testdata'])

    plt.show()

    if scaler=='ss':
        print("Using standard scaler...")
        ss = StandardScaler()
        ss_train_x = ss.fit_transform(train_x.T).T
        ss_test_x = ss.fit_transform(test_x.T).T
        fig = plt.figure()
        sns.histplot(data=np.mean(ss_train_x, axis=0), kde=True, color=colors[3],edgecolor=None)
        sns.histplot(data=np.mean(ss_test_x, axis=0), kde=True, color=colors[7],edgecolor=None)
        plt.legend(title='datatype', labels=['trainingdata', 'testdata'])

        plt.show()

        return ss_train_x, train_y.values, ss_test_x, genename, celltypes, samplename

    elif scaler == 'mms':
        print("Using minmax scaler...")
        mms = MinMaxScaler()
        mms_train_x = mms.fit_transform(train_x.T).T
        mms_test_x = mms.fit_transform(test_x.T).T
        sns.histplot(data=np.mean(mms_train_x, axis=0), kde=True, color=colors[3],edgecolor=None)
        sns.histplot(data=np.mean(mms_test_x, axis=0), kde=True, color=colors[7],edgecolor=None)
        plt.legend(title='datatype', labels=['trainingdata', 'testdata'])

        plt.show()

        return mms_train_x, train_y.values, mms_test_x, genename, celltypes, samplename


def L1error(pred, true):
    return np.mean(np.abs(pred - true))


def CCCscore(y_pred, y_true, mode='all'):
    # pred: shape{n sample, m cell}
    if mode == 'all':
        y_pred = y_pred.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)
    elif mode == 'avg':
        pass
    ccc_value = 0
    for i in range(y_pred.shape[1]):
        r = np.corrcoef(y_pred[:, i], y_true[:, i])[0, 1]
        # Mean
        mean_true = np.mean(y_true[:, i])
        mean_pred = np.mean(y_pred[:, i])
        # Variance
        var_true = np.var(y_true[:, i])
        var_pred = np.var(y_pred[:, i])
        # Standard deviation
        sd_true = np.std(y_true[:, i])
        sd_pred = np.std(y_pred[:, i])
        # Calculate CCC
        numerator = 2 * r * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
        ccc = numerator / denominator
        ccc_value += ccc
    return ccc_value / y_pred.shape[1]


def score(pred, label):
    print('L1 error is', L1error(pred, label))
    print('CCC is ', CCCscore(pred, label))


def showloss(loss):
    #sns.set()
    plt.plot(loss)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()


def transformation(train_x, test_x):
    sigma_2 = np.sum((train_x - np.mean(train_x, axis=0)) ** 2, axis=0) / (train_x.shape[0] + 1)
    sigma = np.sqrt(sigma_2)
    test_x = ((test_x - np.mean(test_x, axis=0)) / np.std(test_x, axis=0)) * sigma + np.mean(train_x, axis=0)
    return test_x
